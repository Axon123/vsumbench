# vsumbench

[![English](https://img.shields.io/badge/README-English-blue)](README.md)
[![简体中文](https://img.shields.io/badge/README-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-orange)](README.zh-CN.md)

**A unified experiment framework for video summarization research.**

Video summarization evaluation has been fragmented for a decade: every paper ships its own dataset loading code, its own metric implementation, and its own splits — so setting up experiments means resurrecting half-a-dozen Keras-era repositories with dead data links, and numbers between papers are often not comparable.

`vsumbench` gives your method one shared, ready-to-run substrate: canonical datasets with checksummed downloads, one explicit evaluation protocol, and a results manifest that makes every number traceable. It is the `lm-evaluation-harness` for the video summarization community — you bring the method, we provide the experiment environment:

```bash
pip install vsumbench

vsumm download tvsum
vsumm eval --model mine --dataset tvsum --splits tvsum-canonical-5
```

For your own model, all you need is a class that maps a video to per-frame importance scores:

```python
from vsumbench import Bench

bench = Bench(dataset="tvsum", splits="tvsum-canonical-5")
bench.register("mine", my_model)

bench.run()  # → per-fold F1 under the shared protocol + run manifest
```

Comparison numbers for other methods go into your table straight from their papers, as always — but now your own numbers are computed under one documented, reproducible protocol instead of yet another private variant.

## How an evaluation runs

You provide **one thing**: a model that maps a video to per-frame importance scores. The framework runs everything else:

| Stage | Who | What happens |
|---|---|---|
| Get the video | framework | `vsumm download` fetches the checksummed h5; the loader hands your model a `Video` |
| Score | **you** | `Model.predict(video) -> Scores`, frame-level |
| Select | framework | frame scores → shot scores → 0/1 knapsack under the budget → boolean summary mask |
| Measure | framework | F1 vs every user summary, τ/ρ vs `gtscore`; collapse per config, aggregate across folds |

### What your model receives

A `Video` carries the whole evaluation substrate (TVSum `video_1` shown as a concrete example):

| Field | Shape (example) | Meaning |
|---|---|---|
| `features` | `[707, 1024]` | GoogLeNet pool5 features, one row per 15-frame step |
| `picks` | `[707]` | original-frame index of each feature row |
| `shots` | 71 shots | KTS boundaries shipped with the dataset |
| `user_summaries` | 20 × `[10597]` | per-user binary frame masks (the ground truth F1 compares against) |
| `gt_score` | `[707]` | the dataset's aggregated importance (training target; reference-protocol rank GT) |
| `n_frames` | 10597 | length of the original frame space |

There are **no raw frames**: the eccv16 lineage convention — and therefore every published comparison number — is built on these precomputed features. Methods that need raw video are outside this substrate for now.

### The model contract

```python
import numpy as np
from vsumbench.models.score_space import expand_step_scores
from vsumbench.types import Scores


class MyModel:
    name = "mine"

    def fit(self, videos, config) -> None:
        # Receives the train-split Videos plus the full EvalConfig.
        # Hyperparameters live in config.training; all randomness must
        # derive from config.seed (no hidden entropy sources).
        ...

    def predict(self, video) -> Scores:
        step_scores = my_scoring(video.features)  # [n_steps] — your method
        return Scores(
            video.video_id,
            expand_step_scores(step_scores, video.picks, video.n_frames),  # [n_frames]
        )
```

Two rules: `predict` must return **frame-level** scores — step scorers expand with the shared `expand_step_scores` (the lineage forward-fill) so every model meets the protocol in the same space; and the model must be deterministic for a fixed config.

Under the hood, selection is a faithful 0/1 knapsack over shots: shot value = mean (or max) of its frame scores, weight = shot length in frames, capacity = `floor(n_frames * budget)`; the selected shots' frames become the machine summary. The pieces are public, so you can also drive them directly:

```python
from vsumbench.metrics.f1 import F1_METRIC
from vsumbench.models.score_space import expand_step_scores
from vsumbench.registry import DATASETS, SELECTORS
from vsumbench.data.download import cache_dir
from vsumbench.types import Budget, Scores
import numpy as np

video = DATASETS.get("tvsum").load(cache_dir())[0]
frame = expand_step_scores(my_scoring(video.features), video.picks, video.n_frames)
mask = SELECTORS.get("knapsack").select(
    Scores(video.video_id, frame), video.shots, Budget(0.15), aggregation="mean"
)
f1 = float(np.max(F1_METRIC.score(mask, video)))  # per-user F1s → gt_aggregation
```

### Running via the Python API

```python
from vsumbench import Bench

bench = Bench(
    dataset="tvsum",
    splits="tvsum-canonical-5",                          # the lineage standard
    metrics=("f1", "kendall_tau_gtscore", "spearman_rho_gtscore"),
    budgets=(0.15,),                                     # the lineage reports 15%
)
bench.register("mine", MyModel())                        # register + select in one call
result = bench.run()                                     # → runs/<timestamp>-tvsum-mine-knapsack/
print(result.summary["f1"]["0.15"])   # {'mean': ..., 'std': ..., 'max': ..., 'n_folds': 5}
```

Per fold the bench deep-copies your model (no cross-fold leakage), fits on the train ids, predicts the test ids, selects at each budget, computes per-user metrics and collapses them per `gt_aggregation`.

### Running via the CLI

`bench.register` is process-local. To make a model visible to `vsumm eval`, declare it in your own package — this single line is the whole integration, no vsumbench source changes:

```toml
[project.entry-points."vsumbench.models"]
mine = "my_pkg.model:MY_MODEL"   # an instance, not a class
```

```bash
vsumm download tvsum
vsumm eval --dataset tvsum --model mine --splits tvsum-canonical-5 \
           --metrics f1,kendall_tau_gtscore,spearman_rho_gtscore --budgets 0.15
vsumm report runs/               # scan manifests → markdown comparison table
```

Data lands in `$VSUM_CACHE_DIR` (default `~/.cache/vsumbench`), checksum-verified; `--data-root` points runs at an existing local copy for offline use.

### Making your numbers comparable

- `gt_aggregation` defaults to `max` (the eccv16 original-paper convention). The KaiyangZhou/VASNet-lineage code hard-codes `avg` for TVSum — switch with `--gt-aggregation mean` when comparing against those tables.
- Prefer canonical 5-fold splits and explicit budgets; both are recorded in the manifest.
- Manifests record the git SHA (plus a dirty flag), the full config and data checksums — run from a clean tree so every number answers *which code, which protocol parameters, which data*.

## Design principles

- **Faithful protocol.** The evaluation pipeline (score expansion, shot selection, budgets, F1) is transcribed from the reference implementations and verified against them on the real datasets (see `tests/alignment/`). Deviations are documented, not hidden.
- **Explicit evaluation.** Contested protocol choices — frame-selection strategy, budget, GT aggregation, splits, normalization — are explicit parameters with recommended defaults. Disagreements become configurable, not silent.
- **Minimal.** Core depends on `numpy` + `h5py` only; frameworks are optional.
- **Data never bundled.** `vsumm download` fetches datasets from their original sources with checksum verification. We distribute loaders, not data.

## Scope

| Layer | First release (v0.1) | Later |
|---|---|---|
| Datasets | TVSum, SumMe, TVSum+SumMe | OVP, VTW, QFVS |
| Metrics | F1@{10,15,30,50}% (shot-based / knapsack selection), Kendall's τ-b and Spearman's ρ vs the dataset's `gtscore` (the Zhang et al. 2016 reference protocol) | per-user τ/ρ (needs the original ydata annotations), extended protocols |
| Models | your own, via the registry (plus a no-learning `uniform` reference for smoke tests) | community ports of published methods |
| Splits | TVSum & SumMe canonical 5-fold, seed-controlled random splits | community splits |

## Status

🚧 **Early development.** The data layer (`vsumm download`), the protocol layer (selectors, F1, reference-protocol τ/ρ against `gtscore`, alignment tests) and the orchestrator (`vsumm eval` / `vsumm report`, run manifests, the `Bench` API) are implemented — see [docs/design/dev-plan.md](docs/design/dev-plan.md) for progress. The protocol archaeology for the TVSum/SumMe lineage lives in [docs/protocols](docs/protocols/). Remaining for v0.1: docs site, PyPI release.

## Contributing

New models join through the registry — see [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide: development setup, per-component extension contracts (datasets, models, metrics, selectors), testing requirements, and how to port a published method with alignment tests.

## License

[MIT](LICENSE)
