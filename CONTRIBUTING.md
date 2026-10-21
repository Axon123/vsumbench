# Contributing to vsumbench

There are two distinct ways to contribute, and most people want the first one:

1. **Run your own method** — this needs *no changes to this repository*. Write a
   model class in your own package and register it via an entry point (see the
   [README](README.md) usage section). Your package, your license, your release
   cycle.
2. **Extend the framework itself** — new datasets, metrics, selectors, protocol
   documentation, or core improvements. That is what this file is about.

The design rule behind everything: **new components = implement a Protocol +
register + tests, never modify core code paths** ([AGENTS.md](AGENTS.md) rule 3).
If a contribution forces a change in a core path to accommodate it, that is a
design discussion first, not a PR.

## Development setup

```bash
git clone https://github.com/Axon123/vsumbench.git
cd vsumbench
pip install -e ".[dev]"
pre-commit install
```

Quality gates (all three must be green before a PR):

```bash
ruff check . && ruff format --check .
mypy            # strict, configured over src/ and tests/
pytest          # unit + alignment; integration is opt-in below
pytest -m integration   # real datasets; run `vsumm download tvsum` / `summe` first
```

Also worth knowing:

- **Minimal dependencies**: core code may import `numpy`, `h5py` and the
  standard library only. Anything else belongs in an extra (rule 1).
- **Layering is one-way** (`cli/reporting/bench → selection/metrics/models →
  data → types/config`) and enforced by a test — low layers must not import
  upward (rule 2).
- **Protocol changes are explicit**: anything that affects evaluation numbers
  (aggregation, normalization, budgets, selection, splits) must be an
  `EvalConfig` parameter, never implicit logic (rule 4).
- **Data is never committed**: the repo ships loaders and checksums, not data
  (rule 6).

Interface contracts and their rationale live in
[docs/design/core-contracts.md](docs/design/core-contracts.md) — read it before
touching any Protocol or `EvalConfig`; update it in the same PR if a contract
changes.

## Adding a model (to your own package)

Implement the `Model` protocol:

```python
from vsumbench.models.score_space import expand_step_scores
from vsumbench.types import Scores


class MyModel:
    name = "mine"

    def fit(self, videos, config) -> None: ...  # train split + full EvalConfig

    def predict(self, video) -> Scores:
        step_scores = my_scoring(video.features)
        return Scores(video.video_id, expand_step_scores(step_scores, video.picks, video.n_frames))
```

Rules: `predict` returns **frame-level** scores (step scorers expand with the
shared `expand_step_scores`); the model must be deterministic for a fixed
config — all randomness derives from `config.seed`; the registry holds
**instances**, not classes. Register from your package:

```toml
[project.entry-points."vsumbench.models"]
mine = "my_pkg.model:MY_MODEL"
```

Learning models are never built into vsumbench itself — they join as
third-party packages.

## Adding a dataset (to this repo)

1. **Loader** — implement `DatasetAdapter.load(root) -> list[Video]` and
   register it under `vsumbench.datasets`. Loaders *only load*: no downloading
   (that is `vsumbench.data.download`'s job, driven by the `SOURCES` mirror
   table with sha256 specs), and no protocol processing — per-user annotations
   are carried raw; every aggregation choice belongs to `EvalConfig` (the GT
   rawness principle, core-contracts §2.2).
2. **Splits** — canonical published splits go into `vsumbench.data.splits`
   (transcribed verbatim); seed-controlled random splits work out of the box.
3. **Tests** — a synthetic-fixture unit test with zero network and zero real
   data (see `tests/unit/_synthetic.py` + `tests/unit/test_eccv16.py` for the
   pattern), plus an opt-in integration test asserting checksums and full
   loads on the real files.

## Adding a metric

Implement `SummaryMetric` (machine summary vs per-user summaries) or
`RankMetric` (predicted scores vs ground-truth importance) and register under
`vsumbench.metrics` / `vsumbench.rank_metrics`. The contract: **one value per
annotation, never aggregated inside the metric** — collapsing to the reported
number is the orchestrator's job via `EvalConfig.gt_aggregation`. Undefined
values are NaN (they propagate and are written as `null` in manifests).

Hand-computed unit test cases are mandatory (small arrays, expected values
derived by hand in the test comments) — see `tests/unit/test_metrics.py`.

## Adding a selector

Implement `Selector.select(scores, shots, budget, *, aggregation) -> Mask` and
register under `vsumbench.selectors`. The frame→segment aggregation is passed
in by the orchestrator; selectors never choose it themselves, must be
deterministic, and must honor `Budget.frame_limit` rounding exactly. Unit tests
cover the boundary cases: full and minimal budgets, single-shot videos, tied
scores, exact division (see `tests/unit/test_selection.py`).

## Porting a published method? Ship alignment tests

vsumbench does not reproduce anyone's learning method — but if you port one as
a third-party package so it runs under the shared protocol, **consider shipping
alignment tests against the original implementation**. A port can silently
diverge (a mistyped layer, a skipped preprocessing step), and unverified numbers
would just be another private variant.

The method this project uses for its own protocol layer (see
`tests/alignment/`): transcribe the reference implementation's evaluation code
*verbatim* into the test, run both on the real datasets, and assert identical
masks/metrics with tight tolerances (e.g. `rtol 1e-12`). Where you must deviate
from the original, document the reason in a `docs/protocols/`-style note inside
your package before the code. This is a recommendation, not a requirement for
registration — but it is what lets others trust your port.

## Pull request checklist

- [ ] `ruff check`, `ruff format --check`, `mypy` and `pytest` all green.
- [ ] New components register through entry points; no core path was modified.
- [ ] Metrics have hand-computed cases; selectors have boundary cases; datasets
      have synthetic-fixture cases (core-contracts §9).
- [ ] If the change can affect evaluation numbers: `tests/alignment/` is green
      and any deviation is recorded under `docs/protocols/` first.
- [ ] If any Protocol or `EvalConfig` field changed: `core-contracts.md` is
      updated in the same PR.

## License

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
