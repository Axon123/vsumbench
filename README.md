# vsumbench

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

🚧 **Early development.** The data layer (`vsumm download`), the protocol layer (selectors, F1, reference-protocol τ/ρ against `gtscore`, alignment tests) and the orchestrator (`vsumm eval` / `vsumm report`, run manifests, the `Bench` API) are implemented — see [docs/design/dev-plan.md](docs/design/dev-plan.md) for progress. The protocol archaeology for the TVSum/SumMe lineage lives in [docs/protocols](docs/protocols/). Remaining for v0.1: docs site, CONTRIBUTING, PyPI release.

## Contributing

New models join through the registry — see [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon). If you port a published method, consider shipping alignment tests against its original implementation so others can trust the port.

## License

[MIT](LICENSE)
