# vsumbench

**A unified, reproducible evaluation toolkit for video summarization research.**

Video summarization evaluation has been fragmented for a decade: every paper ships its own metric implementation, its own dataset loading code, and its own splits. Numbers between papers are often not comparable, and reproducing baselines means resurrecting half-a-dozen Keras-era repositories with dead data links.

`vsumbench` is the missing piece — the `lm-evaluation-harness` for the video summarization community:

```bash
pip install vsumbench

vsumm eval --model pgslum --dataset tvsum --split canonical --metrics all
```

And for your own model, all you need is a function that maps video features to per-frame importance scores:

```python
from vsumbench import Bench

bench = Bench(dataset="tvsum", split="canonical")
bench.register("mine", my_model)

bench.compare(["uniform", "vasnet", "pglsum", "mine"])
# → the comparison table for your paper, under one shared protocol
```

## Design principles

- **Faithful.** Every ported baseline is numerically aligned against its original implementation (fixed seeds, tolerance-checked, run in CI). Deviations are documented, not hidden.
- **Explicit evaluation.** Contested protocol choices — frame-selection strategy, budget, splits, normalization — are explicit parameters with recommended defaults. Disagreements become configurable, not silent.
- **Minimal.** Core depends on `numpy` + `h5py` only; frameworks are optional.
- **Data never bundled.** `vsumm download` fetches datasets from their original sources with checksum verification. We distribute loaders, not data.

## Scope

| Layer | First release (v0.1) | Later |
|---|---|---|
| Datasets | TVSum, SumMe, TVSum+SumMe | OVP, VTW, QFVS |
| Metrics | F1@{10,15,30,50}% (shot-based / knapsack selection), Kendall's τ | Spearman's ρ, extended protocols |
| Baselines | uniform, bi-LSTM, VASNet, PGLSUM | SUM-GAN family, DR-DSN, CSNet |
| Splits | TVSum canonical 5-fold | community splits, seed-controlled random splits |

## Status

🚧 **Early development.** The protocol archaeology (documenting every implicit evaluation choice in existing repositories) is underway — findings will be published in the [docs](docs/) directory and summarized in a write-up.

## Contributing

New models join through the registry — see [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon). Ported baselines must pass alignment tests against their originals before merging.

## License

[MIT](LICENSE)
