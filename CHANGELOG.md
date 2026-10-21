# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Data layer.** `vsumm download` with multi-mirror sources, sha256 verification, a cache directory (`VSUM_CACHE_DIR`, overridable for offline use); TVSum and SumMe adapters over the canonical eccv16 h5 files, carrying both frame spaces raw (step-space `features` + `picks`, frame-space per-user annotations) plus the dataset's aggregated `gt_score`; canonical 5-fold splits transcribed verbatim from the lineage and seed-controlled random k-fold splits.
- **Protocol layer.** `EvalConfig` as the single source of protocol truth (GT aggregation, segment-score aggregation, budget rounding, step-score expansion, score normalization, splits, seed) with strict JSON round-tripping; `greedy` (shot-based) and `knapsack` selectors reproducing the reference tie and flooring semantics bit-for-bit; the per-user `f1` (ratio convention, empty-summary deviation documented) and tie-corrected per-user `kendall_tau`.
- **Reference-protocol rank metrics.** `kendall_tau_gtscore` and `spearman_rho_gtscore` against the dataset's aggregated `gtscore` in step space (the Zhang et al. 2016 convention): rank correlation is invariant to the monotone transforms of the upstream aggregation, so this runs on today's canonical files with no per-user data; predictions are sampled at `picks`.
- **Orchestration.** `Bench` API (explicit `EvalConfig`, or convenience kwargs with recommended-protocol defaults; `register()` registers and selects in one call), `vsumm eval` / `vsumm report` CLI, run manifests (`vsumbench.run-manifest/1`: git SHA + full config + per-file data checksums + per-fold results, non-finite floats written as `null`), and the no-learning `uniform` reference model for smoke tests.
- **Alignment tests.** Reference implementations transcribed verbatim and compared on the 75 real TVSum/SumMe videos (selection masks and per-user F1 at `rtol 1e-12`; rank correlations cross-checked against scipy).
- **Protocol archaeology.** Per-repository records of the eccv16 lineage's implicit evaluation choices and the original TVSum/SumMe papers' protocols (`docs/protocols/`), including the `video_name` asymmetry relevant to joining the original ydata annotations.
- **Docs.** CONTRIBUTING guide (extension contracts, testing admission, porting with alignment tests); bilingual README (English / 简体中文) with a full usage walkthrough.

[Unreleased]: https://github.com/Axon123/vsumbench/commits/main
