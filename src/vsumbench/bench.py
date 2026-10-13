"""Bench orchestrator: one EvalConfig from raw files to a run manifest.

The loop composes the M1/M2 building blocks without adding any numerical
choice of its own: load (dataset adapter) -> split (``data.splits``) -> per
fold: deepcopy + ``Model.fit`` on the train ids -> ``Model.predict`` on the
test ids -> ``Selector.select`` per budget -> per-user metric values ->
collapse across users with ``EvalConfig.gt_aggregation``. Aggregations that
*do* live here (fold means, mean/std/max across folds) never touch the
per-video protocol numbers; std over fold means is the sample std (ddof=1).

Every run lands a JSON manifest (git SHA + full config + data checksums,
``vsumbench.reporting.manifest``); non-finite floats are recorded as null.
NaN per-user values (e.g. an undefined Kendall tau) propagate through the
aggregations rather than being silently dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from vsumbench.config import (
    BudgetRounding,
    CanonicalSplits,
    EvalConfig,
    GTAggregation,
    RandomSplits,
    SegmentAggregation,
    SplitSpec,
    StepExpansion,
    TrainingConfig,
)
from vsumbench.data.download import ChecksumError, cache_dir, sha256_of
from vsumbench.data.sources import SOURCES
from vsumbench.data.splits import CANONICAL_SPLIT_NAMES, build_splits
from vsumbench.registry import DATASETS, MODELS, RANK_METRICS, SELECTORS, SUMMARY_METRICS
from vsumbench.reporting.manifest import (
    NO_BUDGET,
    ManifestFile,
    build_manifest,
    write_manifest,
)
from vsumbench.types import Budget, Model, RankMetric, Selector, SummaryMetric, Video

__all__ = ["Bench", "RunResult"]

#: Recommended-protocol defaults behind the Bench convenience constructor
#: (EvalConfig has no defaults for these — they name the run's subject).
DEFAULT_MODEL = "uniform"
DEFAULT_SELECTOR = "knapsack"
DEFAULT_METRICS: tuple[str, ...] = ("f1",)
DEFAULT_BUDGETS: tuple[float, ...] = (0.1, 0.15, 0.3, 0.5)

_RunDir = str | Path


@dataclass(frozen=True)
class RunResult:
    """What :meth:`Bench.run` produced: the manifest and where it lives."""

    manifest_path: Path
    manifest: dict[str, Any]

    @property
    def results(self) -> dict[str, Any]:
        """The manifest's ``results`` block (per-fold values + summary)."""
        results = self.manifest["results"]
        assert isinstance(results, dict)
        return cast("dict[str, Any]", results)

    @property
    def summary(self) -> dict[str, Any]:
        """Cross-fold summary: metric -> budget -> {mean, std, max, n_folds}."""
        summary = self.results["summary"]
        assert isinstance(summary, dict)
        return cast("dict[str, Any]", summary)


class Bench:
    """Runs one :class:`~vsumbench.config.EvalConfig` end to end.

    Two constructor shapes:

    - ``Bench(config)`` — the full protocol, explicit; the manifest records
      exactly this config.
    - ``Bench(dataset="tvsum", splits="tvsum-canonical-5", ...)`` — keyword
      convenience. Unspecified fields take the recommended-protocol defaults
      (model ``uniform`` is only a smoke reference — bring a real method via
      :meth:`register` or an entry point); a splits string becomes a
      :class:`~vsumbench.config.CanonicalSplits`, ``None`` a seeded
      :class:`~vsumbench.config.RandomSplits`.

    ``data_root`` is runtime-only (where the data lives, not how it is
    evaluated) and may accompany either shape.
    """

    def __init__(
        self,
        config: EvalConfig | None = None,
        *,
        dataset: str | None = None,
        model: str | None = None,
        selector: str | None = None,
        metrics: Sequence[str] | None = None,
        budgets: Sequence[float] | None = None,
        splits: str | SplitSpec | None = None,
        seed: int | None = None,
        gt_aggregation: GTAggregation | None = None,
        segment_score_aggregation: SegmentAggregation | None = None,
        budget_rounding: BudgetRounding | None = None,
        step_score_expansion: StepExpansion | None = None,
        training: TrainingConfig | None = None,
        data_root: _RunDir | None = None,
    ) -> None:
        overrides: dict[str, Any] = {
            key: value
            for key, value in (
                ("dataset", dataset),
                ("model", model),
                ("selector", selector),
                ("metrics", tuple(metrics) if metrics is not None else None),
                ("budgets", tuple(budgets) if budgets is not None else None),
                ("splits", splits),
                ("seed", seed),
                ("gt_aggregation", gt_aggregation),
                ("segment_score_aggregation", segment_score_aggregation),
                ("budget_rounding", budget_rounding),
                ("step_score_expansion", step_score_expansion),
                ("training", training),
            )
            if value is not None
        }
        if config is not None:
            if overrides:
                raise ValueError(
                    "pass either config= or individual protocol fields, not both "
                    f"(got {sorted(overrides)})"
                )
            self._config = config
        else:
            if dataset is None:
                raise ValueError("Bench needs dataset= (or an explicit config=)")
            overrides["splits"] = _coerce_splits(splits)
            overrides.setdefault("model", DEFAULT_MODEL)
            overrides.setdefault("selector", DEFAULT_SELECTOR)
            overrides.setdefault("metrics", DEFAULT_METRICS)
            overrides.setdefault("budgets", DEFAULT_BUDGETS)
            self._config = EvalConfig(**overrides)
        self._root = None if data_root is None else Path(data_root)

    @property
    def config(self) -> EvalConfig:
        return self._config

    def register(self, name: str, model: Model) -> None:
        """Register ``model`` in the models registry *and* select it for runs.

        The dual effect is what makes the README snippet work: an in-process
        model goes from unknown to the run's subject in one call. The
        registration is global (explicit registrations win over entry
        points); registering the same name twice raises.
        """
        MODELS.register(name, model)
        self._config = replace(self._config, model=name)

    def run(
        self,
        output_dir: _RunDir | None = None,
        *,
        report: Callable[[str], None] | None = None,
    ) -> RunResult:
        """Execute the config; always lands a manifest under ``output_dir``.

        ``output_dir`` defaults to ``runs/`` under the current directory
        (one fresh timestamped sub-directory per run); ``report`` receives
        human-readable progress lines if given.
        """
        say = report or (lambda _msg: None)
        config = self._config
        root = self._root if self._root is not None else cache_dir()
        # Resolve every registry name before touching data, so bad config
        # errors surface as themselves instead of as a half-started run.
        adapter = DATASETS.get(config.dataset)
        selector = SELECTORS.get(config.selector)
        summary_metrics, rank_metrics = _resolve_metrics(config.metrics)
        model_template = MODELS.get(config.model)

        say(f"dataset {config.dataset}: loading from {root}")
        videos = adapter.load(root)
        by_id = {video.video_id: video for video in videos}
        folds = build_splits(config.splits, list(by_id))

        per_fold: list[dict[str, Any]] = []
        fold_means: dict[str, dict[str, list[float]]] = {}
        for index, fold in enumerate(folds):
            say(
                f"fold {index + 1}/{len(folds)}: fit {config.model} "
                f"on {len(fold.train)} videos, evaluate {len(fold.test)}"
            )
            # Fresh instance per fold: cross-fold state (learned or cached)
            # must not leak between folds (core-contracts.md §3.1).
            model = deepcopy(model_template)
            model.fit([by_id[vid] for vid in fold.train], config)
            fold_videos: dict[str, dict[str, dict[str, float]]] = {}
            acc: dict[str, dict[str, list[float]]] = {}
            for vid in fold.test:
                values = _evaluate_video(
                    model,
                    by_id[vid],
                    selector=selector,
                    summary_metrics=summary_metrics,
                    rank_metrics=rank_metrics,
                    config=config,
                )
                fold_videos[vid] = values
                for metric_name, metric_values in values.items():
                    for key, value in metric_values.items():
                        acc.setdefault(metric_name, {}).setdefault(key, []).append(value)
            per_fold.append(
                {
                    "fold": index,
                    "test": list(fold.test),
                    # Per-video values live inside their fold: canonical
                    # splits may test the same video in several folds, where
                    # a learned model can score it differently each time.
                    "videos": fold_videos,
                    "metrics": {
                        metric: {
                            key: {"mean": _mean(values), "n_videos": len(values)}
                            for key, values in budget_map.items()
                        }
                        for metric, budget_map in acc.items()
                    },
                }
            )
            for metric, budget_map in acc.items():
                for key, video_values in budget_map.items():
                    fold_means.setdefault(metric, {}).setdefault(key, []).append(
                        _mean(video_values)
                    )

        results = {
            "per_fold": per_fold,
            "summary": {
                metric: {key: _fold_stats(means) for key, means in budget_map.items()}
                for metric, budget_map in fold_means.items()
            },
        }
        manifest = build_manifest(
            config,
            results,
            dataset_root=root,
            files=_dataset_files(config.dataset, root),
            splits=[
                {"fold": i, "train": list(fold.train), "test": list(fold.test)}
                for i, fold in enumerate(folds)
            ],
        )
        path = write_manifest(manifest, output_dir if output_dir is not None else "runs")
        say(f"manifest: {path}")
        return RunResult(manifest_path=path, manifest=manifest)


def _coerce_splits(splits: str | SplitSpec | None) -> SplitSpec:
    if splits is None:
        return RandomSplits()
    if isinstance(splits, str):
        if splits not in CANONICAL_SPLIT_NAMES:
            known = ", ".join(sorted(CANONICAL_SPLIT_NAMES)) or "<none>"
            raise ValueError(
                f"unknown canonical split {splits!r}; available: {known} "
                "(or pass a SplitSpec for random splits)"
            )
        return CanonicalSplits(name=splits)
    return splits


def _resolve_metrics(
    names: Sequence[str],
) -> tuple[list[SummaryMetric], list[RankMetric]]:
    """Split config metric names into summary and rank metrics.

    A name in both registries (or neither) is an error: silently picking one
    would hide a registration bug.
    """
    summary_names = set(SUMMARY_METRICS.names())
    rank_names = set(RANK_METRICS.names())
    summary: list[SummaryMetric] = []
    rank: list[RankMetric] = []
    for name in names:
        in_summary = name in summary_names
        in_rank = name in rank_names
        if in_summary and in_rank:
            raise ValueError(
                f"metric {name!r} is registered in both vsumbench.metrics and "
                "vsumbench.rank_metrics; the config is ambiguous"
            )
        if in_summary:
            summary.append(SUMMARY_METRICS.get(name))
        elif in_rank:
            rank.append(RANK_METRICS.get(name))
        else:
            raise ValueError(
                f"unknown metric {name!r}; registered: {sorted(summary_names | rank_names)}"
            )
    return summary, rank


def _evaluate_video(
    model: Model,
    video: Video,
    *,
    selector: Selector,
    summary_metrics: list[SummaryMetric],
    rank_metrics: list[RankMetric],
    config: EvalConfig,
) -> dict[str, dict[str, float]]:
    scores = model.predict(video)
    if scores.video_id != video.video_id:
        raise ValueError(
            f"model {config.model!r} returned scores for {scores.video_id!r} "
            f"but was asked about {video.video_id!r}"
        )
    if scores.values.shape[0] != video.n_frames:
        raise ValueError(
            f"model {config.model!r} returned {scores.values.shape[0]} scores for "
            f"{video.video_id!r} which has {video.n_frames} frames"
        )

    values: dict[str, dict[str, float]] = {}
    for fraction in config.budgets:
        mask = selector.select(
            scores,
            video.shots,
            Budget(fraction, config.budget_rounding),
            aggregation=config.segment_score_aggregation,
        )
        for summary_metric in summary_metrics:
            per_user = summary_metric.score(mask, video)
            values.setdefault(summary_metric.name, {})[str(fraction)] = _collapse(
                per_user, config.gt_aggregation
            )
    for rank_metric in rank_metrics:
        per_user = rank_metric.score(scores, video)
        values[rank_metric.name] = {NO_BUDGET: _collapse(per_user, config.gt_aggregation)}
    return values


def _mean(values: list[float]) -> float:
    # np.asarray first: numpy's mean overloads don't accept bare list[float]
    # under strict typing, and float64 round-trips Python floats exactly.
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _fold_stats(means: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(means, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "max": float(np.max(arr)),
        "n_folds": int(arr.size),
    }


def _collapse(values: np.ndarray, gt_aggregation: GTAggregation) -> float:
    """One number from per-user values; NaNs propagate (documented choice)."""
    if gt_aggregation == "max":
        return float(np.max(values))
    return float(np.mean(values))


def _dataset_files(dataset: str, root: Path) -> tuple[ManifestFile, ...] | None:
    """Checksum the canonical files of ``dataset``; None when unknown.

    Digests of datasets listed in :data:`vsumbench.data.sources` are checked
    against the spec — a mismatch fails the run rather than reporting
    numbers built on unverified bytes.
    """
    spec = SOURCES.get(dataset)
    if spec is None:
        return None
    files: list[ManifestFile] = []
    for file_spec in spec.files:
        path = root / file_spec.filename
        digest = sha256_of(path)
        if digest != file_spec.sha256:
            raise ChecksumError(
                f"{path} does not match its canonical sha256 — rerun "
                f"`vsumm download {dataset}` or point data_root at a verified copy"
            )
        files.append(
            ManifestFile(
                path=str(path),
                sha256=digest,
                expected_sha256=file_spec.sha256,
                matches_spec=True,
            )
        )
    return tuple(files)
