"""Core data structures and extension protocols.

Together with :mod:`vsumbench.config` (which it imports), this module sits at
the bottom of the package: everything else builds on the types defined here,
and this module must never import higher layers (AGENTS.md rule 2).

Conventions
-----------
- Frame indices are ``0 .. n_frames - 1``; shot ``i`` covers the half-open
  range ``[boundaries[i], boundaries[i + 1])``.
- :class:`Video` carries two frame spaces when the source subsamples: the
  **original** space (``n_frames`` frames: shots, user annotations) and the
  **step** space (``n_steps`` rows: ``features``), linked by ``picks``.
  Which space scores live in and how ``picks`` expands between them is a
  protocol choice deferred to M2 archaeology; loaders store both spaces raw.
- :class:`Scores` are always **per-frame**. Segment-level quantities are
  derived through the explicitly configured aggregation
  (``EvalConfig.segment_score_aggregation``), never implicitly inside a
  model or selector — VASNet scores segments natively while PGLSUM scores
  frames, and the frame level is the only faithful meeting point.
- Arrays passed to constructors are accepted as numpy arrays or (nested)
  sequences, then copied and marked read-only: shared fixtures cannot be
  mutated through one view.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from vsumbench.config import BudgetRounding, EvalConfig, SegmentAggregation

__all__ = [
    "Budget",
    "DatasetAdapter",
    "Mask",
    "Model",
    "RankMetric",
    "Scores",
    "Selector",
    "ShotTable",
    "Split",
    "SummaryMetric",
    "Video",
]


# -- array helpers (private) --------------------------------------------------


def _as_array(values: object, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, (list, tuple)):
        arr: np.ndarray = np.asarray(values)
        return arr
    raise TypeError(
        f"{name} must be a numpy array or a (nested) sequence, got {type(values).__name__}"
    )


def _freeze(
    values: object,
    *,
    name: str,
    ndim: int,
    category: type[np.generic],
    length: int | None = None,
) -> np.ndarray:
    arr = _as_array(values, name)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, category):
        raise TypeError(f"{name} must hold {category.__name__} data, got dtype {arr.dtype}")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}")
    frozen: np.ndarray = arr.copy()
    frozen.flags.writeable = False
    return frozen


def _freeze_int_boundaries(values: object, *, name: str) -> np.ndarray:
    arr = _as_array(values, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.integer):
        frozen = arr.astype(np.int64, copy=True)
    elif np.issubdtype(arr.dtype, np.floating) and bool(np.all(arr == np.rint(arr))):
        frozen = arr.astype(np.int64)
    else:
        raise TypeError(f"{name} must contain only integers, got dtype {arr.dtype}")
    frozen.flags.writeable = False
    return frozen


def _require_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


# -- data structures -----------------------------------------------------------


@dataclass(frozen=True)
class ShotTable:
    """Frame segmentation of a video into consecutive shots (segments).

    ``boundaries`` is a strictly increasing ``int64`` array of length
    ``n_shots + 1`` with ``boundaries[0] == 0`` and
    ``boundaries[-1] == n_frames``. Empty shots are not allowed; loaders
    must merge them away.
    """

    boundaries: np.ndarray
    n_frames: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_frames, bool)
            or not isinstance(self.n_frames, int)
            or self.n_frames < 1
        ):
            raise ValueError(
                f"ShotTable.n_frames must be a positive integer, got {self.n_frames!r}"
            )
        object.__setattr__(
            self, "boundaries", _freeze_int_boundaries(self.boundaries, name="ShotTable.boundaries")
        )
        b = self.boundaries
        if b.size < 2:
            raise ValueError(f"ShotTable.boundaries needs at least 2 entries, got {b.size}")
        if b[0] != 0:
            raise ValueError(f"ShotTable.boundaries must start at 0, got {b[0]}")
        if b[-1] != self.n_frames:
            raise ValueError(
                f"ShotTable.boundaries must end at n_frames={self.n_frames}, got {b[-1]}"
            )
        if bool(np.any(np.diff(b) <= 0)):
            raise ValueError("ShotTable.boundaries must be strictly increasing (no empty shots)")

    @property
    def n_shots(self) -> int:
        return int(self.boundaries.size) - 1

    def shot_span(self, index: int) -> tuple[int, int]:
        """Half-open frame range ``[start, end)`` of shot ``index``."""
        if not 0 <= index < self.n_shots:
            raise IndexError(f"shot index {index} out of range for {self.n_shots} shots")
        return int(self.boundaries[index]), int(self.boundaries[index + 1])


@dataclass(frozen=True)
class Video:
    """A single video with everything evaluation needs.

    Ground-truth annotations stay raw and per-user: every aggregation or
    conversion choice (max vs mean, scores -> summary) belongs to
    ``EvalConfig``, never to the loader (AGENTS.md rule 4). At least one of
    ``user_scores`` / ``user_summaries`` is expected for evaluation, but not
    enforced — score-only experiments may carry none.

    The eccv16 TVSum/SumMe files subsample frames (one GoogLeNet feature
    every 15 original frames), so a faithful video lives in two spaces:
    annotations index the *original* ``n_frames`` frames while ``features``
    has one row per *step*, i.e. per entry of ``picks``. ``picks is None``
    means the two spaces coincide (``n_steps == n_frames``). How scores move
    between the spaces is pinned in M2 (core-contracts.md §8), not here.

    Attributes:
        dataset: dataset registry name, e.g. ``"tvsum"``.
        video_id: dataset-unique identifier.
        n_frames: number of original frames; ``shots`` and every per-user
            annotation have this length.
        shots: frame segmentation of the original frame space.
        features: optional ``[n_steps, dim]`` float array of frame features;
            rows correspond 1:1 to ``picks``.
        picks: optional ``int64[n_steps]`` original-frame index of each
            feature row; strictly increasing within ``[0, n_frames)``.
        fps: optional native frame rate; only second-based protocols need it.
        user_scores: per-user frame importance arrays (original space).
        user_summaries: per-user boolean frame masks (original space).
    """

    dataset: str
    video_id: str
    n_frames: int
    shots: ShotTable
    features: np.ndarray | None = None
    picks: np.ndarray | None = None
    fps: float | None = None
    user_scores: tuple[np.ndarray, ...] = ()
    user_summaries: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.dataset, "Video.dataset")
        _require_id(self.video_id, "Video.video_id")
        if (
            isinstance(self.n_frames, bool)
            or not isinstance(self.n_frames, int)
            or self.n_frames < 1
        ):
            raise ValueError(f"Video.n_frames must be a positive integer, got {self.n_frames!r}")
        if not isinstance(self.shots, ShotTable):
            raise TypeError(f"Video.shots must be a ShotTable, got {type(self.shots).__name__}")
        if self.shots.n_frames != self.n_frames:
            raise ValueError(
                f"Video.shots indexes {self.shots.n_frames} frames "
                f"but Video.n_frames is {self.n_frames}"
            )
        if self.picks is not None:
            object.__setattr__(
                self, "picks", _freeze_int_boundaries(self.picks, name="Video.picks")
            )
            picks = self.picks
            if picks.size < 1:
                raise ValueError("Video.picks must hold at least one index")
            if picks[0] < 0 or picks[-1] >= self.n_frames:
                raise ValueError(
                    f"Video.picks must lie within [0, n_frames={self.n_frames}), "
                    f"got [{picks[0]}, {picks[-1]}]"
                )
            if bool(np.any(np.diff(picks) <= 0)):
                raise ValueError("Video.picks must be strictly increasing")
        if self.features is not None:
            expected_rows = self.picks.shape[0] if self.picks is not None else self.n_frames
            object.__setattr__(
                self,
                "features",
                _freeze(
                    self.features,
                    name="Video.features",
                    ndim=2,
                    category=np.floating,
                    length=expected_rows,
                ),
            )
        if self.fps is not None:
            if (
                isinstance(self.fps, bool)
                or not isinstance(self.fps, (int, float))
                or self.fps <= 0
            ):
                raise ValueError(f"Video.fps must be a positive number, got {self.fps!r}")
            object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(
            self,
            "user_scores",
            tuple(
                _freeze(
                    s,
                    name=f"Video.user_scores[{i}]",
                    ndim=1,
                    category=np.floating,
                    length=self.n_frames,
                )
                for i, s in enumerate(self.user_scores)
            ),
        )
        object.__setattr__(
            self,
            "user_summaries",
            tuple(
                _freeze(
                    m,
                    name=f"Video.user_summaries[{i}]",
                    ndim=1,
                    category=np.bool_,
                    length=self.n_frames,
                )
                for i, m in enumerate(self.user_summaries)
            ),
        )

    @property
    def n_steps(self) -> int:
        """Length of the step space: one entry per feature row / pick."""
        if self.picks is not None:
            return int(self.picks.shape[0])
        if self.features is not None:
            return int(self.features.shape[0])
        return self.n_frames


@dataclass(frozen=True)
class Scores:
    """Per-frame importance scores produced by a model.

    ``values`` is a 1-D float array whose length must equal the
    ``n_frames`` of the video the scores belong to; producers are
    responsible for that alignment, consumers may re-validate against
    ``Video.n_frames``.
    """

    video_id: str
    values: np.ndarray

    def __post_init__(self) -> None:
        _require_id(self.video_id, "Scores.video_id")
        object.__setattr__(
            self, "values", _freeze(self.values, name="Scores.values", ndim=1, category=np.floating)
        )


@dataclass(frozen=True)
class Mask:
    """A machine summary as a read-only per-frame boolean selection mask."""

    video_id: str
    frames: np.ndarray

    def __post_init__(self) -> None:
        _require_id(self.video_id, "Mask.video_id")
        object.__setattr__(
            self, "frames", _freeze(self.frames, name="Mask.frames", ndim=1, category=np.bool_)
        )


@dataclass(frozen=True)
class Budget:
    """Summary budget as a fraction of total frames, e.g. ``0.15`` for F1@15%.

    ``rounding`` is the protocol rule that maps the fraction to the allowed
    number of selected frames (AGENTS.md rule 4). Every eccv16-lineage
    repository floors — ``int(n_frames * fraction)`` — so that is the
    default; see docs/protocols/eccv16-summary-protocol.md. ``round`` uses
    Python's round-half-to-even.
    """

    fraction: float
    rounding: BudgetRounding = "floor"

    def __post_init__(self) -> None:
        if isinstance(self.fraction, bool) or not isinstance(self.fraction, (int, float)):
            raise TypeError(f"Budget.fraction must be a number, got {self.fraction!r}")
        if not 0.0 < float(self.fraction) <= 1.0:
            raise ValueError(f"Budget.fraction must be in (0, 1], got {self.fraction!r}")
        if self.rounding not in ("floor", "round", "ceil"):
            raise ValueError(
                f"Budget.rounding must be one of ('floor', 'round', 'ceil'), got {self.rounding!r}"
            )
        object.__setattr__(self, "fraction", float(self.fraction))

    def frame_limit(self, n_frames: int) -> int:
        """Allowed number of selected frames for a video of ``n_frames`` frames.

        Computed exactly like the lineage code (``n_frames * fraction`` first,
        then rounded), so float quirks of the reference are reproduced; the
        result is clamped at 0.
        """
        raw = n_frames * self.fraction
        if self.rounding == "floor":
            limit = math.floor(raw)
        elif self.rounding == "ceil":
            limit = math.ceil(raw)
        else:
            limit = round(raw)
        return max(int(limit), 0)


@dataclass(frozen=True)
class Split:
    """A train/test split over dataset video ids."""

    train: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", _check_ids(self.train, "Split.train"))
        object.__setattr__(self, "test", _check_ids(self.test, "Split.test"))
        if not self.train or not self.test:
            raise ValueError("Split.train and Split.test must both be non-empty")
        overlap = sorted(set(self.train) & set(self.test))
        if overlap:
            raise ValueError(f"Split.train and Split.test overlap on {overlap}")


def _check_ids(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of video ids")
    ids = tuple(_require_id(v, f"{name} entry") for v in values)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate ids")
    return ids


# -- extension protocols --------------------------------------------------------


@runtime_checkable
class Model(Protocol):
    """Maps videos to per-frame importance scores (:class:`Scores`).

    ``fit`` receives the training split and the full protocol configuration
    (hyperparameters under ``config.training``, seed under ``config.seed``);
    it may be a no-op for learning-free baselines. Implementations must be
    deterministic for a fixed config — all randomness enters through the
    config's seed.
    """

    name: str

    def fit(self, videos: Sequence[Video], config: EvalConfig) -> None: ...

    def predict(self, video: Video) -> Scores: ...


@runtime_checkable
class Selector(Protocol):
    """Turns per-frame scores into a boolean summary mask under a budget.

    ``aggregation`` tells the selector how to reduce frame scores to
    segment scores; the orchestrator passes it from
    ``EvalConfig.segment_score_aggregation`` so the choice stays explicit
    (AGENTS.md rule 4). Implementations must not introduce randomness.
    """

    name: str

    def select(
        self, scores: Scores, shots: ShotTable, budget: Budget, *, aggregation: SegmentAggregation
    ) -> Mask: ...


@runtime_checkable
class SummaryMetric(Protocol):
    """Compares a machine summary against per-user ground-truth summaries.

    ``score`` returns one float per annotation (length =
    ``len(video.user_summaries)``). Collapsing those values into the
    reported number is the orchestrator's job
    (``EvalConfig.gt_aggregation``), never the metric's.
    """

    name: str

    def score(self, summary: Mask, video: Video) -> np.ndarray: ...


@runtime_checkable
class RankMetric(Protocol):
    """Compares predicted per-frame scores against per-user ground-truth scores.

    ``score`` returns one float per annotation (length =
    ``len(video.user_scores)``).
    """

    name: str

    def score(self, scores: Scores, video: Video) -> np.ndarray: ...


@runtime_checkable
class DatasetAdapter(Protocol):
    """Loads a dataset from a local directory in canonical layout.

    Adapters never download or verify checksums — that is
    ``vsumbench.data.download``'s job (M1) — and never apply protocol
    processing: no score normalization, no ground-truth aggregation, no
    segmentation choices. They map raw files to :class:`Video` records and
    nothing else.
    """

    name: str

    def load(self, root: Path) -> list[Video]: ...
