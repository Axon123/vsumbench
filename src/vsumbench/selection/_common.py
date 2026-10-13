"""Shared pieces for shot-level selectors.

The per-shot reduction loops over shot spans and calls ``np.mean`` /
``np.max`` on each slice, mirroring the lineage implementations
(``frame_scores[shot[0]:shot[1]+1].mean()``) instead of a vectorized
``reduceat``: with float32 inputs the pairwise summation inside ``np.mean``
reproduces the reference shot scores bit-for-bit, which a ``reduceat``
sequential sum would not.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from vsumbench.config import SegmentAggregation
from vsumbench.types import Mask, Scores, ShotTable

__all__ = ["mask_from_shots", "shot_lengths", "shot_scores", "validate_alignment"]


def validate_alignment(scores: Scores, shots: ShotTable) -> None:
    if scores.values.shape[0] != shots.n_frames:
        raise ValueError(
            f"scores for {scores.video_id!r} have {scores.values.shape[0]} frames "
            f"but the shot table covers {shots.n_frames}"
        )


def shot_scores(scores: Scores, shots: ShotTable, aggregation: SegmentAggregation) -> np.ndarray:
    """Reduce frame scores to one score per shot using ``aggregation``."""
    validate_alignment(scores, shots)
    values = scores.values
    out = np.empty(shots.n_shots, dtype=values.dtype)
    for i in range(shots.n_shots):
        start, end = shots.shot_span(i)
        segment = values[start:end]
        out[i] = segment.mean() if aggregation == "mean" else segment.max()
    return out


def shot_lengths(shots: ShotTable) -> np.ndarray:
    """Number of frames in each shot."""
    return np.diff(shots.boundaries)


def mask_from_shots(video_id: str, shots: ShotTable, selected: Iterable[int]) -> Mask:
    """Boolean frame mask marking every frame of every selected shot."""
    frames = np.zeros(shots.n_frames, dtype=bool)
    for i in selected:
        start, end = shots.shot_span(i)
        frames[start:end] = True
    return Mask(video_id=video_id, frames=frames)
