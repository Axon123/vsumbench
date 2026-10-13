"""Expansion of step-space scores into the original frame space.

Models that score one step per ``picks`` entry (VASNet, PGL-SUM, SUM-GAN,
DR-DSN — the whole eccv16 lineage) must still emit frame-level
:class:`~vsumbench.types.Scores` (core-contracts.md §2.3). This module is the
single shared implementation of that expansion so every model — and every
alignment test — moves between the two spaces identically.

``forward_fill`` replicates the lineage ``generate_summary`` exactly: each
step score is held until the next pick, and the tail after the last pick
keeps the last score (verified against e-apostolidis/PGL-SUM
``evaluation/generate_summary.py`` and KaiyangZhou/pytorch-vsumm-reinforce
``vsum_tools.py``; the ``i == len(scores)`` zero branch in those files is
dead code for well-formed inputs — it only guards length mismatches).
``zero_fill`` writes scores at the picks and zeros elsewhere, the variant
used by repos that skip the expansion loop.

The output dtype is float32 by default, matching the lineage's
``np.zeros(n_frames, dtype=np.float32)``; shot means computed downstream
then round identically to the reference implementations.
"""

from __future__ import annotations

import numpy as np

from vsumbench.config import StepExpansion

__all__ = ["expand_step_scores"]


def expand_step_scores(
    step_scores: np.ndarray,
    picks: np.ndarray,
    n_frames: int,
    *,
    method: StepExpansion = "forward_fill",
    dtype: np.dtype[np.floating] | type[np.float32] | type[np.float64] = np.float32,
) -> np.ndarray:
    """Expand per-step scores to a length-``n_frames`` frame-score array.

    Args:
        step_scores: 1-D float array, one entry per pick.
        picks: 1-D integer array; strictly increasing within ``[0, n_frames)``.
        n_frames: length of the original frame space.
        method: ``"forward_fill"`` (lineage default) or ``"zero_fill"``.
        dtype: output dtype; float32 matches the reference lineage.

    Returns:
        A new read-write float array of length ``n_frames`` (callers wrap it
        in ``Scores``, which freezes its own copy).
    """
    scores = np.asarray(step_scores)
    if scores.ndim != 1 or not np.issubdtype(scores.dtype, np.floating):
        raise ValueError(
            f"step_scores must be a 1-D float array, got shape {scores.shape} dtype {scores.dtype}"
        )
    idx = np.asarray(picks)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"picks must be a 1-D integer array, got shape {idx.shape}")
    if scores.shape[0] != idx.shape[0]:
        raise ValueError(
            f"got {scores.shape[0]} step scores for {idx.shape[0]} picks; lengths must match"
        )
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError(f"n_frames must be a positive integer, got {n_frames!r}")
    if idx.size < 1 or idx[0] < 0 or idx[-1] >= n_frames or bool(np.any(np.diff(idx) <= 0)):
        raise ValueError(
            f"picks must be strictly increasing within [0, n_frames={n_frames}), "
            f"got [{idx[0]}, {idx[-1]}]"
        )

    frame_scores = np.zeros(n_frames, dtype=dtype)
    if method == "forward_fill":
        widths = np.diff(np.append(idx.astype(np.int64, copy=False), n_frames))
        frame_scores[int(idx[0]) : n_frames] = np.repeat(scores.astype(dtype, copy=False), widths)
    else:  # zero_fill
        frame_scores[idx] = scores.astype(dtype, copy=False)
    return frame_scores
