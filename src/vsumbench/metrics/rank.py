"""Kendall's tau-b between predicted scores and each per-user score array.

Kendall's tau enters the field through the eccv16 paper (Zhang et al. 2016
report tau / Spearman's rho between predicted importance and the h5
``gtscore``), not through any of the fetched lineage repositories — none of
them computes a rank correlation. We implement tau-b (tie-corrected, the
``scipy.stats.kendalltau`` / MATLAB ``corr(..., 'type', 'Kendall')``
definition) because predicted scores regularly contain exact ties.

Ground truth here is *per-user*: the canonical eccv16 h5 files carry only a
cross-user aggregated ``gtscore`` and no per-user importance arrays, so
videos loaded by the tvsum/summe adapters have empty ``user_scores`` and
this metric raises for them until per-user scores are wired in (core
contracts §2.2; v0.1+).
"""

from __future__ import annotations

import math

import numpy as np

from vsumbench.types import Scores, Video

__all__ = ["KENDALL_TAU_METRIC", "KendallTauMetric", "kendall_tau_b"]

_CHUNK = 256


class KendallTauMetric:
    """Kendall's tau-b, one value per user score array (NaN when undefined)."""

    name = "kendall_tau"

    def score(self, scores: Scores, video: Video) -> np.ndarray:
        if scores.video_id != video.video_id:
            raise ValueError(
                f"scores id {scores.video_id!r} does not match video id {video.video_id!r}"
            )
        if scores.values.shape[0] != video.n_frames:
            raise ValueError(
                f"scores have {scores.values.shape[0]} frames but video "
                f"{video.video_id!r} has {video.n_frames}"
            )
        if not video.user_scores:
            raise ValueError(
                f"video {video.video_id!r} carries no user_scores; rank metrics need "
                "per-user importance arrays (not in the canonical eccv16 h5 files)"
            )
        return np.array(
            [kendall_tau_b(scores.values, user) for user in video.user_scores],
            dtype=np.float64,
        )


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau-b for two equal-length 1-D arrays of finite values.

    ``tau_b = (P - Q) / sqrt((n0 - n1) * (n0 - n2))`` with P/Q the concordant
    and discordant pair counts, n0 all pairs, n1/n2 pairs tied in x/y.
    Returns NaN when fewer than two samples or when either array is constant
    (the denominator vanishes), matching scipy's convention.
    """
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"x and y must have equal length, got {n} and {y.shape[0]}")
    if n < 2:
        return math.nan
    concordant = 0
    discordant = 0
    for start in range(0, n, _CHUNK):
        block = slice(start, start + _CHUNK)
        dx = np.sign(x[block, None] - x[None, :])
        dy = np.sign(y[block, None] - y[None, :])
        product = dx * dy
        # Every unordered pair (i, j) is seen twice ((i, j) and (j, i)); the
        # diagonal contributes 0, so the halved counts are exact.
        concordant += int(np.count_nonzero(product > 0))
        discordant += int(np.count_nonzero(product < 0))
    p_pairs = concordant // 2
    q_pairs = discordant // 2
    n0 = n * (n - 1) // 2
    _, x_ties = np.unique(x, return_counts=True)
    _, y_ties = np.unique(y, return_counts=True)
    n1 = int((x_ties * (x_ties - 1) // 2).sum())
    n2 = int((y_ties * (y_ties - 1) // 2).sum())
    # One sqrt over the exact integer product: sqrt(a)*sqrt(b) would lose
    # the last ulp and turn a perfect ranking into 1.0000000000000002.
    denominator = math.sqrt(float(n0 - n1) * float(n0 - n2))
    if denominator == 0.0:
        return math.nan
    return (p_pairs - q_pairs) / denominator


KENDALL_TAU_METRIC = KendallTauMetric()
