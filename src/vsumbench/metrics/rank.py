"""Rank correlation between predicted scores and ground-truth importance.

Ground truth comes in two shapes, and so do the metrics:

- **Per-user** (``kendall_tau``): one value per ``Video.user_scores`` entry.
  The canonical eccv16 h5 files carry no per-user importance, so tvsum/summe
  videos raise until per-user scores are wired in (core contracts §2.2;
  v0.1+).
- **Reference protocol** (``kendall_tau_gtscore`` / ``spearman_rho_gtscore``):
  Zhang et al. (ECCV 2016) — the paper the lineage's τ convention comes
  from — compare predictions against the h5's aggregated ``gtscore`` in
  *step* space. That needs no per-user data, so it runs on tvsum/summe
  today. Rank correlation is invariant to the monotone transforms of any
  upstream aggregation, which is exactly why the undocumented ``gtscore``
  pipeline is harmless *here* (it would not be for F1-style overlap).
  Frame-level predictions are mapped to step space by sampling at
  ``picks`` — the canonical inverse of the forward-fill expansion, and the
  regime the lineage models (native step scorers) were evaluated in.

Both correlations are tie-corrected (tau-b; average-rank Pearson for rho —
the ``scipy.stats.kendalltau`` / ``spearmanr`` / MATLAB ``corr`` conventions)
because predicted scores regularly contain exact ties; each returns NaN
when undefined (fewer than two samples or a constant side).
"""

from __future__ import annotations

import math

import numpy as np

from vsumbench.types import Scores, Video

__all__ = [
    "KENDALL_TAU_GTSCORE_METRIC",
    "KENDALL_TAU_METRIC",
    "SPEARMAN_RHO_GTSCORE_METRIC",
    "KendallTauGtscoreMetric",
    "KendallTauMetric",
    "SpearmanRhoGtscoreMetric",
    "kendall_tau_b",
    "spearman_rho",
]

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


def _step_space_pair(scores: Scores, video: Video) -> tuple[np.ndarray, np.ndarray]:
    """(predictions sampled at picks, gt_score) in step space.

    Sampling at ``picks`` is the canonical inverse of the forward-fill
    expansion (models/score_space.py) and reproduces the lineage regime
    where models score steps natively.
    """
    if scores.video_id != video.video_id:
        raise ValueError(
            f"scores id {scores.video_id!r} does not match video id {video.video_id!r}"
        )
    if scores.values.shape[0] != video.n_frames:
        raise ValueError(
            f"scores have {scores.values.shape[0]} frames but video "
            f"{video.video_id!r} has {video.n_frames}"
        )
    if video.gt_score is None:
        raise ValueError(
            f"video {video.video_id!r} carries no gt_score; reference-protocol rank "
            "metrics need the dataset's aggregated importance array"
        )
    predicted = scores.values[video.picks] if video.picks is not None else scores.values
    if predicted.shape[0] != video.gt_score.shape[0]:
        raise ValueError(
            f"sampling at picks gave {predicted.shape[0]} step scores but gt_score "
            f"has {video.gt_score.shape[0]} entries"
        )
    return predicted, video.gt_score


class KendallTauGtscoreMetric:
    """tau-b against ``gt_score`` in step space (Zhang et al. 2016 protocol).

    Returns a length-1 array: there is one aggregated ground truth, so the
    orchestrator's ``gt_aggregation`` collapse is the identity.
    """

    name = "kendall_tau_gtscore"

    def score(self, scores: Scores, video: Video) -> np.ndarray:
        predicted, gt = _step_space_pair(scores, video)
        return np.array([kendall_tau_b(predicted, gt)], dtype=np.float64)


class SpearmanRhoGtscoreMetric:
    """Spearman's rho against ``gt_score`` in step space (Zhang et al. 2016).

    Returns a length-1 array, like :class:`KendallTauGtscoreMetric`.
    """

    name = "spearman_rho_gtscore"

    def score(self, scores: Scores, video: Video) -> np.ndarray:
        predicted, gt = _step_space_pair(scores, video)
        return np.array([spearman_rho(predicted, gt)], dtype=np.float64)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based average ranks; tied values share the mean of their positions."""
    n = x.shape[0]
    order = np.argsort(x, kind="stable")
    boundaries = np.flatnonzero(np.diff(x[order])) + 1  # group starts, 0 excluded
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    positions = np.arange(1, n + 1, dtype=np.float64)
    cumsum = np.concatenate(([0.0], np.cumsum(positions)))
    averages = (cumsum[ends] - cumsum[starts]) / (ends - starts)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.repeat(averages, ends - starts)
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho (Pearson over average ranks) for two equal-length arrays.

    Returns NaN when fewer than two samples or when either array is constant
    (zero variance), matching ``scipy.stats.spearmanr``.
    """
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError(f"x and y must have equal length, got {n} and {y.shape[0]}")
    if n < 2:
        return math.nan
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    # One sqrt over the exact dot products: sqrt(a)*sqrt(b) could round a
    # perfect ranking past 1.0 (same rationale as kendall_tau_b).
    denominator = math.sqrt(float(dx @ dx) * float(dy @ dy))
    if denominator == 0.0:
        return math.nan
    return float(dx @ dy) / denominator


KENDALL_TAU_GTSCORE_METRIC = KendallTauGtscoreMetric()
SPEARMAN_RHO_GTSCORE_METRIC = SpearmanRhoGtscoreMetric()
