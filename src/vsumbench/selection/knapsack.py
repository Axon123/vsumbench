"""0/1 knapsack shot selection (the lineage default ``method='knapsack'``).

The dynamic program reproduces both lineage implementations —
e-apostolidis/PGL-SUM ``evaluation/knapsack_implementation.py`` (Bhavya
Jain's classic table + backtrack) and KaiyangZhou/pytorch-vsumm-reinforce
``knapsack.py::knapsack_dp`` — including their shared tie rule: when taking
a shot leaves the table value unchanged, the shot is **not** taken, and the
backtrack then prefers the earliest such shots. Values are shot mean/max
scores (never multiplied by length); weights are shot lengths in frames;
the capacity is the budget's frame limit. The table is float64, matching
the PGL-SUM Python-float arithmetic.
"""

from __future__ import annotations

import numpy as np

from vsumbench.config import SegmentAggregation
from vsumbench.selection._common import mask_from_shots, shot_lengths, shot_scores
from vsumbench.selection._common import validate_alignment as _validate_alignment
from vsumbench.types import Budget, Mask, Scores, ShotTable

__all__ = ["KNAPSACK_SELECTOR", "KnapsackSelector", "solve_knapsack"]


class KnapsackSelector:
    """Deterministic knapsack selector over whole shots under a frame budget."""

    name = "knapsack"

    def select(
        self, scores: Scores, shots: ShotTable, budget: Budget, *, aggregation: SegmentAggregation
    ) -> Mask:
        _validate_alignment(scores, shots)
        values = shot_scores(scores, shots, aggregation)
        weights = shot_lengths(shots)
        limit = budget.frame_limit(shots.n_frames)
        selected = solve_knapsack(
            values.astype(np.float64, copy=False), [int(w) for w in weights], limit
        )
        return mask_from_shots(scores.video_id, shots, selected)


def solve_knapsack(values: np.ndarray, weights: list[int], capacity: int) -> list[int]:
    """Maximize ``sum(values)`` subject to ``sum(weights) <= capacity``.

    Returns the selected item indices in ascending order. Ties keep the
    earlier items (lineage backtrack semantics; see module docstring).
    """
    n = len(values)
    if n == 0 or capacity <= 0:
        return []
    table = np.zeros((n + 1, capacity + 1), dtype=np.float64)
    for i in range(1, n + 1):
        previous = table[i - 1]
        row = previous.copy()
        weight = weights[i - 1]
        if weight <= capacity:
            # candidate[w] = values[i-1] + previous[w - weight] for w >= weight
            candidate = values[i - 1] + previous[: capacity + 1 - weight]
            row[weight:] = np.where(candidate > previous[weight:], candidate, previous[weight:])
        table[i] = row
    selected: list[int] = []
    w = capacity
    for i in range(n, 0, -1):
        if table[i, w] != table[i - 1, w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    selected.reverse()
    return selected


KNAPSACK_SELECTOR = KnapsackSelector()
