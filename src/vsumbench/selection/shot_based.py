"""Greedy shot-based selection (the lineage ``method='rank'``).

Ordering and the budget check replicate the reference exactly
(KaiyangZhou/pytorch-vsumm-reinforce ``vsum_tools.generate_summary``, kept
verbatim in the ok1zjf/VASNet fork):

- ``order = np.argsort(seg_score)[::-1]`` — descending score; the reversal
  of a stable ascending sort resolves ties towards the **later** shot;
- a shot is taken only while ``total + length < limit`` (strict), so the
  selection can stop well short of the limit and never reaches it exactly.
"""

from __future__ import annotations

import numpy as np

from vsumbench.config import SegmentAggregation
from vsumbench.selection._common import mask_from_shots, shot_lengths, shot_scores
from vsumbench.selection._common import validate_alignment as _validate_alignment
from vsumbench.types import Budget, Mask, Scores, ShotTable

__all__ = ["GREEDY_SELECTOR", "GreedySelector"]


class GreedySelector:
    """Deterministic greedy selector: best-scoring shots first, strict budget."""

    name = "greedy"

    def select(
        self, scores: Scores, shots: ShotTable, budget: Budget, *, aggregation: SegmentAggregation
    ) -> Mask:
        _validate_alignment(scores, shots)
        seg = shot_scores(scores, shots, aggregation)
        lengths = shot_lengths(shots)
        limit = budget.frame_limit(shots.n_frames)
        selected: list[int] = []
        total = 0
        for i in np.argsort(seg)[::-1]:
            shot = int(i)
            if total + int(lengths[shot]) < limit:
                selected.append(shot)
                total += int(lengths[shot])
        return mask_from_shots(scores.video_id, shots, sorted(selected))


GREEDY_SELECTOR = GreedySelector()
