"""Unit tests for selectors: hand-computed cases and boundary conditions.

Hand-derived expectations follow the lineage dynamic program / greedy
(see docs/protocols/eccv16-summary-protocol.md); each docstring states the
arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from vsumbench.config import SegmentAggregation
from vsumbench.selection._common import shot_scores
from vsumbench.selection.knapsack import KNAPSACK_SELECTOR, solve_knapsack
from vsumbench.selection.shot_based import GREEDY_SELECTOR
from vsumbench.types import Budget, Mask, Scores, Selector, ShotTable


def make_scores(video_id: str, values: list[float]) -> Scores:
    return Scores(video_id=video_id, values=np.asarray(values, dtype=np.float32))


def select(
    selector: Selector,
    scores: Scores,
    shots: ShotTable,
    budget: Budget,
    *,
    aggregation: SegmentAggregation = "mean",
) -> Mask:
    return selector.select(scores, shots, budget, aggregation=aggregation)


def make_shots(lengths: list[int]) -> ShotTable:
    boundaries = np.cumsum([0, *lengths]).astype(np.int64)
    return ShotTable(boundaries=boundaries, n_frames=int(boundaries[-1]))


class TestBudgetFrameLimit:
    def test_floor_matches_int_multiplication(self) -> None:
        # int(n * f) in every lineage repo; 0.15 is not exact binary.
        for n, fraction, expected in [(1000, 0.15, 150), (11, 0.15, 1), (7, 0.5, 3)]:
            assert Budget(fraction).frame_limit(n) == expected

    def test_round_uses_half_to_even(self) -> None:
        assert Budget(0.25, rounding="round").frame_limit(10) == 2  # 2.5 -> 2
        assert Budget(0.35, rounding="round").frame_limit(10) == 4  # 3.5 -> 4
        assert Budget(0.26, rounding="round").frame_limit(10) == 3

    def test_ceil_and_clamping(self) -> None:
        assert Budget(0.26, rounding="ceil").frame_limit(10) == 3
        assert Budget(0.01).frame_limit(5) == 0

    def test_rejects_bad_rounding(self) -> None:
        with pytest.raises(ValueError, match="rounding"):
            Budget(0.15, rounding="nearest")  # type: ignore[arg-type]


class TestShotScores:
    def test_mean_and_max(self) -> None:
        shots = make_shots([2, 3])
        scores = make_scores("v", [0.2, 0.8, 0.4, 0.4, 0.4])
        assert shot_scores(scores, shots, "mean").tolist() == pytest.approx([0.5, 0.4])
        assert shot_scores(scores, shots, "max").tolist() == pytest.approx([0.8, 0.4])

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="shot table covers"):
            shot_scores(make_scores("v", [1.0, 2.0]), make_shots([3]), "mean")


class TestKnapsack:
    def test_picks_best_value_under_budget(self) -> None:
        # shots (len, mean): (3, 0.5), (4, 0.9), (3, 0.5); limit 5.
        # Any two shots exceed 5 frames, so shot 1 alone (0.9) is optimal.
        scores = make_scores("v", [0.5] * 3 + [0.9] * 4 + [0.5] * 3)
        shots = make_shots([3, 4, 3])
        mask = select(KNAPSACK_SELECTOR, scores, shots, Budget(0.5))
        assert mask.frames.tolist() == [False] * 3 + [True] * 4 + [False] * 3

    def test_packs_two_smaller_shots(self) -> None:
        # (2, 0.5), (3, 0.7), (2, 0.5); limit = floor(7*0.6) = 4.
        # {0, 2} = 4 frames, value 1.0 beats {1} = 0.7.
        scores = make_scores("v", [0.5] * 2 + [0.7] * 3 + [0.5] * 2)
        mask = select(KNAPSACK_SELECTOR, scores, make_shots([2, 3, 2]), Budget(0.6))
        assert mask.frames.tolist() == [True, True, False, False, False, True, True]

    def test_tie_keeps_earliest_shot(self) -> None:
        # (2, 0.5), (2, 0.5); limit 2: exactly one fits, both give 0.5.
        # Lineage backtracking takes the earlier shot.
        scores = make_scores("v", [0.5] * 4)
        mask = select(KNAPSACK_SELECTOR, scores, make_shots([2, 2]), Budget(0.5))
        assert mask.frames.tolist() == [True, True, False, False]

    def test_full_budget_selects_everything(self) -> None:
        scores = make_scores("v", [0.3, 0.9, 0.1])
        mask = select(KNAPSACK_SELECTOR, scores, make_shots([1, 1, 1]), Budget(1.0))
        assert mask.frames.all()

    def test_zero_limit_selects_nothing(self) -> None:
        scores = make_scores("v", [0.5] * 6)
        mask = select(KNAPSACK_SELECTOR, scores, make_shots([2, 2, 2]), Budget(0.1))
        # floor(6*0.1) = 0: nothing fits.
        assert not mask.frames.any()

    def test_all_negative_scores_select_nothing(self) -> None:
        # DP keeps value 0 (= select nothing) over negative totals.
        scores = make_scores("v", [-1.0] * 4)
        mask = select(KNAPSACK_SELECTOR, scores, make_shots([2, 2]), Budget(0.5))
        assert not mask.frames.any()

    def test_single_shot_video(self) -> None:
        # limit 2 fits the only shot exactly.
        scores = make_scores("v", [0.7, 0.7])
        mask = select(KNAPSACK_SELECTOR, scores, make_shots([2]), Budget(1.0))
        assert mask.frames.tolist() == [True, True]

    def test_solve_knapsack_reference_example(self) -> None:
        # The example embedded in PGL-SUM's knapsack_implementation.py:
        # val=[4,4,2,2,2,4], wt=[2,2,1,1,1,2], W=7. Optimum 14; among the
        # tieing optima the lineage backtrack keeps {0, 1, 2, 3, 4}
        # (w=7, v=14) — verified by transcribing the reference DP verbatim.
        selected = solve_knapsack(np.asarray([4.0, 4.0, 2.0, 2.0, 2.0, 4.0]), [2, 2, 1, 1, 1, 2], 7)
        assert selected == [0, 1, 2, 3, 4]

    def test_mask_is_read_only_and_id_propagates(self) -> None:
        scores = make_scores("video_9", [1.0, 0.0])
        mask = select(KNAPSACK_SELECTOR, scores, make_shots([1, 1]), Budget(0.5))
        assert mask.video_id == "video_9"
        with pytest.raises(ValueError, match="read-only"):
            mask.frames[0] = False

    def test_deterministic(self) -> None:
        scores = make_scores("v", [0.1, 0.9, 0.5, 0.5, 0.2])
        shots = make_shots([2, 3])
        first = select(KNAPSACK_SELECTOR, scores, shots, Budget(0.5))
        for _ in range(3):
            assert (
                select(KNAPSACK_SELECTOR, scores, shots, Budget(0.5)).frames == first.frames
            ).all()


class TestGreedy:
    def test_takes_highest_scoring_shot_first(self) -> None:
        # limit 5: shot 1 (0.9, 4 frames) first; the 3-frame shots no longer
        # fit strictly below the limit.
        scores = make_scores("v", [0.5] * 3 + [0.9] * 4 + [0.5] * 3)
        mask = select(GREEDY_SELECTOR, scores, make_shots([3, 4, 3]), Budget(0.5))
        assert mask.frames.tolist() == [False] * 3 + [True] * 4 + [False] * 3

    def test_strict_limit_never_fills_exactly(self) -> None:
        # limit 2, shots of 2 frames: 0 + 2 < 2 is false for every shot —
        # the reference quirk produces an empty summary.
        scores = make_scores("v", [0.9, 0.8, 0.8, 0.8])
        mask = select(GREEDY_SELECTOR, scores, make_shots([2, 2]), Budget(0.5))
        assert not mask.frames.any()

    def test_tie_prefers_later_shot(self) -> None:
        # argsort ascending is stable, [::-1] reverses it: equal scores are
        # visited later-index-first. limit 4: shot 1 taken, then shot 0 does
        # not fit strictly below the limit.
        scores = make_scores("v", [0.5, 0.5, 0.5, 0.5])
        mask = select(GREEDY_SELECTOR, scores, make_shots([2, 2]), Budget(1.0))
        assert mask.frames.tolist() == [False, False, True, True]

    def test_keeps_scanning_after_first_misfit(self) -> None:
        # order [0(0.9), 1(0.8), 2(0.7)]; limit 6: shot 0 (4 frames) taken,
        # shot 1 (3 frames) cannot follow strictly below 6, but the 1-frame
        # tail shot still fits (4+1 < 6).
        scores = make_scores("v", [0.9] * 4 + [0.8] * 3 + [0.7])
        mask = select(GREEDY_SELECTOR, scores, make_shots([4, 3, 1]), Budget(0.75))
        assert mask.frames.tolist() == [True] * 4 + [False] * 3 + [True]

    def test_differs_from_knapsack_when_packing_wins(self) -> None:
        # limit = floor(6*0.7) = 4: greedy takes shot 0 (0.9) then stops
        # (2+2 < 4 fails); knapsack packs {0, 1} (value 1.7).
        scores = make_scores("v", [0.9] * 2 + [0.8] * 2 + [0.8] * 2)
        shots = make_shots([2, 2, 2])
        budget = Budget(0.7)
        greedy_mask = select(GREEDY_SELECTOR, scores, shots, budget)
        knapsack_mask = select(KNAPSACK_SELECTOR, scores, shots, budget)
        assert greedy_mask.frames.tolist() == [True, True, False, False, False, False]
        assert knapsack_mask.frames.tolist() == [True, True, True, True, False, False]

    def test_aggregation_max_changes_selection(self) -> None:
        # shot 0 = [0.6] (mean/max 0.6), shot 1 = [0.3, 0.8] (mean 0.55,
        # max 0.8); limit 3 admits shot 0 alone under mean but shot 1 under
        # max (the 1-frame shot then no longer fits strictly below 3).
        scores = make_scores("v", [0.6, 0.3, 0.8])
        shots = make_shots([1, 2])
        budget = Budget(1.0)
        by_mean = select(GREEDY_SELECTOR, scores, shots, budget, aggregation="mean")
        by_max = select(GREEDY_SELECTOR, scores, shots, budget, aggregation="max")
        assert by_mean.frames.tolist() == [True, False, False]
        assert by_max.frames.tolist() == [False, True, True]

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="shot table covers"):
            select(GREEDY_SELECTOR, make_scores("v", [1.0]), make_shots([2]), Budget(0.5))
