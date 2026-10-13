"""Unit tests for step-score expansion (models/score_space.py)."""

from __future__ import annotations

import numpy as np
import pytest

from vsumbench.models.score_space import expand_step_scores


class TestForwardFill:
    def test_each_score_held_until_next_pick(self) -> None:
        out = expand_step_scores(np.array([1.0, 2.0, 3.0]), np.array([0, 3, 6]), 10)
        assert out.tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

    def test_tail_keeps_last_score(self) -> None:
        # [picks[-1], n_frames) carries the last score — the lineage rule;
        # the zero branch in the reference loop is dead code for valid input.
        out = expand_step_scores(np.array([7.0, 8.0]), np.array([0, 4]), 11)
        assert out.tolist() == [7.0] * 4 + [8.0] * 7

    def test_frames_before_first_pick_stay_zero(self) -> None:
        out = expand_step_scores(np.array([7.0, 8.0]), np.array([2, 5]), 6)
        assert out.tolist() == [0, 0, 7, 7, 7, 8]


class TestZeroFill:
    def test_scores_land_on_picks_only(self) -> None:
        out = expand_step_scores(
            np.array([1.0, 2.0, 3.0]), np.array([0, 3, 6]), 10, method="zero_fill"
        )
        assert out.tolist() == [1, 0, 0, 2, 0, 0, 3, 0, 0, 0]


class TestProperties:
    def test_default_dtype_is_float32(self) -> None:
        out = expand_step_scores(np.array([1.0]), np.array([0]), 3)
        assert out.dtype == np.float32

    def test_single_step_covers_everything(self) -> None:
        out = expand_step_scores(np.array([0.5]), np.array([0]), 4)
        assert out.tolist() == [0.5] * 4

    def test_full_length_picks(self) -> None:
        picks = np.arange(5)
        scores = np.linspace(0, 1, 5)
        assert expand_step_scores(scores, picks, 5).tolist() == pytest.approx(scores.tolist())


class TestValidation:
    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="lengths must match"):
            expand_step_scores(np.array([1.0, 2.0]), np.array([0, 5, 9]), 10)

    def test_rejects_decreasing_picks(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            expand_step_scores(np.array([1.0, 2.0]), np.array([5, 3]), 10)

    def test_rejects_out_of_range_picks(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, n_frames=10\)"):
            expand_step_scores(np.array([1.0]), np.array([10]), 10)

    def test_rejects_non_positive_n_frames(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            expand_step_scores(np.array([1.0]), np.array([0]), 0)

    def test_rejects_integer_scores(self) -> None:
        with pytest.raises(ValueError, match="float array"):
            expand_step_scores(np.array([1, 2]), np.array([0, 4]), 8)
