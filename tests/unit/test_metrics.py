"""Unit tests for metrics: hand-computed F1 and Kendall's tau-b cases."""

from __future__ import annotations

import numpy as np
import pytest

from vsumbench.metrics.f1 import F1_METRIC
from vsumbench.metrics.rank import KENDALL_TAU_METRIC, kendall_tau_b
from vsumbench.types import Mask, Scores, ShotTable, Video


def make_video(
    video_id: str = "video_1",
    n_frames: int | None = None,
    user_summaries: tuple[list[int], ...] | None = None,
    user_scores: tuple[list[float], ...] | None = None,
) -> Video:
    summaries = tuple(np.asarray(s, dtype=bool) for s in (user_summaries or ()))
    scores = tuple(np.asarray(s, dtype=np.float64) for s in (user_scores or ()))
    if n_frames is None:
        first = summaries[0] if summaries else (scores[0] if scores else None)
        if first is None:
            raise ValueError("test helper needs n_frames or an annotation to infer it")
        n_frames = int(first.shape[0])
    return Video(
        dataset="synthetic",
        video_id=video_id,
        n_frames=n_frames,
        shots=ShotTable(boundaries=np.arange(0, n_frames + 1, dtype=np.int64), n_frames=n_frames),
        user_summaries=summaries,
        user_scores=scores,
    )


def make_mask(video_id: str, frames: list[int]) -> Mask:
    return Mask(video_id=video_id, frames=np.asarray(frames, dtype=bool))


class TestF1HandComputed:
    def test_basic_overlap(self) -> None:
        # S = [1,0,1,1], G = [1,1,0,1]: overlap 2, P = 2/3, R = 2/3,
        # F1 = 2*(2/3)*(2/3)/(4/3) = 2/3.
        video = make_video(user_summaries=([1, 1, 0, 1],))
        scores = F1_METRIC.score(make_mask("video_1", [1, 0, 1, 1]), video)
        assert scores.shape == (1,)
        assert scores[0] == pytest.approx(2.0 / 3.0)

    def test_one_value_per_user(self) -> None:
        # user 1: overlap 2, P = 2/3, R = 2/3 -> F1 = 2/3
        # user 2 ([0,0,1,0]): overlap 1, P = 1/3, R = 1 -> F1 = 0.5
        video = make_video(user_summaries=([1, 1, 0, 1], [0, 0, 1, 0]))
        scores = F1_METRIC.score(make_mask("video_1", [1, 0, 1, 1]), video)
        assert scores.tolist() == pytest.approx([2.0 / 3.0, 0.5])

    def test_perfect_match_is_one(self) -> None:
        video = make_video(user_summaries=([1, 0, 1],))
        assert F1_METRIC.score(make_mask("video_1", [1, 0, 1]), video)[0] == 1.0

    def test_disjoint_summaries_are_zero(self) -> None:
        video = make_video(user_summaries=([1, 1, 0, 0],))
        assert F1_METRIC.score(make_mask("video_1", [0, 0, 1, 1]), video)[0] == 0.0

    def test_empty_machine_summary_is_zero(self) -> None:
        # Documented deviation: lineage variants crash (PGL-SUM) or use a
        # 1e-8 epsilon (DR-DSN); we return 0.
        video = make_video(user_summaries=([1, 0, 1],))
        assert F1_METRIC.score(make_mask("video_1", [0, 0, 0]), video)[0] == 0.0

    def test_empty_user_summary_is_zero(self) -> None:
        video = make_video(user_summaries=([0, 0, 0],))
        assert F1_METRIC.score(make_mask("video_1", [1, 0, 0]), video)[0] == 0.0


class TestF1Validation:
    def test_rejects_id_mismatch(self) -> None:
        video = make_video(video_id="video_1", user_summaries=([1, 0],))
        with pytest.raises(ValueError, match="does not match"):
            F1_METRIC.score(make_mask("video_2", [1, 0]), video)

    def test_rejects_length_mismatch(self) -> None:
        video = make_video(n_frames=4, user_summaries=([1, 0, 1, 0],))
        with pytest.raises(ValueError, match="frames"):
            F1_METRIC.score(make_mask("video_1", [1, 0]), video)

    def test_rejects_video_without_user_summaries(self) -> None:
        with pytest.raises(ValueError, match="no user_summaries"):
            F1_METRIC.score(make_mask("video_1", [1, 0]), make_video(n_frames=2))


class TestKendallTauHandComputed:
    def test_perfect_and_reverse(self) -> None:
        x = np.asarray([1.0, 2.0, 3.0, 4.0])
        assert kendall_tau_b(x, np.asarray([10.0, 20.0, 30.0, 40.0])) == 1.0
        assert kendall_tau_b(x, np.asarray([40.0, 30.0, 20.0, 10.0])) == -1.0

    def test_one_discordant_pair(self) -> None:
        # x = [1,2,3,4], y = [1,3,2,4]: 6 pairs, 5 concordant, 1 discordant;
        # tau = (5-1)/6 = 2/3.
        tau = kendall_tau_b(np.asarray([1.0, 2.0, 3.0, 4.0]), np.asarray([1.0, 3.0, 2.0, 4.0]))
        assert tau == pytest.approx(2.0 / 3.0)

    def test_tau_b_partial_with_ties(self) -> None:
        # x = [1,1,2], y = [1,2,2]: (0,1) ties in x, (1,2) ties in y, only
        # (0,2) is concordant; n0 = 3, n1 = n2 = 1 -> tau-b = 1/sqrt(4) = 0.5.
        tau = kendall_tau_b(np.asarray([1.0, 1.0, 2.0]), np.asarray([1.0, 2.0, 2.0]))
        assert tau == 0.5

    def test_tau_b_negative_with_ties(self) -> None:
        # x = [1,1,2], y = [2,1,1]: (0,1) ties in x, (1,2) ties in y, only
        # (0,2) is discordant; tau-b = (0-1)/sqrt(4) = -0.5.
        tau = kendall_tau_b(np.asarray([1.0, 1.0, 2.0]), np.asarray([2.0, 1.0, 1.0]))
        assert tau == -0.5

    def test_constant_input_is_nan(self) -> None:
        assert np.isnan(kendall_tau_b(np.asarray([1.0, 1.0, 1.0]), np.asarray([1.0, 2.0, 3.0])))
        assert np.isnan(kendall_tau_b(np.asarray([1.0, 2.0, 3.0]), np.asarray([5.0, 5.0, 5.0])))

    def test_single_sample_is_nan(self) -> None:
        assert np.isnan(kendall_tau_b(np.asarray([1.0]), np.asarray([1.0])))

    def test_matches_naive_pairwise_implementation(self) -> None:
        # Independent O(n^2) reference over random data with repeated values.
        rng = np.random.default_rng(42)
        x = rng.choice(np.linspace(0, 1, 7), size=60)
        y = rng.choice(np.linspace(0, 1, 7), size=60)
        p = q = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                dx, dy = x[i] - x[j], y[i] - y[j]
                if dx * dy > 0:
                    p += 1
                elif dx * dy < 0:
                    q += 1
        n0 = len(x) * (len(x) - 1) // 2
        _, cx = np.unique(x, return_counts=True)
        _, cy = np.unique(y, return_counts=True)
        n1 = int((cx * (cx - 1) // 2).sum())
        n2 = int((cy * (cy - 1) // 2).sum())
        expected = (p - q) / np.sqrt((n0 - n1) * (n0 - n2))
        assert kendall_tau_b(x, y) == pytest.approx(float(expected))


class TestKendallTauMetric:
    def test_one_value_per_user_score(self) -> None:
        video = make_video(
            n_frames=4,
            user_scores=([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]),
        )
        scores = KENDALL_TAU_METRIC.score(
            Scores("video_1", np.asarray([1.0, 2.0, 3.0, 4.0])), video
        )
        assert scores.shape == (2,)
        assert scores.tolist() == [1.0, -1.0]

    def test_rejects_video_without_user_scores(self) -> None:
        # The canonical tvsum/summe h5 files carry no per-user importance.
        with pytest.raises(ValueError, match="no user_scores"):
            KENDALL_TAU_METRIC.score(
                Scores("video_1", np.asarray([1.0, 2.0])), make_video(n_frames=2)
            )

    def test_rejects_id_mismatch(self) -> None:
        video = make_video(n_frames=2, user_scores=([1.0, 2.0],))
        with pytest.raises(ValueError, match="does not match"):
            KENDALL_TAU_METRIC.score(Scores("video_9", np.asarray([1.0, 2.0])), video)

    def test_rejects_length_mismatch(self) -> None:
        video = make_video(n_frames=4, user_scores=([1.0, 2.0, 3.0, 4.0],))
        with pytest.raises(ValueError, match="frames"):
            KENDALL_TAU_METRIC.score(Scores("video_1", np.asarray([1.0, 2.0])), video)
