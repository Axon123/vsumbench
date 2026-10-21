"""Unit tests for metrics: hand-computed F1, Kendall's tau-b and Spearman's rho."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vsumbench.metrics.f1 import F1_METRIC
from vsumbench.metrics.rank import (
    KENDALL_TAU_GTSCORE_METRIC,
    KENDALL_TAU_METRIC,
    SPEARMAN_RHO_GTSCORE_METRIC,
    kendall_tau_b,
    spearman_rho,
)
from vsumbench.types import Mask, Scores, ShotTable, Video


def make_video(
    video_id: str = "video_1",
    n_frames: int | None = None,
    user_summaries: tuple[list[int], ...] | None = None,
    user_scores: tuple[list[float], ...] | None = None,
    picks: list[int] | None = None,
    gt_score: list[float] | None = None,
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
        picks=None if picks is None else np.asarray(picks, dtype=np.int64),
        gt_score=None if gt_score is None else np.asarray(gt_score, dtype=np.float64),
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


class TestSpearmanRhoHandComputed:
    def test_perfect_and_reverse(self) -> None:
        x = np.asarray([1.0, 2.0, 3.0, 4.0])
        assert spearman_rho(x, np.asarray([10.0, 20.0, 30.0, 40.0])) == 1.0
        assert spearman_rho(x, np.asarray([40.0, 30.0, 20.0, 10.0])) == -1.0

    def test_rho_with_ties(self) -> None:
        # x = [1,2,2,4] -> ranks [1, 2.5, 2.5, 4]; y = [1,2,3,4] -> [1,2,3,4].
        # Pearson over ranks: dx=[-1.5,0,0,1.5], dy=[-1.5,-0.5,0.5,1.5];
        # sum(dx*dy) = 4.5, sum(dx^2) = 4.5, sum(dy^2) = 5
        # -> rho = 4.5/sqrt(4.5*5) = 3/sqrt(10).
        rho = spearman_rho(np.asarray([1.0, 2.0, 2.0, 4.0]), np.asarray([1.0, 2.0, 3.0, 4.0]))
        assert rho == pytest.approx(3.0 / math.sqrt(10.0))

    def test_rho_one_swap(self) -> None:
        # ranks [1,2,3] vs [1,3,2]: d = [0,-1,1], sum d^2 = 2
        # -> rho = 1 - 6*2/(3*(9-1)) = 0.5.
        rho = spearman_rho(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 3.0, 2.0]))
        assert rho == pytest.approx(0.5)

    def test_constant_input_is_nan(self) -> None:
        assert np.isnan(spearman_rho(np.asarray([1.0, 1.0, 1.0]), np.asarray([1.0, 2.0, 3.0])))
        assert np.isnan(spearman_rho(np.asarray([1.0, 2.0, 3.0]), np.asarray([5.0, 5.0, 5.0])))

    def test_single_sample_is_nan(self) -> None:
        assert np.isnan(spearman_rho(np.asarray([1.0]), np.asarray([1.0])))

    def test_rejects_unequal_lengths(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            spearman_rho(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0, 3.0]))

    def test_matches_naive_average_rank_implementation(self) -> None:
        # Independent reference: sort-based average ranks + textbook Pearson.
        rng = np.random.default_rng(3)
        x = rng.choice(np.linspace(0, 1, 6), size=40)
        y = rng.choice(np.linspace(0, 1, 6), size=40)

        def naive_ranks(v: np.ndarray) -> np.ndarray:
            # average rank = mean of a value's 1-based positions in sorted order
            order = np.sort(v)
            return np.asarray([np.mean(np.nonzero(order == value)[0] + 1) for value in v])

        rx, ry = naive_ranks(x), naive_ranks(y)
        dx, dy = rx - rx.mean(), ry - ry.mean()
        expected = float(dx @ dy) / math.sqrt(float(dx @ dx) * float(dy @ dy))
        assert spearman_rho(x, y) == pytest.approx(expected)


class TestGtscoreRankMetrics:
    # 6-frame video, picks [0, 2, 5] -> step space of 3; gtscore [1, 2, 3].
    def _video(self, **kwargs: object) -> Video:
        defaults: dict[str, object] = {
            "n_frames": 6,
            "picks": [0, 2, 5],
            "gt_score": [1.0, 2.0, 3.0],
        }
        defaults.update(kwargs)
        return make_video(**defaults)  # type: ignore[arg-type]

    def test_tau_samples_predictions_at_picks(self) -> None:
        # frame scores 1,*,2,*,*,3 -> step scores [1,2,3]: perfect ranking.
        values = np.asarray([1.0, 9.9, 2.0, 9.9, 9.9, 3.0])
        out = KENDALL_TAU_GTSCORE_METRIC.score(Scores("video_1", values), self._video())
        assert out.shape == (1,)
        assert out[0] == 1.0

    def test_tau_swapped_steps(self) -> None:
        # step scores [1,3,2] vs gtscore [1,2,3]: one discordant pair of
        # three -> tau-b = (2-1)/3 = 1/3.
        values = np.asarray([1.0, 9.9, 3.0, 9.9, 9.9, 2.0])
        out = KENDALL_TAU_GTSCORE_METRIC.score(Scores("video_1", values), self._video())
        assert out[0] == pytest.approx(1.0 / 3.0)

    def test_rho_samples_predictions_at_picks(self) -> None:
        # step scores [1,3,2] vs [1,2,3]: ranks [1,2,3] vs [1,3,2] -> rho = 0.5.
        values = np.asarray([1.0, 9.9, 3.0, 9.9, 9.9, 2.0])
        out = SPEARMAN_RHO_GTSCORE_METRIC.score(Scores("video_1", values), self._video())
        assert out.shape == (1,)
        assert out[0] == pytest.approx(0.5)

    def test_rho_perfect(self) -> None:
        values = np.asarray([1.0, 9.9, 2.0, 9.9, 9.9, 3.0])
        out = SPEARMAN_RHO_GTSCORE_METRIC.score(Scores("video_1", values), self._video())
        assert out[0] == 1.0

    def test_constant_predictions_are_nan(self) -> None:
        # uniform-style constant scores are an undefined ranker.
        values = np.full(6, 0.5)
        for metric in (KENDALL_TAU_GTSCORE_METRIC, SPEARMAN_RHO_GTSCORE_METRIC):
            assert np.isnan(metric.score(Scores("video_1", values), self._video())[0])

    def test_without_picks_frame_space_is_used(self) -> None:
        # picks None -> spaces coincide; gt_score indexes frames directly.
        video = self._video(picks=None, gt_score=[1.0, 2.0, 3.0, 4.0], n_frames=4)
        out = SPEARMAN_RHO_GTSCORE_METRIC.score(
            Scores("video_1", np.asarray([4.0, 3.0, 2.0, 1.0])), video
        )
        assert out[0] == -1.0

    def test_rejects_missing_gt_score(self) -> None:
        with pytest.raises(ValueError, match="no gt_score"):
            KENDALL_TAU_GTSCORE_METRIC.score(
                Scores("video_1", np.asarray([1.0, 2.0])), make_video(n_frames=2)
            )

    def test_rejects_id_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            KENDALL_TAU_GTSCORE_METRIC.score(Scores("video_9", np.zeros(6)), self._video())

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="frames"):
            KENDALL_TAU_GTSCORE_METRIC.score(Scores("video_1", np.zeros(5)), self._video())


class TestScipyAgreement:
    """Cross-check the numpy implementations against scipy's conventions."""

    def _tied_pair(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        grid = np.linspace(0.0, 1.0, 9)
        return rng.choice(grid, size=80), rng.choice(grid, size=80)

    def test_kendall_tau_b_matches_scipy(self) -> None:
        scipy_stats = pytest.importorskip("scipy.stats")
        for seed in range(10):
            x, y = self._tied_pair(seed)
            expected = scipy_stats.kendalltau(x, y).statistic
            assert kendall_tau_b(x, y) == pytest.approx(float(expected), rel=1e-12)
        x, y = self._tied_pair(11)
        assert np.isnan(kendall_tau_b(np.ones_like(x), y)) and np.isnan(
            scipy_stats.kendalltau(np.ones_like(x), y).statistic
        )

    def test_spearman_rho_matches_scipy(self) -> None:
        scipy_stats = pytest.importorskip("scipy.stats")
        for seed in range(10):
            x, y = self._tied_pair(seed)
            expected = scipy_stats.spearmanr(x, y).statistic
            assert spearman_rho(x, y) == pytest.approx(float(expected), rel=1e-12)
        x, y = self._tied_pair(11)
        assert np.isnan(spearman_rho(np.ones_like(x), y)) and np.isnan(
            scipy_stats.spearmanr(np.ones_like(x), y).statistic
        )
