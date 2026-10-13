"""Unit tests for vsumbench.types: invariants, validation, read-only arrays."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vsumbench.types import Budget, Mask, Scores, ShotTable, Split, Video


def make_video() -> Video:
    return Video(
        dataset="tvsum",
        video_id="v0",
        n_frames=10,
        shots=ShotTable(boundaries=np.array([0, 4, 10]), n_frames=10),
        features=np.arange(30, dtype=np.float32).reshape(10, 3),
        fps=30.0,
        user_scores=(np.linspace(1.0, 5.0, 10), np.full(10, 3.0)),
        user_summaries=(
            np.array([True, True, True, False, False, False, False, False, False, False]),
            np.array([False, False, False, False, False, True, True, True, True, True]),
        ),
    )


class TestShotTable:
    def test_happy_path(self) -> None:
        table = ShotTable(boundaries=[0, 4, 10], n_frames=10)  # type: ignore[arg-type]
        assert table.n_shots == 2
        assert table.boundaries.dtype == np.int64
        assert table.shot_span(0) == (0, 4)
        assert table.shot_span(1) == (4, 10)

    def test_arrays_are_read_only(self) -> None:
        table = ShotTable(boundaries=np.array([0, 4, 10]), n_frames=10)
        with pytest.raises(ValueError, match="read-only"):
            table.boundaries[0] = 5

    def test_accepts_integral_floats(self) -> None:
        table = ShotTable(boundaries=np.array([0.0, 3.0, 10.0]), n_frames=10)
        assert table.boundaries.dtype == np.int64

    def test_rejects_fractional_boundaries(self) -> None:
        with pytest.raises(TypeError, match="only integers"):
            ShotTable(boundaries=np.array([0.0, 3.5, 10.0]), n_frames=10)

    def test_must_start_at_zero(self) -> None:
        with pytest.raises(ValueError, match="start at 0"):
            ShotTable(boundaries=np.array([1, 4, 10]), n_frames=10)

    def test_must_end_at_n_frames(self) -> None:
        with pytest.raises(ValueError, match="end at n_frames"):
            ShotTable(boundaries=np.array([0, 4, 9]), n_frames=10)

    def test_rejects_empty_shots(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            ShotTable(boundaries=np.array([0, 4, 4, 10]), n_frames=10)

    def test_rejects_decreasing_boundaries(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            ShotTable(boundaries=np.array([0, 6, 4, 10]), n_frames=10)

    def test_needs_at_least_two_entries(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            ShotTable(boundaries=np.array([0]), n_frames=1)

    def test_rejects_non_positive_n_frames(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            ShotTable(boundaries=np.array([0, 1]), n_frames=0)

    def test_shot_span_out_of_range(self) -> None:
        table = ShotTable(boundaries=np.array([0, 4, 10]), n_frames=10)
        with pytest.raises(IndexError, match="out of range"):
            table.shot_span(2)


class TestVideo:
    def test_happy_path(self) -> None:
        video = make_video()
        assert video.n_frames == 10
        assert len(video.user_scores) == 2
        assert len(video.user_summaries) == 2
        assert video.user_scores[0].shape == (10,)
        assert video.user_summaries[0].dtype == np.bool_

    def test_is_frozen(self) -> None:
        video = make_video()
        with pytest.raises(dataclasses.FrozenInstanceError):
            video.dataset = "summe"  # type: ignore[misc]

    def test_features_are_read_only(self) -> None:
        video = make_video()
        assert video.features is not None
        with pytest.raises(ValueError, match="read-only"):
            video.features[0, 0] = 1.0

    def test_features_optional(self) -> None:
        video = Video(
            dataset="tvsum",
            video_id="v1",
            n_frames=5,
            shots=ShotTable(boundaries=[0, 5], n_frames=5),  # type: ignore[arg-type]
        )
        assert video.features is None
        assert video.fps is None
        assert video.user_scores == ()

    def test_rejects_feature_row_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length 10"):
            Video(
                dataset="tvsum",
                video_id="v1",
                n_frames=10,
                shots=ShotTable(boundaries=np.array([0, 10]), n_frames=10),
                features=np.zeros((9, 3), dtype=np.float32),
            )

    def test_rejects_feature_ndim(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            Video(
                dataset="tvsum",
                video_id="v1",
                n_frames=10,
                shots=ShotTable(boundaries=np.array([0, 10]), n_frames=10),
                features=np.zeros(10, dtype=np.float32),
            )

    def test_rejects_non_float_features(self) -> None:
        with pytest.raises(TypeError, match="float"):
            Video(
                dataset="tvsum",
                video_id="v1",
                n_frames=10,
                shots=ShotTable(boundaries=np.array([0, 10]), n_frames=10),
                features=np.zeros((10, 3), dtype=np.int32),
            )

    def test_rejects_shots_mismatch(self) -> None:
        with pytest.raises(ValueError, match="indexes"):
            Video(
                dataset="tvsum",
                video_id="v1",
                n_frames=10,
                shots=ShotTable(boundaries=np.array([0, 8]), n_frames=8),
            )

    def test_rejects_non_bool_summaries(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            Video(
                dataset="tvsum",
                video_id="v1",
                n_frames=4,
                shots=ShotTable(boundaries=np.array([0, 4]), n_frames=4),
                user_summaries=(np.array([1, 0, 1, 0]),),
            )

    def test_rejects_wrong_annotation_length(self) -> None:
        with pytest.raises(ValueError, match="length 4"):
            Video(
                dataset="tvsum",
                video_id="v1",
                n_frames=4,
                shots=ShotTable(boundaries=np.array([0, 4]), n_frames=4),
                user_scores=(np.array([1.0, 2.0, 3.0]),),
            )

    def test_rejects_bad_fps(self) -> None:
        for fps in (0, -30.0):
            with pytest.raises(ValueError, match="fps"):
                Video(
                    dataset="tvsum",
                    video_id="v1",
                    n_frames=4,
                    shots=ShotTable(boundaries=np.array([0, 4]), n_frames=4),
                    fps=fps,
                )

    def test_rejects_empty_ids(self) -> None:
        with pytest.raises(ValueError, match="dataset"):
            Video(
                dataset="",
                video_id="v1",
                n_frames=4,
                shots=ShotTable(boundaries=np.array([0, 4]), n_frames=4),
            )


class TestVideoPicks:
    """Dual frame space: original frames vs subsampled feature steps."""

    def make(self, **overrides: object) -> Video:
        kwargs: dict[str, object] = {
            "dataset": "tvsum",
            "video_id": "v0",
            "n_frames": 10,
            "shots": ShotTable(boundaries=np.array([0, 10]), n_frames=10),
            "features": np.zeros((4, 3), dtype=np.float32),
            "picks": np.array([0, 3, 6, 9]),
        }
        kwargs.update(overrides)
        return Video(**kwargs)  # type: ignore[arg-type]

    def test_happy_path(self) -> None:
        video = self.make()
        assert video.n_steps == 4
        assert video.n_frames == 10
        assert video.picks is not None and video.picks.dtype == np.int64
        with pytest.raises(ValueError, match="read-only"):
            video.picks[0] = 1

    def test_n_steps_without_picks(self) -> None:
        video = Video(
            dataset="tvsum",
            video_id="v1",
            n_frames=6,
            shots=ShotTable(boundaries=np.array([0, 6]), n_frames=6),
            features=np.zeros((6, 2), dtype=np.float32),
        )
        assert video.n_steps == 6
        assert (
            Video(
                dataset="tvsum",
                video_id="v1",
                n_frames=6,
                shots=ShotTable(boundaries=np.array([0, 6]), n_frames=6),
            ).n_steps
            == 6
        )

    def test_accepts_integral_float_picks(self) -> None:
        video = self.make(picks=np.array([0.0, 3.0, 6.0, 9.0]))
        assert video.picks is not None and video.picks.dtype == np.int64

    def test_rejects_out_of_range_picks(self) -> None:
        with pytest.raises(ValueError, match=r"within \[0, n_frames"):
            self.make(picks=np.array([0, 3, 6, 10]))

    def test_rejects_non_increasing_picks(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            self.make(picks=np.array([0, 3, 3, 9]))

    def test_rejects_empty_picks(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            self.make(picks=np.array([], dtype=np.int64), features=np.zeros((0, 3)))

    def test_rejects_picks_features_row_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length 4"):
            self.make(features=np.zeros((5, 3), dtype=np.float32))

    def test_features_without_picks_must_match_n_frames(self) -> None:
        with pytest.raises(ValueError, match="length 6"):
            Video(
                dataset="tvsum",
                video_id="v1",
                n_frames=6,
                shots=ShotTable(boundaries=np.array([0, 6]), n_frames=6),
                features=np.zeros((4, 3), dtype=np.float32),
            )


class TestScoresMaskBudget:
    def test_scores_happy(self) -> None:
        scores = Scores(video_id="v0", values=np.linspace(0.0, 1.0, 5))
        assert scores.values.shape == (5,)
        with pytest.raises(ValueError, match="read-only"):
            scores.values[0] = 2.0

    def test_scores_rejects_non_float(self) -> None:
        with pytest.raises(TypeError, match="float"):
            Scores(video_id="v0", values=np.array([1, 2, 3]))

    def test_scores_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            Scores(video_id="v0", values=np.zeros((2, 3)))

    def test_mask_happy(self) -> None:
        mask = Mask(video_id="v0", frames=[True, False, True])  # type: ignore[arg-type]
        assert mask.frames.dtype == np.bool_

    def test_mask_rejects_non_bool(self) -> None:
        with pytest.raises(TypeError, match="bool"):
            Mask(video_id="v0", frames=np.array([1.0, 0.0]))


class TestBudget:
    def test_accepts_fractions(self) -> None:
        assert Budget(fraction=0.15).fraction == 0.15
        assert Budget(fraction=1).fraction == 1.0

    def test_rejects_out_of_range(self) -> None:
        for bad in (0, -0.1, 1.5, float("nan")):
            with pytest.raises(ValueError):
                Budget(fraction=bad)


class TestSplit:
    def test_happy_path_from_lists(self) -> None:
        split = Split(train=["a", "b"], test=["c"])  # type: ignore[arg-type]
        assert split.train == ("a", "b")
        assert split.test == ("c",)

    def test_rejects_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            Split(train=("a", "b"), test=("b", "c"))

    def test_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            Split(train=("a", "a"), test=("b",))

    def test_rejects_empty_side(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Split(train=("a",), test=())
