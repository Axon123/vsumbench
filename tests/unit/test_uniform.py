"""Tests for the built-in ``uniform`` reference model."""

from __future__ import annotations

import numpy as np
from _synthetic import SyntheticSpec, build_arrays

from vsumbench.config import CanonicalSplits, EvalConfig
from vsumbench.registry import MODELS
from vsumbench.types import ShotTable, Video


def _config() -> EvalConfig:
    return EvalConfig(
        dataset="synthetic",
        model="uniform",
        selector="knapsack",
        metrics=("f1",),
        budgets=(0.15,),
        splits=CanonicalSplits(name="tvsum-canonical-5"),
    )


def _video() -> Video:
    arrays = build_arrays(SyntheticSpec(n_frames=32, step=4, dim=4, n_users=2, shot_len=8))
    boundaries = np.concatenate([arrays["change_points"][:, 0], [32]])
    return Video(
        dataset="synthetic",
        video_id="video_1",
        n_frames=32,
        shots=ShotTable(boundaries=boundaries, n_frames=32),
    )


def test_uniform_discovers_via_entry_point() -> None:
    assert "uniform" in MODELS.names()
    model = MODELS.get("uniform")
    assert model.name == "uniform"


def test_uniform_fit_accepts_train_split_and_config() -> None:
    model = MODELS.get("uniform")
    model.fit([_video()], _config())  # no-op: must not raise


def test_uniform_predict_scores_every_frame_equally() -> None:
    model = MODELS.get("uniform")
    scores = model.predict(_video())
    assert scores.video_id == "video_1"
    assert scores.values.shape == (32,)
    assert scores.values.dtype == np.float32
    assert bool(np.all(scores.values == np.float32(1.0)))
    assert not scores.values.flags.writeable
