"""Synthetic eccv16-format h5 fixtures: tiny, deterministic, fully offline.

The builder mirrors the schema of the canonical TVSum/SumMe files (see
``vsumbench.data._eccv16``) including the deliberately-ignored keys, so
loader tests exercise exactly the code paths real data takes. Ground-truth
arrays are returned for assertion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class SyntheticSpec:
    """Parameters of one synthetic video; defaults give a minimal valid video."""

    video_id: str = "video_1"
    n_frames: int = 64
    step: int = 4  # picks stride -> n_steps = ceil(n_frames / step)
    dim: int = 8
    n_users: int = 3
    shot_len: int = 16  # uniform shots (last one may be shorter)
    seed: int = 0


def build_arrays(spec: SyntheticSpec) -> dict[str, np.ndarray]:
    """All arrays of one synthetic video, keyed as they land in the h5 file."""
    rng = np.random.default_rng(spec.seed)
    picks = np.arange(0, spec.n_frames, spec.step, dtype=np.int64)
    features = rng.random((picks.size, spec.dim), dtype=np.float32) * 0.5
    selected = rng.random((spec.n_users, spec.n_frames)) < 0.3
    for row in selected:  # every annotator selects at least one frame
        if not row.any():
            row[int(rng.integers(spec.n_frames))] = True
    # the canonical files store summaries as float32 0/1, so mirror that here
    user_summary = selected.astype(np.float32)
    starts = np.arange(0, spec.n_frames, spec.shot_len, dtype=np.int64)
    ends = np.minimum(starts + spec.shot_len - 1, spec.n_frames - 1)
    change_points = np.stack([starts, ends], axis=1)
    gtscore = rng.random(picks.size, dtype=np.float32)
    return {
        "picks": picks,
        "features": features,
        "user_summary": user_summary,
        "change_points": change_points,
        "n_frame_per_seg": (ends - starts + 1),
        "n_frames": np.int64(spec.n_frames),
        "n_steps": np.int64(picks.size),
        "gtscore": gtscore,
        "gtsummary": (gtscore > 0.5).astype(np.float32),
        "v_length": np.float32(spec.n_frames / 30.0),
    }


def write_eccv16_h5(path: Path, specs: Sequence[SyntheticSpec]) -> dict[str, dict[str, np.ndarray]]:
    """Write a synthetic eccv16 h5 file; returns the arrays per video id."""
    ground_truth: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "w") as handle:
        for spec in specs:
            arrays = build_arrays(spec)
            ground_truth[spec.video_id] = arrays
            group = handle.create_group(spec.video_id)
            for key, value in arrays.items():
                group.create_dataset(key, data=value)
    return ground_truth
