"""Shared loader for eccv16-format TVSum/SumMe h5 files.

Schema (per video group, keys ``video_1 .. video_N``), as verified against
the canonical files byte-for-byte:

==================  =======================  ================================
dataset             shape                    frame space
==================  =======================  ================================
``features``        ``[n_steps, 1024]`` f32  step (one per ``picks`` entry)
``picks``           ``[n_steps]`` i64        index into original frames
``change_points``   ``[n_shots, 2]`` i64     original; inclusive ends tiling
``user_summary``    ``[n_users, n_frames]``  original; strictly 0/1 floats
``n_frames``        scalar i64               original frame count
``n_frame_per_seg`` ``[n_shots]`` i64        = shot lengths (not loaded)
``gtscore``         ``[n_steps]``            aggregated; carried raw as ``Video.gt_score``
``gtsummary``       ``[n_steps]``            aggregated; ignored (nothing consumes it)
``n_steps``, ``v_length``            redundant; ignored
==================  =======================  ================================

``picks`` is exactly ``range(0, n_frames, 15)`` in both canonical files;
loaders keep it as data, never as an assumed constant. Neither file carries
per-user importance scores, so :attr:`Video.user_scores` stays empty. The
aggregated ``gtscore`` is loaded verbatim (present in both canonical files;
a file without that key loads with ``gt_score=None``). ``video_name`` exists
only in the SumMe file (25/25 groups, e.g. ``b'Air_Force_One'``); TVSum
groups carry no such key — the asymmetry that matters when joining the
original ydata annotations (docs/protocols/tvsum-summe-originals.md §3).
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from vsumbench.types import ShotTable, Video

__all__ = ["Eccv16Adapter"]

_REQUIRED_KEYS = ("features", "picks", "change_points", "user_summary", "n_frames")
_VIDEO_KEY = re.compile(r"video_(\d+)")


class Eccv16Adapter:
    """DatasetAdapter over one eccv16-format h5 file.

    Only format搬运 happens here: no normalization, no per-user aggregation,
    no gtscore->summary conversion (AGENTS.md rule 4). ``root`` must already
    contain the canonical h5 file — fetching and checksums belong to
    :mod:`vsumbench.data.download`.
    """

    def __init__(self, dataset_name: str, h5_filename: str) -> None:
        self.name = dataset_name
        self._h5_filename = h5_filename

    def load(self, root: Path) -> list[Video]:
        path = root / self._h5_filename
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; run `vsumm download {self.name}` or point the "
                f"loader at a directory holding {self._h5_filename}"
            )
        videos: list[Video] = []
        with h5py.File(path, "r") as handle:
            for key in _video_keys(handle):
                videos.append(_load_video(handle[key], self.name, key))
        if not videos:
            raise ValueError(f"{path} contains no video groups")
        return videos

    def __repr__(self) -> str:
        return f"Eccv16Adapter({self.name!r}, {self._h5_filename!r})"


def _video_keys(handle: h5py.File) -> list[str]:
    groups = [k for k in handle if isinstance(handle[k], h5py.Group)]

    def sort_key(key: str) -> tuple[int, int, str]:
        match = _VIDEO_KEY.fullmatch(key)
        return (0, int(match.group(1)), "") if match else (1, 0, key)

    return sorted(groups, key=sort_key)


def _load_video(group: h5py.Group, dataset: str, key: str) -> Video:
    where = f"{dataset}/{key}"
    missing = [k for k in _REQUIRED_KEYS if k not in group]
    if missing:
        raise ValueError(f"{where}: missing required datasets {missing}")

    n_frames = int(group["n_frames"][()])
    if n_frames < 1:
        raise ValueError(f"{where}: n_frames must be positive, got {n_frames}")

    features = np.asarray(group["features"][()])
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f"{where}: features must be a 2-D float array, got {features.shape}")
    picks = np.asarray(group["picks"][()])
    if picks.ndim != 1 or not np.issubdtype(picks.dtype, np.integer):
        raise ValueError(f"{where}: picks must be a 1-D integer array")
    if picks.shape[0] != features.shape[0]:
        raise ValueError(
            f"{where}: features has {features.shape[0]} rows but picks has {picks.shape[0]} entries"
        )
    if picks[0] != 0 or picks[-1] >= n_frames or bool(np.any(np.diff(picks) <= 0)):
        raise ValueError(
            f"{where}: picks must start at 0, stay strictly increasing and end "
            f"below n_frames={n_frames}"
        )

    user_summary = np.asarray(group["user_summary"][()])
    if user_summary.ndim != 2 or user_summary.shape[1] != n_frames:
        raise ValueError(
            f"{where}: user_summary must be [n_users, n_frames={n_frames}], "
            f"got {user_summary.shape}"
        )
    if not np.isin(user_summary, (0, 1)).all():
        raise ValueError(f"{where}: user_summary must be strictly binary")

    boundaries = _shot_boundaries(np.asarray(group["change_points"][()]), where, n_frames)

    return Video(
        dataset=dataset,
        video_id=key,
        n_frames=n_frames,
        shots=ShotTable(boundaries=boundaries, n_frames=n_frames),
        features=features,
        picks=picks,
        gt_score=_gt_score(group, where, picks.shape[0]),
        user_summaries=tuple(np.asarray(row, dtype=bool) for row in user_summary),
    )


def _gt_score(group: h5py.Group, where: str, n_steps: int) -> np.ndarray | None:
    """Load the file's aggregated importance scores, carried verbatim.

    ``gtscore`` is the dataset's own cross-user aggregate (how it was built
    is undocumented upstream) in step space; loading it does not make it the
    ground truth for anything by default (rule 4) — the reference-protocol
    rank metrics opt in explicitly. Optional in the file: absence leaves
    ``Video.gt_score`` as None.
    """
    if "gtscore" not in group:
        return None
    values = np.asarray(group["gtscore"][()])
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{where}: gtscore must be a 1-D float array, got {values.shape}")
    if values.shape[0] != n_steps:
        raise ValueError(f"{where}: gtscore has {values.shape[0]} entries for {n_steps} picks")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{where}: gtscore must be finite")
    return values


def _shot_boundaries(change_points: np.ndarray, where: str, n_frames: int) -> np.ndarray:
    """Convert the file's inclusive-end segments to half-open boundaries.

    The canonical files tile ``[0, n_frames)`` with ``[start, end]`` rows
    (``change_points[i+1, 0] == change_points[i, 1] + 1``); anything else is
    rejected as corrupt rather than silently re-interpreted.
    """
    if change_points.ndim != 2 or change_points.shape[1] != 2:
        raise ValueError(f"{where}: change_points must be [n_shots, 2], got {change_points.shape}")
    if not np.issubdtype(change_points.dtype, np.integer):
        raise ValueError(f"{where}: change_points must be integers")
    if change_points[0, 0] != 0 or change_points[-1, 1] != n_frames - 1:
        raise ValueError(
            f"{where}: change_points must span [0, n_frames-1], got "
            f"[{change_points[0, 0]}, {change_points[-1, 1]}]"
        )
    if not bool(np.all(change_points[1:, 0] == change_points[:-1, 1] + 1)):
        raise ValueError(f"{where}: change_points rows must tile without gaps or overlaps")
    return np.concatenate([change_points[:, 0], [change_points[-1, 1] + 1]])
