"""Protocol-level alignment: vsumbench vs. the eccv16-lineage reference code.

The reference pipeline below is a verbatim transcription (py3, attribution
in comments) of:

- ``generate_summary`` / ``knapSack`` — e-apostolidis/PGL-SUM
  ``evaluation/generate_summary.py`` + ``knapsack_implementation.py``
  (byte-identical files also ship in CA-SUM and SUM-GAN-AAE/sl);
- the ``method='rank'`` greedy branch and the per-user F1 loop —
  KaiyangZhou/pytorch-vsumm-reinforce ``vsum_tools.py`` (courtesy header in
  the ok1zjf/VASNet fork confirms the lineage).

Using each canonical h5's ``gtscore`` as the "model scores" (exactly what
the repos feed their evaluator at test time), the full path
scores -> frame expansion -> shot means -> floor(15%) budget -> selection
-> per-user F1 must agree with ours on every video. Model-level golden
fixtures from Docker-pinned original environments follow in M3
(see README.md).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from vsumbench.data import download as download_mod
from vsumbench.data.sources import SOURCES
from vsumbench.metrics.f1 import F1_METRIC
from vsumbench.models.score_space import expand_step_scores
from vsumbench.registry import DATASETS, SELECTORS
from vsumbench.selection.knapsack import KNAPSACK_SELECTOR
from vsumbench.types import Budget, Scores, Video

pytestmark = pytest.mark.alignment

# --- reference transcriptions --------------------------------------------------


def _reference_knapsack(
    W: int,  # noqa: N803 -- verbatim reference naming
    wt: list[int],
    val: list[float],
    n: int,
) -> list[int]:
    # e-apostolidis/PGL-SUM evaluation/knapsack_implementation.py (Bhavya Jain)
    K: list[list[float]] = [[0 for _ in range(W + 1)] for _ in range(n + 1)]  # noqa: N806
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
    selected: list[int] = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]
    return selected


def _reference_select(
    ypred: np.ndarray,
    cps: np.ndarray,
    n_frames: int,
    nfps: list[int],
    positions: np.ndarray,
    proportion: float = 0.15,
    method: str = "knapsack",
) -> np.ndarray:
    # generate_summary, knapsack branch from e-apostolidis/PGL-SUM, rank
    # branch from KaiyangZhou/pytorch-vsumm-reinforce vsum_tools.py.
    n_segs = cps.shape[0]
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1] + 1)
        seg_score.append(float(frame_scores[start:end].mean()))

    limits = int(n_frames * proportion)
    if method == "knapsack":
        picks = _reference_knapsack(limits, nfps, seg_score, n_segs)
    elif method == "rank":
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]

    summary = np.zeros(n_frames, dtype=np.float32)
    for seg_idx in range(n_segs):
        if seg_idx in picks:
            start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1] + 1)
            summary[start:end] = 1
    return summary


def _reference_user_f1s(machine_summary: np.ndarray, user_summary: np.ndarray) -> list[float]:
    # Per-user loop of evaluate_summary (PGL-SUM shape, DR-DSN formula).
    machine_summary = machine_summary.astype(bool)
    f_scores = []
    for user in range(user_summary.shape[0]):
        gt = user_summary[user].astype(bool)
        overlapped = np.logical_and(machine_summary, gt).sum()
        precision = overlapped / machine_summary.sum()
        recall = overlapped / gt.sum()
        if precision == 0 and recall == 0:
            f_scores.append(0.0)
        else:
            f_scores.append(2 * precision * recall / (precision + recall))
    return f_scores


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(params=["tvsum", "summe"])
def dataset(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    path = download_mod.cache_dir() / SOURCES[name].files[0].filename
    if not path.is_file():
        pytest.skip(f"{path} not present; run `vsumm download {name}`")
    return name


# --- tests ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "selector_name"),
    [("knapsack", "knapsack"), ("rank", "greedy")],
)
def test_selection_matches_reference(dataset: str, method: str, selector_name: str) -> None:
    h5_path = download_mod.cache_dir() / SOURCES[dataset].files[0].filename
    videos = {v.video_id: v for v in DATASETS.get(dataset).load(download_mod.cache_dir())}
    selector = SELECTORS.get(selector_name)
    with h5py.File(h5_path, "r") as hdf:
        for key, video in videos.items():
            assert video.picks is not None
            gtscore = np.asarray(hdf[f"{key}/gtscore"][()], dtype=np.float32)
            reference = _reference_select(
                gtscore,
                video_to_change_points(video),
                video.n_frames,
                list(np.diff(video.shots.boundaries)),
                video.picks,
                method=method,
            )
            frame_scores = expand_step_scores(gtscore, video.picks, video.n_frames)
            ours = selector.select(
                Scores(video.video_id, frame_scores),
                video.shots,
                Budget(0.15),
                aggregation="mean",
            )
            assert np.array_equal(
                ours.frames, reference.astype(bool)
            ), f"{dataset}/{key}: selection differs from reference ({method})"


def test_f1_matches_reference(dataset: str) -> None:
    h5_path = download_mod.cache_dir() / SOURCES[dataset].files[0].filename
    videos = {v.video_id: v for v in DATASETS.get(dataset).load(download_mod.cache_dir())}
    with h5py.File(h5_path, "r") as hdf:
        for key, video in videos.items():
            assert video.picks is not None
            gtscore = np.asarray(hdf[f"{key}/gtscore"][()], dtype=np.float32)
            reference = _reference_select(
                gtscore,
                video_to_change_points(video),
                video.n_frames,
                list(np.diff(video.shots.boundaries)),
                video.picks,
            )
            reference_f1s = _reference_user_f1s(
                reference, np.asarray(hdf[f"{key}/user_summary"][()]).astype(bool)
            )
            frame_scores = expand_step_scores(gtscore, video.picks, video.n_frames)
            mask = KNAPSACK_SELECTOR.select(
                Scores(video.video_id, frame_scores),
                video.shots,
                Budget(0.15),
                aggregation="mean",
            )
            ours = F1_METRIC.score(mask, video)
            assert np.allclose(ours, reference_f1s, rtol=1e-12), (
                f"{dataset}/{key}: per-user F1 differs\n"
                f"reference={np.asarray(reference_f1s)}\nours={ours}"
            )


def video_to_change_points(video: Video) -> np.ndarray:
    """Rebuild inclusive [start, end] change points from our half-open shots."""
    boundaries = video.shots.boundaries
    return np.stack([boundaries[:-1], boundaries[1:] - 1], axis=1)
