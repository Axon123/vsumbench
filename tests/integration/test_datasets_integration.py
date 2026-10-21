"""Integration tests against the real canonical datasets.

Opt-in via ``-m integration``; skipped when the canonical file is not in the
cache root. Get the data with ``vsumm download {tvsum,summe}`` (or drop the
h5 into ``$VSUM_CACHE_DIR`` manually and let the checksum verify it).
"""

from __future__ import annotations

import numpy as np
import pytest

from vsumbench.config import CanonicalSplits
from vsumbench.data import download as download_mod
from vsumbench.data.sources import SOURCES
from vsumbench.data.splits import build_splits
from vsumbench.registry import DATASETS

pytestmark = pytest.mark.integration

#: (dataset, n_videos, annotator counts seen in the canonical files)
EXPECTED = {"tvsum": (50, {20}), "summe": (25, {15, 16, 17, 18})}


@pytest.fixture(params=sorted(EXPECTED))
def dataset(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    path = download_mod.cache_dir() / SOURCES[name].files[0].filename
    if not path.is_file():
        pytest.skip(f"{path} not present; run `vsumm download {name}`")
    return name


def test_checksum_verifies(dataset: str) -> None:
    download_mod.verify_dataset(dataset)


def test_load_full_dataset(dataset: str) -> None:
    n_videos, user_counts = EXPECTED[dataset]
    videos = DATASETS.get(dataset).load(download_mod.cache_dir())

    assert len(videos) == n_videos
    assert [v.video_id for v in videos] == [f"video_{i}" for i in range(1, n_videos + 1)]
    for video in videos:
        assert video.dataset == dataset
        assert video.fps is None and video.user_scores == ()
        assert len(video.user_summaries) in user_counts
        for summary in video.user_summaries:
            assert summary.shape == (video.n_frames,) and summary.dtype == np.bool_
        assert video.features is not None and video.features.shape[1] == 1024
        assert video.n_steps == video.features.shape[0]
        assert video.picks is not None
        # the canonical files subsample every 15th frame (verified archaeology)
        np.testing.assert_array_equal(video.picks, np.arange(0, video.n_frames, 15, dtype=np.int64))


def test_canonical_splits_match_loaded_ids(dataset: str) -> None:
    videos = DATASETS.get(dataset).load(download_mod.cache_dir())
    folds = build_splits(
        CanonicalSplits(name=f"{dataset}-canonical-5"), [v.video_id for v in videos]
    )
    assert len(folds) == 5


def test_gtscore_rank_metrics_echo_is_perfect(dataset: str) -> None:
    # Re-predicting the dataset's own gtscore (expanded to frame space) and
    # sampling it back at picks must rank perfectly: tau-b = rho = 1 for
    # every real video, tying the metric, the loader's gt_score and the
    # step-space sampling convention together on canonical data.
    from vsumbench.metrics.rank import (
        KENDALL_TAU_GTSCORE_METRIC,
        SPEARMAN_RHO_GTSCORE_METRIC,
    )
    from vsumbench.models.score_space import expand_step_scores
    from vsumbench.types import Scores

    videos = DATASETS.get(dataset).load(download_mod.cache_dir())
    for video in videos:
        assert video.gt_score is not None and video.picks is not None
        echoed = expand_step_scores(video.gt_score, video.picks, video.n_frames)
        scores = Scores(video_id=video.video_id, values=echoed)
        assert KENDALL_TAU_GTSCORE_METRIC.score(scores, video)[0] == 1.0
        assert SPEARMAN_RHO_GTSCORE_METRIC.score(scores, video)[0] == 1.0
