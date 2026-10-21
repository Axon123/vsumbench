"""Unit tests for the eccv16 TVSum/SumMe adapters over synthetic h5 fixtures."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from _synthetic import SyntheticSpec, write_eccv16_h5

from vsumbench.data._eccv16 import Eccv16Adapter
from vsumbench.data.summe import SUMME_ADAPTER
from vsumbench.data.tvsum import TVSUM_ADAPTER
from vsumbench.types import DatasetAdapter

ADAPTERS = pytest.mark.parametrize(
    "adapter", [TVSUM_ADAPTER, SUMME_ADAPTER], ids=["tvsum", "summe"]
)


@ADAPTERS
def test_adapter_satisfies_protocol(adapter: Eccv16Adapter) -> None:
    assert isinstance(adapter, DatasetAdapter)
    assert adapter.name in ("tvsum", "summe")


@ADAPTERS
def test_load_maps_raw_fields_untouched(adapter: Eccv16Adapter, tmp_path: Path) -> None:
    h5_path = tmp_path / f"eccv16_dataset_{adapter.name}_google_pool5.h5"
    truth = write_eccv16_h5(
        h5_path, [SyntheticSpec(video_id="video_1"), SyntheticSpec(video_id="video_2", seed=1)]
    )
    videos = adapter.load(tmp_path)

    assert [v.video_id for v in videos] == ["video_1", "video_2"]
    for video in videos:
        arrays = truth[video.video_id]
        assert video.dataset == adapter.name
        assert video.n_frames == int(arrays["n_frames"])
        assert video.n_steps == arrays["picks"].size
        assert video.features is not None
        assert video.features.shape == arrays["features"].shape
        np.testing.assert_array_equal(video.features, arrays["features"])
        np.testing.assert_array_equal(video.picks, arrays["picks"])
        # GT 原始性: per-user summaries verbatim, in original frame space;
        # the file's aggregated gtscore is carried raw (reference-protocol
        # rank metrics opt in), gtsummary stays behind.
        assert len(video.user_summaries) == arrays["user_summary"].shape[0]
        assert video.user_summaries[0].dtype == np.bool_
        for got, expected in zip(video.user_summaries, arrays["user_summary"], strict=True):
            np.testing.assert_array_equal(got, expected)
        assert video.user_scores == ()
        assert video.gt_score is not None
        np.testing.assert_array_equal(video.gt_score, arrays["gtscore"])
        assert video.fps is None
        # half-open boundaries derived from the file's inclusive ends
        cp = arrays["change_points"]
        np.testing.assert_array_equal(
            video.shots.boundaries, np.concatenate([cp[:, 0], [cp[-1, 1] + 1]])
        )
        with pytest.raises(ValueError, match="read-only"):
            video.features[0, 0] = 1.0


def test_natural_video_order(tmp_path: Path) -> None:
    h5_path = tmp_path / "eccv16_dataset_tvsum_google_pool5.h5"
    write_eccv16_h5(
        h5_path,
        [SyntheticSpec(video_id=f"video_{i}", seed=i) for i in (1, 2, 10, 3)],
    )
    ids = [v.video_id for v in TVSUM_ADAPTER.load(tmp_path)]
    assert ids == ["video_1", "video_2", "video_3", "video_10"]


def test_missing_gtscore_loads_with_gt_score_none(tmp_path: Path) -> None:
    # gtscore is optional in the file: absence disables the reference-protocol
    # rank metrics (they raise) but must not break loading.
    h5_path = tmp_path / "eccv16_dataset_tvsum_google_pool5.h5"
    write_eccv16_h5(h5_path, [SyntheticSpec()])
    with h5py.File(h5_path, "r+") as handle:
        del handle["video_1"]["gtscore"]
    videos = TVSUM_ADAPTER.load(tmp_path)
    assert videos[0].gt_score is None


def test_missing_file_names_download_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="vsumm download tvsum"):
        TVSUM_ADAPTER.load(tmp_path)


def _with_h5(tmp_path: Path, spec: SyntheticSpec, mutate: str) -> Path:
    h5_path = tmp_path / "eccv16_dataset_tvsum_google_pool5.h5"
    write_eccv16_h5(h5_path, [spec])
    with h5py.File(h5_path, "r+") as handle:
        _MUTATIONS[mutate](handle["video_1"])
    return h5_path


def _drop_user_summary(group: h5py.Group) -> None:
    del group["user_summary"]


def _gap_in_change_points(group: h5py.Group) -> None:
    group["change_points"][1, 0] += 1


def _wrong_span(group: h5py.Group) -> None:
    group["change_points"][-1, 1] -= 1


def _bad_picks_start(group: h5py.Group) -> None:
    group["picks"][0] = 1


def _picks_beyond_end(group: h5py.Group) -> None:
    group["picks"][-1] = group["n_frames"][()]


def _pick_row_mismatch(group: h5py.Group) -> None:
    shortened = np.asarray(group["picks"][()])[:-1]
    del group["picks"]
    group.create_dataset("picks", data=shortened)


def _non_binary_summary(group: h5py.Group) -> None:
    group["user_summary"][0, 0] = 0.5


def _summary_width_mismatch(group: h5py.Group) -> None:
    del group["user_summary"]
    group.create_dataset("user_summary", data=np.zeros((2, 3), dtype=np.float32))


def _non_integer_change_points(group: h5py.Group) -> None:
    cp = np.asarray(group["change_points"][()]).astype(np.float64)
    cp[0, 0] = 0.5
    del group["change_points"]
    group.create_dataset("change_points", data=cp)


def _gtscore_row_mismatch(group: h5py.Group) -> None:
    shortened = np.asarray(group["gtscore"][()])[:-1]
    del group["gtscore"]
    group.create_dataset("gtscore", data=shortened)


def _non_finite_gtscore(group: h5py.Group) -> None:
    group["gtscore"][0] = np.nan


_MUTATIONS = {
    "drop_user_summary": _drop_user_summary,
    "gap_in_change_points": _gap_in_change_points,
    "wrong_span": _wrong_span,
    "bad_picks_start": _bad_picks_start,
    "picks_beyond_end": _picks_beyond_end,
    "pick_row_mismatch": _pick_row_mismatch,
    "non_binary_summary": _non_binary_summary,
    "summary_width_mismatch": _summary_width_mismatch,
    "non_integer_change_points": _non_integer_change_points,
    "gtscore_row_mismatch": _gtscore_row_mismatch,
    "non_finite_gtscore": _non_finite_gtscore,
}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("drop_user_summary", "missing required datasets"),
        ("gap_in_change_points", "tile"),
        ("wrong_span", "span"),
        ("bad_picks_start", "start at 0"),
        ("picks_beyond_end", "below n_frames"),
        ("pick_row_mismatch", "rows but picks has"),
        ("non_binary_summary", "binary"),
        ("summary_width_mismatch", "n_frames"),
        ("non_integer_change_points", "integers"),
        ("gtscore_row_mismatch", "picks"),
        ("non_finite_gtscore", "finite"),
    ],
)
def test_corrupt_files_rejected(tmp_path: Path, mutation: str, message: str) -> None:
    _with_h5(tmp_path, SyntheticSpec(), mutation)
    with pytest.raises(ValueError, match=message):
        TVSUM_ADAPTER.load(tmp_path)
