"""Bench orchestration tests on synthetic, fully offline datasets.

The hand-computed case pins the whole chain (uniform scores -> knapsack
selection at a floor budget -> per-user F1 -> cross-user aggregation); the
h5-based cases re-derive expected numbers through the same verified
components the bench composes, which catches wiring mistakes (wrong budget,
wrong aggregation, fold leakage) without re-testing the components.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _synthetic import SyntheticSpec, write_eccv16_h5

from vsumbench.bench import Bench
from vsumbench.config import EvalConfig, RandomSplits
from vsumbench.data._eccv16 import Eccv16Adapter
from vsumbench.data.download import ChecksumError
from vsumbench.data.sources import SOURCES, DatasetSpec, FileSpec
from vsumbench.data.splits import canonical_splits
from vsumbench.models.uniform import UNIFORM_MODEL
from vsumbench.registry import DATASETS, RANK_METRICS, SELECTORS, SUMMARY_METRICS
from vsumbench.types import Budget, Scores, ShotTable, Video

# -- fixtures ---------------------------------------------------------------


def _tiny_video(video_id: str) -> Video:
    # 8 frames, two 4-frame shots; user 1 summarized shot 1, user 2 shot 2.
    user1 = np.array([True, True, True, True, False, False, False, False])
    user2 = np.array([False, False, False, False, True, True, True, True])
    return Video(
        dataset="bench-tiny",
        video_id=video_id,
        n_frames=8,
        shots=ShotTable(boundaries=np.array([0, 4, 8], dtype=np.int64), n_frames=8),
        user_summaries=(user1, user2),
    )


class _TinyAdapter:
    name = "bench-tiny"

    def load(self, root: Path) -> list[Video]:
        return [_tiny_video("v1"), _tiny_video("v2")]


def _register_once(name: str, adapter: Any) -> None:
    if name not in DATASETS.names():
        DATASETS.register(name, adapter)


_register_once("bench-tiny", _TinyAdapter())


class _StatefulModel:
    """Learning-free model that records fit calls on its instance."""

    name = "bench-stateful"

    def __init__(self) -> None:
        self.fit_sizes: list[int] = []

    def fit(self, videos: Any, config: Any) -> None:
        self.fit_sizes.append(len(videos))

    def predict(self, video: Video) -> Scores:
        return Scores(video_id=video.video_id, values=np.full(video.n_frames, 0.5, np.float32))


class _WrongIdModel:
    name = "bench-wrong-id"

    def fit(self, videos: Any, config: Any) -> None: ...

    def predict(self, video: Video) -> Scores:
        return Scores(video_id="not-this-video", values=np.ones(video.n_frames, np.float32))


class _WrongLengthModel:
    name = "bench-wrong-length"

    def fit(self, videos: Any, config: Any) -> None: ...

    def predict(self, video: Video) -> Scores:
        return Scores(video_id=video.video_id, values=np.ones(video.n_frames + 1, np.float32))


class _DupMetric:
    name = "bench-dup"

    def score(self, value: Any, video: Any) -> np.ndarray:
        return np.zeros(1)


@pytest.fixture()
def h5_root(tmp_path: Path) -> Path:
    specs = tuple(
        SyntheticSpec(
            video_id=f"video_{i}", n_frames=48, step=4, dim=6, n_users=3, shot_len=12, seed=i
        )
        for i in range(1, 7)
    )
    write_eccv16_h5(tmp_path / "synth.h5", specs)
    _register_once("bench-h5", Eccv16Adapter("bench-h5", "synth.h5"))
    return tmp_path


# -- constructor ------------------------------------------------------------


def test_convenience_constructor_applies_recommended_defaults() -> None:
    config = Bench(dataset="bench-tiny").config
    assert config.model == "uniform"
    assert config.selector == "knapsack"
    assert config.metrics == ("f1",)
    assert config.budgets == (0.1, 0.15, 0.3, 0.5)
    assert config.gt_aggregation == "max"
    assert config.splits == RandomSplits(n_folds=5, seed=0)


def test_explicit_config_cannot_be_mixed_with_overrides() -> None:
    config = Bench(dataset="bench-tiny").config
    with pytest.raises(ValueError, match="not both"):
        Bench(config, model="uniform")
    # data_root is runtime-only and allowed next to a config
    Bench(config, data_root="/tmp")


def test_dataset_is_required_without_config() -> None:
    with pytest.raises(ValueError, match="dataset"):
        Bench()


def test_splits_string_must_be_a_known_canonical_name() -> None:
    with pytest.raises(ValueError, match="unknown canonical split"):
        Bench(dataset="bench-tiny", splits="bogus-5")


def test_register_selects_the_model_for_runs() -> None:
    bench = Bench(dataset="bench-tiny")
    assert bench.config.model == "uniform"
    model = _StatefulModel()
    bench.register("bench-stateful", model)
    assert bench.config.model == "bench-stateful"


# -- the hand-computed end-to-end case ---------------------------------------


def _video_value(result: Any, vid: str, metric: str, key: str) -> Any:
    """Fetch one per-video value from whichever fold tested the video."""
    for fold in result.results["per_fold"]:
        videos = fold["videos"]
        assert isinstance(videos, dict)
        if vid in videos:
            return videos[vid][metric][key]
    raise AssertionError(f"{vid} was never tested")


def _tested_ids(result: Any) -> set[str]:
    return {
        vid
        for fold in result.results["per_fold"]
        for vid in (fold["videos"] if isinstance(fold["videos"], dict) else {})
    }


def test_hand_computed_f1_end_to_end(tmp_path: Path) -> None:
    # Budget 0.5 of 8 frames floors to 4; both shots are 4 frames scored 1.0,
    # knapsack takes the first shot -> machine summary == frames 0..3.
    # User 1 selected exactly those frames (F1 = 1.0), user 2 the others
    # (F1 = 0.0): gt max -> 1.0, gt mean -> 0.5.
    result = Bench(
        dataset="bench-tiny",
        budgets=(0.5,),
        splits=RandomSplits(n_folds=2),
        data_root=tmp_path,
    ).run(output_dir=tmp_path / "runs")
    assert _video_value(result, "v1", "f1", "0.5") == pytest.approx(1.0)
    assert _video_value(result, "v2", "f1", "0.5") == pytest.approx(1.0)
    summary = result.summary["f1"]["0.5"]
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["std"] == pytest.approx(0.0)
    assert summary["max"] == pytest.approx(1.0)
    assert summary["n_folds"] == 2


def test_gt_aggregation_mean_changes_the_number(tmp_path: Path) -> None:
    result = Bench(
        dataset="bench-tiny",
        budgets=(0.5,),
        gt_aggregation="mean",
        splits=RandomSplits(n_folds=2),
        data_root=tmp_path,
    ).run(output_dir=tmp_path / "runs")
    assert _video_value(result, "v1", "f1", "0.5") == pytest.approx(0.5)


def test_run_writes_manifest_with_expected_fields(tmp_path: Path) -> None:
    result = Bench(
        dataset="bench-tiny",
        budgets=(0.5,),
        splits=RandomSplits(n_folds=2),
        data_root=tmp_path,
    ).run(output_dir=tmp_path / "runs")
    manifest = result.manifest
    assert manifest["config"]["budgets"] == [0.5]
    # bench-tiny is unknown to SOURCES -> checksums recorded as unavailable
    assert manifest["data"]["files"] is None
    assert len(manifest["splits"]) == 2
    folds = manifest["splits"]
    assert isinstance(folds, list)
    assert sorted(folds[0]["train"]) == ["v2"] and sorted(folds[0]["test"]) == ["v1"]
    assert result.manifest_path.is_file()
    assert result.manifest_path.parent.parent == tmp_path / "runs"


# -- h5 datasets: wiring equals direct computation ---------------------------


def test_h5_run_matches_direct_computation(h5_root: Path, tmp_path: Path) -> None:
    result = Bench(
        dataset="bench-h5",
        selector="greedy",
        metrics=("f1",),
        budgets=(0.15, 0.5),
        gt_aggregation="mean",
        splits=RandomSplits(n_folds=3),
        data_root=h5_root,
    ).run(output_dir=tmp_path / "runs")
    selector = SELECTORS.get("greedy")
    f1 = SUMMARY_METRICS.get("f1")
    videos = {v.video_id: v for v in DATASETS.get("bench-h5").load(h5_root)}
    assert _tested_ids(result) == set(videos)
    for vid, video in videos.items():
        scores = UNIFORM_MODEL.predict(video)
        for fraction in (0.15, 0.5):
            mask = selector.select(
                scores, video.shots, Budget(fraction, "floor"), aggregation="mean"
            )
            expected = float(np.mean(f1.score(mask, video)))
            assert _video_value(result, vid, "f1", str(fraction)) == pytest.approx(
                expected
            ), f"{vid} @ {fraction}"
    assert len(result.results["per_fold"]) == 3
    test_sets = [set(fold["test"]) for fold in result.results["per_fold"]]
    assert set().union(*test_sets) == set(videos)
    assert not test_sets[0] & test_sets[1]
    assert set(result.summary["f1"]) == {"0.15", "0.5"}
    assert result.summary["f1"]["0.15"]["n_folds"] == 3


def test_canonical_splits_flow_end_to_end(tmp_path: Path) -> None:
    specs = tuple(
        SyntheticSpec(
            video_id=f"video_{i}", n_frames=24, step=4, dim=4, n_users=2, shot_len=8, seed=i
        )
        for i in range(1, 51)
    )
    write_eccv16_h5(tmp_path / "synth.h5", specs)
    _register_once("bench-canonical", Eccv16Adapter("bench-canonical", "synth.h5"))
    result = Bench(
        dataset="bench-canonical",
        splits="tvsum-canonical-5",
        budgets=(0.15,),
        data_root=tmp_path,
    ).run(output_dir=tmp_path / "runs")
    assert [len(fold["test"]) for fold in result.results["per_fold"]] == [10] * 5
    # The tvsum canonical folds OVERLAP on their test sets (verbatim
    # transcription: 50 slots, 34 distinct videos) — per-fold video records
    # keep each evaluation instead of overwriting a flat per-video map.
    canonical = canonical_splits("tvsum-canonical-5")
    expected_tested = {vid for fold in canonical for vid in fold.test}
    assert _tested_ids(result) == expected_tested
    assert len(expected_tested) == 34
    splits = result.manifest["splits"]
    assert isinstance(splits, list) and len(splits) == 5
    assert len(splits[0]["train"]) == 40


# -- data checksums -----------------------------------------------------------


def test_manifest_records_verified_checksums(h5_root: Path, tmp_path: Path) -> None:
    h5_path = h5_root / "synth.h5"
    SOURCES["bench-h5"] = DatasetSpec(
        name="bench-h5",
        files=(
            FileSpec(
                filename="synth.h5",
                sha256=hashlib.sha256(h5_path.read_bytes()).hexdigest(),
                size=h5_path.stat().st_size,
            ),
        ),
        archive_urls=(),
    )
    try:
        result = Bench(
            dataset="bench-h5",
            budgets=(0.5,),
            splits=RandomSplits(n_folds=2),
            data_root=h5_root,
        ).run(output_dir=tmp_path / "runs")
        files = result.manifest["data"]["files"]
        assert isinstance(files, list) and len(files) == 1
        assert files[0]["matches_spec"] is True
        assert files[0]["sha256"] == SOURCES["bench-h5"].files[0].sha256
    finally:
        del SOURCES["bench-h5"]


def test_checksum_mismatch_fails_the_run(h5_root: Path, tmp_path: Path) -> None:
    SOURCES["bench-h5"] = DatasetSpec(
        name="bench-h5",
        files=(FileSpec(filename="synth.h5", sha256="0" * 64, size=1),),
        archive_urls=(),
    )
    try:
        with pytest.raises(ChecksumError, match="does not match"):
            Bench(
                dataset="bench-h5",
                budgets=(0.5,),
                splits=RandomSplits(n_folds=2),
                data_root=h5_root,
            ).run(output_dir=tmp_path / "runs")
    finally:
        del SOURCES["bench-h5"]


# -- model lifecycle and validation -------------------------------------------


def test_registry_singleton_never_mutated_and_fit_sees_train_only(
    h5_root: Path, tmp_path: Path
) -> None:
    seen: list[list[str]] = []

    class _RecordingModel:
        name = "bench-recording"

        def __init__(self) -> None:
            self.fit_sizes: list[int] = []

        def fit(self, videos: Any, config: Any) -> None:
            self.fit_sizes.append(len(videos))
            seen.append(sorted(v.video_id for v in videos))

        def predict(self, video: Video) -> Scores:
            return Scores(video_id=video.video_id, values=np.full(video.n_frames, 1.0, np.float32))

    singleton = _RecordingModel()
    bench = Bench(
        dataset="bench-h5",
        budgets=(0.5,),
        splits=RandomSplits(n_folds=2),
        data_root=h5_root,
    )
    bench.register("bench-recording", singleton)
    bench.run(output_dir=tmp_path / "runs")

    # two folds fitted, each on the 3 videos the fold does not test
    assert len(seen) == 2 and all(len(fold) == 3 for fold in seen)
    assert set().union(*[set(fold) for fold in seen]) == {f"video_{i}" for i in range(1, 7)}
    # deepcopy per fold: the registered singleton stayed pristine
    assert singleton.fit_sizes == []


def test_wrong_video_id_scores_rejected(tmp_path: Path) -> None:
    bench = Bench(
        dataset="bench-tiny",
        budgets=(0.5,),
        splits=RandomSplits(n_folds=2),
        data_root=tmp_path,
    )
    bench.register("bench-wrong-id", _WrongIdModel())
    with pytest.raises(ValueError, match="was asked about"):
        bench.run(output_dir=tmp_path / "runs")


def test_wrong_length_scores_rejected(tmp_path: Path) -> None:
    bench = Bench(
        dataset="bench-tiny",
        budgets=(0.5,),
        splits=RandomSplits(n_folds=2),
        data_root=tmp_path,
    )
    bench.register("bench-wrong-length", _WrongLengthModel())
    with pytest.raises(ValueError, match="frames"):
        bench.run(output_dir=tmp_path / "runs")


# -- metric resolution ---------------------------------------------------------


def test_unknown_metric_rejected(tmp_path: Path) -> None:
    bench = Bench(dataset="bench-tiny", metrics=("nope",), data_root=tmp_path)
    with pytest.raises(ValueError, match="unknown metric"):
        bench.run(output_dir=tmp_path / "runs")


def test_ambiguous_metric_name_rejected(tmp_path: Path) -> None:
    RANK_METRICS.register("bench-dup", _DupMetric())
    SUMMARY_METRICS.register("bench-dup", _DupMetric())
    bench = Bench(dataset="bench-tiny", metrics=("bench-dup",), data_root=tmp_path)
    with pytest.raises(ValueError, match="both"):
        bench.run(output_dir=tmp_path / "runs")


# -- non-finite values ------------------------------------------------------------


def test_nan_metric_values_land_as_null(tmp_path: Path) -> None:
    class _NanAdapter:
        name = "bench-nan"

        def load(self, root: Path) -> list[Video]:
            videos = []
            for vid in ("n1", "n2"):
                videos.append(
                    Video(
                        dataset="bench-nan",
                        video_id=vid,
                        n_frames=8,
                        shots=ShotTable(boundaries=np.array([0, 8], dtype=np.int64), n_frames=8),
                        user_scores=(
                            np.full(8, 3.0),  # constant -> tau undefined (NaN)
                            np.arange(8, dtype=float),
                        ),
                        user_summaries=(np.array([True] * 8),),
                    )
                )
            return videos

    _register_once("bench-nan", _NanAdapter())
    result = Bench(
        dataset="bench-nan",
        metrics=("kendall_tau",),
        budgets=(0.5,),
        splits=RandomSplits(n_folds=2),
        data_root=tmp_path,
    ).run(output_dir=tmp_path / "runs")
    # uniform scores are constant -> tau NaN against every user -> null
    assert _video_value(result, "n1", "kendall_tau", "none") is None
    assert result.summary["kendall_tau"]["none"]["mean"] is None


def test_explicit_config_object_runs(tmp_path: Path) -> None:
    # greedy uses the lineage strict-less-than rule: a shot exactly filling
    # the budget is skipped, so budget 0.5 of 8 frames selects nothing while
    # 0.75 (limit 6) takes the first 4-frame shot.
    config = EvalConfig(
        dataset="bench-tiny",
        model="uniform",
        selector="greedy",
        metrics=("f1",),
        budgets=(0.5, 0.75),
        splits=RandomSplits(n_folds=2),
    )
    result = Bench(config, data_root=tmp_path).run(output_dir=tmp_path / "runs")
    assert result.summary["f1"]["0.5"]["mean"] == pytest.approx(0.0)
    assert result.summary["f1"]["0.75"]["mean"] == pytest.approx(1.0)
