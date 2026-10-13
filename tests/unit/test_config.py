"""Unit tests for vsumbench.config: validation defaults and JSON round-trips."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from vsumbench.config import (
    CanonicalSplits,
    EvalConfig,
    RandomSplits,
    ScoreNormalization,
    TrainingConfig,
)


def make_config() -> EvalConfig:
    return EvalConfig(
        dataset="tvsum",
        model="uniform",
        selector="shot_based",
        metrics=("f1",),
        budgets=(0.1, 0.15),
        splits=CanonicalSplits(name="tvsum-canonical-5"),
    )


def make_full_config() -> EvalConfig:
    return EvalConfig(
        dataset="summe",
        model="pglsum",
        selector="knapsack",
        metrics=("f1", "kendall_tau"),
        budgets=(0.1, 0.15, 0.3, 0.5),
        splits=RandomSplits(n_folds=5, seed=7),
        seed=123,
        gt_aggregation="mean",
        segment_score_aggregation="max",
        score_normalization=ScoreNormalization(method="min_max"),
        training=TrainingConfig(
            epochs=10,
            learning_rate=0.001,
            batch_size=4,
            device="cuda",
            extra={"hidden_size": 128, "layers": [1, 2]},
        ),
    )


class TestValidation:
    def test_rejects_empty_metrics(self) -> None:
        with pytest.raises(ValueError, match="metrics"):
            dataclasses.replace(make_config(), metrics=())

    def test_rejects_empty_budgets(self) -> None:
        with pytest.raises(ValueError, match="budgets"):
            dataclasses.replace(make_config(), budgets=())

    def test_rejects_out_of_range_budgets(self) -> None:
        for bad in ((0.0,), (-0.1,), (1.5,), (float("nan"),)):
            with pytest.raises(ValueError, match="budgets"):
                dataclasses.replace(make_config(), budgets=bad)

    def test_rejects_empty_strings(self) -> None:
        with pytest.raises(ValueError, match="dataset"):
            dataclasses.replace(make_config(), dataset="")

    def test_rejects_bad_gt_aggregation(self) -> None:
        with pytest.raises(ValueError, match="gt_aggregation"):
            dataclasses.replace(make_config(), gt_aggregation="median")  # type: ignore[arg-type]

    def test_rejects_bad_segment_aggregation(self) -> None:
        with pytest.raises(ValueError, match="segment_score_aggregation"):
            dataclasses.replace(make_config(), segment_score_aggregation="sum")  # type: ignore[arg-type]

    def test_rejects_bad_split_type(self) -> None:
        with pytest.raises(TypeError, match="splits"):
            dataclasses.replace(make_config(), splits="canonical")  # type: ignore[arg-type]

    def test_rejects_bad_training(self) -> None:
        for kwargs in (
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"batch_size": -1},
            {"device": "gpu"},
        ):
            with pytest.raises(ValueError):
                TrainingConfig(**kwargs)

    def test_rejects_bad_random_splits(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            RandomSplits(n_folds=1)

    def test_rejects_bad_normalization(self) -> None:
        with pytest.raises(ValueError, match="method"):
            ScoreNormalization(method="l2")  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_config().dataset = "summe"  # type: ignore[misc]


class TestDefaults:
    def test_recommended_defaults(self) -> None:
        config = make_config()
        assert config.seed == 0
        assert config.gt_aggregation == "max"
        assert config.segment_score_aggregation == "mean"
        assert config.score_normalization.method == "none"
        assert config.training.epochs == 100
        assert config.training.device == "cpu"
        assert dict(config.training.extra) == {}


class TestJsonRoundTrip:
    def test_full_round_trip(self) -> None:
        config = make_full_config()
        assert EvalConfig.from_json(config.to_json()) == config

    def test_defaults_round_trip(self) -> None:
        config = make_config()
        assert EvalConfig.from_json(config.to_json()) == config

    def test_splits_encoded_with_kind_tag(self) -> None:
        data = make_full_config().to_dict()
        assert data["splits"] == {"kind": "random", "n_folds": 5, "seed": 7}
        data = make_config().to_dict()
        assert data["splits"] == {"kind": "canonical", "name": "tvsum-canonical-5"}

    def test_json_is_plain_serializable(self) -> None:
        text = make_full_config().to_json()
        assert isinstance(json.loads(text), dict)

    def test_save_and_load(self, tmp_path: Any) -> None:
        config = make_full_config()
        path = tmp_path / "config.json"
        config.save(path)
        assert EvalConfig.load(path) == config
        assert path.read_text(encoding="utf-8").endswith("\n")


class TestFromDict:
    def base_dict(self) -> dict[str, Any]:
        return dict(make_full_config().to_dict())

    def test_rejects_unknown_top_level_key(self) -> None:
        data = self.base_dict()
        data["oops"] = 1
        with pytest.raises(ValueError, match="unknown key"):
            EvalConfig.from_dict(data)

    def test_rejects_unknown_nested_key(self) -> None:
        data = self.base_dict()
        data["training"]["oops"] = 1
        with pytest.raises(ValueError, match="unknown key"):
            EvalConfig.from_dict(data)

    def test_missing_required_key(self) -> None:
        data = self.base_dict()
        del data["dataset"]
        with pytest.raises(ValueError, match="missing required key"):
            EvalConfig.from_dict(data)

    def test_optional_keys_fall_back_to_defaults(self) -> None:
        data = self.base_dict()
        for key in (
            "seed",
            "gt_aggregation",
            "segment_score_aggregation",
            "score_normalization",
            "training",
        ):
            data.pop(key)
        config = EvalConfig.from_dict(data)
        assert config.seed == 0
        assert config.gt_aggregation == "max"
        assert config.segment_score_aggregation == "mean"
        assert config.score_normalization.method == "none"
        assert config.training.epochs == 100

    def test_rejects_unknown_splits_kind(self) -> None:
        data = self.base_dict()
        data["splits"] = {"kind": "hierarchical"}
        with pytest.raises(ValueError, match="kind"):
            EvalConfig.from_dict(data)

    def test_rejects_invalid_json_text(self) -> None:
        with pytest.raises(ValueError, match="invalid EvalConfig JSON"):
            EvalConfig.from_json("{not json")
