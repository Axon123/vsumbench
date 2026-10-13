"""Unit tests for vsumbench.registry: registration, lookup, entry-point discovery."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import EntryPoint
from pathlib import Path

import numpy as np
import pytest

import vsumbench.registry as registry_module
from vsumbench.config import EvalConfig, SegmentAggregation
from vsumbench.registry import (
    DATASETS,
    MODELS,
    RANK_METRICS,
    SELECTORS,
    SUMMARY_METRICS,
    Registry,
)
from vsumbench.types import (
    Budget,
    DatasetAdapter,
    Mask,
    Model,
    RankMetric,
    Scores,
    Selector,
    ShotTable,
    SummaryMetric,
    Video,
)


class FakeModel:
    name = "fake-model"

    def fit(self, videos: Sequence[Video], config: EvalConfig) -> None:
        return None

    def predict(self, video: Video) -> Scores:
        return Scores(video_id=video.video_id, values=np.zeros(video.n_frames, dtype=np.float32))


class FakeSelector:
    name = "fake-selector"

    def select(
        self, scores: Scores, shots: ShotTable, budget: Budget, *, aggregation: SegmentAggregation
    ) -> Mask:
        return Mask(video_id=scores.video_id, frames=np.zeros(shots.n_frames, dtype=bool))


class FakeSummaryMetric:
    name = "fake-f1"

    def score(self, summary: Mask, video: Video) -> np.ndarray:
        return np.zeros(len(video.user_summaries))


class FakeRankMetric:
    name = "fake-tau"

    def score(self, scores: Scores, video: Video) -> np.ndarray:
        return np.zeros(len(video.user_scores))


class FakeDataset:
    name = "fake-dataset"

    def load(self, root: Path) -> list[Video]:
        return []


def patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, eps_by_group: dict[str, list[EntryPoint]]
) -> None:
    def fake_entry_points(*, group: str) -> list[EntryPoint]:
        return eps_by_group.get(group, [])

    monkeypatch.setattr(registry_module, "entry_points", fake_entry_points)


class TestRegistration:
    def test_register_and_get(self) -> None:
        registry: Registry[Model] = Registry("test.models", Model)
        model = FakeModel()
        registry.register("uniform", model)
        assert registry.get("uniform") is model
        assert registry.names() == ("uniform",)

    def test_duplicate_registration_rejected(self) -> None:
        registry: Registry[Model] = Registry("test.models", Model)
        registry.register("uniform", FakeModel())
        with pytest.raises(ValueError, match="already registered"):
            registry.register("uniform", FakeModel())

    def test_empty_name_rejected(self) -> None:
        registry: Registry[Model] = Registry("test.models", Model)
        with pytest.raises(ValueError, match="non-empty string"):
            registry.register("", FakeModel())

    def test_non_protocol_object_rejected(self) -> None:
        registry: Registry[Model] = Registry("test.models", Model)
        with pytest.raises(TypeError, match="does not implement"):
            registry.register("not-a-model", object())  # type: ignore[arg-type]

    def test_cross_protocol_object_rejected(self) -> None:
        registry: Registry[Model] = Registry("test.models", Model)
        with pytest.raises(TypeError, match="does not implement"):
            registry.register("it-selects", FakeSelector())  # type: ignore[arg-type]

    def test_unknown_name_lists_registered(self) -> None:
        registry: Registry[Model] = Registry("test.models", Model)
        registry.register("uniform", FakeModel())
        with pytest.raises(
            KeyError, match=r"unknown test\.models entry 'nope'; registered: uniform"
        ):
            registry.get("nope")


class TestEntryPointDiscovery:
    def test_discovers_entry_points_lazily(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = EntryPoint(
            name="epfake",
            value="_fake_registry_plugin:FAKE_MODEL",
            group="test.models",
        )
        registry: Registry[Model] = Registry("test.models", Model)
        patch_entry_points(monkeypatch, {"test.models": [ep]})
        assert registry.names() == ("epfake",)

    def test_explicit_registration_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = EntryPoint(
            name="dup",
            value="_fake_registry_plugin:FAKE_MODEL",
            group="test.models",
        )
        registry: Registry[Model] = Registry("test.models", Model)
        manual = FakeModel()
        registry.register("dup", manual)
        patch_entry_points(monkeypatch, {"test.models": [ep]})
        assert registry.get("dup") is manual

    def test_protocol_violating_entry_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = EntryPoint(name="pi", value="math:pi", group="test.models")
        registry: Registry[Model] = Registry("test.models", Model)
        patch_entry_points(monkeypatch, {"test.models": [ep]})
        with pytest.raises(TypeError, match="does not implement"):
            registry.names()

    def test_broken_entry_point_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ep = EntryPoint(
            name="broken",
            value="definitely_missing_module_9x:thing",
            group="test.models",
        )
        registry: Registry[Model] = Registry("test.models", Model)
        patch_entry_points(monkeypatch, {"test.models": [ep]})
        with pytest.raises(RuntimeError, match="failed to load entry point"):
            registry.names()


class TestModuleRegistries:
    def test_all_protocols_are_runtime_checkable(self) -> None:
        for protocol, fake in (
            (Model, FakeModel()),
            (Selector, FakeSelector()),
            (SummaryMetric, FakeSummaryMetric()),
            (RankMetric, FakeRankMetric()),
            (DatasetAdapter, FakeDataset()),
        ):
            assert isinstance(fake, protocol)

    def test_builtin_registries_content_matches_milestones(self) -> None:
        # M1 ships the dataset adapters; M2 adds selectors and metrics; M3
        # (orchestration) adds the uniform reference model. Subset, not
        # equality: other test modules register in-process components in the
        # same global registries before this one runs.
        assert {"summe", "tvsum"} <= set(DATASETS.names())
        assert "uniform" in MODELS.names()
        assert {"greedy", "knapsack"} <= set(SELECTORS.names())
        assert "f1" in SUMMARY_METRICS.names()
        assert "kendall_tau" in RANK_METRICS.names()
