"""EvalConfig — the single source of truth for evaluation protocol parameters.

Every choice that can change a reported number (ground-truth aggregation,
score normalization, budget definition, splits, ...) must live here as an
explicit field (AGENTS.md rule 4). All types in this module are JSON-safe
dataclasses: numpy objects never appear, so configs round-trip through JSON
and land verbatim in run manifests.

This module is the leaf of the dependency graph: it imports nothing from
vsumbench.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

__all__ = [
    "BudgetRounding",
    "CanonicalSplits",
    "Device",
    "EvalConfig",
    "GTAggregation",
    "JsonValue",
    "NormalizationMethod",
    "RandomSplits",
    "ScoreNormalization",
    "SegmentAggregation",
    "SplitSpec",
    "StepExpansion",
    "TrainingConfig",
]

#: How frame scores are reduced to segment (shot) scores before selection.
SegmentAggregation = Literal["mean", "max"]

#: How per-user metric values are collapsed into the reported number.
GTAggregation = Literal["max", "mean"]

#: How a budget fraction maps to the allowed number of selected frames.
#: "floor" is the eccv16-lineage rule (``int(n_frames * fraction)``).
BudgetRounding = Literal["floor", "round", "ceil"]

#: How per-step model scores expand to the original frame space.
#: "forward_fill" holds each step score until the next pick (the
#: eccv16-lineage rule; the tail after the last pick keeps the last score).
StepExpansion = Literal["forward_fill", "zero_fill"]

NormalizationMethod = Literal["none", "min_max", "z_score"]

Device = Literal["cpu", "cuda", "mps"]

#: JSON-representable value; the closed world allowed inside an EvalConfig.
JsonValue: TypeAlias = bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"] | None

_SEGMENT_AGGREGATIONS: tuple[str, ...] = ("mean", "max")
_GT_AGGREGATIONS: tuple[str, ...] = ("max", "mean")
_BUDGET_ROUNDINGS: tuple[str, ...] = ("floor", "round", "ceil")
_STEP_EXPANSIONS: tuple[str, ...] = ("forward_fill", "zero_fill")
_NORMALIZATION_METHODS: tuple[str, ...] = ("none", "min_max", "z_score")
_DEVICES: tuple[str, ...] = ("cpu", "cuda", "mps")


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class CanonicalSplits:
    """Use a published, named split collection (e.g. ``tvsum-canonical-5``)."""

    name: str

    def __post_init__(self) -> None:
        _require_str(self.name, "CanonicalSplits.name")


@dataclass(frozen=True)
class RandomSplits:
    """Seeded k-fold random splits over dataset video ids.

    ``seed`` defaults independently of ``EvalConfig.seed`` so that a fixed
    split can be reused across otherwise different runs.
    """

    n_folds: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.n_folds, bool) or not isinstance(self.n_folds, int) or self.n_folds < 2:
            raise ValueError(f"RandomSplits.n_folds must be an integer >= 2, got {self.n_folds!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"RandomSplits.seed must be an integer, got {self.seed!r}")


SplitSpec = CanonicalSplits | RandomSplits


@dataclass(frozen=True)
class ScoreNormalization:
    """Normalization applied to scores before selection / ranking.

    Method semantics are pinned per source-repository in
    ``docs/protocols/`` (M2); the default is "no normalization".
    """

    method: NormalizationMethod = "none"

    def __post_init__(self) -> None:
        if self.method not in _NORMALIZATION_METHODS:
            raise ValueError(
                f"ScoreNormalization.method must be one of {_NORMALIZATION_METHODS}, "
                f"got {self.method!r}"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for learned models (M3 revisits defaults per baseline).

    ``extra`` carries model-specific parameters; it is JSON-safe and lands in
    the run manifest, so nothing model-specific can hide outside the config.
    """

    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 1
    device: Device = "cpu"
    extra: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_positive_int(self.epochs, "TrainingConfig.epochs")
        _require_positive_int(self.batch_size, "TrainingConfig.batch_size")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)):
            raise ValueError(
                f"TrainingConfig.learning_rate must be a number, got {self.learning_rate!r}"
            )
        if float(self.learning_rate) <= 0:
            raise ValueError(
                f"TrainingConfig.learning_rate must be positive, got {self.learning_rate!r}"
            )
        if self.device not in _DEVICES:
            raise ValueError(
                f"TrainingConfig.device must be one of {_DEVICES}, got {self.device!r}"
            )
        if not isinstance(self.extra, Mapping):
            raise TypeError("TrainingConfig.extra must be a mapping of JSON values")
        object.__setattr__(self, "extra", dict(self.extra))


@dataclass(frozen=True)
class EvalConfig:
    """Complete description of one evaluation protocol run.

    Field defaults are the *recommended protocol* (see docs/design/core-contracts.md
    §4 for per-field rationale). Instances are immutable and JSON round-trippable;
    ``save``/``to_json`` output is what run manifests record verbatim.
    """

    dataset: str
    model: str
    selector: str
    metrics: tuple[str, ...]
    #: Summary budgets as fractions of total frames, e.g. (0.1, 0.15, 0.3, 0.5)
    #: for F1@{10,15,30,50}%.
    budgets: tuple[float, ...]
    splits: SplitSpec
    seed: int = 0
    gt_aggregation: GTAggregation = "max"
    segment_score_aggregation: SegmentAggregation = "mean"
    #: Rounding rule turning a budget fraction into the allowed frame count
    #: (see docs/protocols/eccv16-summary-protocol.md: every lineage repo
    #: floors, i.e. ``int(n_frames * fraction)``).
    budget_rounding: BudgetRounding = "floor"
    #: How step-space model scores expand to the frame space of ``Scores``
    #: (docs/protocols/eccv16-summary-protocol.md; lineage default keeps each
    #: score until the next pick, tail included).
    step_score_expansion: StepExpansion = "forward_fill"
    score_normalization: ScoreNormalization = field(default_factory=ScoreNormalization)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        _require_str(self.dataset, "EvalConfig.dataset")
        _require_str(self.model, "EvalConfig.model")
        _require_str(self.selector, "EvalConfig.selector")
        metrics = tuple(_require_str(m, "EvalConfig.metrics entry") for m in self.metrics)
        object.__setattr__(self, "metrics", metrics)
        if not metrics:
            raise ValueError("EvalConfig.metrics must not be empty")
        budgets = []
        for b in self.budgets:
            if isinstance(b, bool) or not isinstance(b, (int, float)) or not 0.0 < float(b) <= 1.0:
                raise ValueError(f"EvalConfig.budgets entries must be numbers in (0, 1], got {b!r}")
            budgets.append(float(b))
        object.__setattr__(self, "budgets", tuple(budgets))
        if not budgets:
            raise ValueError("EvalConfig.budgets must not be empty")
        if not isinstance(self.splits, (CanonicalSplits, RandomSplits)):
            raise TypeError(
                "EvalConfig.splits must be CanonicalSplits or RandomSplits, "
                f"got {type(self.splits).__name__}"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"EvalConfig.seed must be an integer, got {self.seed!r}")
        if self.gt_aggregation not in _GT_AGGREGATIONS:
            raise ValueError(
                f"EvalConfig.gt_aggregation must be one of {_GT_AGGREGATIONS}, "
                f"got {self.gt_aggregation!r}"
            )
        if self.segment_score_aggregation not in _SEGMENT_AGGREGATIONS:
            raise ValueError(
                f"EvalConfig.segment_score_aggregation must be one of {_SEGMENT_AGGREGATIONS}, "
                f"got {self.segment_score_aggregation!r}"
            )
        if self.budget_rounding not in _BUDGET_ROUNDINGS:
            raise ValueError(
                f"EvalConfig.budget_rounding must be one of {_BUDGET_ROUNDINGS}, "
                f"got {self.budget_rounding!r}"
            )
        if self.step_score_expansion not in _STEP_EXPANSIONS:
            raise ValueError(
                f"EvalConfig.step_score_expansion must be one of {_STEP_EXPANSIONS}, "
                f"got {self.step_score_expansion!r}"
            )
        if not isinstance(self.score_normalization, ScoreNormalization):
            raise TypeError("EvalConfig.score_normalization must be a ScoreNormalization")
        if not isinstance(self.training, TrainingConfig):
            raise TypeError("EvalConfig.training must be a TrainingConfig")

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, JsonValue]:
        """Encode to a JSON-safe dict (nested tagged unions carry a "kind" key)."""
        return {f.name: _encode(getattr(self, f.name)) for f in fields(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalConfig:
        """Decode from a JSON-like dict. Unknown keys are rejected (typo guard)."""
        _reject_unknown_keys(data, _EVAL_CONFIG_KEYS, "EvalConfig")
        try:
            kwargs: dict[str, Any] = {
                "dataset": _get_str(data, "dataset"),
                "model": _get_str(data, "model"),
                "selector": _get_str(data, "selector"),
                "metrics": tuple(
                    _as_str(m, "EvalConfig.metrics entry") for m in _get_list(data, "metrics")
                ),
                "budgets": tuple(
                    _as_number(b, "EvalConfig.budgets entry") for b in _get_list(data, "budgets")
                ),
                "splits": _decode_splits(data["splits"]),
            }
            if "seed" in data:
                kwargs["seed"] = _get_int(data, "seed")
            if "gt_aggregation" in data:
                kwargs["gt_aggregation"] = _get_str(data, "gt_aggregation")
            if "segment_score_aggregation" in data:
                kwargs["segment_score_aggregation"] = _get_str(data, "segment_score_aggregation")
            if "budget_rounding" in data:
                kwargs["budget_rounding"] = _get_str(data, "budget_rounding")
            if "step_score_expansion" in data:
                kwargs["step_score_expansion"] = _get_str(data, "step_score_expansion")
            if "score_normalization" in data:
                kwargs["score_normalization"] = _decode_normalization(data["score_normalization"])
            if "training" in data:
                kwargs["training"] = _decode_training(data["training"])
            return cls(**kwargs)
        except KeyError as exc:
            raise ValueError(f"EvalConfig is missing required key {exc}") from None

    @classmethod
    def from_json(cls, text: str) -> EvalConfig:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid EvalConfig JSON: {exc}") from None
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> EvalConfig:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


_EVAL_CONFIG_KEYS = frozenset(
    {
        "dataset",
        "model",
        "selector",
        "metrics",
        "budgets",
        "splits",
        "seed",
        "gt_aggregation",
        "segment_score_aggregation",
        "budget_rounding",
        "step_score_expansion",
        "score_normalization",
        "training",
    }
)


# -- JSON codec helpers -------------------------------------------------------


def _encode(value: Any) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, CanonicalSplits):
        return {"kind": "canonical", "name": value.name}
    if isinstance(value, RandomSplits):
        return {"kind": "random", "n_folds": value.n_folds, "seed": value.seed}
    if isinstance(value, (ScoreNormalization, TrainingConfig)):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    raise TypeError(f"{type(value).__name__} is not representable inside an EvalConfig")


def _reject_unknown_keys(
    data: Mapping[str, Any], expected: frozenset[str] | set[str], where: str
) -> None:
    unknown = sorted(set(data) - expected)
    if unknown:
        raise ValueError(f"{where} got unknown key(s) {unknown}; allowed: {sorted(expected)}")


def _as_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected a non-empty string for {name}, got {value!r}")
    return value


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected an integer for {name}, got {value!r}")
    return value


def _as_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number for {name}, got {value!r}")
    return float(value)


def _get_str(data: Mapping[str, Any], key: str) -> str:
    return _as_str(data[key], repr(key))


def _get_int(data: Mapping[str, Any], key: str) -> int:
    return _as_int(data[key], repr(key))


def _get_number(data: Mapping[str, Any], key: str) -> float:
    return _as_number(data[key], repr(key))


def _get_list(data: Mapping[str, Any], key: str) -> list[Any]:
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"expected a list for {key!r}, got {type(value).__name__}")
    return value


def _decode_splits(data: object) -> SplitSpec:
    if not isinstance(data, Mapping):
        raise ValueError("EvalConfig.splits must be an object with a 'kind' key")
    kind = data.get("kind")
    if kind == "canonical":
        _reject_unknown_keys(data, {"kind", "name"}, "CanonicalSplits")
        return CanonicalSplits(name=_get_str(data, "name"))
    if kind == "random":
        _reject_unknown_keys(data, {"kind", "n_folds", "seed"}, "RandomSplits")
        kwargs: dict[str, Any] = {}
        if "n_folds" in data:
            kwargs["n_folds"] = _get_int(data, "n_folds")
        if "seed" in data:
            kwargs["seed"] = _get_int(data, "seed")
        return RandomSplits(**kwargs)
    raise ValueError(f"EvalConfig.splits.kind must be 'canonical' or 'random', got {kind!r}")


def _decode_normalization(data: object) -> ScoreNormalization:
    if not isinstance(data, Mapping):
        raise ValueError("EvalConfig.score_normalization must be an object")
    _reject_unknown_keys(data, {"method"}, "ScoreNormalization")
    # Runtime validation of the literal happens in ScoreNormalization.__post_init__.
    return ScoreNormalization(method=cast(NormalizationMethod, _get_str(data, "method")))


def _decode_training(data: object) -> TrainingConfig:
    if not isinstance(data, Mapping):
        raise ValueError("EvalConfig.training must be an object")
    _reject_unknown_keys(
        data, {"epochs", "learning_rate", "batch_size", "device", "extra"}, "TrainingConfig"
    )
    kwargs: dict[str, Any] = {}
    if "epochs" in data:
        kwargs["epochs"] = _get_int(data, "epochs")
    if "learning_rate" in data:
        kwargs["learning_rate"] = _get_number(data, "learning_rate")
    if "batch_size" in data:
        kwargs["batch_size"] = _get_int(data, "batch_size")
    if "device" in data:
        kwargs["device"] = _get_str(data, "device")
    if "extra" in data:
        extra = data["extra"]
        if not isinstance(extra, Mapping):
            raise ValueError("TrainingConfig.extra must be an object")
        kwargs["extra"] = dict(extra)
    return TrainingConfig(**kwargs)
