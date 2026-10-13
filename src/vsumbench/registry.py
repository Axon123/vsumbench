"""Entry-point based registries for datasets, models, selectors and metrics.

Two registration paths, one namespace per component kind:

1. **Entry points** (for distributions): a third-party package declares::

       [project.entry-points."vsumbench.models"]
       my-model = "my_pkg.model:MY_MODEL"

   and is discovered lazily on the first lookup — no vsumbench source
   changes needed (AGENTS.md rule 3). Built-in components register through
   the exact same mechanism in vsumbench's own ``pyproject.toml``.

2. **Explicit ``register()``** (for in-process use: tests, notebooks,
   one-off experiments). Explicit registrations win over entry points of
   the same name.

The value behind an entry point (or passed to ``register``) must be an
*instance* implementing the protocol for that group; the runtime check is
attribute/method presence only (``isinstance`` against a runtime-checkable
protocol), so signature compatibility is enforced by mypy, not at runtime.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Generic, TypeVar

from vsumbench.types import DatasetAdapter, Model, RankMetric, Selector, SummaryMetric

__all__ = [
    "DATASETS",
    "MODELS",
    "RANK_METRICS",
    "SELECTORS",
    "SUMMARY_METRICS",
    "Registry",
]

T = TypeVar("T")


class Registry(Generic[T]):
    """Name -> component lookup with lazy entry-point discovery."""

    def __init__(self, group: str, protocol: Any) -> None:
        # ``protocol`` is a runtime-checkable Protocol class used only for
        # isinstance checks (mypy rejects protocols as ``type[T]`` values);
        # the static contract is the generic parameter T.
        self._group = group
        self._protocol = protocol
        self._objects: dict[str, T] = {}
        self._discovered = False

    @property
    def group(self) -> str:
        return self._group

    def register(self, name: str, obj: T) -> None:
        """Register ``obj`` under ``name``; explicit wins over entry points."""
        if not isinstance(name, str) or not name:
            raise ValueError(f"registry name must be a non-empty string, got {name!r}")
        if not isinstance(obj, self._protocol):
            raise TypeError(
                f"{name!r} does not implement the {self._group} protocol "
                f"({self._protocol.__name__} attributes/methods missing)"
            )
        if name in self._objects:
            raise ValueError(f"{name!r} is already registered in {self._group}")
        self._objects[name] = obj

    def get(self, name: str) -> T:
        self._ensure_discovered()
        try:
            return self._objects[name]
        except KeyError:
            available = ", ".join(sorted(self._objects)) or "<none>"
            raise KeyError(
                f"unknown {self._group} entry {name!r}; registered: {available}"
            ) from None

    def names(self) -> tuple[str, ...]:
        self._ensure_discovered()
        return tuple(sorted(self._objects))

    def _ensure_discovered(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        for ep in entry_points(group=self._group):
            try:
                obj = ep.load()
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load entry point {ep.name!r} in group {self._group}"
                ) from exc
            if not isinstance(obj, self._protocol):
                raise TypeError(
                    f"entry point {ep.name!r} in group {self._group} does not implement "
                    f"the {self._protocol.__name__} protocol"
                )
            self._objects.setdefault(ep.name, obj)

    def __repr__(self) -> str:
        return f"Registry({self._group!r}, {list(self.names())})"


MODELS: Registry[Model] = Registry("vsumbench.models", Model)
SELECTORS: Registry[Selector] = Registry("vsumbench.selectors", Selector)
#: Summary-overlap metrics (e.g. F1@budget) — compare a :class:`~vsumbench.types.Mask`.
SUMMARY_METRICS: Registry[SummaryMetric] = Registry("vsumbench.metrics", SummaryMetric)
#: Rank-correlation metrics (e.g. Kendall's tau) — compare raw :class:`~vsumbench.types.Scores`.
RANK_METRICS: Registry[RankMetric] = Registry("vsumbench.rank_metrics", RankMetric)
DATASETS: Registry[DatasetAdapter] = Registry("vsumbench.datasets", DatasetAdapter)
