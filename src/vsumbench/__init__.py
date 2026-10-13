"""vsumbench — unified, reproducible evaluation toolkit for video summarization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.0.1"

__all__ = ["Bench", "RunResult", "__version__"]

if TYPE_CHECKING:
    from vsumbench.bench import Bench, RunResult


def __getattr__(name: str) -> Any:
    # Bench is re-exported lazily (PEP 562): `from vsumbench import Bench`
    # works as documented, while `import vsumbench.types` etc. stay light
    # and the layering guard keeps its teeth.
    if name in ("Bench", "RunResult"):
        from vsumbench.bench import Bench, RunResult

        return {"Bench": Bench, "RunResult": RunResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
