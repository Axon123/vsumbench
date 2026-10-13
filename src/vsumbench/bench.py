"""Bench orchestrator.

The evaluation loop (splits -> fit -> predict -> select -> aggregate metrics
-> run manifest) lands across M2-M4; this module pins the public entry point
early so the API shape is reviewable before the machinery exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

if TYPE_CHECKING:
    from vsumbench.config import EvalConfig

__all__ = ["Bench"]


class Bench:
    """Runs one :class:`~vsumbench.config.EvalConfig` end to end (M4)."""

    def __init__(self, config: EvalConfig) -> None:
        self._config = config

    @property
    def config(self) -> EvalConfig:
        return self._config

    def run(self) -> NoReturn:
        raise NotImplementedError(
            "Bench.run lands in the orchestration milestone (see docs/design/dev-plan.md): "
            "selectors and metrics shipped in M2; models come from the registry "
            "(built-in learned baselines were descoped)."
        )
