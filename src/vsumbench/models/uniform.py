"""The ``uniform`` reference model: every frame equally important.

A learning-free model shipped for CLI smoke tests and examples (dev-plan M3).
It makes no claim about any published method and carries **no alignment
promise** (tests/alignment/ does not guard it): with constant scores the
selector's tie-breaking fully determines the summary, which is exactly what
makes it a good smoke signal — deterministic, dataset-agnostic, and any
non-trivial model must beat it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from vsumbench.types import Scores, Video

if TYPE_CHECKING:
    from vsumbench.config import EvalConfig

__all__ = ["UNIFORM_MODEL", "UniformModel"]


class UniformModel:
    """Scores every frame 1.0; ``fit`` is a no-op."""

    name = "uniform"

    def fit(self, videos: Sequence[Video], config: EvalConfig) -> None:
        return None

    def predict(self, video: Video) -> Scores:
        return Scores(
            video_id=video.video_id,
            values=np.full(video.n_frames, 1.0, dtype=np.float32),
        )


UNIFORM_MODEL = UniformModel()
