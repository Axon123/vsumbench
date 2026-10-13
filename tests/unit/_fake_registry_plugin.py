"""Importable plugin for entry-point discovery tests (not a test module).

Exposed through a synthetic ``EntryPoint(value="_fake_registry_plugin:FAKE_MODEL")``
so registry discovery can be exercised without installing a real distribution.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from vsumbench.config import EvalConfig
from vsumbench.types import Scores, Video


class PluginModel:
    name = "plugin-model"

    def fit(self, videos: Sequence[Video], config: EvalConfig) -> None:
        return None

    def predict(self, video: Video) -> Scores:
        return Scores(video_id=video.video_id, values=np.ones(video.n_frames, dtype=np.float32))


FAKE_MODEL: PluginModel = PluginModel()
FAKE_MODEL_ALT: PluginModel = PluginModel()
