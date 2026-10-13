"""TVSum adapter over the canonical eccv16 h5 file (50 videos, 20 annotators)."""

from __future__ import annotations

from vsumbench.data._eccv16 import Eccv16Adapter
from vsumbench.data.sources import SOURCES

__all__ = ["TVSUM_ADAPTER"]

TVSUM_ADAPTER = Eccv16Adapter(
    dataset_name="tvsum",
    h5_filename=SOURCES["tvsum"].files[0].filename,
)
