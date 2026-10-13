"""SumMe adapter over the canonical eccv16 h5 file (25 videos, 15+ annotators)."""

from __future__ import annotations

from vsumbench.data._eccv16 import Eccv16Adapter
from vsumbench.data.sources import SOURCES

__all__ = ["SUMME_ADAPTER"]

SUMME_ADAPTER = Eccv16Adapter(
    dataset_name="summe",
    h5_filename=SOURCES["summe"].files[0].filename,
)
