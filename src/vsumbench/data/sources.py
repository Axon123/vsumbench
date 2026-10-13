"""Canonical dataset sources: mirrors, archive layout and sha256 checksums.

The ``eccv16`` h5 files are the de-facto standard TVSum/SumMe distribution
used by the VASNet-lineage repos (VASNet, SUM-GAN, PGL-SUM, CA-SUM, ...):
GoogLeNet pool5 frame features extracted by Ke Zhang and Wei-Lun Chao for
"Video Summarization with Long Short-term Memory" (ECCV 2016), packaged as
h5 by Kaiyang Zhou (pytorch-vsumm-reinforce). We mirror them through the
rar archives committed to the data folders of the e-apostolidis repos and
trust the *extracted h5 bytes* (sha256 below), not any particular archive.

Only loader code and checksums ship with the package — never the data
itself (AGENTS.md rule 6).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SOURCES",
    "DatasetSpec",
    "FileSpec",
]


@dataclass(frozen=True)
class FileSpec:
    """One canonical file of a dataset, identified by sha256."""

    filename: str
    sha256: str
    size: int


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset's canonical files come from.

    ``archive_urls`` each serve a rar archive that extracts to the canonical
    ``files``; extraction is best-effort through any available system tool
    (see :mod:`vsumbench.data.download`). Verification always targets the
    extracted files, so mirror archives are interchangeable.
    """

    name: str
    files: tuple[FileSpec, ...]
    archive_urls: tuple[str, ...]


SOURCES: dict[str, DatasetSpec] = {
    "tvsum": DatasetSpec(
        name="tvsum",
        files=(
            FileSpec(
                filename="eccv16_dataset_tvsum_google_pool5.h5",
                sha256="ec9cbcc715887d30ba2ad46764cb810f07eb6d6241e96c0eac5fca51d398228b",
                size=125_126_256,
            ),
        ),
        archive_urls=(
            "https://raw.githubusercontent.com/e-apostolidis/PGL-SUM/master/data/TVSum/"
            "eccv16_dataset_tvsum_google_pool5.rar",
            "https://raw.githubusercontent.com/e-apostolidis/CA-SUM/main/data/TVSum/"
            "eccv16_dataset_tvsum_google_pool5.rar",
        ),
    ),
    "summe": DatasetSpec(
        name="summe",
        files=(
            FileSpec(
                filename="eccv16_dataset_summe_google_pool5.h5",
                sha256="db3816d2504f322c1107fd5e4d8d612380b1f3ca9cb1e5ce06a113f6515ab278",
                size=37_099_544,
            ),
        ),
        archive_urls=(
            "https://raw.githubusercontent.com/e-apostolidis/PGL-SUM/master/data/SumMe/"
            "eccv16_dataset_summe_google_pool5.rar",
            "https://raw.githubusercontent.com/e-apostolidis/CA-SUM/main/data/SumMe/"
            "eccv16_dataset_summe_google_pool5.rar",
        ),
    ),
}
