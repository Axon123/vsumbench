"""Official raw-video archives and the ``video_id`` mapping.

Static table derived from the official archives on 2026-11-02; see
``docs/design/raw-media.md`` for provenance and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = ["MEDIA_ENTRIES", "MEDIA_SOURCES", "MediaEntry", "MediaSource"]


@dataclass(frozen=True)
class MediaEntry:
    """One ``video_id`` -> official archive member mapping.

    ``filename`` is the canonical local basename (always ``<video_id>.mp4``);
    ``archive_member`` is the exact path inside ``MediaSource.inner_archive``
    (or inside the archive itself for zip sources). ``original_name`` is the
    YouTube id (TVSum) or the h5 ``video_name`` (SumMe), kept for provenance
    only.
    """

    dataset: str
    video_id: str
    filename: str
    archive_member: str
    original_name: str
    source_url: str | None = None
    size: int | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("dataset", "video_id", "filename", "archive_member", "original_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"MediaEntry.{name} must be a non-empty string")
        if "/" in self.filename or "\\" in self.filename:
            raise ValueError(f"MediaEntry.filename must be a basename, got {self.filename!r}")
        if ".." in Path(self.archive_member).parts:
            raise ValueError("MediaEntry.archive_member must stay inside the archive")
        if self.size is not None and (
            isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 1
        ):
            raise ValueError("MediaEntry.size must be a positive integer")
        if self.sha256 is not None and not _is_sha256(self.sha256):
            raise ValueError("MediaEntry.sha256 must be 64 hex characters")


@dataclass(frozen=True)
class MediaSource:
    """The official archive holding every mapped video of one dataset."""

    dataset: str
    homepage: str
    license: str
    archive_urls: tuple[str, ...]
    archive_filename: str
    archive_size: int
    archive_sha256: str
    archive_format: Literal["zip", "tar.gz"]
    inner_archive: str | None = None

    def __post_init__(self) -> None:
        if self.archive_format not in ("zip", "tar.gz"):
            raise ValueError(f"unsupported archive_format: {self.archive_format!r}")
        if not self.archive_urls or not all(
            isinstance(url, str) and url.startswith(("https://", "http://", "file://"))
            for url in self.archive_urls
        ):
            raise ValueError("MediaSource.archive_urls must be non-empty URL strings")
        if not _is_sha256(self.archive_sha256):
            raise ValueError("MediaSource.archive_sha256 must be 64 hex characters")
        if self.archive_size < 1:
            raise ValueError("MediaSource.archive_size must be positive")
        if self.archive_format == "tar.gz" and not self.inner_archive:
            raise ValueError("tar.gz sources must name their inner_archive")


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


MEDIA_SOURCES: dict[str, MediaSource] = {
    "tvsum": MediaSource(
        dataset="tvsum",
        homepage="https://github.com/yalesong/tvsum",
        license=(
            "Videos: CC-BY 3.0 per the dataset README; the bundled Yahoo! Webscope "
            "readme additionally restricts use to academic research and forbids "
            "redistribution. See docs/design/raw-media.md."
        ),
        archive_urls=("https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz",),
        archive_filename="tvsum50_ver_1_1.tgz",
        archive_size=671779858,
        archive_sha256="407d340bcd06fdc6d17374ebe6760b4a96816bcace228559c8283d9fb2520dea",
        archive_format="tar.gz",
        inner_archive="ydata-tvsum50-v1_1/ydata-tvsum50-video.zip",
    ),
    "summe": MediaSource(
        dataset="summe",
        homepage="https://data.vision.ee.ethz.ch/cvl/SumMe/",
        license=("Non-commercial research use only; see README.txt inside the official archive."),
        archive_urls=("https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip",),
        archive_filename="SumMe.zip",
        archive_size=2396688224,
        archive_sha256="42aac7668888ae30885abdb61bb4a10a2e82543695354a54de82e48868b8fb05",
        archive_format="zip",
    ),
}


MEDIA_ENTRIES: dict[str, dict[str, MediaEntry]] = {
    "tvsum": {
        "video_1": MediaEntry(
            dataset="tvsum",
            video_id="video_1",
            filename="video_1.mp4",
            archive_member="video/AwmHb44_ouw.mp4",
            original_name="AwmHb44_ouw",
            source_url="https://www.youtube.com/watch?v=AwmHb44_ouw",
            size=19246027,
            sha256="82ecca1de0363253060bb3c19e2f5dbd4080bfc0cd790db23c3e73ffd76a602f",
        ),
        "video_2": MediaEntry(
            dataset="tvsum",
            video_id="video_2",
            filename="video_2.mp4",
            archive_member="video/98MoyGZKHXc.mp4",
            original_name="98MoyGZKHXc",
            source_url="https://www.youtube.com/watch?v=98MoyGZKHXc",
            size=9817234,
            sha256="e293ceb35a8f1ec514707c579ab9b78e678a0c686f895d25945b9a50432c000c",
        ),
        "video_3": MediaEntry(
            dataset="tvsum",
            video_id="video_3",
            filename="video_3.mp4",
            archive_member="video/J0nA4VgnoCo.mp4",
            original_name="J0nA4VgnoCo",
            source_url="https://www.youtube.com/watch?v=J0nA4VgnoCo",
            size=31626560,
            sha256="65c9490a2166d17736c38ed077f250290c548b1da8f1801f10ff53fad8e84a5b",
        ),
        "video_4": MediaEntry(
            dataset="tvsum",
            video_id="video_4",
            filename="video_4.mp4",
            archive_member="video/gzDbaEs1Rlg.mp4",
            original_name="gzDbaEs1Rlg",
            source_url="https://www.youtube.com/watch?v=gzDbaEs1Rlg",
            size=15419041,
            sha256="6682bdee7edc1d697a1807bb3f84f50e3cacd98266fc93d1deb38adbb8d125c2",
        ),
        "video_5": MediaEntry(
            dataset="tvsum",
            video_id="video_5",
            filename="video_5.mp4",
            archive_member="video/XzYM3PfTM4w.mp4",
            original_name="XzYM3PfTM4w",
            source_url="https://www.youtube.com/watch?v=XzYM3PfTM4w",
            size=6010860,
            sha256="c758df53e237aadc9e893a4389078b25f58000a782d3c69b8490195aae30dac0",
        ),
        "video_6": MediaEntry(
            dataset="tvsum",
            video_id="video_6",
            filename="video_6.mp4",
            archive_member="video/HT5vyqe0Xaw.mp4",
            original_name="HT5vyqe0Xaw",
            source_url="https://www.youtube.com/watch?v=HT5vyqe0Xaw",
            size=17564055,
            sha256="85fdd93e61fb670b5437666c9eb90aa1798889edbe07394749c00ca0d044c886",
        ),
        "video_7": MediaEntry(
            dataset="tvsum",
            video_id="video_7",
            filename="video_7.mp4",
            archive_member="video/sTEELN-vY30.mp4",
            original_name="sTEELN-vY30",
            source_url="https://www.youtube.com/watch?v=sTEELN-vY30",
            size=8065363,
            sha256="b6767caf3ddb2a1750672427315fe61c63a854de26ca707f5e5041bf45557d48",
        ),
        "video_8": MediaEntry(
            dataset="tvsum",
            video_id="video_8",
            filename="video_8.mp4",
            archive_member="video/vdmoEJ5YbrQ.mp4",
            original_name="vdmoEJ5YbrQ",
            source_url="https://www.youtube.com/watch?v=vdmoEJ5YbrQ",
            size=17926114,
            sha256="77f719f5de3ca948691dd34e7dac5ed5783d1b98814fea035aa7cce6d21edb72",
        ),
        "video_9": MediaEntry(
            dataset="tvsum",
            video_id="video_9",
            filename="video_9.mp4",
            archive_member="video/xwqBXPGE9pQ.mp4",
            original_name="xwqBXPGE9pQ",
            source_url="https://www.youtube.com/watch?v=xwqBXPGE9pQ",
            size=12666392,
            sha256="71bb39923112962e30006918c9e3a9dbcd248786f275568447e8a29ebdb380ff",
        ),
        "video_10": MediaEntry(
            dataset="tvsum",
            video_id="video_10",
            filename="video_10.mp4",
            archive_member="video/akI8YFjEmUw.mp4",
            original_name="akI8YFjEmUw",
            source_url="https://www.youtube.com/watch?v=akI8YFjEmUw",
            size=7222459,
            sha256="28f8d4d89a14b30f9066ad5146e8b219c67d45291b9a1588af6d1f6c455acffe",
        ),
        "video_11": MediaEntry(
            dataset="tvsum",
            video_id="video_11",
            filename="video_11.mp4",
            archive_member="video/i3wAGJaaktw.mp4",
            original_name="i3wAGJaaktw",
            source_url="https://www.youtube.com/watch?v=i3wAGJaaktw",
            size=8413810,
            sha256="7a8cc3f5415702ce254200db00650cb42981ba0c0fa5b5ef4a910f7c66c69c21",
        ),
        "video_12": MediaEntry(
            dataset="tvsum",
            video_id="video_12",
            filename="video_12.mp4",
            archive_member="video/Bhxk-O1Y7Ho.mp4",
            original_name="Bhxk-O1Y7Ho",
            source_url="https://www.youtube.com/watch?v=Bhxk-O1Y7Ho",
            size=24536587,
            sha256="a66202fe5943daee0a0984fd708ff3126168ded35750c099a54ae67705e1ff3f",
        ),
        "video_13": MediaEntry(
            dataset="tvsum",
            video_id="video_13",
            filename="video_13.mp4",
            archive_member="video/0tmA_C6XwfM.mp4",
            original_name="0tmA_C6XwfM",
            source_url="https://www.youtube.com/watch?v=0tmA_C6XwfM",
            size=7573592,
            sha256="63a2f711416d2a911a4f00166fedb7ce01e2eaab18946c0222071322111b961b",
        ),
        "video_14": MediaEntry(
            dataset="tvsum",
            video_id="video_14",
            filename="video_14.mp4",
            archive_member="video/3eYKfiOEJNs.mp4",
            original_name="3eYKfiOEJNs",
            source_url="https://www.youtube.com/watch?v=3eYKfiOEJNs",
            size=10488602,
            sha256="d8c2669fbd08e8b2f35ebce82dfe5aa6cbf7e8fbee56df9e971ca12b7dd9e322",
        ),
        "video_15": MediaEntry(
            dataset="tvsum",
            video_id="video_15",
            filename="video_15.mp4",
            archive_member="video/xxdtq8mxegs.mp4",
            original_name="xxdtq8mxegs",
            source_url="https://www.youtube.com/watch?v=xxdtq8mxegs",
            size=7367068,
            sha256="fe3200fd13dfbf74f5c02d73ec9acd7bd0c1f68eb876968bc74230549320c327",
        ),
        "video_16": MediaEntry(
            dataset="tvsum",
            video_id="video_16",
            filename="video_16.mp4",
            archive_member="video/WG0MBPpPC6I.mp4",
            original_name="WG0MBPpPC6I",
            source_url="https://www.youtube.com/watch?v=WG0MBPpPC6I",
            size=21473331,
            sha256="0d68d94612da8efd687419dc030a485d61419c30290865d9ec1f9ed1062040f9",
        ),
        "video_17": MediaEntry(
            dataset="tvsum",
            video_id="video_17",
            filename="video_17.mp4",
            archive_member="video/Hl-__g2gn_A.mp4",
            original_name="Hl-__g2gn_A",
            source_url="https://www.youtube.com/watch?v=Hl-__g2gn_A",
            size=13134659,
            sha256="9386d64a62463f36213a043e6e74c76faf5eff746d7ef53035146c7bead40bd1",
        ),
        "video_18": MediaEntry(
            dataset="tvsum",
            video_id="video_18",
            filename="video_18.mp4",
            archive_member="video/Yi4Ij2NM7U4.mp4",
            original_name="Yi4Ij2NM7U4",
            source_url="https://www.youtube.com/watch?v=Yi4Ij2NM7U4",
            size=21884159,
            sha256="26ec9f71ea771e5601a1f5c963fa425681ae2f86b3dfca483a64d58b5a65feb9",
        ),
        "video_19": MediaEntry(
            dataset="tvsum",
            video_id="video_19",
            filename="video_19.mp4",
            archive_member="video/37rzWOQsNIw.mp4",
            original_name="37rzWOQsNIw",
            source_url="https://www.youtube.com/watch?v=37rzWOQsNIw",
            size=10156252,
            sha256="c4fb034705c03a890fed5cc46ff79d1e9aef5cd9edec1d22dfe7cc653b5fd98a",
        ),
        "video_20": MediaEntry(
            dataset="tvsum",
            video_id="video_20",
            filename="video_20.mp4",
            archive_member="video/LRw_obCPUt0.mp4",
            original_name="LRw_obCPUt0",
            source_url="https://www.youtube.com/watch?v=LRw_obCPUt0",
            size=13697882,
            sha256="ce57a731477468ddf799c29cfcbd11469e30cb96eb844c372a8447a90e1db858",
        ),
        "video_21": MediaEntry(
            dataset="tvsum",
            video_id="video_21",
            filename="video_21.mp4",
            archive_member="video/cjibtmSLxQ4.mp4",
            original_name="cjibtmSLxQ4",
            source_url="https://www.youtube.com/watch?v=cjibtmSLxQ4",
            size=35194393,
            sha256="43183ef7f35bfcbc39c122be797fdee5031103862c7c7da3b9ab74d222c79c9d",
        ),
        "video_22": MediaEntry(
            dataset="tvsum",
            video_id="video_22",
            filename="video_22.mp4",
            archive_member="video/b626MiF1ew4.mp4",
            original_name="b626MiF1ew4",
            source_url="https://www.youtube.com/watch?v=b626MiF1ew4",
            size=12624955,
            sha256="66d6dc7720db2ae750c9c85a7f6a0d55416a9a23ce6ca7d151969501d8606e1c",
        ),
        "video_23": MediaEntry(
            dataset="tvsum",
            video_id="video_23",
            filename="video_23.mp4",
            archive_member="video/XkqCExn6_Us.mp4",
            original_name="XkqCExn6_Us",
            source_url="https://www.youtube.com/watch?v=XkqCExn6_Us",
            size=10210795,
            sha256="ee3e102d0cda34b30c8eadaa162157616c398213dc88d0198ff4cc6476dd5fc9",
        ),
        "video_24": MediaEntry(
            dataset="tvsum",
            video_id="video_24",
            filename="video_24.mp4",
            archive_member="video/GsAD1KT1xo8.mp4",
            original_name="GsAD1KT1xo8",
            source_url="https://www.youtube.com/watch?v=GsAD1KT1xo8",
            size=7794439,
            sha256="a195fc9d8e5330ee55c62dfc0a99eb5d130bf0444bb412992d1204eecd8f3a86",
        ),
        "video_25": MediaEntry(
            dataset="tvsum",
            video_id="video_25",
            filename="video_25.mp4",
            archive_member="video/PJrm840pAUI.mp4",
            original_name="PJrm840pAUI",
            source_url="https://www.youtube.com/watch?v=PJrm840pAUI",
            size=14766069,
            sha256="790c560f5132928d456803ca5e23b03ca38b1eb8d3383b619afbed60dd5be86f",
        ),
        "video_26": MediaEntry(
            dataset="tvsum",
            video_id="video_26",
            filename="video_26.mp4",
            archive_member="video/91IHQYk1IQM.mp4",
            original_name="91IHQYk1IQM",
            source_url="https://www.youtube.com/watch?v=91IHQYk1IQM",
            size=5830349,
            sha256="327f8cd1ff27ecec1de291e396ac7c40521253fa208f2e3c477c5b1e84964f05",
        ),
        "video_27": MediaEntry(
            dataset="tvsum",
            video_id="video_27",
            filename="video_27.mp4",
            archive_member="video/RBCABdttQmI.mp4",
            original_name="RBCABdttQmI",
            source_url="https://www.youtube.com/watch?v=RBCABdttQmI",
            size=19590609,
            sha256="2411ac74f1e8a049072063bd579b4aa034e5c01a388c67cdcc7959763562b36b",
        ),
        "video_28": MediaEntry(
            dataset="tvsum",
            video_id="video_28",
            filename="video_28.mp4",
            archive_member="video/z_6gVvQb2d0.mp4",
            original_name="z_6gVvQb2d0",
            source_url="https://www.youtube.com/watch?v=z_6gVvQb2d0",
            size=15043855,
            sha256="fe77605cdf8c234219ce423a90a93e44e003e55567bdb9bb87a3fb5fec0063cd",
        ),
        "video_29": MediaEntry(
            dataset="tvsum",
            video_id="video_29",
            filename="video_29.mp4",
            archive_member="video/fWutDQy1nnY.mp4",
            original_name="fWutDQy1nnY",
            source_url="https://www.youtube.com/watch?v=fWutDQy1nnY",
            size=31792596,
            sha256="d50d740e77517611ea231e844d409ebdf51e05037ef8cd17a22cd983d2464fc3",
        ),
        "video_30": MediaEntry(
            dataset="tvsum",
            video_id="video_30",
            filename="video_30.mp4",
            archive_member="video/4wU_LUjG5Ic.mp4",
            original_name="4wU_LUjG5Ic",
            source_url="https://www.youtube.com/watch?v=4wU_LUjG5Ic",
            size=8815620,
            sha256="2779762f2c332931c9fb4e61783b81df6e34b487dc3a94c882781c981e6c4869",
        ),
        "video_31": MediaEntry(
            dataset="tvsum",
            video_id="video_31",
            filename="video_31.mp4",
            archive_member="video/VuWGsYPqAX8.mp4",
            original_name="VuWGsYPqAX8",
            source_url="https://www.youtube.com/watch?v=VuWGsYPqAX8",
            size=11537377,
            sha256="9476a6d8bc6fdcff909fa2885ba82ba7696b3d05fca2f070e313139e434eabe3",
        ),
        "video_32": MediaEntry(
            dataset="tvsum",
            video_id="video_32",
            filename="video_32.mp4",
            archive_member="video/JKpqYvAdIsw.mp4",
            original_name="JKpqYvAdIsw",
            source_url="https://www.youtube.com/watch?v=JKpqYvAdIsw",
            size=8201511,
            sha256="da63a55d05c49f042699b366f3026ba77c8695208b40eb40b506867604caf905",
        ),
        "video_33": MediaEntry(
            dataset="tvsum",
            video_id="video_33",
            filename="video_33.mp4",
            archive_member="video/xmEERLqJ2kU.mp4",
            original_name="xmEERLqJ2kU",
            source_url="https://www.youtube.com/watch?v=xmEERLqJ2kU",
            size=24129938,
            sha256="14184d35578cc9e672428f693ae74328ecb54b003d85857ff2afbf8354a90081",
        ),
        "video_34": MediaEntry(
            dataset="tvsum",
            video_id="video_34",
            filename="video_34.mp4",
            archive_member="video/byxOvuiIJV0.mp4",
            original_name="byxOvuiIJV0",
            source_url="https://www.youtube.com/watch?v=byxOvuiIJV0",
            size=8234183,
            sha256="b94f772069a7479b54c27b50650bccdfb0b75bb573abf8bda27dba1a5af75abd",
        ),
        "video_35": MediaEntry(
            dataset="tvsum",
            video_id="video_35",
            filename="video_35.mp4",
            archive_member="video/_xMr-HKMfVA.mp4",
            original_name="_xMr-HKMfVA",
            source_url="https://www.youtube.com/watch?v=_xMr-HKMfVA",
            size=8040777,
            sha256="1d9cd42e19931934ea362f7509bf69f0e98a13da801414998d58296fca4bcbbe",
        ),
        "video_36": MediaEntry(
            dataset="tvsum",
            video_id="video_36",
            filename="video_36.mp4",
            archive_member="video/WxtbjNsCQ8A.mp4",
            original_name="WxtbjNsCQ8A",
            source_url="https://www.youtube.com/watch?v=WxtbjNsCQ8A",
            size=14014038,
            sha256="b6c658e4d5258fcba10207b8c49ed624cabfe951ed9c9a96aa4ed6fada96d388",
        ),
        "video_37": MediaEntry(
            dataset="tvsum",
            video_id="video_37",
            filename="video_37.mp4",
            archive_member="video/uGu_10sucQo.mp4",
            original_name="uGu_10sucQo",
            source_url="https://www.youtube.com/watch?v=uGu_10sucQo",
            size=8256585,
            sha256="302cfeab45f174688d2a709cc4528087a5ba17d4fb9880ceac8fe8db87007b51",
        ),
        "video_38": MediaEntry(
            dataset="tvsum",
            video_id="video_38",
            filename="video_38.mp4",
            archive_member="video/EE-bNr36nyA.mp4",
            original_name="EE-bNr36nyA",
            source_url="https://www.youtube.com/watch?v=EE-bNr36nyA",
            size=5274150,
            sha256="0e11bb5bb60ac4842b2ee500c9cf985395a083ae92c3f498f1e91f12aa8e1691",
        ),
        "video_39": MediaEntry(
            dataset="tvsum",
            video_id="video_39",
            filename="video_39.mp4",
            archive_member="video/Se3oxnaPsz0.mp4",
            original_name="Se3oxnaPsz0",
            source_url="https://www.youtube.com/watch?v=Se3oxnaPsz0",
            size=7584394,
            sha256="87bf58a787009bce2add1a743bca81b98b33e7bb8edee80c1ff7c7070ae368e6",
        ),
        "video_40": MediaEntry(
            dataset="tvsum",
            video_id="video_40",
            filename="video_40.mp4",
            archive_member="video/oDXZc0tZe04.mp4",
            original_name="oDXZc0tZe04",
            source_url="https://www.youtube.com/watch?v=oDXZc0tZe04",
            size=20654776,
            sha256="dd2cddeaeb6bc44dea976337af5c8b79b0030abbe2438ed8549fe9304621efd2",
        ),
        "video_41": MediaEntry(
            dataset="tvsum",
            video_id="video_41",
            filename="video_41.mp4",
            archive_member="video/qqR6AEXwxoQ.mp4",
            original_name="qqR6AEXwxoQ",
            source_url="https://www.youtube.com/watch?v=qqR6AEXwxoQ",
            size=14617342,
            sha256="77987b565d016441183ec928ef985f1a1d805b4c36c833a0b0b2705fff6a092d",
        ),
        "video_42": MediaEntry(
            dataset="tvsum",
            video_id="video_42",
            filename="video_42.mp4",
            archive_member="video/EYqVtI9YWJA.mp4",
            original_name="EYqVtI9YWJA",
            source_url="https://www.youtube.com/watch?v=EYqVtI9YWJA",
            size=10713251,
            sha256="6ffcc966ab00a6096d7b256cffd09408efa7db376df7732644f875b8ff27f390",
        ),
        "video_43": MediaEntry(
            dataset="tvsum",
            video_id="video_43",
            filename="video_43.mp4",
            archive_member="video/eQu1rNs0an0.mp4",
            original_name="eQu1rNs0an0",
            source_url="https://www.youtube.com/watch?v=eQu1rNs0an0",
            size=8817667,
            sha256="7a3daec4c22feace99c4e193ce72e4a3e17bbedd4f0c6feb39e3e9030e6fb05f",
        ),
        "video_44": MediaEntry(
            dataset="tvsum",
            video_id="video_44",
            filename="video_44.mp4",
            archive_member="video/JgHubY5Vw3Y.mp4",
            original_name="JgHubY5Vw3Y",
            source_url="https://www.youtube.com/watch?v=JgHubY5Vw3Y",
            size=7682265,
            sha256="dff35488cc8dfd1b4962d82d8a96f475a051f163ed7dea4be8a0d10456e5b7c5",
        ),
        "video_45": MediaEntry(
            dataset="tvsum",
            video_id="video_45",
            filename="video_45.mp4",
            archive_member="video/iVt07TCkFM0.mp4",
            original_name="iVt07TCkFM0",
            source_url="https://www.youtube.com/watch?v=iVt07TCkFM0",
            size=5538282,
            sha256="7c34621e37ff0cc6d7f8838e8c6468e67eea934484e3c7ecf8eca7409c35e3f7",
        ),
        "video_46": MediaEntry(
            dataset="tvsum",
            video_id="video_46",
            filename="video_46.mp4",
            archive_member="video/E11zDS9XGzg.mp4",
            original_name="E11zDS9XGzg",
            source_url="https://www.youtube.com/watch?v=E11zDS9XGzg",
            size=27524696,
            sha256="f4370b8fd985c3b41bf5590988669b18794ea64fea3921a8109962a91c324986",
        ),
        "video_47": MediaEntry(
            dataset="tvsum",
            video_id="video_47",
            filename="video_47.mp4",
            archive_member="video/NyBmCxDoHJU.mp4",
            original_name="NyBmCxDoHJU",
            source_url="https://www.youtube.com/watch?v=NyBmCxDoHJU",
            size=10213204,
            sha256="6960e8850f04eae077a85e3befddaed36ce1196ac84d9efd8f8560ac90e557b7",
        ),
        "video_48": MediaEntry(
            dataset="tvsum",
            video_id="video_48",
            filename="video_48.mp4",
            archive_member="video/kLxoNp-UchI.mp4",
            original_name="kLxoNp-UchI",
            source_url="https://www.youtube.com/watch?v=kLxoNp-UchI",
            size=6970411,
            sha256="0c91bfa880e3af0e1716e7456ada0a304a97911eef76b1cd4fdce5066a52e8f7",
        ),
        "video_49": MediaEntry(
            dataset="tvsum",
            video_id="video_49",
            filename="video_49.mp4",
            archive_member="video/jcoYJXDG9sw.mp4",
            original_name="jcoYJXDG9sw",
            source_url="https://www.youtube.com/watch?v=jcoYJXDG9sw",
            size=14753114,
            sha256="3354cee170c1b9ace768883f058150869874ed27c2b60cf08b9f2272b5aa1db9",
        ),
        "video_50": MediaEntry(
            dataset="tvsum",
            video_id="video_50",
            filename="video_50.mp4",
            archive_member="video/-esJrBWj2d8.mp4",
            original_name="-esJrBWj2d8",
            source_url="https://www.youtube.com/watch?v=-esJrBWj2d8",
            size=12547350,
            sha256="811186cdec40ef58049f0cc951c07a9f6f7d2b5cb27511e01094b829bbd10a9e",
        ),
    },
    "summe": {
        "video_1": MediaEntry(
            dataset="summe",
            video_id="video_1",
            filename="video_1.mp4",
            archive_member="videos/Air_Force_One.mp4",
            original_name="Air_Force_One",
            size=60482840,
            sha256="5b84d4bed1fcfd036f701a582a1a1ef04cc443071e81a2d9347b756d82860c4a",
        ),
        "video_2": MediaEntry(
            dataset="summe",
            video_id="video_2",
            filename="video_2.mp4",
            archive_member="videos/Base jumping.mp4",
            original_name="Base jumping",
            size=52150848,
            sha256="b7156368c6b69a3ffc69249b3b54b1449143d531d1feb128019463f346e11428",
        ),
        "video_3": MediaEntry(
            dataset="summe",
            video_id="video_3",
            filename="video_3.mp4",
            archive_member="videos/Bearpark_climbing.mp4",
            original_name="Bearpark_climbing",
            size=124375584,
            sha256="5b67be9f486b31aced8665385427eb1ccc1104395a4b932a10305ec0f7db91c8",
        ),
        "video_4": MediaEntry(
            dataset="summe",
            video_id="video_4",
            filename="video_4.mp4",
            archive_member="videos/Bike Polo.mp4",
            original_name="Bike Polo",
            size=52347202,
            sha256="5992807a550df1b8e1a834d7d17ed9bde679a4f0c56b45f7d14f00a54f5aae66",
        ),
        "video_5": MediaEntry(
            dataset="summe",
            video_id="video_5",
            filename="video_5.mp4",
            archive_member="videos/Bus_in_Rock_Tunnel.mp4",
            original_name="Bus_in_Rock_Tunnel",
            size=47928740,
            sha256="346dbc70da3fe9d5e7fb2a9759eb8f5fc0fce420e8480f79ca01764772657ac3",
        ),
        "video_6": MediaEntry(
            dataset="summe",
            video_id="video_6",
            filename="video_6.mp4",
            archive_member="videos/Car_railcrossing.mp4",
            original_name="Car_railcrossing",
            size=63667468,
            sha256="ac45c745f5f3c54d1f05245a57e84e4673744a43c2a9346905320cffb073ff9d",
        ),
        "video_7": MediaEntry(
            dataset="summe",
            video_id="video_7",
            filename="video_7.mp4",
            archive_member="videos/Cockpit_Landing.mp4",
            original_name="Cockpit_Landing",
            size=93533028,
            sha256="a979e281572f5bec168b1e6635b3de7c7ecff5edcd3fad2fb750d4c22638b541",
        ),
        "video_8": MediaEntry(
            dataset="summe",
            video_id="video_8",
            filename="video_8.mp4",
            archive_member="videos/Cooking.mp4",
            original_name="Cooking",
            size=19948168,
            sha256="8ab50eecccf13b56b55a5c747ce11c7978bac9f202b1863a62894783b10600cf",
        ),
        "video_9": MediaEntry(
            dataset="summe",
            video_id="video_9",
            filename="video_9.mp4",
            archive_member="videos/Eiffel Tower.mp4",
            original_name="Eiffel Tower",
            size=134154334,
            sha256="8b41b90c9b15af8ab5c3bffa659301945d58dd40294c6f31e578f787331efeed",
        ),
        "video_10": MediaEntry(
            dataset="summe",
            video_id="video_10",
            filename="video_10.mp4",
            archive_member="videos/Excavators river crossing.mp4",
            original_name="Excavators river crossing",
            size=99182802,
            sha256="c62e40513295f8453798a536d146d123dda27ecc6943594bcdce2584a82ff215",
        ),
        "video_11": MediaEntry(
            dataset="summe",
            video_id="video_11",
            filename="video_11.mp4",
            archive_member="videos/Fire Domino.mp4",
            original_name="Fire Domino",
            size=13257686,
            sha256="547ce33f6a66da4c38c2b9b4252e87c04ba5834e97e363e6358ecd17ccd774c9",
        ),
        "video_12": MediaEntry(
            dataset="summe",
            video_id="video_12",
            filename="video_12.mp4",
            archive_member="videos/Jumps.mp4",
            original_name="Jumps",
            size=8499852,
            sha256="72e8dfabd30945feae89348469bd54797239ce405f5b87b191cac6bb5188f776",
        ),
        "video_13": MediaEntry(
            dataset="summe",
            video_id="video_13",
            filename="video_13.mp4",
            archive_member="videos/Kids_playing_in_leaves.mp4",
            original_name="Kids_playing_in_leaves",
            size=59231206,
            sha256="3186ee7b58064f754f70938bc9264e9b88bf3ee0296f79c8ac5dbabe7dadb9bc",
        ),
        "video_14": MediaEntry(
            dataset="summe",
            video_id="video_14",
            filename="video_14.mp4",
            archive_member="videos/Notre_Dame.mp4",
            original_name="Notre_Dame",
            size=108412268,
            sha256="12b47ab76499175e06a04d0f544ed38fdca73ba5471c8c5c295cd1b7c68137cd",
        ),
        "video_15": MediaEntry(
            dataset="summe",
            video_id="video_15",
            filename="video_15.mp4",
            archive_member="videos/Paintball.mp4",
            original_name="Paintball",
            size=68042258,
            sha256="af4779351898759eaa9c617f2d0255ddfad6b1cd54fc3072c382e548bdeefc11",
        ),
        "video_16": MediaEntry(
            dataset="summe",
            video_id="video_16",
            filename="video_16.mp4",
            archive_member="videos/Playing_on_water_slide.mp4",
            original_name="Playing_on_water_slide",
            size=26500418,
            sha256="926a0f0c0b1f47b107a6ede2e1666edf20c80f0355610cc8047f7639b6248e5e",
        ),
        "video_17": MediaEntry(
            dataset="summe",
            video_id="video_17",
            filename="video_17.mp4",
            archive_member="videos/Saving dolphines.mp4",
            original_name="Saving dolphines",
            size=91065842,
            sha256="6c9652190786a3a3ed3a389dfded81b70b34c242be3d54e6a8bc27053c6dc912",
        ),
        "video_18": MediaEntry(
            dataset="summe",
            video_id="video_18",
            filename="video_18.mp4",
            archive_member="videos/Scuba.mp4",
            original_name="Scuba",
            size=40467580,
            sha256="f497553350c6ea377c87c51f9512c435d34fb265df573d9512b04e9c5f378d0f",
        ),
        "video_19": MediaEntry(
            dataset="summe",
            video_id="video_19",
            filename="video_19.mp4",
            archive_member="videos/St Maarten Landing.mp4",
            original_name="St Maarten Landing",
            size=5448842,
            sha256="f12ba91597115508acc240d4368708cfce89d145b81c64e37e47446e022e893f",
        ),
        "video_20": MediaEntry(
            dataset="summe",
            video_id="video_20",
            filename="video_20.mp4",
            archive_member="videos/Statue of Liberty.mp4",
            original_name="Statue of Liberty",
            size=39865444,
            sha256="e0a76e5cef877c498b1668228d1db7c6e89e1860b70573324ce5bc8da0fb3937",
        ),
        "video_21": MediaEntry(
            dataset="summe",
            video_id="video_21",
            filename="video_21.mp4",
            archive_member="videos/Uncut_Evening_Flight.mp4",
            original_name="Uncut_Evening_Flight",
            size=129527376,
            sha256="3e4171accac9df560a4617253424bca2134d1c63a4f8ff107eed3a8371aa0b30",
        ),
        "video_22": MediaEntry(
            dataset="summe",
            video_id="video_22",
            filename="video_22.mp4",
            archive_member="videos/Valparaiso_Downhill.mp4",
            original_name="Valparaiso_Downhill",
            size=127484968,
            sha256="89e01f692b5bfeb0521f602e4b4f2d6e249e42a31b234cd2d652de82b453326b",
        ),
        "video_23": MediaEntry(
            dataset="summe",
            video_id="video_23",
            filename="video_23.mp4",
            archive_member="videos/car_over_camera.mp4",
            original_name="car_over_camera",
            size=28137398,
            sha256="a2e97294b381c15eb98d3dbd444943c70de3584b579ae13d6684c4a11f796b3b",
        ),
        "video_24": MediaEntry(
            dataset="summe",
            video_id="video_24",
            filename="video_24.mp4",
            archive_member="videos/paluma_jump.mp4",
            original_name="paluma_jump",
            size=29224748,
            sha256="f68a6a7ae01870d6baed552fc4614c21dc8a9c6e5f56cc5d50247c4cc132a10e",
        ),
        "video_25": MediaEntry(
            dataset="summe",
            video_id="video_25",
            filename="video_25.mp4",
            archive_member="videos/playing_ball.mp4",
            original_name="playing_ball",
            size=16916384,
            sha256="68a64cca5752e60e315aa68d7c17f909b1841deb0dd00b6be15a69a1a289ea9f",
        ),
    },
}
