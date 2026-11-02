"""Raw video media: official archives, ``video_id`` mapping and local resolution.

The evaluation substrate is the eccv16 h5 feature files handled by
:mod:`vsumbench.data.download`; raw frames are an **optional side channel**
for methods that need pixels. This module downloads the official video
archives, installs each video under its h5 ``video_id`` and resolves that id
back to a local path.

Alignment contract
------------------
Local media files are named ``<dataset>/<video_id>.mp4`` and the shipped
mapping (:mod:`vsumbench.data.media_sources`) pins every ``video_id`` to an
exact member of an official archive. Nothing here ever sorts a directory or
uses file-manifest order: a missing file is an error, not an invitation to
guess. This is the whole point — "the i-th file in the folder" is not a
stable identity.

Cache layout
------------
The media root resolves as ``override`` > ``VSUM_MEDIA_DIR`` >
``$VSUM_CACHE_DIR/media`` > ``~/.cache/vsumbench/media``. Archives are
downloaded whole (the official distributions are monolithic), used to
extract the requested members, and discarded unless ``keep_archive=True``.

Sources and licensing are documented in ``docs/design/raw-media.md``. This
module never scrapes YouTube and never invokes yt-dlp: TVSum's official
tgz and SumMe's official zip are the only download paths, both stdlib.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from vsumbench.data.download import cache_dir, sha256_of
from vsumbench.data.media_sources import MEDIA_ENTRIES, MEDIA_SOURCES, MediaEntry, MediaSource

__all__ = [
    "MEDIA_ENTRIES",
    "MEDIA_SOURCES",
    "MediaDownloadError",
    "MediaEntry",
    "MediaNotAvailableError",
    "MediaSource",
    "download_media",
    "expected_media_path",
    "media_dir",
    "media_entry",
    "media_index",
    "media_path",
    "media_source",
]

_CHUNK_BYTES = 1 << 20
_READ_TIMEOUT_S = 120
_ATTEMPTS_PER_MIRROR = 2
_PROGRESS_EVERY_BYTES = 64 << 20


class MediaDownloadError(RuntimeError):
    """The official media archive could not be fetched or extracted."""


class MediaNotAvailableError(RuntimeError):
    """No local media file for this ``video_id`` (or it failed verification)."""


def media_dir(override: Path | None = None) -> Path:
    """Resolve the media root (``override`` > ``VSUM_MEDIA_DIR`` > cache/media)."""
    if override is not None:
        return override
    env = os.environ.get("VSUM_MEDIA_DIR")
    if env:
        return Path(env)
    return cache_dir() / "media"


def media_source(dataset: str) -> MediaSource:
    """Return the official archive description for ``dataset``."""
    try:
        return MEDIA_SOURCES[dataset]
    except KeyError:
        known = ", ".join(sorted(MEDIA_SOURCES)) or "<none>"
        raise ValueError(
            f"no raw-media source for dataset {dataset!r}; available: {known}"
        ) from None


def media_entry(dataset: str, video_id: str) -> MediaEntry:
    """Return the mapping record for ``(dataset, video_id)``."""
    media_source(dataset)
    try:
        return MEDIA_ENTRIES[dataset][video_id]
    except KeyError:
        known = MEDIA_ENTRIES[dataset]
        raise ValueError(
            f"unknown video_id {video_id!r} for dataset {dataset!r} "
            f"({len(known)} mapped ids, e.g. {', '.join(list(known)[:3])})"
        ) from None


def expected_media_path(dataset: str, video_id: str, root: Path | None = None) -> Path:
    """Canonical local path for ``video_id``, whether or not the file exists."""
    entry = media_entry(dataset, video_id)
    return media_dir(root) / dataset / entry.filename


def media_path(
    dataset: str,
    video_id: str,
    root: Path | None = None,
    *,
    verify_sha256: bool = False,
) -> Path:
    """Resolve ``video_id`` to a verified local video file.

    Raises :class:`MediaNotAvailableError` when the file is absent, truncated
    or fails its content hash. The error names the exact archive member and
    the command that installs it; it never falls back to another file.
    """
    entry = media_entry(dataset, video_id)
    source = media_source(dataset)
    path = media_dir(root) / dataset / entry.filename
    if not path.is_file():
        raise MediaNotAvailableError(_missing_message(entry, source, path))
    _verify_local(entry, path, verify_sha256=verify_sha256)
    return path


def media_index(
    dataset: str, root: Path | None = None, *, verify_sha256: bool = False
) -> dict[str, Path]:
    """Map every locally present, canonical ``video_id`` to its path.

    Only files whose names are in the shipped mapping are considered; extra
    files are ignored (and can never shift the identity of a mapped id).
    """
    media_source(dataset)
    base = media_dir(root) / dataset
    present: dict[str, Path] = {}
    for video_id, entry in MEDIA_ENTRIES[dataset].items():
        path = base / entry.filename
        if not path.is_file():
            continue
        _verify_local(entry, path, verify_sha256=verify_sha256)
        present[video_id] = path
    return present


def download_media(
    dataset: str,
    root: Path | None = None,
    *,
    video_ids: Iterable[str] | None = None,
    report: Callable[[str], None] | None = None,
    verify_sha256: bool = True,
    keep_archive: bool = False,
) -> list[Path]:
    """Download the official archive and install the requested videos.

    ``video_ids=None`` installs every mapped video. Already-installed files
    are size/hash-checked and skipped; corrupt ones are replaced atomically.
    The archive is removed after extraction unless ``keep_archive`` is true
    (a kept archive is verified and reused on later calls).

    Returns the local paths in the requested order.
    """
    source = media_source(dataset)
    entries = _select_entries(dataset, video_ids)
    say = report if report is not None else (lambda _msg: None)
    base = media_dir(root)
    dataset_dir = base / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    destinations = {entry.video_id: dataset_dir / entry.filename for entry in entries}
    pending: list[MediaEntry] = []
    for entry in entries:
        path = destinations[entry.video_id]
        if path.is_file() and _local_is_valid(entry, path, verify_sha256=verify_sha256):
            say(f"already present and verified: {path}")
        else:
            pending.append(entry)
    if not pending:
        return [destinations[entry.video_id] for entry in entries]

    with tempfile.TemporaryDirectory(prefix=".vsumbench-media-", dir=base) as tmp_name:
        tmp = Path(tmp_name)
        archive = _obtain_archive(source, base, tmp, keep_archive=keep_archive, say=say)
        with _member_zip(source, archive, tmp, say) as members:
            staged = tmp / "staged"
            staged.mkdir()
            for entry in pending:
                _stage_member(members, source, entry, staged, say)
            for entry in pending:
                os.replace(staged / entry.filename, destinations[entry.video_id])
                say(f"installed: {destinations[entry.video_id]}")

    return [destinations[entry.video_id] for entry in entries]


# -- selection and local verification ------------------------------------------


def _select_entries(dataset: str, video_ids: Iterable[str] | None) -> list[MediaEntry]:
    table = MEDIA_ENTRIES[dataset]
    if video_ids is None:
        return list(table.values())
    if isinstance(video_ids, str):
        raise TypeError("video_ids must be an iterable of ids, not a single string")
    ids = list(video_ids)
    if not ids:
        raise ValueError("video_ids must not be empty")
    unknown = [video_id for video_id in ids if video_id not in table]
    if unknown:
        raise ValueError(
            f"unknown video_id(s) for dataset {dataset!r}: {', '.join(unknown)}; "
            f"mapped ids are {', '.join(table)}"
        )
    return [table[video_id] for video_id in dict.fromkeys(ids)]


def _local_is_valid(entry: MediaEntry, path: Path, *, verify_sha256: bool) -> bool:
    try:
        _verify_local(entry, path, verify_sha256=verify_sha256)
    except MediaNotAvailableError:
        return False
    return True


def _verify_local(entry: MediaEntry, path: Path, *, verify_sha256: bool) -> None:
    if entry.size is not None:
        actual_size = path.stat().st_size
        if actual_size != entry.size:
            raise MediaNotAvailableError(
                f"raw media for {entry.dataset}/{entry.video_id} has the wrong size: "
                f"{path} is {actual_size} bytes, expected {entry.size}.\n"
                f"Delete it and run `vsumm download-media {entry.dataset} "
                f"--video-id {entry.video_id}`."
            )
    if verify_sha256 and entry.sha256 is not None:
        actual = sha256_of(path)
        if actual != entry.sha256:
            raise MediaNotAvailableError(
                f"raw media for {entry.dataset}/{entry.video_id} failed its sha256 check: "
                f"{path}\n  expected: {entry.sha256}\n  actual:   {actual}\n"
                f"Delete it and run `vsumm download-media {entry.dataset} "
                f"--video-id {entry.video_id}`."
            )


def _missing_message(entry: MediaEntry, source: MediaSource, path: Path) -> str:
    inner = f" (inside {source.inner_archive})" if source.inner_archive else ""
    return (
        f"raw media for {entry.dataset}/{entry.video_id} is not available at {path}.\n"
        f"  install it from the official archive:\n"
        f"    vsumm download-media {entry.dataset} --video-id {entry.video_id}\n"
        f"  or place the official file there yourself; it is\n"
        f"    {entry.archive_member}{inner}\n"
        f"  in {source.archive_urls[0]} (sha256 {source.archive_sha256}).\n"
        f"  Media files are addressed by video_id; never align a directory "
        f"listing by sort order."
    )


# -- archive acquisition --------------------------------------------------------


def _obtain_archive(
    source: MediaSource,
    base: Path,
    tmp: Path,
    *,
    keep_archive: bool,
    say: Callable[[str], None],
) -> Path:
    cached = base / ".archives" / source.archive_filename
    if cached.is_file():
        try:
            _verify_archive(source, cached)
        except MediaDownloadError:
            say(f"cached archive failed verification, re-downloading: {cached}")
            cached.unlink()
        else:
            say(f"using verified cached archive: {cached}")
            return cached

    downloads = [
        (url, attempt)
        for url in source.archive_urls
        for attempt in range(1, _ATTEMPTS_PER_MIRROR + 1)
    ]
    failures: list[str] = []
    for url, attempt in downloads:
        target = tmp / source.archive_filename
        target.unlink(missing_ok=True)
        try:
            _download_file(url, target, say)
            _verify_archive(source, target)
        except (MediaDownloadError, urllib.error.URLError, OSError) as exc:
            failures.append(f"{url}: {exc}")
            if attempt < _ATTEMPTS_PER_MIRROR:
                say(f"attempt {attempt} failed ({exc}), retrying {url}")
            continue
        if keep_archive:
            cached.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, cached)
            say(f"kept verified archive: {cached}")
            return cached
        return target
    raise MediaDownloadError(
        f"could not fetch the {source.dataset!r} media archive from any mirror:\n  "
        + "\n  ".join(failures)
    )


def _verify_archive(source: MediaSource, path: Path) -> None:
    if not path.is_file():
        raise MediaDownloadError(f"missing archive: {path}")
    actual_size = path.stat().st_size
    if actual_size != source.archive_size:
        raise MediaDownloadError(
            f"archive size mismatch for {path}: expected {source.archive_size} bytes, "
            f"got {actual_size}"
        )
    actual = sha256_of(path)
    if actual != source.archive_sha256:
        raise MediaDownloadError(
            f"archive checksum mismatch for {path}\n"
            f"  expected sha256: {source.archive_sha256}\n"
            f"  actual   sha256: {actual}"
        )


def _download_file(url: str, dest: Path, say: Callable[[str], None]) -> None:
    say(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "vsumbench"})
    done = 0
    with urllib.request.urlopen(request, timeout=_READ_TIMEOUT_S) as response:
        total = response.headers.get("Content-Length")
        with open(dest, "wb") as fh:
            while True:
                chunk = response.read(_CHUNK_BYTES)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if done % _PROGRESS_EVERY_BYTES < _CHUNK_BYTES:
                    say(f"  {done / 1e6:.0f} MB")
    if total is not None and done != int(total):
        raise MediaDownloadError(f"truncated download ({done} of {total} bytes) from {url}")


@contextmanager
def _member_zip(
    source: MediaSource, archive: Path, tmp: Path, say: Callable[[str], None]
) -> Iterator[zipfile.ZipFile]:
    if source.archive_format == "zip":
        with zipfile.ZipFile(archive) as members:
            yield members
        return
    if source.inner_archive is None:
        raise MediaDownloadError(
            f"{source.dataset}: {source.archive_format} source is missing inner_archive"
        )
    inner = tmp / Path(source.inner_archive).name
    _extract_tar_member(archive, source.inner_archive, inner, say)
    with zipfile.ZipFile(inner) as members:
        yield members


def _extract_tar_member(
    archive: Path, member_name: str, dest: Path, say: Callable[[str], None]
) -> None:
    wanted = _normalize_member(member_name)
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if _normalize_member(member.name) != wanted:
                continue
            if not member.isfile():
                raise MediaDownloadError(f"archive member is not a file: {member.name}")
            source = tar.extractfile(member)
            if source is None:  # pragma: no cover - guarded by member.isfile()
                raise MediaDownloadError(f"cannot read archive member: {member.name}")
            say(f"extracting {member.name} ({member.size / 1e6:.0f} MB)")
            with source, open(dest, "wb") as fh:
                shutil.copyfileobj(source, fh, _CHUNK_BYTES)
            return
    raise MediaDownloadError(
        f"{archive} does not contain {member_name!r}; the official archive layout "
        f"changed — update vsumbench.data.media_sources rather than guessing by order"
    )


def _stage_member(
    members: zipfile.ZipFile,
    source: MediaSource,
    entry: MediaEntry,
    staged: Path,
    say: Callable[[str], None],
) -> None:
    wanted = _normalize_member(entry.archive_member)
    info = next(
        (item for item in members.infolist() if _normalize_member(item.filename) == wanted),
        None,
    )
    if info is None:
        raise MediaDownloadError(
            f"{source.archive_filename} does not contain {entry.archive_member!r} "
            f"for {entry.dataset}/{entry.video_id}; the official archive layout "
            f"changed — update vsumbench.data.media_sources rather than guessing by order"
        )
    if info.is_dir():
        raise MediaDownloadError(f"archive member is a directory: {info.filename}")
    if entry.size is not None and info.file_size != entry.size:
        raise MediaDownloadError(
            f"{info.filename}: size {info.file_size} does not match the mapped "
            f"{entry.size}; refusing to install a different video under "
            f"{entry.dataset}/{entry.video_id}"
        )
    digest = hashlib.sha256()
    out = staged / entry.filename
    with members.open(info) as src, open(out, "wb") as dst:
        while chunk := src.read(_CHUNK_BYTES):
            digest.update(chunk)
            dst.write(chunk)
    if entry.sha256 is not None and digest.hexdigest() != entry.sha256:
        raise MediaDownloadError(
            f"{entry.archive_member}: sha256 mismatch\n"
            f"  expected: {entry.sha256}\n  actual:   {digest.hexdigest()}"
        )
    say(f"staged: {entry.dataset}/{entry.video_id} <- {entry.archive_member}")


def _normalize_member(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name.lstrip("/")
