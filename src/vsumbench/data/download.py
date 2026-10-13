"""Download, cache and verify canonical dataset files.

Resolution order for the cache root: explicit ``override`` argument >
``VSUM_CACHE_DIR`` environment variable > ``~/.cache/vsumbench``. Files that
already exist in the root are checksum-verified and never re-downloaded
(the offline mode: obtain the file by any means, ``vsumm download`` only
verifies).

Everything here is stdlib-only: no vsumbench imports besides the source
table, no network at import time. Archives (rar) are extracted through
whichever system tool is available.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from vsumbench.data.sources import SOURCES, DatasetSpec, FileSpec

__all__ = [
    "ChecksumError",
    "DownloadError",
    "cache_dir",
    "download_dataset",
    "sha256_of",
    "verify_dataset",
    "verify_file",
]

_CHUNK_BYTES = 1 << 20
_READ_TIMEOUT_S = 120
_ATTEMPTS_PER_MIRROR = 2

#: (tool, argv template) pairs tried in order; ``{archive}``/``{dest}`` are
#: substituted per run. All of them extract every member of the archive.
#: ``tar`` also covers plain tar archives (used by offline tests).
_EXTRACTORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bsdtar", ("-xf", "{archive}", "-C", "{dest}")),
    ("tar", ("-xf", "{archive}", "-C", "{dest}")),
    ("unrar", ("x", "-o+", "-y", "{archive}", f"{{dest}}{os.sep}")),
    ("unar", ("-f", "-o", "{dest}", "{archive}")),
    ("7z", ("x", "-y", "-o{dest}", "{archive}")),
)


class DownloadError(RuntimeError):
    """A dataset file could not be fetched or extracted."""


class ChecksumError(DownloadError):
    """A local file does not match its canonical sha256."""


def cache_dir(override: Path | None = None) -> Path:
    """Resolve the dataset cache root (``override`` > ``VSUM_CACHE_DIR`` > default)."""
    if override is not None:
        return override
    env = os.environ.get("VSUM_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "vsumbench"


def sha256_of(path: str | Path) -> str:
    """Stream a file through sha256 and return the hex digest."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(spec: FileSpec, path: Path) -> None:
    """Raise :class:`ChecksumError` unless ``path`` matches ``spec`` exactly."""
    if not path.is_file():
        raise ChecksumError(f"missing file: {path}")
    actual_size = path.stat().st_size
    if actual_size != spec.size:
        raise ChecksumError(
            f"size mismatch for {path}: expected {spec.size} bytes, got {actual_size}"
        )
    actual = sha256_of(path)
    if actual != spec.sha256:
        raise ChecksumError(
            f"checksum mismatch for {path}\n  expected sha256: {spec.sha256}\n"
            f"  actual   sha256: {actual}\n"
            f"The file is corrupt or from an incompatible source; delete it and "
            f"run `vsumm download` again."
        )


def verify_dataset(name: str, root: Path | None = None) -> list[Path]:
    """Verify every canonical file of dataset ``name`` under ``root``."""
    spec = _spec(name)
    base = cache_dir(root)
    paths = [base / f.filename for f in spec.files]
    for file_spec, path in zip(spec.files, paths, strict=True):
        verify_file(file_spec, path)
    return paths


def download_dataset(
    name: str, root: Path | None = None, *, report: Callable[[str], None] | None = None
) -> list[Path]:
    """Ensure dataset ``name`` is present and verified under the cache root.

    Existing files are only verified; missing ones are fetched from the
    first reachable mirror archive, extracted, verified against their
    sha256, and moved into place atomically.
    """
    spec = _spec(name)
    base = cache_dir(root)
    base.mkdir(parents=True, exist_ok=True)
    say = report if report is not None else (lambda _msg: None)

    targets = {f.filename: base / f.filename for f in spec.files}
    for file_spec in spec.files:
        path = targets[file_spec.filename]
        if path.exists():
            verify_file(file_spec, path)
            say(f"already present and verified: {path}")

    missing = [f for f in spec.files if not targets[f.filename].exists()]
    if missing:
        _fetch_from_mirrors(spec, base, targets, missing, say)
    return [targets[f.filename] for f in spec.files]


def _spec(name: str) -> DatasetSpec:
    try:
        return SOURCES[name]
    except KeyError:
        known = ", ".join(sorted(SOURCES)) or "<none>"
        raise ValueError(f"unknown dataset {name!r}; available: {known}") from None


def _fetch_from_mirrors(
    spec: DatasetSpec,
    base: Path,
    targets: dict[str, Path],
    missing: list[FileSpec],
    say: Callable[[str], None],
) -> None:
    failures: list[str] = []
    for url in spec.archive_urls:
        # Slow or flaky links stall between reads; one retry reconnects.
        for attempt in range(1, _ATTEMPTS_PER_MIRROR + 1):
            # Keep archive + extraction on the same filesystem as the cache so
            # the final move is an atomic os.replace.
            with tempfile.TemporaryDirectory(prefix=".vsumbench-tmp-", dir=base) as tmp_name:
                tmp = Path(tmp_name)
                archive = tmp / "download.rar"
                try:
                    _download_file(url, archive, say)
                    extracted = tmp / "extracted"
                    extracted.mkdir()
                    _extract_archive(archive, extracted)
                    for file_spec in missing:
                        found = _locate(extracted, file_spec.filename)
                        if found is None:
                            raise DownloadError(
                                f"archive from {url} does not contain {file_spec.filename}"
                            )
                        verify_file(file_spec, found)
                        os.replace(found, targets[file_spec.filename])
                        say(f"verified and installed: {targets[file_spec.filename]}")
                    return
                except (
                    DownloadError,
                    urllib.error.URLError,
                    OSError,
                    subprocess.SubprocessError,
                ) as exc:
                    failures.append(f"{url}: {exc}")
                    if attempt < _ATTEMPTS_PER_MIRROR:
                        say(f"attempt {attempt} failed ({exc}), retrying {url}")
    raise DownloadError(
        f"could not fetch dataset {spec.name!r} from any mirror:\n  " + "\n  ".join(failures)
    )


def _download_file(url: str, dest: Path, say: Callable[[str], None]) -> None:
    say(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "vsumbench"})
    with urllib.request.urlopen(request, timeout=_READ_TIMEOUT_S) as response:
        total = response.headers.get("Content-Length")
        done = 0
        with open(dest, "wb") as fh:
            while True:
                chunk = response.read(_CHUNK_BYTES)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
    if total is not None and done != int(total):
        raise DownloadError(f"truncated download ({done} of {total} bytes) from {url}")


def _extract_archive(archive: Path, dest: Path) -> None:
    failures: list[str] = []
    for tool, template in _EXTRACTORS:
        path = shutil.which(tool)
        if path is None:
            continue
        argv = [path, *(part.format(archive=str(archive), dest=str(dest)) for part in template)]
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode == 0 and any(dest.iterdir()):
            return
        failures.append(f"{tool} exited with code {result.returncode}")
    if failures:
        raise DownloadError(
            f"archive extraction failed ({'; '.join(failures)}); the download is probably corrupt"
        )
    raise DownloadError(
        "no rar extractor available (tried bsdtar, unrar, unar, 7z); install one of "
        "them (e.g. `brew install sevenzip` / `apt install unrar-free`), or extract "
        "the downloaded archive manually into the cache directory"
    )


def _locate(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None
