"""Offline unit tests for download/verify/cache logic.

Mirrors are simulated with a local HTTP server serving tar archives (the
``tar``/``bsdtar`` extractor branch of :mod:`vsumbench.data.download`), so
the whole fetch -> extract -> verify -> install path runs without network.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import shutil
import tarfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from vsumbench.data import download as dl
from vsumbench.data.sources import SOURCES, DatasetSpec, FileSpec

FAKE_NAME = "fake"
FAKE_FILENAME = "fake_data.h5"
FAKE_BYTES = b"canonical-bytes-of-the-fake-dataset"


# -- local mirror machinery ------------------------------------------------------


def _tar_bytes(filename: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


@contextmanager
def serve_paths(files: dict[str, bytes]) -> Iterator[str]:
    """Serve ``{path: body}`` over localhost HTTP; yields the base URL."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = files.get(self.path.lstrip("/"))
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def fake_spec() -> Iterator[DatasetSpec]:
    spec = DatasetSpec(
        name=FAKE_NAME,
        files=(
            FileSpec(
                filename=FAKE_FILENAME,
                sha256=hashlib.sha256(FAKE_BYTES).hexdigest(),
                size=len(FAKE_BYTES),
            ),
        ),
        archive_urls=(),  # tests register mirrors per case
    )
    SOURCES[FAKE_NAME] = spec
    yield spec
    del SOURCES[FAKE_NAME]


def _register_mirrors(spec: DatasetSpec, urls: tuple[str, ...]) -> DatasetSpec:
    return DatasetSpec(name=spec.name, files=spec.files, archive_urls=urls)


# -- cache root ------------------------------------------------------------------


def test_cache_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert dl.cache_dir(tmp_path) == tmp_path
    monkeypatch.setenv("VSUM_CACHE_DIR", str(tmp_path / "env"))
    assert dl.cache_dir() == tmp_path / "env"
    assert dl.cache_dir(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("VSUM_CACHE_DIR")
    assert dl.cache_dir() == Path.home() / ".cache" / "vsumbench"


def test_sha256_known_vector(tmp_path: Path) -> None:
    path = tmp_path / "abc"
    path.write_bytes(b"abc")
    assert dl.sha256_of(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


# -- verification ----------------------------------------------------------------


def test_verify_file_ok(tmp_path: Path, fake_spec: DatasetSpec) -> None:
    path = tmp_path / FAKE_FILENAME
    path.write_bytes(FAKE_BYTES)
    dl.verify_file(fake_spec.files[0], path)  # no exception


def test_verify_file_missing(tmp_path: Path, fake_spec: DatasetSpec) -> None:
    with pytest.raises(dl.ChecksumError, match="missing file"):
        dl.verify_file(fake_spec.files[0], tmp_path / FAKE_FILENAME)


def test_verify_file_size_mismatch(tmp_path: Path, fake_spec: DatasetSpec) -> None:
    path = tmp_path / FAKE_FILENAME
    path.write_bytes(FAKE_BYTES[:-1])
    with pytest.raises(dl.ChecksumError, match="size mismatch"):
        dl.verify_file(fake_spec.files[0], path)


def test_verify_file_checksum_mismatch(tmp_path: Path, fake_spec: DatasetSpec) -> None:
    path = tmp_path / FAKE_FILENAME
    tampered = bytearray(FAKE_BYTES)
    tampered[0] ^= 0xFF  # same size, different content
    path.write_bytes(bytes(tampered))
    with pytest.raises(dl.ChecksumError, match="checksum mismatch"):
        dl.verify_file(fake_spec.files[0], path)


def test_unknown_dataset_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        dl.download_dataset("nope")


# -- download flow ----------------------------------------------------------------


def test_download_installs_and_reverifies(fake_spec: DatasetSpec, tmp_path: Path) -> None:
    with serve_paths({"mirror.tar": _tar_bytes(FAKE_FILENAME, FAKE_BYTES)}) as base:
        SOURCES[FAKE_NAME] = _register_mirrors(fake_spec, (f"{base}/mirror.tar",))
        messages: list[str] = []
        paths = dl.download_dataset(FAKE_NAME, tmp_path, report=messages.append)
    assert paths[0] == tmp_path / FAKE_FILENAME
    assert paths[0].read_bytes() == FAKE_BYTES
    assert not list(tmp_path.glob(".vsumbench-tmp-*")), "temporary dir left behind"
    # second call takes the offline path: verify only
    messages.clear()
    dl.download_dataset(FAKE_NAME, tmp_path, report=messages.append)
    assert any("already present and verified" in m for m in messages)


def test_download_corrupt_local_file_errors(fake_spec: DatasetSpec, tmp_path: Path) -> None:
    tampered = bytearray(FAKE_BYTES)
    tampered[0] ^= 0xFF  # same size, different content
    (tmp_path / FAKE_FILENAME).write_bytes(bytes(tampered))
    with pytest.raises(dl.ChecksumError, match="checksum mismatch"):
        dl.download_dataset(FAKE_NAME, tmp_path)


def test_download_falls_back_to_next_mirror(fake_spec: DatasetSpec, tmp_path: Path) -> None:
    with serve_paths(
        {
            "good.tar": _tar_bytes(FAKE_FILENAME, FAKE_BYTES),
        }
    ) as base:
        SOURCES[FAKE_NAME] = _register_mirrors(
            fake_spec, (f"{base}/broken.tar", f"{base}/good.tar")
        )
        paths = dl.download_dataset(FAKE_NAME, tmp_path)
    assert paths[0].read_bytes() == FAKE_BYTES


def test_download_rejects_corrupt_mirror_content(fake_spec: DatasetSpec, tmp_path: Path) -> None:
    # same size, wrong bytes -> sha mismatch inside the mirror chain
    tampered = bytearray(FAKE_BYTES)
    tampered[-1] ^= 0x01
    with serve_paths({"bad.tar": _tar_bytes(FAKE_FILENAME, bytes(tampered))}) as base:
        SOURCES[FAKE_NAME] = _register_mirrors(fake_spec, (f"{base}/bad.tar",))
        with pytest.raises(dl.DownloadError, match="could not fetch"):
            dl.download_dataset(FAKE_NAME, tmp_path)
    assert not (tmp_path / FAKE_FILENAME).exists(), "corrupt content must not be installed"


def test_all_mirrors_unreachable(fake_spec: DatasetSpec, tmp_path: Path) -> None:
    SOURCES[FAKE_NAME] = _register_mirrors(fake_spec, ("http://127.0.0.1:1/x.tar",))
    with pytest.raises(dl.DownloadError, match="mirror"):
        dl.download_dataset(FAKE_NAME, tmp_path)


# -- shipped source table ---------------------------------------------------------


def test_builtin_sources_wellformed() -> None:
    for name in ("tvsum", "summe"):
        spec = SOURCES[name]
        assert spec.name == name
        assert len(spec.files) == 1
        file_spec = spec.files[0]
        assert file_spec.filename.startswith(f"eccv16_dataset_{name}_")
        assert len(file_spec.sha256) == 64 and int(file_spec.sha256, 16) >= 0
        assert file_spec.size > 0
        assert all(url.startswith("https://") for url in spec.archive_urls)


def test_an_extractor_exists() -> None:
    assert any(shutil.which(tool) for tool, _ in dl._EXTRACTORS)
