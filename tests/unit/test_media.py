"""Offline tests for the raw-media mapping, resolution and download flow.

The real archives are hundreds of megabytes; every download path here runs
against a local HTTP server serving synthetic zip / tar.gz archives. The
mapping itself is checked for completeness and spot-checked against the
verified ``video_N`` joins (see docs/design/raw-media.md).
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import test_download as td

from vsumbench import cli
from vsumbench.data import download as dl
from vsumbench.data import media as md
from vsumbench.data.media_sources import MEDIA_ENTRIES, MEDIA_SOURCES, MediaEntry, MediaSource

FAKE_DATASET = "fakemedia"
FAKE_PAYLOADS = {
    "video_1": b"fake mp4 payload for video one",
    "video_2": b"fake mp4 payload for video two (longer)",
}
FAKE_MEMBERS = {
    "video_1": "videos/one.mp4",
    "video_2": "videos/two.mp4",
}


# -- synthetic archives ---------------------------------------------------------


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _tar_gz_bytes(name: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _entries() -> dict[str, MediaEntry]:
    return {
        video_id: MediaEntry(
            dataset=FAKE_DATASET,
            video_id=video_id,
            filename=f"{video_id}.mp4",
            archive_member=FAKE_MEMBERS[video_id],
            original_name=Path(FAKE_MEMBERS[video_id]).stem,
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for video_id, payload in FAKE_PAYLOADS.items()
    }


def _source(
    url: str, archive: bytes, *, archive_format: str = "zip", inner: str | None = None
) -> MediaSource:
    return MediaSource(
        dataset=FAKE_DATASET,
        homepage="https://example.invalid/dataset",
        license="test-only",
        archive_urls=(url,),
        archive_filename="bundle.zip" if archive_format == "zip" else "bundle.tgz",
        archive_size=len(archive),
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        archive_format=archive_format,  # type: ignore[arg-type]
        inner_archive=inner,
    )


@pytest.fixture
def fake_entries() -> Iterator[dict[str, MediaEntry]]:
    entries = _entries()
    MEDIA_SOURCES[FAKE_DATASET] = MediaSource(
        dataset=FAKE_DATASET,
        homepage="https://example.invalid/dataset",
        license="test-only",
        archive_urls=("https://example.invalid/bundle.zip",),
        archive_filename="bundle.zip",
        archive_size=1,
        archive_sha256="0" * 64,
        archive_format="zip",
    )
    MEDIA_ENTRIES[FAKE_DATASET] = entries
    yield entries
    MEDIA_SOURCES.pop(FAKE_DATASET, None)
    MEDIA_ENTRIES.pop(FAKE_DATASET, None)


def _install_fake_source(
    archive: bytes, url: str, *, archive_format: str = "zip", inner: str | None = None
) -> None:
    MEDIA_SOURCES[FAKE_DATASET] = _source(url, archive, archive_format=archive_format, inner=inner)


# -- shipped mapping ------------------------------------------------------------


def test_mapping_covers_both_datasets_with_contiguous_ids() -> None:
    assert set(MEDIA_SOURCES) == {"tvsum", "summe"}
    assert set(MEDIA_ENTRIES) == set(MEDIA_SOURCES)
    expected_counts = {"tvsum": 50, "summe": 25}
    for dataset, count in expected_counts.items():
        table = MEDIA_ENTRIES[dataset]
        assert list(table) == [f"video_{index}" for index in range(1, count + 1)]


def test_mapping_entries_are_wellformed_and_unique() -> None:
    for dataset, table in MEDIA_ENTRIES.items():
        filenames: set[str] = set()
        members: set[str] = set()
        for video_id, entry in table.items():
            assert entry.dataset == dataset
            assert entry.video_id == video_id
            assert entry.filename == f"{video_id}.mp4"
            assert entry.filename not in filenames
            assert entry.archive_member not in members
            filenames.add(entry.filename)
            members.add(entry.archive_member)
            assert entry.original_name
            assert entry.size is not None and entry.size > 0
            assert entry.sha256 is not None and len(entry.sha256) == 64
            int(entry.sha256, 16)


def test_mapping_sources_are_wellformed() -> None:
    for dataset, source in MEDIA_SOURCES.items():
        assert source.dataset == dataset
        assert source.archive_urls and all(
            url.startswith("https://") for url in source.archive_urls
        )
        assert source.archive_size > 0
        assert len(source.archive_sha256) == 64
        int(source.archive_sha256, 16)
        assert source.archive_format in ("zip", "tar.gz")
        if source.archive_format == "tar.gz":
            assert source.inner_archive


def test_mapping_spot_checks_verified_joins() -> None:
    # TVSum: row order of ydata-tvsum50-info.tsv == h5 video_N; validated by
    # comparing n_frames / reported duration (all 50 land at 24/25/30 fps).
    tvsum = MEDIA_ENTRIES["tvsum"]
    assert tvsum["video_1"].original_name == "AwmHb44_ouw"
    assert tvsum["video_1"].source_url == "https://www.youtube.com/watch?v=AwmHb44_ouw"
    assert tvsum["video_50"].original_name == "-esJrBWj2d8"
    # SumMe: h5 group attribute video_name == videos/<name>.mp4 in the archive.
    summe = MEDIA_ENTRIES["summe"]
    assert summe["video_1"].original_name == "Air_Force_One"
    assert summe["video_1"].archive_member == "videos/Air_Force_One.mp4"
    assert summe["video_10"].original_name == "Excavators river crossing"


# -- root resolution ------------------------------------------------------------


def test_media_dir_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_entries: dict[str, MediaEntry]
) -> None:
    monkeypatch.delenv("VSUM_MEDIA_DIR", raising=False)
    assert md.media_dir(tmp_path) == tmp_path
    monkeypatch.setenv("VSUM_MEDIA_DIR", str(tmp_path / "env"))
    assert md.media_dir() == tmp_path / "env"
    assert md.media_dir(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("VSUM_MEDIA_DIR")
    assert md.media_dir() == dl.cache_dir() / "media"


# -- id -> path resolution ------------------------------------------------------


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_media_path_resolves_by_id_not_by_directory_order(
    tmp_path: Path, fake_entries: dict[str, MediaEntry]
) -> None:
    # Only video_2 exists, plus an unrelated file that sorts first.
    _write(tmp_path / FAKE_DATASET / "video_2.mp4", FAKE_PAYLOADS["video_2"])
    _write(tmp_path / FAKE_DATASET / "0001.mp4", b"not a mapped video")
    assert md.media_path(FAKE_DATASET, "video_2", tmp_path) == (
        tmp_path / FAKE_DATASET / "video_2.mp4"
    )
    assert md.expected_media_path(FAKE_DATASET, "video_1", tmp_path) == (
        tmp_path / FAKE_DATASET / "video_1.mp4"
    )
    with pytest.raises(md.MediaNotAvailableError, match="is not available"):
        md.media_path(FAKE_DATASET, "video_1", tmp_path)
    assert md.media_index(FAKE_DATASET, tmp_path) == {
        "video_2": tmp_path / FAKE_DATASET / "video_2.mp4"
    }


def test_missing_media_error_is_actionable(
    tmp_path: Path, fake_entries: dict[str, MediaEntry]
) -> None:
    with pytest.raises(md.MediaNotAvailableError) as excinfo:
        md.media_path(FAKE_DATASET, "video_1", tmp_path)
    message = str(excinfo.value)
    assert "vsumm download-media fakemedia --video-id video_1" in message
    assert "videos/one.mp4" in message
    assert "https://example.invalid/bundle.zip" in message
    assert "never align a directory listing by sort order" in message
    assert str(tmp_path / FAKE_DATASET / "video_1.mp4") in message


def test_media_path_checks_size_and_optional_sha256(
    tmp_path: Path, fake_entries: dict[str, MediaEntry]
) -> None:
    good = _write(tmp_path / FAKE_DATASET / "video_1.mp4", FAKE_PAYLOADS["video_1"])
    assert md.media_path(FAKE_DATASET, "video_1", tmp_path) == good

    # Same size, different bytes: only the explicit sha256 check catches it.
    tampered = bytearray(FAKE_PAYLOADS["video_1"])
    tampered[0] ^= 0xFF
    good.write_bytes(bytes(tampered))
    assert md.media_path(FAKE_DATASET, "video_1", tmp_path) == good
    with pytest.raises(md.MediaNotAvailableError, match="sha256"):
        md.media_path(FAKE_DATASET, "video_1", tmp_path, verify_sha256=True)

    good.write_bytes(FAKE_PAYLOADS["video_1"][:-1])
    with pytest.raises(md.MediaNotAvailableError, match="wrong size"):
        md.media_path(FAKE_DATASET, "video_1", tmp_path)


def test_unknown_dataset_and_id_fail_loudly(fake_entries: dict[str, MediaEntry]) -> None:
    with pytest.raises(ValueError, match="no raw-media source"):
        md.media_entry("nope", "video_1")
    with pytest.raises(ValueError, match="unknown video_id"):
        md.media_entry(FAKE_DATASET, "video_99")


# -- download flows -------------------------------------------------------------


def test_download_installs_by_mapping_and_skips_on_repeat(
    tmp_path: Path, fake_entries: dict[str, MediaEntry], capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _zip_bytes(
        {FAKE_MEMBERS[video_id]: payload for video_id, payload in FAKE_PAYLOADS.items()}
    )
    messages: list[str] = []
    with td.serve_paths({"bundle.zip": archive}) as base:
        _install_fake_source(archive, f"{base}/bundle.zip")
        paths = md.download_media(FAKE_DATASET, tmp_path, report=messages.append)
    assert paths == [
        tmp_path / FAKE_DATASET / "video_1.mp4",
        tmp_path / FAKE_DATASET / "video_2.mp4",
    ]
    assert [path.read_bytes() for path in paths] == list(FAKE_PAYLOADS.values())
    assert not list(tmp_path.glob(".vsumbench-media-*"))

    # Server is gone; valid local files mean no network access is attempted.
    messages.clear()
    assert md.download_media(FAKE_DATASET, tmp_path, report=messages.append) == paths
    assert sum("already present and verified" in message for message in messages) == 2


def test_download_subset_by_video_id(tmp_path: Path, fake_entries: dict[str, MediaEntry]) -> None:
    archive = _zip_bytes(
        {FAKE_MEMBERS[video_id]: payload for video_id, payload in FAKE_PAYLOADS.items()}
    )
    with td.serve_paths({"bundle.zip": archive}) as base:
        _install_fake_source(archive, f"{base}/bundle.zip")
        paths = md.download_media(FAKE_DATASET, tmp_path, video_ids=["video_2"])
    assert paths == [tmp_path / FAKE_DATASET / "video_2.mp4"]
    assert not (tmp_path / FAKE_DATASET / "video_1.mp4").exists()


def test_download_rejects_unknown_ids_before_network(
    tmp_path: Path, fake_entries: dict[str, MediaEntry]
) -> None:
    with pytest.raises(ValueError, match="unknown video_id"):
        md.download_media(FAKE_DATASET, tmp_path, video_ids=["video_1", "video_99"])
    with pytest.raises(TypeError, match="not a single string"):
        md.download_media(FAKE_DATASET, tmp_path, video_ids="video_1")
    with pytest.raises(ValueError, match="must not be empty"):
        md.download_media(FAKE_DATASET, tmp_path, video_ids=[])


def test_download_nested_tar_gz_archive(
    tmp_path: Path, fake_entries: dict[str, MediaEntry]
) -> None:
    inner = _zip_bytes(
        {FAKE_MEMBERS[video_id]: payload for video_id, payload in FAKE_PAYLOADS.items()}
    )
    outer = _tar_gz_bytes("bundle/videos.zip", inner)
    with td.serve_paths({"bundle.tgz": outer}) as base:
        _install_fake_source(
            outer,
            f"{base}/bundle.tgz",
            archive_format="tar.gz",
            inner="bundle/videos.zip",
        )
        paths = md.download_media(FAKE_DATASET, tmp_path)
    assert [path.read_bytes() for path in paths] == list(FAKE_PAYLOADS.values())


def test_download_refuses_archive_with_changed_layout(
    tmp_path: Path, fake_entries: dict[str, MediaEntry]
) -> None:
    archive = _zip_bytes({"unexpected/name.mp4": FAKE_PAYLOADS["video_1"]})
    with td.serve_paths({"bundle.zip": archive}) as base:
        _install_fake_source(archive, f"{base}/bundle.zip")
        with pytest.raises(md.MediaDownloadError, match="does not contain"):
            md.download_media(FAKE_DATASET, tmp_path)
    assert not (tmp_path / FAKE_DATASET / "video_1.mp4").exists(), "partial install must not happen"


def test_download_rejects_archive_checksum_mismatch(
    tmp_path: Path, fake_entries: dict[str, MediaEntry]
) -> None:
    archive = _zip_bytes(
        {FAKE_MEMBERS[video_id]: payload for video_id, payload in FAKE_PAYLOADS.items()}
    )
    with td.serve_paths({"bundle.zip": archive}) as base:
        _install_fake_source(archive, f"{base}/bundle.zip")
        source = MEDIA_SOURCES[FAKE_DATASET]
        MEDIA_SOURCES[FAKE_DATASET] = MediaSource(
            dataset=source.dataset,
            homepage=source.homepage,
            license=source.license,
            archive_urls=source.archive_urls,
            archive_filename=source.archive_filename,
            archive_size=source.archive_size,
            archive_sha256="0" * 64,
            archive_format=source.archive_format,
            inner_archive=source.inner_archive,
        )
        with pytest.raises(md.MediaDownloadError, match="checksum mismatch"):
            md.download_media(FAKE_DATASET, tmp_path)
    assert not (tmp_path / FAKE_DATASET / "video_1.mp4").exists()


def test_download_keeps_and_reuses_verified_archive(
    tmp_path: Path, fake_entries: dict[str, MediaEntry]
) -> None:
    archive = _zip_bytes(
        {FAKE_MEMBERS[video_id]: payload for video_id, payload in FAKE_PAYLOADS.items()}
    )
    with td.serve_paths({"bundle.zip": archive}) as base:
        _install_fake_source(archive, f"{base}/bundle.zip")
        paths = md.download_media(FAKE_DATASET, tmp_path, keep_archive=True)
    kept = tmp_path / ".archives" / "bundle.zip"
    assert kept.read_bytes() == archive
    paths[0].unlink()
    # No server: the kept archive must satisfy the request on its own.
    assert md.download_media(FAKE_DATASET, tmp_path, video_ids=["video_1"]) == [paths[0]]
    assert paths[0].read_bytes() == FAKE_PAYLOADS["video_1"]


# -- CLI ------------------------------------------------------------------------


def test_cli_download_media_and_media_path(
    tmp_path: Path, fake_entries: dict[str, MediaEntry], capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _zip_bytes(
        {FAKE_MEMBERS[video_id]: payload for video_id, payload in FAKE_PAYLOADS.items()}
    )
    with td.serve_paths({"bundle.zip": archive}) as base:
        _install_fake_source(archive, f"{base}/bundle.zip")
        code = cli.main(["download-media", FAKE_DATASET, "--media-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "media root:" in captured.out
    assert "docs/design/raw-media.md" in captured.out
    assert f"ready: {tmp_path / FAKE_DATASET / 'video_1.mp4'}" in captured.out

    code = cli.main(["media-path", FAKE_DATASET, "video_1", "--media-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert captured.out.strip() == str(tmp_path / FAKE_DATASET / "video_1.mp4")

    code = cli.main(
        [
            "media-path",
            FAKE_DATASET,
            "video_1",
            "--media-root",
            str(tmp_path),
            "--verify-sha256",
        ]
    )
    assert code == 0, capsys.readouterr().err


def test_cli_media_path_missing_is_an_error(
    tmp_path: Path, fake_entries: dict[str, MediaEntry], capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["media-path", FAKE_DATASET, "video_1", "--media-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert "is not available" in captured.err
    assert "download-media" in captured.err


def test_cli_media_unknown_id_is_usage_error(
    tmp_path: Path, fake_entries: dict[str, MediaEntry], capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["media-path", FAKE_DATASET, "video_99", "--media-root", str(tmp_path)])
    assert code == 2
    assert "unknown video_id" in capsys.readouterr().err
