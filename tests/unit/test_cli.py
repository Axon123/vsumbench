"""CLI smoke tests: the ``vsumm`` console script identifies itself."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from vsumbench import __version__, cli


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert f"vsumbench {__version__}" in capsys.readouterr().out


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 2
    assert "usage: vsumm" in capsys.readouterr().err


def test_console_script_smoke() -> None:
    # Resolve next to the running interpreter: the script lives in the same
    # bin/ directory when tests run inside the installed environment.
    vsumm = Path(sys.executable).with_name("vsumm")
    assert vsumm.exists(), f"{vsumm} missing — install with `pip install -e '.[dev]'`"
    proc = subprocess.run([str(vsumm), "--version"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert __version__ in proc.stdout


def test_download_missing_dataset_arg_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["download"])
    assert excinfo.value.code == 2


def test_download_unknown_dataset_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["download", "ovp"])
    assert excinfo.value.code == 2


def test_download_reports_corrupt_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "eccv16_dataset_tvsum_google_pool5.h5").write_bytes(b"not the real thing")
    assert cli.main(["download", "tvsum", "--root", str(tmp_path)]) == 1
    assert "mismatch" in capsys.readouterr().err


def test_download_fetches_from_mirror(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import hashlib

    import test_download as td

    from vsumbench.data.sources import SOURCES, DatasetSpec, FileSpec

    with td.serve_paths({"m.tar": td._tar_bytes(td.FAKE_FILENAME, td.FAKE_BYTES)}) as base:
        SOURCES[td.FAKE_NAME] = DatasetSpec(
            name=td.FAKE_NAME,
            files=(
                FileSpec(
                    filename=td.FAKE_FILENAME,
                    sha256=hashlib.sha256(td.FAKE_BYTES).hexdigest(),
                    size=len(td.FAKE_BYTES),
                ),
            ),
            archive_urls=(f"{base}/m.tar",),
        )
        try:
            assert cli.main(["download", td.FAKE_NAME, "--root", str(tmp_path)]) == 0
        finally:
            del SOURCES[td.FAKE_NAME]
    assert (tmp_path / td.FAKE_FILENAME).read_bytes() == td.FAKE_BYTES
    out = capsys.readouterr().out
    assert "cache root" in out
    assert f"ready: {tmp_path / td.FAKE_FILENAME}" in out
