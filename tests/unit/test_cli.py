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
