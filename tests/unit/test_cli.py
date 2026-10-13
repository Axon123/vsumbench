"""CLI smoke tests: download / eval / report through ``cli.main`` in-process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from _synthetic import SyntheticSpec, write_eccv16_h5

from vsumbench import __version__, cli
from vsumbench.config import EvalConfig, RandomSplits
from vsumbench.data._eccv16 import Eccv16Adapter
from vsumbench.registry import DATASETS

_CLI_DATASET = "clisynth"


@pytest.fixture()
def synth_root(tmp_path: Path) -> Path:
    specs = tuple(
        SyntheticSpec(
            video_id=f"video_{i}", n_frames=48, step=4, dim=4, n_users=2, shot_len=12, seed=i
        )
        for i in range(1, 7)
    )
    write_eccv16_h5(tmp_path / "synth.h5", specs)
    if _CLI_DATASET not in DATASETS.names():
        DATASETS.register(_CLI_DATASET, Eccv16Adapter(_CLI_DATASET, "synth.h5"))
    return tmp_path


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


# -- eval ---------------------------------------------------------------------


def test_eval_runs_and_prints_summary(
    synth_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "runs"
    code = cli.main(
        [
            "eval",
            "--dataset",
            _CLI_DATASET,
            "--splits",
            "random",
            "--n-folds",
            "2",
            "--budgets",
            "0.5",
            "--data-root",
            str(synth_root),
            "--out",
            str(out),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "fold 1/2" in captured.out
    assert "manifest" in captured.out
    assert "f1@0.5" in captured.out
    assert "uniform" in captured.out
    manifests = list(out.rglob("manifest.json"))
    assert len(manifests) == 1


def test_eval_unknown_dataset_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["eval", "--dataset", "nope", "--out", str(tmp_path)])
    assert code == 1
    assert "error" in capsys.readouterr().err


def test_eval_requires_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["eval", "--out", str(tmp_path)])
    assert code == 2
    assert "--dataset" in capsys.readouterr().err


def test_eval_unknown_canonical_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["eval", "--dataset", _CLI_DATASET, "--splits", "bogus", "--out", str(tmp_path)]
    )
    assert code == 1
    assert "unknown canonical split" in capsys.readouterr().err


def test_eval_n_folds_rejected_for_canonical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            "eval",
            "--dataset",
            _CLI_DATASET,
            "--splits",
            "tvsum-canonical-5",
            "--n-folds",
            "3",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 1
    assert "--splits random" in capsys.readouterr().err


def test_eval_with_config_file(
    synth_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    EvalConfig(
        dataset=_CLI_DATASET,
        model="uniform",
        selector="greedy",
        metrics=("f1",),
        budgets=(0.5,),
        splits=RandomSplits(n_folds=2),
    ).save(config_path)
    code = cli.main(
        [
            "eval",
            "--config",
            str(config_path),
            "--data-root",
            str(synth_root),
            "--out",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "greedy" in captured.out


def test_eval_config_conflicts_with_protocol_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    code = cli.main(["eval", "--config", str(config_path), "--seed", "1", "--out", str(tmp_path)])
    assert code == 2
    assert "--config cannot be combined" in capsys.readouterr().err


# -- report ---------------------------------------------------------------------


def test_report_aggregates_run_dirs(
    synth_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "runs"
    base = [
        "eval",
        "--dataset",
        _CLI_DATASET,
        "--splits",
        "random",
        "--n-folds",
        "2",
        "--budgets",
        "0.5",
        "--data-root",
        str(synth_root),
        "--out",
        str(out),
    ]
    assert cli.main(base) == 0
    assert cli.main([*base, "--gt-aggregation", "mean"]) == 0
    capsys.readouterr()
    code = cli.main(["report", str(out)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "| run | dataset | model | selector | gt_agg | seed | f1@0.5 |" in captured.out
    assert captured.out.count("| uniform | knapsack |") == 2


def test_report_empty_dir_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["report", str(tmp_path)])
    assert code == 1
    assert "no run manifests" in capsys.readouterr().err
