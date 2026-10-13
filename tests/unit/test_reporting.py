"""Tests for manifest aggregation into comparison tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vsumbench.reporting.manifest import write_manifest
from vsumbench.reporting.table import collect_manifests, runs_table


def _manifest(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "dataset": "tvsum",
        "model": "uniform",
        "selector": "knapsack",
        "metrics": ["f1"],
        "budgets": [0.15],
        "splits": {"kind": "random", "n_folds": 5, "seed": 0},
        "seed": 0,
        "gt_aggregation": "max",
        "segment_score_aggregation": "mean",
        "budget_rounding": "floor",
        "step_score_expansion": "forward_fill",
        "score_normalization": {"method": "none"},
        "training": {
            "epochs": 100,
            "learning_rate": 0.0001,
            "batch_size": 1,
            "device": "cpu",
            "extra": {},
        },
    }
    config.update(overrides)
    return {
        "schema": "vsumbench.run-manifest/1",
        "config": config,
        "results": {
            "summary": {
                "f1": {"0.15": {"mean": 0.5312, "std": 0.0123, "max": 0.55, "n_folds": 5}},
                "kendall_tau": {"none": {"mean": 0.3, "std": None, "max": 0.3, "n_folds": 1}},
            }
        },
    }


def test_runs_table_rendes_metric_columns() -> None:
    table = runs_table([("runs/a", _manifest())])
    lines = table.splitlines()
    assert lines[0].split("|") == [
        "",
        " run ",
        " dataset ",
        " model ",
        " selector ",
        " gt_agg ",
        " seed ",
        " f1@0.15 ",
        " kendall_tau ",
        "",
    ]
    assert lines[1] == "|" + "---|" * 8
    assert (
        "| runs/a | tvsum | uniform | knapsack | max | 0 | 0.5312 ± 0.0123 | 0.3000 |" in (lines[2])
    )


def test_runs_table_marks_missing_metrics() -> None:
    other = _manifest(model="mine")
    other["results"]["summary"] = {
        "f1": {"0.15": {"mean": 0.42, "std": None, "max": 0.42, "n_folds": 1}}
    }
    table = runs_table([("runs/a", _manifest()), ("runs/b", other)])
    lines = table.splitlines()
    assert len(lines) == 4  # header, separator, two rows
    row_b = lines[3]
    assert "mine" in row_b
    cells = row_b.split("|")
    assert cells[-3].strip() == "0.4200"  # f1@0.15, no std for a single fold
    assert cells[-2].strip() == "n/a"  # kendall_tau absent from this run


def test_collect_manifests_from_files_and_dirs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    first = write_manifest(_manifest(), runs)
    second = write_manifest(_manifest(), runs)
    labels = [label for label, _ in collect_manifests([first, runs])]
    # the file once, plus both manifests found by scanning the directory
    assert labels.count(str(first.parent)) == 2
    assert str(second.parent) in labels


def test_collect_manifests_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_manifests([tmp_path / "nope"])


def test_collect_manifests_rejects_foreign_json(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.json"
    foreign.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not a vsumbench run manifest"):
        collect_manifests([foreign])
