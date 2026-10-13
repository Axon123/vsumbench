"""Tests for run manifest construction, writing and loading."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from vsumbench.config import CanonicalSplits, EvalConfig
from vsumbench.reporting.manifest import (
    MANIFEST_SCHEMA,
    build_manifest,
    code_info,
    load_manifest,
    write_manifest,
)


def _config() -> EvalConfig:
    return EvalConfig(
        dataset="synthetic",
        model="uniform",
        selector="knapsack",
        metrics=("f1",),
        budgets=(0.15,),
        splits=CanonicalSplits(name="tvsum-canonical-5"),
    )


def _manifest(results: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(
        build_manifest(
            _config(),
            results if results is not None else {"summary": {"f1": {"0.15": {"mean": 0.5}}}},
            dataset_root=Path("/tmp/data"),
            files=None,
            splits=[{"fold": 0, "train": ["video_1"], "test": ["video_2"]}],
        )
    )


def test_manifest_records_config_code_and_splits() -> None:
    manifest = _manifest()
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["config"] == _config().to_dict()
    assert manifest["splits"][0]["fold"] == 0
    code = manifest["code"]
    assert isinstance(code, dict)
    assert set(code) == {"vsumbench", "git_sha", "git_dirty", "python", "platform", "numpy"}
    data = manifest["data"]
    assert isinstance(data, dict)
    assert data["dataset"] == "synthetic"
    assert data["files"] is None
    assert "unavailable" in str(data["files_note"])


def test_manifest_serializes_nonfinite_floats_as_null(tmp_path: Path) -> None:
    manifest = _manifest({"summary": {"kendall_tau": {"none": {"mean": math.nan}}}})
    path = write_manifest(manifest, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    loaded = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    summary = loaded["results"]["summary"]["kendall_tau"]["none"]
    assert summary["mean"] is None


def test_write_manifest_creates_fresh_run_dirs(tmp_path: Path) -> None:
    first = write_manifest(_manifest(), tmp_path)
    second = write_manifest(_manifest(), tmp_path)
    assert first.parent != second.parent
    assert (first.parent / "manifest.json") == first
    assert first.parent.name.split("-", 1)[-1] == "synthetic-uniform-knapsack"
    assert load_manifest(first) == _manifest()


def test_write_manifest_sanitizes_unsafe_names(tmp_path: Path) -> None:
    manifest = _manifest()
    config = manifest["config"]
    assert isinstance(config, dict)
    config["model"] = "my model/../x"
    path = write_manifest(manifest, tmp_path)
    assert "/" not in path.parent.name
    assert path.parent.exists()


def test_load_manifest_rejects_foreign_json(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a vsumbench run manifest"):
        load_manifest(other)


def test_code_info_git_fields_from_this_repo() -> None:
    info = code_info()
    sha = info["git_sha"]
    if sha is not None:
        assert isinstance(sha, str)
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    dirty = info["git_dirty"]
    assert dirty is None or isinstance(dirty, bool)
