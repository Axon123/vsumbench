"""Run manifests: the traceability record of one evaluation run.

A manifest answers the three questions behind every number (AGENTS.md
quality gate): which code produced it (package version + git SHA + dirty
flag), under which protocol (the complete :class:`~vsumbench.config.EvalConfig`,
verbatim), on which data (per-file sha256 checked against
:mod:`vsumbench.data.sources`). Everything is plain JSON — manifests are
meant to be diffed, grepped and aggregated by ``vsumm report``.

Non-finite floats (NaN / inf, e.g. a Kendall tau that is undefined for a
degenerate annotator) are written as ``null``: ``json.dumps`` would emit
``NaN``, which is not valid JSON and breaks strict parsers.
"""

from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from vsumbench.config import JsonValue

if TYPE_CHECKING:
    from vsumbench.config import EvalConfig

__all__ = [
    "MANIFEST_SCHEMA",
    "NO_BUDGET",
    "ManifestFile",
    "build_manifest",
    "load_manifest",
    "write_manifest",
]

#: Bump on any breaking change to the manifest shape (semver for manifests).
MANIFEST_SCHEMA = "vsumbench.run-manifest/1"

#: Results-block key under which budget-independent (rank) metric values
#: live; summary metrics key theirs by the budget fraction (``"0.15"``).
NO_BUDGET = "none"

_MANIFEST_FILENAME = "manifest.json"
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def code_info() -> dict[str, JsonValue]:
    """Identify the running code: version, git SHA/dirty state, interpreter."""
    # Imported lazily: vsumbench/__init__ pulls in bench -> this module, so a
    # module-level import would be circular.
    import vsumbench

    sha, dirty = _git_state()
    return {
        "vsumbench": vsumbench.__version__,
        "git_sha": sha,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _git_state() -> tuple[str | None, bool | None]:
    """SHA and dirty flag of the repository containing this package, if any.

    Returns ``(None, None)`` for non-repo installs (e.g. a wheel in
    site-packages): there is nothing to lie about, the field is absent.
    """
    # ``git -C`` needs a directory it can chdir into; the package's parent
    # is inside the working tree for editable/dev installs.
    root = Path(__file__).resolve().parent
    try:
        sha_proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if sha_proc.returncode != 0:
            return None, None
        status_proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    dirty = None
    if status_proc.returncode == 0:
        dirty = bool(status_proc.stdout.strip())
    return sha_proc.stdout.strip() or None, dirty


@dataclass(frozen=True)
class ManifestFile:
    """One data file the run consumed, with its verified checksum."""

    path: str
    sha256: str
    expected_sha256: str | None
    matches_spec: bool | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "expected_sha256": self.expected_sha256,
            "matches_spec": self.matches_spec,
        }


def build_manifest(
    config: EvalConfig,
    results: dict[str, Any],
    *,
    dataset_root: Path,
    files: tuple[ManifestFile, ...] | None,
    splits: Sequence[Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    """Assemble the manifest of a finished run.

    ``results`` is the results block produced by :mod:`vsumbench.bench`;
    ``files`` is ``None`` for datasets unknown to :mod:`vsumbench.data.sources`
    (recorded as such rather than pretending the data was verified).
    """
    return {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": code_info(),
        "data": {
            "dataset": config.dataset,
            "root": str(dataset_root),
            "files": [f.to_dict() for f in files] if files is not None else None,
            "files_note": (
                "dataset not listed in vsumbench.data.sources; checksums unavailable for this run"
                if files is None
                else None
            ),
        },
        "config": config.to_dict(),
        "splits": [dict(s) for s in splits],
        "results": _jsonify(results),
    }


def _jsonify(value: Any) -> JsonValue:
    """Recursively map non-finite floats to None; everything else passes."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return cast(JsonValue, value)


def write_manifest(manifest: dict[str, JsonValue], base_dir: str | Path) -> Path:
    """Write the manifest into a fresh timestamped run dir; return its path.

    Run dirs are ``<UTC stamp>-<dataset>-<model>-<selector>`` under
    ``base_dir``; same-second reruns get a numeric suffix instead of
    overwriting, so consecutive runs stay diffable.
    """
    config = manifest["config"]
    assert isinstance(config, dict)
    label = _UNSAFE_CHARS.sub(
        "_", f"{config['dataset']}-{config['model']}-{config['selector']}"
    ).strip("._-")[:120]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base_dir) / f"{stamp}-{label}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(base_dir) / f"{stamp}-{label}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    path = run_dir / _MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> dict[str, JsonValue]:
    """Read back a manifest file (the inverse of :func:`write_manifest`)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"{path} is not a vsumbench run manifest (expected schema {MANIFEST_SCHEMA!r})"
        )
    return data
