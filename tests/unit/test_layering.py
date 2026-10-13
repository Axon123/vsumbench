"""Guard the one-way dependency rule (AGENTS.md rule 2).

Each check imports one module in a fresh interpreter and asserts exactly
which vsumbench modules got pulled in, so an accidental upward import fails
loudly instead of rotting the architecture.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

ALLOWED_IMPORTS: dict[str, set[str]] = {
    "vsumbench": {"vsumbench"},
    "vsumbench.config": {"vsumbench", "vsumbench.config"},
    "vsumbench.types": {"vsumbench", "vsumbench.config", "vsumbench.types"},
    "vsumbench.registry": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.registry",
    },
    "vsumbench.data": {"vsumbench", "vsumbench.data"},
    "vsumbench.data._canonical_splits": {
        "vsumbench",
        "vsumbench.data",
        "vsumbench.data._canonical_splits",
    },
    "vsumbench.data.sources": {"vsumbench", "vsumbench.data", "vsumbench.data.sources"},
    "vsumbench.data.download": {
        "vsumbench",
        "vsumbench.data",
        "vsumbench.data.sources",
        "vsumbench.data.download",
    },
    "vsumbench.data.splits": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.data",
        "vsumbench.data._canonical_splits",
        "vsumbench.data.splits",
    },
    "vsumbench.data._eccv16": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.data",
        "vsumbench.data._eccv16",
    },
    "vsumbench.data.tvsum": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.data",
        "vsumbench.data._eccv16",
        "vsumbench.data.sources",
        "vsumbench.data.tvsum",
    },
    "vsumbench.data.summe": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.data",
        "vsumbench.data._eccv16",
        "vsumbench.data.sources",
        "vsumbench.data.summe",
    },
    "vsumbench.models": {"vsumbench", "vsumbench.models"},
    "vsumbench.models.score_space": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.models",
        "vsumbench.models.score_space",
    },
    "vsumbench.selection": {"vsumbench", "vsumbench.selection"},
    "vsumbench.selection._common": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.selection",
        "vsumbench.selection._common",
    },
    "vsumbench.selection.shot_based": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.selection",
        "vsumbench.selection._common",
        "vsumbench.selection.shot_based",
    },
    "vsumbench.selection.knapsack": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.selection",
        "vsumbench.selection._common",
        "vsumbench.selection.knapsack",
    },
    "vsumbench.metrics": {"vsumbench", "vsumbench.metrics"},
    "vsumbench.metrics.f1": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.metrics",
        "vsumbench.metrics.f1",
    },
    "vsumbench.metrics.rank": {
        "vsumbench",
        "vsumbench.config",
        "vsumbench.types",
        "vsumbench.metrics",
        "vsumbench.metrics.rank",
    },
    "vsumbench.reporting": {"vsumbench", "vsumbench.reporting"},
    "vsumbench.bench": {"vsumbench", "vsumbench.bench"},
    "vsumbench.cli": {
        "vsumbench",
        "vsumbench.data",
        "vsumbench.data.sources",
        "vsumbench.data.download",
        "vsumbench.cli",
    },
}


@pytest.mark.parametrize(("module", "allowed"), sorted(ALLOWED_IMPORTS.items()))
def test_module_imports_only_allowed_layers(module: str, allowed: set[str]) -> None:
    code = (
        "import json, sys\n"
        f"import {module}\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('vsumbench'))))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"importing {module} failed:\n{proc.stderr}"
    loaded = set(json.loads(proc.stdout))
    assert loaded <= allowed, f"{module} pulled in unexpected modules: {sorted(loaded - allowed)}"
