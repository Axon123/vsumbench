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
    "vsumbench.models": {"vsumbench", "vsumbench.models"},
    "vsumbench.selection": {"vsumbench", "vsumbench.selection"},
    "vsumbench.metrics": {"vsumbench", "vsumbench.metrics"},
    "vsumbench.reporting": {"vsumbench", "vsumbench.reporting"},
    "vsumbench.bench": {"vsumbench", "vsumbench.bench"},
    "vsumbench.cli": {"vsumbench", "vsumbench.cli"},
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
