"""Aggregation of run manifests into comparison tables.

``vsumm report`` reads manifests (files, or directories scanned for
``manifest.json``) and prints one row per run: identifying config fields
plus a ``mean ± std`` cell for every metric@budget present in any of them.
Comparison numbers for *other people's* methods do not belong here — those
come from their papers; this table only summarizes runs computed under
vsumbench protocols (dev-plan M3 scope).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from vsumbench.reporting.manifest import NO_BUDGET, load_manifest

__all__ = ["collect_manifests", "runs_table"]


def collect_manifests(paths: Iterable[str | Path]) -> list[tuple[str, dict[str, Any]]]:
    """Resolve user-supplied paths to ``(run dir, manifest)`` pairs.

    A path that is a file is loaded directly; a directory is scanned
    recursively for ``manifest.json``. Missing paths raise FileNotFoundError.
    """
    collected: list[tuple[str, dict[str, Any]]] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"no such path: {path}")
        found = [path] if path.is_file() else sorted(path.rglob("manifest.json"))
        for manifest_path in found:
            label = str(manifest_path.parent)
            collected.append((label, load_manifest(manifest_path)))
    return collected


def runs_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    """Render ``(run dir, manifest)`` pairs as a GitHub-flavored markdown table."""
    columns = _metric_columns(manifest for _, manifest in rows)
    header = [
        "run",
        "dataset",
        "model",
        "selector",
        "gt_agg",
        "seed",
        *(_column_label(metric, key) for metric, key in columns),
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, manifest in rows:
        config = manifest["config"]
        assert isinstance(config, dict)
        cells = [
            label,
            str(config["dataset"]),
            str(config["model"]),
            str(config["selector"]),
            str(config["gt_aggregation"]),
            str(config["seed"]),
        ]
        summary = _summary_of(manifest)
        for metric, key in columns:
            cells.append(_cell(summary.get(metric, {}).get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _column_label(metric: str, key: str) -> str:
    return metric if key == NO_BUDGET else f"{metric}@{key}"


def _summary_of(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = manifest.get("results")
    if not isinstance(results, dict):
        raise ValueError("manifest carries no results block")
    summary = results.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("manifest carries no results summary")
    return summary


def _metric_columns(manifests: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    """Union of (metric, budget key) pairs across manifests, in stable order."""
    pairs: set[tuple[str, str]] = set()
    for manifest in manifests:
        for metric, budget_map in _summary_of(manifest).items():
            for key in budget_map:
                pairs.add((metric, key))
    return sorted(pairs)


def _cell(stats: dict[str, Any] | None) -> str:
    if stats is None:
        return "n/a"
    mean = stats.get("mean")
    if mean is None:
        return "n/a"
    std = stats.get("std")
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"
