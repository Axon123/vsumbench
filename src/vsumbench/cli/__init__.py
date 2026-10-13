"""``vsumm`` command line interface.

Subcommands: ``download`` (M1, data layer), ``eval`` and ``report``
(M3, orchestrator). ``eval`` builds a Bench from the flags (or reads a full
EvalConfig JSON with ``--config``), runs it and prints the summary; every
run lands a manifest under ``--out`` that ``report`` can aggregate later.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vsumbench import __version__
from vsumbench.bench import Bench
from vsumbench.config import CanonicalSplits, EvalConfig, RandomSplits
from vsumbench.data import download as download_mod
from vsumbench.data.sources import SOURCES
from vsumbench.data.splits import CANONICAL_SPLIT_NAMES
from vsumbench.reporting.table import collect_manifests, runs_table

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsumm",
        description="vsumbench: unified, reproducible evaluation for video summarization",
    )
    parser.add_argument("--version", action="version", version=f"vsumbench {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    download = subparsers.add_parser(
        "download",
        help="fetch and verify a dataset into the cache (or verify if present)",
        description=(
            "Ensure the canonical files of DATASET are present under the cache root "
            "and match their sha256. Files already in place are verified, never "
            "re-downloaded — put a manually obtained file there for offline use."
        ),
    )
    download.add_argument("dataset", choices=sorted(SOURCES), help="dataset registry name")
    download.add_argument(
        "--root",
        type=str,
        default=None,
        metavar="DIR",
        help="cache root (default: $VSUM_CACHE_DIR or ~/.cache/vsumbench)",
    )

    evaluate = subparsers.add_parser(
        "eval",
        help="run one evaluation under an explicit protocol",
        description=(
            "Evaluate MODEL on DATASET: per-fold fit/predict, selection under each "
            "budget, metrics collapsed across annotations, and a run manifest "
            "recording git SHA + full config + data checksums."
        ),
    )
    evaluate.add_argument(
        "--dataset", metavar="NAME", help="dataset registry name (required unless --config)"
    )
    evaluate.add_argument("--model", metavar="NAME", help="models registry name (default: uniform)")
    evaluate.add_argument(
        "--selector", metavar="NAME", help="selectors registry name (default: knapsack)"
    )
    evaluate.add_argument(
        "--metrics",
        type=_comma_list,
        default=None,
        metavar="A,B",
        help="comma-separated metric registry names (default: f1)",
    )
    evaluate.add_argument(
        "--budgets",
        type=_comma_floats,
        default=None,
        metavar="F1,F2",
        help="comma-separated budget fractions (default: 0.1,0.15,0.3,0.5)",
    )
    evaluate.add_argument(
        "--splits",
        default=None,
        metavar="NAME|random",
        help=(
            "canonical split name (one of "
            f"{', '.join(sorted(CANONICAL_SPLIT_NAMES))}) or 'random' "
            "(default: random 5-fold, seed 0)"
        ),
    )
    evaluate.add_argument(
        "--n-folds", type=int, default=None, help="fold count for --splits random"
    )
    evaluate.add_argument(
        "--split-seed", type=int, default=None, help="split seed for --splits random"
    )
    evaluate.add_argument("--seed", type=int, default=None, help="global evaluation seed")
    evaluate.add_argument(
        "--gt-aggregation",
        choices=("max", "mean"),
        default=None,
        help="cross-annotation aggregation (default: max)",
    )
    evaluate.add_argument(
        "--data-root",
        type=str,
        default=None,
        metavar="DIR",
        help="dataset cache root (default: $VSUM_CACHE_DIR or ~/.cache/vsumbench)",
    )
    evaluate.add_argument(
        "--out",
        type=str,
        default="runs",
        metavar="DIR",
        help="directory for run manifests (default: ./runs)",
    )
    evaluate.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "load the full EvalConfig from a JSON file "
            "(mutually exclusive with the protocol flags above)"
        ),
    )

    report = subparsers.add_parser(
        "report",
        help="aggregate run manifests into a comparison table",
        description=(
            "Collect manifests (files, or directories scanned for manifest.json) "
            "and print one markdown row per run. Numbers for methods you did not "
            "run belong in your table from their papers, not here."
        ),
    )
    report.add_argument(
        "paths",
        nargs="*",
        help="manifest files or directories (default: runs)",
    )
    return parser


def _comma_list(text: str) -> list[str]:
    items = [item.strip() for item in text.split(",")]
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError(f"expected a comma-separated list, got {text!r}")
    return items


def _comma_floats(text: str) -> list[float]:
    try:
        return [float(item) for item in _comma_list(text)]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {text!r}"
        ) from None


def _run_download(args: argparse.Namespace) -> int:
    root = None if args.root is None else Path(args.root)
    print(f"cache root: {download_mod.cache_dir(root)}")
    try:
        paths = download_mod.download_dataset(args.dataset, root=root, report=print)
    except download_mod.DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in paths:
        print(f"ready: {path}")
    return 0


def _splits_spec(args: argparse.Namespace) -> CanonicalSplits | RandomSplits | None:
    if args.splits is None:
        if args.n_folds is not None or args.split_seed is not None:
            raise ValueError("--n-folds/--split-seed only apply with --splits random")
        return None
    if args.splits == "random":
        kwargs: dict[str, int] = {}
        if args.n_folds is not None:
            kwargs["n_folds"] = args.n_folds
        if args.split_seed is not None:
            kwargs["seed"] = args.split_seed
        return RandomSplits(**kwargs)
    if args.splits not in CANONICAL_SPLIT_NAMES:
        known = ", ".join(sorted(CANONICAL_SPLIT_NAMES))
        raise ValueError(f"unknown canonical split {args.splits!r}; available: {known}, 'random'")
    if args.n_folds is not None or args.split_seed is not None:
        raise ValueError("--n-folds/--split-seed only apply with --splits random")
    return CanonicalSplits(name=args.splits)


def _run_eval(args: argparse.Namespace) -> int:
    data_root = None if args.data_root is None else Path(args.data_root)
    try:
        if args.config is not None:
            protocol_flags = [
                flag
                for flag, value in (
                    ("--dataset", args.dataset),
                    ("--model", args.model),
                    ("--selector", args.selector),
                    ("--metrics", args.metrics),
                    ("--budgets", args.budgets),
                    ("--splits", args.splits),
                    ("--n-folds", args.n_folds),
                    ("--split-seed", args.split_seed),
                    ("--seed", args.seed),
                    ("--gt-aggregation", args.gt_aggregation),
                )
                if value is not None
            ]
            if protocol_flags:
                print(
                    f"error: --config cannot be combined with {protocol_flags}",
                    file=sys.stderr,
                )
                return 2
            bench = Bench(EvalConfig.load(args.config), data_root=data_root)
        else:
            if args.dataset is None:
                print(
                    "error: --dataset is required (or pass --config FILE)",
                    file=sys.stderr,
                )
                return 2
            bench = Bench(
                dataset=args.dataset,
                model=args.model,
                selector=args.selector,
                metrics=args.metrics,
                budgets=args.budgets,
                splits=_splits_spec(args),
                seed=args.seed,
                gt_aggregation=args.gt_aggregation,
                data_root=data_root,
            )
        result = bench.run(output_dir=args.out, report=print)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print()
    print(runs_table([(str(result.manifest_path.parent), result.manifest)]))
    return 0


def _run_report(args: argparse.Namespace) -> int:
    try:
        rows = collect_manifests(args.paths or ["runs"])
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print("error: no run manifests found (pass files or directories)", file=sys.stderr)
        return 1
    print(runs_table(rows))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        return _run_download(args)
    if args.command == "eval":
        return _run_eval(args)
    if args.command == "report":
        return _run_report(args)
    parser.print_help(sys.stderr)
    return 2
