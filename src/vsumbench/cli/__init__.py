"""``vsumm`` command line interface.

Subcommands grow with the milestones: ``download`` (M1, data layer) and
``eval`` / ``report`` (M4, orchestrator).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vsumbench import __version__
from vsumbench.data import download as download_mod
from vsumbench.data.sources import SOURCES

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
    return parser


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "download":
        return _run_download(args)
    parser.print_help(sys.stderr)
    return 2
