"""``vsumm`` command line interface.

Subcommands arrive with the data layer (``download`` in M1) and the
orchestrator (``eval`` / ``report`` in M4); for now the CLI only identifies
itself so the console script is stable from day one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from vsumbench import __version__

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsumm",
        description="vsumbench: unified, reproducible evaluation for video summarization",
    )
    parser.add_argument("--version", action="version", version=f"vsumbench {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    parser.parse_args(argv)  # --version exits by itself
    parser.print_help(sys.stderr)
    return 2
