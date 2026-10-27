#!/usr/bin/env python3
"""Unattended automated runner for Renslate translation pipelines.

Ergonomic thin delegation wrapper forwarding to renslate.cli.main(["run", ...]).
Preserves 100% backward compatibility for legacy scripts and test harnesses.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path when executed directly
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import renslate.ui.runner as _runner_mod  # noqa: E402
from renslate.cli import main as cli_main  # noqa: E402
from renslate.ui.progress import format_duration, render_progress_bar  # noqa: E402
from renslate.ui.runner import (  # noqa: E402
    bootstrap_if_needed,
    check_server_health,
    detect_server_model,
    query_db_status,
    run_unattended_pipeline,
    wait_for_server,
)
from renslate.ui.theme import (  # noqa: E402
    C_BOLD,
    C_CYAN,
    C_GRAY,
    C_GREEN,
    C_RED,
    C_RESET,
    C_YELLOW,
    IS_TTY,
)

# Re-export module state
_STOP_REQUESTED: bool = False
_INTERRUPT_COUNT: int = 0


def _sig_handler(signum: int, frame: Any) -> None:
    global _STOP_REQUESTED, _INTERRUPT_COUNT
    _runner_mod._STOP_REQUESTED = _STOP_REQUESTED
    _runner_mod._INTERRUPT_COUNT = _INTERRUPT_COUNT
    _runner_mod._sig_handler(signum, frame)
    _STOP_REQUESTED = _runner_mod._STOP_REQUESTED
    _INTERRUPT_COUNT = _runner_mod._INTERRUPT_COUNT


__all__ = [
    "C_BOLD",
    "C_CYAN",
    "C_GRAY",
    "C_GREEN",
    "C_RED",
    "C_RESET",
    "C_YELLOW",
    "IS_TTY",
    "_INTERRUPT_COUNT",
    "_STOP_REQUESTED",
    "_sig_handler",
    "bootstrap_if_needed",
    "check_server_health",
    "detect_server_model",
    "format_duration",
    "main",
    "query_db_status",
    "render_progress_bar",
    "run_unattended_pipeline",
    "wait_for_server",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate CLI execution to native renslate run subcommand."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "translate":
        raw_args = raw_args[1:]
    return cli_main(["run", *raw_args])


if __name__ == "__main__":
    sys.exit(main())
