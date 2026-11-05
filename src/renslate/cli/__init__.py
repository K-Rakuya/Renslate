"""Command-line interface primary adapter (Hexagonal Architecture)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from renslate import __version__
from renslate.cli.parser import build_parser
from renslate.core.errors import RenslateError
from renslate.engine.translate import TranslationPipeline

_build_parser = build_parser

__all__ = ["TranslationPipeline", "__version__", "_build_parser", "build_parser", "main"]


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint."""
    _configure_streams()
    raw_args = list(sys.argv[1:] if argv is None else argv)

    if not raw_args and sys.stdin.isatty() and sys.stdout.isatty():
        from renslate.ui.wizard import run_interactive_wizard

        return run_interactive_wizard(Path.cwd(), interactive_menu=True)

    parser = build_parser()
    args = parser.parse_args(raw_args)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except RenslateError as error:
        print(f"lỗi: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
