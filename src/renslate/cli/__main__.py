"""Executable entrypoint for python -m renslate.cli."""

from __future__ import annotations

import sys

from renslate.cli import main

if __name__ == "__main__":
    sys.exit(main())
