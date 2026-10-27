"""Backward compatibility shim forwarding to renslate.engine.tsvio."""

from __future__ import annotations

import sys

from renslate.engine import tsvio as _target
from renslate.engine.tsvio import *  # noqa: F403
from renslate.engine.tsvio import (
    export_tsv,
    import_tsv,
)

__all__ = [
    "export_tsv",
    "import_tsv",
]

sys.modules[__name__] = _target
