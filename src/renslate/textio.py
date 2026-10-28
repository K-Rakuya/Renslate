"""Backward compatibility shim forwarding to renslate.core.textio."""

from __future__ import annotations

import sys

from renslate.core import textio as _target
from renslate.core.textio import *  # noqa: F403
from renslate.core.textio import (
    TextFile,
    normalize_nfc,
    read_text_file,
    write_atomic,
)

__all__ = [
    "TextFile",
    "normalize_nfc",
    "read_text_file",
    "write_atomic",
]

sys.modules[__name__] = _target
