"""Backward compatibility shim forwarding to renslate.renpy.writer."""

from __future__ import annotations

import sys

from renslate.renpy import writer as _target
from renslate.renpy.writer import *  # noqa: F403
from renslate.renpy.writer import (
    ApplyReport,
    apply_translations,
    setup_vietnamese_support,
)

__all__ = [
    "ApplyReport",
    "apply_translations",
    "setup_vietnamese_support",
]

sys.modules[__name__] = _target
