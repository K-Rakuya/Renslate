"""Backward compatibility shim forwarding to renslate.renpy.extract."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import renslate.renpy.extract as _target

if TYPE_CHECKING:
    from renslate.renpy.extract import (
        FLAG_MARKUP_INVALID,
        FLAG_SOURCE_CHANGED,
        ExtractReport,
        extract,
        extract_project,
        scan_project,
    )
else:
    from renslate.renpy.extract import *  # noqa: F403

__all__ = [
    "FLAG_MARKUP_INVALID",
    "FLAG_SOURCE_CHANGED",
    "ExtractReport",
    "extract",
    "extract_project",
    "scan_project",
]

sys.modules[__name__] = _target
