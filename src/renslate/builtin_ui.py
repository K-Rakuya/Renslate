"""Backward compatibility shim forwarding to renslate.renpy.builtin_ui."""

from __future__ import annotations

import sys

from renslate.renpy import builtin_ui as _target
from renslate.renpy.builtin_ui import *  # noqa: F403
from renslate.renpy.builtin_ui import (
    BUILTIN_UI_SEEDS,
    get_builtin_ui_translation,
)

__all__ = [
    "BUILTIN_UI_SEEDS",
    "get_builtin_ui_translation",
]

sys.modules[__name__] = _target
