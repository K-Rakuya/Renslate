"""Backward compatibility shim forwarding to renslate.renpy.sdk."""

from __future__ import annotations

import sys

from renslate.renpy import sdk as _target
from renslate.renpy.sdk import *  # noqa: F403
from renslate.renpy.sdk import (
    Runner,
    build_command,
    find_launcher,
    generate_translations,
)

__all__ = [
    "Runner",
    "build_command",
    "find_launcher",
    "generate_translations",
]

sys.modules[__name__] = _target
