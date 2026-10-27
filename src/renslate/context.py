"""Backward compatibility shim forwarding to renslate.engine.context."""

from __future__ import annotations

import sys

from renslate.engine import context as _target
from renslate.engine.context import *  # noqa: F403
from renslate.engine.context import (
    QUOTED_RULES,
    ContextBuilder,
    LookbackItem,
    estimate_tokens,
)

__all__ = [
    "QUOTED_RULES",
    "ContextBuilder",
    "LookbackItem",
    "estimate_tokens",
]

sys.modules[__name__] = _target
