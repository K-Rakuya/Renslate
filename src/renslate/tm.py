"""Backward compatibility shim forwarding to renslate.engine.tm."""

from __future__ import annotations

import sys

from renslate.engine import tm as _target
from renslate.engine.tm import *  # noqa: F403
from renslate.engine.tm import (
    FuzzyMatch,
    TranslationMemory,
    normalize_source,
)

__all__ = [
    "FuzzyMatch",
    "TranslationMemory",
    "normalize_source",
]

sys.modules[__name__] = _target
