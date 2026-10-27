"""Backward compatibility shim forwarding to renslate.engine.qa."""

from __future__ import annotations

import sys

from renslate.engine import qa as _target
from renslate.engine.qa import *  # noqa: F403
from renslate.engine.qa import (
    QAChecker,
    clean_translation,
    repair_glossary_leakage,
)

__all__ = [
    "QAChecker",
    "clean_translation",
    "repair_glossary_leakage",
]

sys.modules[__name__] = _target
