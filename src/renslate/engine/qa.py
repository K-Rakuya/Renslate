"""Backward compatibility forwarding shim for renslate.engine.qa package."""

from __future__ import annotations

from renslate.engine.qa import (
    _CJK_PATTERN,
    _ENGLISH_WORDS,
    _WORD_TOKEN,
    QAChecker,
    clean_translation,
    compute_calque_density,
    is_non_translatable_text,
)

__all__ = [
    "_CJK_PATTERN",
    "_ENGLISH_WORDS",
    "_WORD_TOKEN",
    "QAChecker",
    "clean_translation",
    "compute_calque_density",
    "is_non_translatable_text",
]
