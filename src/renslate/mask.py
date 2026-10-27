"""Backward compatibility shim forwarding to renslate.renpy.mask."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import renslate.renpy.mask as _target

if TYPE_CHECKING:
    from renslate.renpy.mask import (
        CLOSE_MARK,
        OPEN_MARK,
        MaskedText,
        MaskToken,
        PlaceholderError,
        PlaceholderIssue,
        Token,
        TokenKind,
        auto_repair_placeholders,
        find_markup_error,
        mask,
        reconcile_translation,
        tokenize,
        unmask,
        validate,
        validate_raw,
    )
else:
    from renslate.renpy.mask import *  # noqa: F403

__all__ = [
    "CLOSE_MARK",
    "OPEN_MARK",
    "MaskToken",
    "MaskedText",
    "PlaceholderError",
    "PlaceholderIssue",
    "Token",
    "TokenKind",
    "auto_repair_placeholders",
    "find_markup_error",
    "mask",
    "reconcile_translation",
    "tokenize",
    "unmask",
    "validate",
    "validate_raw",
]

sys.modules[__name__] = _target
