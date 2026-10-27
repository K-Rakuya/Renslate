"""Backward compatibility shim forwarding to renslate.renpy.parser."""

from __future__ import annotations

import sys

from renslate.renpy import parser as _target
from renslate.renpy.parser import *  # noqa: F403
from renslate.renpy.parser import (
    Issue,
    ParsedUnit,
    SayStatement,
    Severity,
    TranslationScanner,
    fingerprint,
    parse_say,
    parse_string_line,
    string_uid,
)

__all__ = [
    "Issue",
    "ParsedUnit",
    "SayStatement",
    "Severity",
    "TranslationScanner",
    "fingerprint",
    "parse_say",
    "parse_string_line",
    "string_uid",
]

sys.modules[__name__] = _target
