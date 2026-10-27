"""Backward compatibility shim forwarding to renslate.renpy.literals."""

from __future__ import annotations

import sys

from renslate.renpy import literals as _target
from renslate.renpy.literals import *  # noqa: F403
from renslate.renpy.literals import (
    decode_say_text,
    encode_say_text,
    parse_python_string,
    quote_python_text,
)

__all__ = [
    "decode_say_text",
    "encode_say_text",
    "parse_python_string",
    "quote_python_text",
]

sys.modules[__name__] = _target
