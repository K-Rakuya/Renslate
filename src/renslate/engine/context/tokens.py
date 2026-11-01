"""Token budgeting and multi-script token estimation mathematics."""

from __future__ import annotations

import math

__all__ = [
    "_CHARS_PER_TOKEN_CJK",
    "_CHARS_PER_TOKEN_EN",
    "_CHARS_PER_TOKEN_VI",
    "estimate_tokens",
]

_CHARS_PER_TOKEN_CJK = 1.2
_CHARS_PER_TOKEN_VI = 2.8
_CHARS_PER_TOKEN_EN = 3.6


def estimate_tokens(text: str, is_vietnamese: bool = False) -> int:
    """Multi-script token estimation distinguishing CJK (1.2), Vietnamese (2.8), and ASCII (3.6)."""
    if not text:
        return 0
    if not is_vietnamese and text.isascii():
        return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN_EN))

    cjk = 0
    vi = 0
    other = 0
    for ch in text:
        cp = ord(ch)
        if cp < 128:
            if is_vietnamese:
                vi += 1
            else:
                other += 1
        elif (
            0x3000 <= cp <= 0x9FFF
            or 0xAC00 <= cp <= 0xD7AF
            or 0xF900 <= cp <= 0xFAFF
            or 0xFF00 <= cp <= 0xFFEF
        ):
            cjk += 1
        elif is_vietnamese or (
            0x1EA0 <= cp <= 0x1EF9 or 0x00C0 <= cp <= 0x00FF or 0x0100 <= cp <= 0x01BF
        ):
            vi += 1
        else:
            other += 1

    tokens = cjk / _CHARS_PER_TOKEN_CJK + vi / _CHARS_PER_TOKEN_VI + other / _CHARS_PER_TOKEN_EN
    return max(1, math.ceil(tokens))
