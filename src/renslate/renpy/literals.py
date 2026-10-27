"""Codecs for the two string syntaxes found in Ren'Py translation files.

Dialogue lines use Ren'Py's own say-string syntax (see ``renpy.lexer.Lexer.string`` and
``renpy.translation.encode_say_string``). ``old``/``new`` lines in ``strings`` blocks are
plain Python string literals (see ``renpy.translation.quote_unicode``).
"""

from __future__ import annotations

import ast
import re

_SAY_WHITESPACE_RUN = re.compile(r"[ \n]+")
_SAY_ESCAPE = re.compile(r"\\(u([0-9a-fA-F]{1,4})|.)")
_SAY_SPACE_AFTER_SPACE = re.compile(r"(?<= ) ")

_PYTHON_ESCAPES = (
    ("\\", "\\\\"),
    ('"', '\\"'),
    ("\a", "\\a"),
    ("\b", "\\b"),
    ("\f", "\\f"),
    ("\n", "\\n"),
    ("\r", "\\r"),
    ("\t", "\\t"),
    ("\v", "\\v"),
)


def _decode_say_escape(match: re.Match[str]) -> str:
    char = match.group(1)
    if char == "{":
        return "{{"
    if char == "[":
        return "[["
    if char == "%":
        return "%%"
    if char == "n":
        return "\n"
    if char[0] == "u":
        digits = match.group(2)
        return chr(int(digits, 16)) if digits else ""
    return char


def decode_say_text(body: str) -> str:
    """Return the text a say statement produces from the characters between its quotes."""
    collapsed = _SAY_WHITESPACE_RUN.sub(" ", body)
    return _SAY_ESCAPE.sub(_decode_say_escape, collapsed)


def encode_say_text(text: str) -> str:
    """Encode text for the inside of a say-statement string; inverse of ``decode_say_text``."""
    encoded = text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    return _SAY_SPACE_AFTER_SPACE.sub("\\\\ ", encoded)


def quote_python_text(text: str) -> str:
    """Escape text for the inside of a double-quoted ``old``/``new`` literal."""
    for char, escaped in _PYTHON_ESCAPES:
        text = text.replace(char, escaped)
    return text


def parse_python_string(literal: str) -> str | None:
    """Evaluate a Python string literal, returning None unless it is a plain ``str``."""
    try:
        value = ast.literal_eval(literal.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    return value if isinstance(value, str) else None
