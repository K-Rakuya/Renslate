from __future__ import annotations

import random
import re

import pytest

from renslate.literals import (
    decode_say_text,
    encode_say_text,
    parse_python_string,
    quote_python_text,
)


def renpy_encode_say_string(s: str) -> str:
    """Reference implementation copied from renpy.translation.encode_say_string."""
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    s = re.sub(r"(?<= ) ", "\\ ", s)
    return s


SAMPLES = [
    "",
    "Hello",
    'Say "cheese"!',
    "back\\slash",
    "two  spaces   and three",
    " leading and trailing ",
    "line\nbreak",
    "{i}tag{/i} [var] [[ {{ %%",
    "unicode: Đã dịch rồi à? ⟦V1⟧",
    "\\u0041 literal",
    "tab\there",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_say_round_trip(text: str) -> None:
    assert decode_say_text(encode_say_text(text)) == text


@pytest.mark.parametrize("text", SAMPLES)
def test_say_encoding_matches_renpy(text: str) -> None:
    assert encode_say_text(text) == renpy_encode_say_string(text)


def test_say_round_trip_fuzz() -> None:
    rng = random.Random(20261023)
    alphabet = ['"', "\\", " ", " ", "\n", "{", "[", "%", "a", "n", "u", "0", "đ", "\t"]
    for _ in range(2000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        assert decode_say_text(encode_say_text(text)) == text


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (r"a\nb", "a\nb"),
        (r"\{b\}", "{{b}"),
        (r"\[x\]", "[[x]"),
        (r"100\%", "100%%"),
        (r"Aé", "Aé"),
        (r"\ \ x", "  x"),
        ("a   b", "a b"),
        (r"\"q\"", '"q"'),
    ],
)
def test_say_decoding(body: str, expected: str) -> None:
    assert decode_say_text(body) == expected


@pytest.mark.parametrize("text", [*SAMPLES, "\a\b\f\r\v", "'single'"])
def test_python_literal_round_trip(text: str) -> None:
    assert parse_python_string(f'"{quote_python_text(text)}"') == text


@pytest.mark.parametrize("literal", ["", "123", "b'x'", "foo", '"unterminated', "[1]"])
def test_python_literal_rejects_non_strings(literal: str) -> None:
    assert parse_python_string(literal) is None
