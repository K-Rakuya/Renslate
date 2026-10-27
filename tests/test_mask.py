from __future__ import annotations

import pytest

from vitrans.errors import MarkupError
from vitrans.mask import (
    PlaceholderError,
    TokenKind,
    find_markup_error,
    mask,
    tokenize,
    unmask,
    validate,
)

SOURCE = "Hello, [player_name]! {i}Welcome{/i} to the club.{w} 100%% [[ok]"


def test_mask_replaces_every_markup_element() -> None:
    masked = mask(SOURCE)
    assert masked.text == "Hello, ⟦V1⟧! ⟦T1⟧Welcome⟦/T1⟧ to the club.⟦T2⟧ 100⟦E1⟧ ⟦E2⟧ok]"
    assert [t.key for t in masked.tokens] == ["V1", "T1", "T2", "E1", "E2"]
    assert masked.tokens[1].close_raw == "{/i}"
    assert masked.tokens[2].close_raw is None


def test_unmask_round_trip() -> None:
    masked = mask(SOURCE)
    assert unmask(masked, masked.text) == SOURCE


def test_translation_may_reorder_variables() -> None:
    masked = mask("[a] meets [b]")
    assert unmask(masked, "⟦V2⟧ gặp ⟦V1⟧") == "[b] gặp [a]"


def test_nested_and_quoted_interpolation() -> None:
    tokens = [t for t in tokenize('[names["x]"][0]] and [a[1]]') if not isinstance(t, str)]
    assert [t.raw for t in tokens] == ['[names["x]"][0]]', "[a[1]]"]


def test_tag_with_interpolated_argument() -> None:
    masked = mask("{color=[c]}red{/color}")
    assert masked.text == "⟦T1⟧red⟦/T1⟧"
    assert masked.tokens[0].raw == "{color=[c]}"


def test_percent_variable_and_newline() -> None:
    masked = mask("%(name)s\nnext")
    assert [t.kind for t in masked.tokens] == [TokenKind.VARIABLE, TokenKind.NEWLINE]
    assert unmask(masked, masked.text) == "%(name)s\nnext"


def test_same_tag_twice_gets_distinct_pairs() -> None:
    masked = mask("{b}x{/b} {b}y{/b}")
    assert masked.text == "⟦T1⟧x⟦/T1⟧ ⟦T2⟧y⟦/T2⟧"


def test_unclosed_tag_is_standalone() -> None:
    masked = mask("{i}never closed")
    assert masked.text == "⟦T1⟧never closed"
    assert masked.tokens[0].close_raw is None


@pytest.mark.parametrize("text", ["open [var", "open {tag", "[x['a]", "{a=[b}"])
def test_malformed_markup_is_rejected(text: str) -> None:
    with pytest.raises(MarkupError):
        mask(text)
    assert find_markup_error(text) is not None


@pytest.mark.parametrize("text", ["plain", "lone ] and } closers", "[[ {{ %%", ""])
def test_well_formed_markup(text: str) -> None:
    assert find_markup_error(text) is None


@pytest.mark.parametrize(
    ("translated", "code"),
    [
        ("⟦T1⟧Chào⟦/T1⟧", "missing_placeholder"),
        ("⟦V1⟧ ⟦V1⟧ ⟦T1⟧a⟦/T1⟧", "duplicate_placeholder"),
        ("⟦V1⟧ ⟦V9⟧ ⟦T1⟧a⟦/T1⟧", "unknown_placeholder"),
        ("⟦V1⟧ ⟦/T1⟧a⟦T1⟧", "tag_order"),
        ("⟦V1⟧ ⟦T1⟧a⟦/T1⟧ ⟦", "malformed_placeholder"),
    ],
)
def test_validation_detects_damage(translated: str, code: str) -> None:
    masked = mask("[v] {i}a{/i}")
    assert code in {issue.code for issue in validate(masked, translated)}
    with pytest.raises(PlaceholderError):
        unmask(masked, translated)


def test_unmask_rejects_markup_defects_introduced_by_text() -> None:
    masked = mask("[v]")
    with pytest.raises(MarkupError):
        unmask(masked, "⟦V1⟧ {oops")
