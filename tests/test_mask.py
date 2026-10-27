from __future__ import annotations

import pytest

from renslate.errors import MarkupError
from renslate.mask import (
    PlaceholderError,
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


def test_validate_raw_and_reconcile() -> None:
    source = "Hello [player], meet {b}[npc]{/b}!"
    # 1. Natural raw valid
    valid_raw = "Xin chào [player], hãy gặp {b}[npc]{/b}!"
    assert not validate_raw(source, valid_raw)
    target, issues = reconcile_translation(source, valid_raw)
    assert not issues
    assert target == valid_raw

    # 2. Natural raw missing variable
    missing_var = "Xin chào, hãy gặp {b}[npc]{/b}!"
    issues = validate_raw(source, missing_var)
    assert any(i.code == "missing_placeholder" and "[player]" in i.detail for i in issues)

    # 3. Natural raw unknown variable
    unknown_var = "Xin chào [player] và [unknown], gặp {b}[npc]{/b}!"
    issues = validate_raw(source, unknown_var)
    assert any(i.code == "unknown_placeholder" and "[unknown]" in i.detail for i in issues)

    # 4. Natural raw tag order inverted
    bad_order = "Xin chào [player], hãy gặp {/b}[npc]{b}!"
    issues = validate_raw(source, bad_order)
    assert any(i.code == "tag_order" for i in issues)

    # 5. Synthetic masked valid
    masked_target = "Xin chào ⟦V1⟧, hãy gặp ⟦T1⟧⟦V2⟧⟦/T1⟧!"
    target, issues = reconcile_translation(source, masked_target)
    assert not issues
    assert target == "Xin chào [player], hãy gặp {b}[npc]{/b}!"


def test_strict_lifo_tag_order() -> None:
    source = "{b}{i}text{/i}{/b}"
    # Interleaved / crossed tags violates LIFO
    interleaved = "{b}{i}text{/b}{/i}"
    issues = validate_raw(source, interleaved)
    assert any(i.code == "tag_order" and i.detail == "{/b}" for i in issues)

    # Properly nested LIFO passes
    proper = "{b}{i}text{/i}{/b}"
    assert not validate_raw(source, proper)

    # Standalone tag like {w} inside paired tags does not corrupt stack
    source_w = "{b}text{w}more{/b}"
    target_w = "{b}dịch{w}thêm{/b}"
    assert not validate_raw(source_w, target_w)


def test_auto_repair_placeholders_whitespace_and_brackets() -> None:
    source = "Hello, [player]! {b}Welcome{/b}!"
    masked = mask(source)

    # 1. Model output with whitespace in placeholders
    broken_whitespace = "Xin chào ⟦ V1 ⟧! ⟦ T1 ⟧Chào mừng⟦ /T1 ⟧!"
    repaired = auto_repair_placeholders(masked, broken_whitespace)
    assert repaired == "Xin chào ⟦V1⟧! ⟦T1⟧Chào mừng⟦/T1⟧!"
    assert unmask(masked, repaired) == "Xin chào [player]! {b}Chào mừng{/b}!"

    # 2. Model output with ASCII brackets [V1] or {T1}
    bracket_variants = "Xin chào [V1]! {T1}Chào mừng{/T1}!"
    repaired_brackets = auto_repair_placeholders(masked, bracket_variants)
    assert repaired_brackets == "Xin chào ⟦V1⟧! ⟦T1⟧Chào mừng⟦/T1⟧!"
    assert unmask(masked, repaired_brackets) == "Xin chào [player]! {b}Chào mừng{/b}!"


def test_auto_repair_placeholders_newline_and_unclosed_tags() -> None:
    source = "First line\n{i}Second italic line{/i}"
    masked = mask(source)

    # 1. Model output with literal newline and unclosed tag
    broken = "Dòng đầu tiên\n⟦T1⟧Dòng in nghiêng thứ hai"
    repaired = auto_repair_placeholders(masked, broken)
    assert "⟦L1⟧" in repaired
    assert repaired.endswith("⟦/T1⟧")
    restored = unmask(masked, repaired)
    assert restored == "Dòng đầu tiên\n{i}Dòng in nghiêng thứ hai{/i}"


def test_symmetric_style_tags_and_universal_closer_masking() -> None:
    # 1. Symmetric {i}...{i}
    source_sym = "{i}It'll burst!{i}"
    masked_sym = mask(source_sym)
    assert masked_sym.text == "⟦T1⟧It'll burst!⟦/T1⟧"
    assert len(masked_sym.tokens) == 1
    assert masked_sym.tokens[0].close_raw == "{i}"
    assert unmask(masked_sym, "⟦T1⟧Nó sẽ nổ tung!⟦/T1⟧") == "{i}Nó sẽ nổ tung!{i}"

    # 2. Universal closer {/}
    source_slash = "{i}It'll burst!{/}"
    masked_slash = mask(source_slash)
    assert masked_slash.text == "⟦T1⟧It'll burst!⟦/T1⟧"
    assert masked_slash.tokens[0].close_raw == "{/}"
    assert unmask(masked_slash, "⟦T1⟧Nó sẽ nổ tung!⟦/T1⟧") == "{i}Nó sẽ nổ tung!{/}"

    # 3. Mixed symmetric bold and italic
    source_nested = "{b}Bold{b} and {i}Italic{i}"
    masked_nested = mask(source_nested)
    assert masked_nested.text == "⟦T1⟧Bold⟦/T1⟧ and ⟦T2⟧Italic⟦/T2⟧"
    assert unmask(masked_nested, "⟦T1⟧Đậm⟦/T1⟧ và ⟦T2⟧Nghiêng⟦/T2⟧") == "{b}Đậm{b} và {i}Nghiêng{i}"


def test_auto_repair_trailing_and_symmetric_tags() -> None:
    # 1. Trailing {nw} tag dropped by LLM
    source_nw = "Hello, world!{nw}"
    masked_nw = mask(source_nw)
    repaired_nw = auto_repair_placeholders(masked_nw, "Xin chào thế giới!")
    assert repaired_nw == "Xin chào thế giới!⟦T1⟧"
    assert unmask(masked_nw, repaired_nw) == "Xin chào thế giới!{nw}"

    # 2. Dropped closing placeholder on symmetric {i}...{i}
    source_sym = "{i}It'll burst!{i}"
    masked_sym = mask(source_sym)
    repaired_sym = auto_repair_placeholders(masked_sym, "⟦T1⟧Nó sẽ nổ tung!")
    assert repaired_sym == "⟦T1⟧Nó sẽ nổ tung!⟦/T1⟧"
    assert unmask(masked_sym, repaired_sym) == "{i}Nó sẽ nổ tung!{i}"


def test_validate_raw_accepts_symmetric_and_universal_closers() -> None:
    # Source {i}...{i} can be translated with canonical {i}...{/i} or original {i}...{i}
    src = "{i}It'll burst!{i}"
    assert not validate_raw(src, "{i}Nó sẽ nổ tung!{/i}")
    assert not validate_raw(src, "{i}Nó sẽ nổ tung!{i}")
    assert not validate_raw(src, "{i}Nó sẽ nổ tung!{/}")

    # Source {i}...{/} can be translated with canonical {i}...{/i}
    src_slash = "{i}It'll burst!{/}"
    assert not validate_raw(src_slash, "{i}Nó sẽ nổ tung!{/i}")


def test_auto_repair_enclosing_multi_tag_pairs() -> None:
    src = "{size=50}{font=fonts/chinese-rocks.otf}After chatting with Dyna{/font}{/size}"
    m = mask(src)
    repaired = auto_repair_placeholders(m, "Sau khi trò chuyện với Dyna")
    assert repaired == "⟦T1⟧⟦T2⟧Sau khi trò chuyện với Dyna⟦/T2⟧⟦/T1⟧"
    expected = "{size=50}{font=fonts/chinese-rocks.otf}Sau khi trò chuyện với Dyna{/font}{/size}"
    assert unmask(m, repaired) == expected


def test_auto_repair_strips_hallucinated_outer_tags() -> None:
    src = "{color=#950000}Ups...\nTry Again.{/color}"
    m = mask(src)
    raw_model_output = "{font=fonts/f.otf}⟦T1⟧Rơi vào...⟦L1⟧Thử lại.{/font}{/size}"
    repaired = auto_repair_placeholders(m, raw_model_output)
    assert repaired == "⟦T1⟧Rơi vào...⟦L1⟧Thử lại.⟦/T1⟧"
    assert unmask(m, repaired) == "{color=#950000}Rơi vào...\nThử lại.{/color}"
