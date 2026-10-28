"""Comprehensive tests for multi-script token estimation (CJK, Vietnamese, ASCII)."""

from __future__ import annotations

import math

from renslate.engine.context import (
    _CHARS_PER_TOKEN_EN,
    _CHARS_PER_TOKEN_VI,
    estimate_tokens,
)


def test_estimate_tokens_empty_string() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_ascii_fast_path() -> None:
    ascii_text = "Hello world, this is a standard English dialogue line."
    expected = math.ceil(len(ascii_text) / _CHARS_PER_TOKEN_EN)
    assert estimate_tokens(ascii_text) == expected
    assert estimate_tokens("A") == 1
    assert estimate_tokens("1234567890") == 3


def test_estimate_tokens_vietnamese_explicit() -> None:
    vn_text = "Xin chào thế giới, đây là câu thoại tiếng Việt."
    explicit = estimate_tokens(vn_text, is_vietnamese=True)
    expected = math.ceil(len(vn_text) / _CHARS_PER_TOKEN_VI)
    assert explicit == expected


def test_estimate_tokens_vietnamese_implicit_diacritics() -> None:
    vn_text = "Tiếng Việt có dấu: sắc, huyền, hỏi, ngã, nặng và các chữ ă, â, đ, ê, ô, ơ, ư."
    tokens = estimate_tokens(vn_text)
    assert tokens > 0
    # Accented characters are categorized into Vietnamese ratio 2.8
    assert tokens >= math.ceil(len(vn_text) / _CHARS_PER_TOKEN_EN)


def test_estimate_tokens_hiragana_katakana_kanji() -> None:
    # Hiragana: 5 characters -> 5 / 1.2 = 4.167 -> 5 tokens
    hiragana = "こんにちは"
    assert estimate_tokens(hiragana) == 5

    # Katakana: 5 characters -> 5 / 1.2 = 4.167 -> 5 tokens
    katakana = "アイリーン"
    assert estimate_tokens(katakana) == 5

    # Kanji: 7 characters -> 7 / 1.2 = 5.833 -> 6 tokens
    kanji = "日本語能力試験"
    assert len(kanji) == 7
    assert estimate_tokens(kanji) == 6

    # Katakana with fullwidth brackets and exclamation: 8 characters -> 7 tokens
    fullwidth = "「アイリーン！」"  # noqa: RUF001
    assert estimate_tokens(fullwidth) == 7


def test_estimate_tokens_hangul() -> None:
    # Hangul syllables (0xAC00..0xD7AF): 5 chars / 1.2 -> 5 tokens
    hangul = "안녕하세요"
    assert estimate_tokens(hangul) == 5


def test_estimate_tokens_long_japanese_narration() -> None:
    # 42 characters / 1.2 -> 35 tokens
    narration = (
        "春の暖かな日差しが教室の窓から差し込み、"
        "彼女の長い黒髪を透き通るように照らしていた。"
    )
    assert len(narration) == 42
    tokens = estimate_tokens(narration)
    assert tokens == 35


def test_estimate_tokens_mixed_multiscript() -> None:
    mixed = 'd:001 | [Người nói: Eileen (アイリーン)] "Hello, 私の名前はアイリーンです！"'  # noqa: RUF001
    tokens = estimate_tokens(mixed)
    # 18 CJK chars (/1.2 = 15) + 5 VI chars (/2.8 = 1.785) + 35 ASCII (/3.6 = 9.722) -> 27 tokens
    assert tokens == 27


def test_cjk_token_math_prevents_context_underestimation() -> None:
    dialogue_batch = (
        "「おはよう、兄さん！」\n"  # noqa: RUF001
        "「……本当に、これでよかったのかな」\n"
    ) * 10
    tokens = estimate_tokens(dialogue_batch)
    old_formula_tokens = math.ceil(len(dialogue_batch) / _CHARS_PER_TOKEN_EN)
    assert tokens > old_formula_tokens * 2
