"""Adversarial Stress Test Suite for Milestone 2: Stylometrics & Fallback.

Empirically probes:
1. Extreme stutters, chained hyphens, unicode/accented stutters, and interjections.
2. Nested Ren'Py tags ({size=+5}{b}{i}...{/i}{/b}{/size}), variables, malformed markup.
3. Exotic punctuation (CJK fullwidth, emojis, ZWJ sequences, math symbols).
4. Empty, whitespace-only, tag-only, and degenerate single-token dialogue corpora.
5. Algorithmic complexity & ReDoS resistance under pathological inputs and scale (10k lines).
6. Persona archetype deduction stability across exact numerical boundary conditions.
7. Socioeconomic classification and sample dialogue selection edge cases.
8. Persona user prompt assembly under extreme/missing metadata.
9. LLM bootstrap resilient unpacking (stringified JSON, ast literal eval, corrupted dicts).
10. ContextBuilder relational prompt assembly with complex bilateral addressing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from renslate.core.config import Config
from renslate.core.models import (
    CharacterProfile,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.analyze import (
    StylometricSummary,
    bootstrap_characters_with_llm,
    build_persona_user_prompt,
    extract_stylometrics,
    infer_persona_fallback,
)
from renslate.engine.context import ContextBuilder

# ==============================================================================
# 1. Extreme Stutters, Chained Hyphens & Interjections
# ==============================================================================


def test_adversarial_extract_stylometrics_extreme_stutters() -> None:
    """Stress-test stylometrics on extreme stutters, repeated hyphens, and mixed interjections."""
    dialogues = [
        "w-w-w-w-what are you doing?!",
        "N-N-N-No way! It c-c-can't be true!",
        "b-b-b-baka! d-don't look at me like that!",
        "a-a-a-ack! u-uh... w-whoa, hold on a second!",
        "hmph... tch... sigh... eep! oof! urk!",
    ]
    summary = extract_stylometrics(dialogues)

    assert summary.total_lines == 5
    assert summary.hesitation_count >= 10
    assert summary.hesitation_rate > 1.0

    # Ensure captured interjections contain stutter fragments and known exclamations
    top_set = set(summary.top_interjections)
    assert any("-" in item for item in top_set) or "hmph" in top_set or "tch" in top_set

    # Verify boundary cases: lone hyphens, trailing hyphens, repeated hyphens do not crash
    boundary_dialogues = [
        "-",
        "--",
        "---",
        "w-",
        "-what",
        "a--b",
        "c---d",
        "\u2014",  # Em-dash
        "\u2013",  # En-dash
    ]
    b_summary = extract_stylometrics(boundary_dialogues)
    assert b_summary.total_lines == len(boundary_dialogues)
    assert b_summary.summary_text() != ""


# ==============================================================================
# 2. Nested Ren'Py Tags, Variables & Malformed Markup
# ==============================================================================


def test_adversarial_extract_stylometrics_nested_and_malformed_tags() -> None:
    """Probe deep nesting, self-closing tags, variable interpolations, and malformed tags."""
    dialogues = [
        (
            "{size=+10}{b}{i}{color=#ff0000}{font=gui/font.ttf}{u}{s}"
            "Deeply nested text{/s}{/u}{/font}{/color}{/i}{/b}{/size}"
        ),
        "{w=1.0}{nw}{p=0.5}{fast}{cps=20}Dialogue with void timing tags.{cps=*0.5}",
        "{space=30}{vspace=20}{art=1}{image=heart.png}{k=-1.5}Special spacing and image tags.",
        "[player_name] looked at [[escaped_bracket]] and [custom_var.attribute]!",
        "{size=+5 Unclosed tag text should still be processed without crashing",
        "Double open {b{i}nested without close{/b}",
        "Empty tags {} and {   } should be stripped cleanly.",
    ]
    summary = extract_stylometrics(dialogues)

    assert summary.total_lines == len(dialogues)
    assert summary.total_words > 0
    assert summary.unique_words > 0

    # Ensure tag names and hex colors are not leaked as vocabulary tokens
    for tag_keyword in ("ff0000", "gui/font.ttf", "cps=20", "space=30"):
        assert tag_keyword not in summary.summary_text()


# ==============================================================================
# 3. Exotic Punctuation, Emojis & Unicode Barrage
# ==============================================================================


def test_adversarial_extract_stylometrics_exotic_punctuation_and_emojis() -> None:
    """Stress-test with CJK punctuation, emojis, ZWJ sequences, and mathematical symbols."""
    dialogues = [
        "Hello! \U0001f44b\U0001f389 What is this? \U0001f631",
        "本当に…ありがとう\uff01 嬉しいです\uff1f\uff01",
        "C-Cậu đang làm gì thế\uff1f\uff01 Ơ kìa…\uff01",
        "What?!?!?!????!!!! Are you insane?????!!!!!",
        "Wait.......... is that really happening...?",
        "😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰",
        "👨\u200d👩\u200d👧\u200d👦 Family emoji with ZWJ and flags 🇻🇳 🇯🇵 🇺🇸",
        "Math formula: \u2200x \u2208 \\mathbb{R}, \u2211(1/n\u00b2) \u2248 1.6449 \u2260 0!",
    ]
    summary = extract_stylometrics(dialogues)

    assert summary.total_lines == len(dialogues)
    assert summary.exclamation_count >= 8
    assert summary.question_count >= 4
    assert summary.ellipsis_count >= 3
    assert 0.0 <= summary.type_token_ratio <= 1.0

    text = summary.summary_text()
    assert "TTR:" in text
    assert "Độ dài câu TB:" in text


# ==============================================================================
# 4. Empty & Degenerate Corpus (ZeroDivision Safety)
# ==============================================================================


def test_adversarial_extract_stylometrics_degenerate_and_empty() -> None:
    """Verify zero division safety across various degenerate corpora."""
    # 1. Completely empty list
    s_empty = extract_stylometrics([])
    assert s_empty.total_lines == 0
    assert s_empty.total_words == 0
    assert s_empty.type_token_ratio == 0.0
    assert s_empty.avg_sentence_length == 0.0
    assert s_empty.summary_text() == "Không có dữ liệu thoại thống kê"

    # 2. Whitespace-only strings
    s_spaces = extract_stylometrics(["", "   ", "\t\t", "\n\n\n", " \r\n "])
    assert s_spaces.total_lines == 5
    assert s_spaces.total_words == 0
    assert s_spaces.type_token_ratio == 0.0
    assert s_spaces.avg_sentence_length == 0.0
    assert s_spaces.summary_text() == "Không có dữ liệu thoại thống kê"

    # 3. Tag-only and variable-only strings
    s_tags = extract_stylometrics(
        ["{}", "[]", "{b}{/b}", "[player_name]", "{size=+5}[var]{/size}"]
    )
    assert s_tags.total_lines == 5
    assert s_tags.total_words == 0
    assert s_tags.summary_text() == "Không có dữ liệu thoại thống kê"

    # 4. Punctuation-only strings
    s_punct = extract_stylometrics([".", "...", "!?", "----", "~~~"])
    assert s_punct.total_lines == 5
    assert s_punct.total_words == 0

    # 5. Monotonous single word repeated 100 times
    s_mono = extract_stylometrics(["yes"] * 100)
    assert s_mono.total_lines == 100
    assert s_mono.total_words == 100
    assert s_mono.unique_words == 1
    assert s_mono.type_token_ratio == 0.01


# ==============================================================================
# 5. Algorithmic Complexity, ReDoS & Massive Corpus Throughput
# ==============================================================================


def test_adversarial_extract_stylometrics_redos_and_massive_scale() -> None:
    """Ensure linear complexity: no ReDoS on pathological inputs and fast 10k throughput."""
    # 1. Pathological ReDoS strings
    pathological_strings = [
        "{" * 5000 + "test" + "}" * 5000,
        "[" * 5000 + "player" + "]" * 5000,
        "a-" * 5000 + "b",
        "!" * 10000 + "?" * 10000 + "." * 10000,
        "-----" * 2000,
    ]
    t0 = time.perf_counter()
    summary_redos = extract_stylometrics(pathological_strings)
    t1 = time.perf_counter()
    assert (t1 - t0) < 0.5, f"ReDoS vulnerability detected: took {t1 - t0:.4f}s"
    assert summary_redos.total_lines == 5

    # 2. Massive dialogue throughput (10,000 dialogue lines)
    sample_corpus = [
        "{size=+5}W-What are you doing here, [player_name]?!{/size}",
        "I... I was just looking for the ancient spellbook...",
        "Hmph! As if an idiot like you could understand high magic! {i}B-Baka!{/i}",
        "Please teach me, Lady Charlotte! I want to become stronger.",
    ] * 2500  # 10,000 lines

    t_start = time.perf_counter()
    summary_massive = extract_stylometrics(sample_corpus)
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    assert summary_massive.total_lines == 10000
    assert summary_massive.total_words == 90000
    assert elapsed < 1.0, f"Throughput too slow: 10k lines took {elapsed:.4f}s"


# ==============================================================================
# 6. Persona Archetype Deduction Stability & Boundary Conditions
# ==============================================================================


def test_adversarial_infer_persona_fallback_boundary_matrix() -> None:
    """Stress-test archetype deduction around exact numerical thresholds."""

    def _make_sty(
        *,
        excl: float = 0.0,
        ellip: float = 0.0,
        ques: float = 0.0,
        hesit: float = 0.0,
        avg_len: float = 0.0,
        ttr: float = 0.0,
        top: tuple[str, ...] = (),
    ) -> StylometricSummary:
        return StylometricSummary(
            total_lines=10,
            total_words=100,
            unique_words=int(100 * ttr),
            type_token_ratio=ttr,
            total_sentences=10,
            avg_sentence_length=avg_len,
            exclamation_count=int(10 * excl),
            exclamation_rate=excl,
            question_count=int(10 * ques),
            question_rate=ques,
            ellipsis_count=int(10 * ellip),
            ellipsis_rate=ellip,
            hesitation_count=int(10 * hesit),
            hesitation_rate=hesit,
            top_interjections=top,
        )

    # Meta tag overrides in current implementation:
    for meta_tag in ("mc", "player", "hero", "protagonist"):
        p = infer_persona_fallback(meta_tag, [], _make_sty(excl=2.0))
        assert p["archetype"] == "protagonist"

    for meta_tag in ("narrator", "narration", "none"):
        p = infer_persona_fallback(meta_tag, [], _make_sty(excl=2.0))
        assert p["archetype"] == "narrator"

    for meta_tag in ("thought", "thoughts", "thinking", "t", "th"):
        p = infer_persona_fallback(meta_tag, [], _make_sty(excl=2.0))
        assert p["archetype"] == "inner_monologue"

    # Tsundere boundary: excl_rate >= 0.7 and (has_stutter or ellip_rate >= 0.2)
    p_tsun_at = infer_persona_fallback("ch1", [], _make_sty(excl=0.70, ellip=0.20))
    assert p_tsun_at["archetype"] == "tsundere"

    p_tsun_below = infer_persona_fallback("ch1", [], _make_sty(excl=0.69, ellip=0.20))
    assert p_tsun_below["archetype"] != "tsundere"

    p_tsun_interj = infer_persona_fallback("ch1", [], _make_sty(top=("hmph",)))
    assert p_tsun_interj["archetype"] == "tsundere"

    # Dandere vs Kuudere: ellip_rate >= 0.5 and excl_rate < 0.3
    p_dandere = infer_persona_fallback(
        "ch1", [], _make_sty(ellip=0.50, excl=0.29, top=("w-wait",))
    )
    assert p_dandere["archetype"] == "dandere"

    p_kuudere = infer_persona_fallback("ch1", [], _make_sty(ellip=0.50, excl=0.29))
    assert p_kuudere["archetype"] == "kuudere"

    # Rebel Warrior: excl_rate >= 1.0 and ellip_rate < 0.25 (without tsundere trigger)
    p_rebel = infer_persona_fallback("ch1", [], _make_sty(excl=1.00, ellip=0.10))
    assert p_rebel["archetype"] == "rebel_warrior"

    # Elder Mentor: avg_len >= 12.0 and ttr >= 0.70 and excl_rate < 0.4
    p_mentor = infer_persona_fallback(
        "ch1", [], _make_sty(avg_len=12.0, ttr=0.70, excl=0.39)
    )
    assert p_mentor["archetype"] == "elder_mentor"

    p_mentor_fail_ttr = infer_persona_fallback(
        "ch1", [], _make_sty(avg_len=12.0, ttr=0.69, excl=0.39)
    )
    assert p_mentor_fail_ttr["archetype"] != "elder_mentor"

    # Inquisitive Apprentice: ques_rate >= 0.50
    p_apprentice = infer_persona_fallback("ch1", [], _make_sty(ques=0.50))
    assert p_apprentice["archetype"] == "inquisitive_apprentice"

    p_apprentice_below = infer_persona_fallback("ch1", [], _make_sty(ques=0.49))
    assert p_apprentice_below["archetype"] == "supporting_cast"


# ==============================================================================
# 7. Socioeconomic Status & Sample Dialogue Edge Cases
# ==============================================================================


def test_adversarial_infer_persona_socioeconomic_and_samples() -> None:
    """Verify socioeconomic cues and sample dialogue selection across edge cases."""
    dummy_sty = extract_stylometrics([])

    # Socioeconomic cue checks across tag, display_name, role
    test_cases = [
        ("elenor", "Nữ hoàng Elenor", "Ruler", "Quý tộc / Hoàng gia (Nobility)"),
        ("ch1", "Knight Captain", "Commander", "Quân đội / Hiệp sĩ (Military)"),
        (
            "layla",
            "High Sorceress",
            "Đại pháp sư",
            "Học giả ma thuật / Giáo sĩ (Religious / Scholarly)",
        ),
        ("bart", "Innkeeper Tom", "Thương nhân", "Thương nhân / Tiểu thương (Merchant)"),
        ("peasant", "Old Farmer", "Villager", "Thường dân (Commoner)"),
    ]
    for tag, disp, role, expected_status in test_cases:
        p = infer_persona_fallback(tag, [], dummy_sty, role=role, display_name=disp)
        assert p["socioeconomic_status"] == expected_status

    # Sample dialogue edge cases:
    # 1. Short lines <= 5 chars are filtered out
    p_short = infer_persona_fallback("ch1", ["Hi", "No", "Yes", "Bye"], dummy_sty)
    assert p_short["sample_dialogue"] == ()

    # 2. Duplicate lines do not repeat in sample_dialogue
    p_dup = infer_persona_fallback("ch1", ["I will protect this land!"] * 20, dummy_sty)
    assert len(p_dup["sample_dialogue"]) == 1

    # 3. 1, 2, 3 lines are all captured
    for n in (1, 2, 3):
        lines = [f"This is representative dialogue line number {i}" for i in range(n)]
        p_n = infer_persona_fallback("ch1", lines, dummy_sty)
        assert len(p_n["sample_dialogue"]) == n

    # 4. 100 lines: extracts beginning, middle, and end
    lines_100 = [f"Dialogue line {i:03d} for test verification" for i in range(100)]
    p_100 = infer_persona_fallback("ch1", lines_100, dummy_sty)
    assert len(p_100["sample_dialogue"]) == 3
    assert p_100["sample_dialogue"][0] == lines_100[0]
    assert p_100["sample_dialogue"][1] == lines_100[50]
    assert p_100["sample_dialogue"][2] == lines_100[-1]


# ==============================================================================
# 8. Persona User Prompt Assembly Under Extreme Metadata
# ==============================================================================


def test_adversarial_build_persona_user_prompt_stress() -> None:
    """Stress-test build_persona_user_prompt with empty, missing, and oversized inputs."""
    # 1. Empty chunk_tags returns prefix
    p_empty = build_persona_user_prompt([], {}, {}, {})
    assert "Phân tích hồ sơ tính cách" in p_empty

    # 2. Missing profiles and stylometrics fall back cleanly
    p_missing = build_persona_user_prompt(["ghost_tag"], {}, {}, {})
    assert "- Tag: ghost_tag" in p_missing
    assert "Tên: Ghost_tag" in p_missing
    assert "Không có dữ liệu thống kê" in p_missing
    assert "(Không có thoại)" in p_missing

    # 3. Extreme sample dialogues containing quotes, newlines, and unicode
    chars = {"ch1": CharacterProfile(tag="ch1", display_name='Charlotte "The Flame"')}
    samples = {
        "ch1": [
            'Line 1 with "nested quotes" and \n newlines \n\n and {b}tags{/b}',
            "Line 2 with emojis \U0001f631\U0001f525 and [player_name]",
            "A" * 1500,  # Giant string
            "Line 4 standard",
            "Line 5 should be ignored due to slice limit",
        ]
    }
    stys = {"ch1": extract_stylometrics(samples["ch1"])}
    prompt = build_persona_user_prompt(["ch1"], chars, samples, stys)

    assert "- Tag: ch1" in prompt
    assert 'Tên: Charlotte "The Flame"' in prompt
    assert "Line 1 with" in prompt
    assert "Line 5 should be ignored" not in prompt


# ==============================================================================
# 9. LLM Bootstrap Resilient Unpacking & Corrupted Payloads
# ==============================================================================


def test_adversarial_bootstrap_characters_with_llm_resilience(tmp_path: Path) -> None:
    """Verify bootstrap_characters_with_llm survives corrupted or weird LLM outputs."""
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    project = Project.create(tmp_path)

    with project.open_db() as db:
        # Seed test units
        for i in range(5):
            db.save_unit(
                Unit(
                    f"d:{i}",
                    UnitKind.DIALOGUE,
                    i,
                    f"Speaker line {i}",
                    Status.PENDING,
                    speaker="c1",
                )
            )
            db.save_unit(
                Unit(
                    f"d:1{i}",
                    UnitKind.DIALOGUE,
                    10 + i,
                    f"Speaker line {i}",
                    Status.PENDING,
                    speaker="c2",
                )
            )

        db.save_character(CharacterProfile(tag="c1", display_name="Charlotte"))
        db.save_character(CharacterProfile(tag="c2", display_name="Arthur"))

        class CorruptedLLM:
            def complete_json(
                self, system: str, user: str, *args: Any, **kwargs: Any
            ) -> tuple[dict[str, Any], int, int]:
                return (
                    {
                        # Stringified dict with stringified pronoun_map and comma-separated traits
                        "Charlotte": (
                            "{'gender': 'female', 'role': 'Tiểu thư', 'archetype': 'tsundere', "
                            "'personality_traits': 'kiêu hãnh, sắc sảo', "
                            "'pronoun_map': '{\\'mc\\': \\'tôi - cậu\\'}', "
                            "'socioeconomic_status': 'Quý tộc'}"
                        ),
                        # Malformed string that fails literal_eval
                        "c2": "{'gender': 'male', invalid syntax error...",
                        # Non-dict scalar values
                        "scalar_tag": 9999,
                        "none_tag": None,
                    },
                    150,
                    80,
                )

        chars = bootstrap_characters_with_llm(project, db, CorruptedLLM())

        # Charlotte correctly parsed and normalized
        assert "c1" in chars
        c1 = chars["c1"]
        assert c1.archetype == "tsundere"
        assert c1.personality_traits == ("kiêu hãnh", "sắc sảo")
        assert c1.pronoun_map == {"mc": "tôi - cậu"}
        assert c1.socioeconomic_status == "Quý tộc"

        # c2 retained base profile without crash
        assert "c2" in chars


# ==============================================================================
# 10. ContextBuilder Relational Prompt Assembly & Token Budgeting
# ==============================================================================


def test_adversarial_context_builder_relational_prompt_assembly() -> None:
    """Verify ContextBuilder correctly formats relational pronoun frames and respects budgets."""
    cfg = Config()
    cb = ContextBuilder(cfg)

    chars = {
        "mc": CharacterProfile(
            tag="mc",
            display_name="Hero",
            gender="male",
            role="Protagonist",
            pronoun_map={"ch1": "anh - em"},
        ),
        "ch1": CharacterProfile(
            tag="ch1",
            display_name="Charlotte",
            gender="female",
            role="Tsundere Mage",
            archetype="tsundere",
            pronoun_map={"mc": "em - anh"},
        ),
    }

    units = [
        Unit(
            "d:1",
            UnitKind.DIALOGUE,
            1,
            "{size=+10}{b}W-W-What are you doing, [player_name]?!{/b}{/size}",
            Status.PENDING,
            speaker="ch1",
        ),
        Unit(
            "d:2",
            UnitKind.DIALOGUE,
            2,
            "I... I am just studying ancient magic.",
            Status.PENDING,
            speaker="mc",
        ),
        Unit(
            "d:3",
            UnitKind.DIALOGUE,
            3,
            "(What a strange reaction... Did I offend her?)",
            Status.PENDING,
            speaker="thought",
        ),
        Unit(
            "d:4",
            UnitKind.DIALOGUE,
            4,
            "Hmph! \U0001f631 Like you could understand it! \U0001f44b\U0001f389",
            Status.PENDING,
            speaker="ch1",
        ),
        Unit(
            "d:5",
            UnitKind.DIALOGUE,
            5,
            "Narrator voice speaking here...",
            Status.PENDING,
            speaker="narrator",
        ),
        Unit(
            "d:6",
            UnitKind.DIALOGUE,
            6,
            "Empty speaker line with no tag.",
            Status.PENDING,
            speaker=None,
        ),
    ]

    prompt = cb.build_user_prompt(units, characters=chars)

    # 1. Relational pronoun dialogue frame
    assert "QUAN HỆ XƯNG HÔ ĐỐI THOẠI TRONG PHÂN ĐOẠN:" in prompt
    assert "Charlotte và Hero đang đối thoại trực tiếp (cặp ch1 - mc):" in prompt
    assert "Tự xưng (I/me/my) là 'em', gọi đối phương (you/your) là 'anh'" in prompt
    assert "Tự xưng (I/me/my) là 'anh', gọi đối phương (you/your) là 'em'" in prompt

    # 2. Character profile summary
    assert "HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:" in prompt
    assert "Charlotte (tag: ch1): Nữ | Vai trò: Tsundere Mage" in prompt
    assert "Hero (tag: mc): Nam | Vai trò: Protagonist" in prompt

    # 3. Formatted batch lines
    assert "d:1 | [Người nói: Charlotte (ch1)]" in prompt
    assert "d:2 | [Người nói: Hero (mc)]" in prompt
    assert "d:3 | [Độc thoại/Nội tâm: thought]" in prompt
    assert "d:4 | [Người nói: Charlotte (ch1)]" in prompt
    assert "d:5 | [Độc thoại/Nội tâm: narrator]" in prompt
    assert "d:6 | Empty speaker line with no tag." in prompt
