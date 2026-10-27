"""Adversarial stress harness & empirical verification for Milestone 3 (QA Anti-Leak).

Probing:
1. Malformed echoes from LLM:
   - Doubled and tripled speaker tags, relational frames, and monologue frames.
   - Mixed metadata tags (e.g. bracket speaker + relational frame).
   - Conversational filler prefixes and unit ID prefixes (d:101 |, s:ch1 |).
   - Trailing metadata suffixes (speaker, relation, emotion, listener).
   - Multi-layer peeling of nested quotes and prefixes.
   - Solely-metadata targets collapsing cleanly to empty string.
2. Genuine brackets preservation:
   - Ren'Py text formatting tags: {size}, {font}, {color}, {b}, {i}.
   - Ren'Py variable interpolations: [player_name], [hero.name], [score!s].
   - Sound effects: [Sigh], [Laughs], [Gasp], [Chuckles], [Thở dài], [Cười lớn].
   - Pause ellipsis: [...], ...
   - Numeric choice indicators: [1], [2].
   - Escaped brackets: [[...]], {{...}}.
3. Fatal severity validation on QAChecker:
   - All speaker tag leakages emit severity="error" (speaker_tag_leakage).
   - All relational/emotion/listener tag leakages emit severity="error"
     (relation_metadata_leakage).
   - Standalone speaker tags emit severity="error".
   - Speaker name only without dialogue emits severity="error".
   - Empty targets emit severity="error" (empty_target).
   - Unclosed bracket markup emits severity="error" (bad_markup).
4. Latency micro-benchmark:
   - Microsecond-level latency verification for clean_translation (< 15 us).
   - Microsecond-level latency verification for QAChecker.check_unit (< 25 us).
"""

from __future__ import annotations

import time

from renslate.engine.qa import QAChecker, clean_translation


class TestAdversarialMalformedEchoes:
    """Stress tests probing clean_translation under malformed LLM outputs."""

    def test_doubled_relational_frames(self) -> None:
        """Doubled relational frames must be stripped completely."""
        raw = (
            "[Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE] "
            "[Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE] "
            "Xin chào em!"
        )
        cleaned = clean_translation(raw, speaker="ren", source="Hello!")
        assert cleaned == "Xin chào em!"

    def test_doubled_speaker_frames(self) -> None:
        """Doubled simple speaker frames must be stripped completely."""
        raw = "[Người nói: Ren] [Người nói: Ren] Xin chào em!"
        cleaned = clean_translation(raw, speaker="ren", source="Hello!")
        assert cleaned == "Xin chào em!"

    def test_doubled_monologue_frames(self) -> None:
        """Doubled monologue frames must be stripped completely."""
        raw = "[Độc thoại/Nội tâm: Ren] [Độc thoại/Nội tâm: Ren] Mình phải cố gắng."
        cleaned = clean_translation(raw, speaker="ren", source="(I must try.)")
        assert cleaned == "Mình phải cố gắng."

    def test_tripled_bracket_speaker_tags(self) -> None:
        """Tripled short bracket speaker tags must be stripped completely."""
        raw = "[MC] [MC] [MC] Xin chào!"
        cleaned = clean_translation(raw, speaker="mc", source="Hello!")
        assert cleaned == "Xin chào!"

    def test_mixed_bracket_and_speaker_frame(self) -> None:
        """Mixed short bracket and full speaker frame must be stripped cleanly."""
        raw = "[MC] [Người nói: MC] Xin chào!"
        cleaned = clean_translation(raw, speaker="mc", source="Hello!")
        assert cleaned == "Xin chào!"

    def test_speaker_frame_followed_by_relation_frame(self) -> None:
        """Sequential speaker frame and relation frame must both be peeled."""
        raw = "[Người nói: Ren] [Quan hệ: ROMANCE] Xin chào em!"
        cleaned = clean_translation(raw, speaker="ren", source="Hello!")
        assert cleaned == "Xin chào em!"

    def test_monologue_frame_followed_by_speaker_frame(self) -> None:
        """Sequential monologue frame and speaker frame must both be peeled."""
        raw = "[Độc thoại/Nội tâm: Ren] [Người nói: Ren] Xin chào em!"
        cleaned = clean_translation(raw, speaker="ren", source="Hello!")
        assert cleaned == "Xin chào em!"

    def test_doubled_unit_id_prefixes(self) -> None:
        """Doubled unit ID prefixes (d:101 | d:101 |) must be peeled."""
        raw = "d:101 | d:101 | Xin chào em!"
        cleaned = clean_translation(raw, speaker="ren", source="Hello!")
        assert cleaned == "Xin chào em!"

    def test_doubled_translation_prefix_junk(self) -> None:
        """Doubled conversational translation prefixes must be peeled."""
        raw = "Bản dịch: translation: Xin chào em!"
        cleaned = clean_translation(raw, speaker="ren", source="Hello!")
        assert cleaned == "Xin chào em!"

    def test_multilayer_prefix_and_quotes(self) -> None:
        """Multi-layer combination of ID prefix, speaker frame, junk, and quotes."""
        raw = 'd:101 | [Người nói: Ren] Bản dịch: "Xin chào em!"'
        cleaned = clean_translation(raw, speaker="ren", source="Hello!")
        assert cleaned == "Xin chào em!"

    def test_trailing_metadata_suffixes(self) -> None:
        """Trailing metadata suffixes (relation, emotion, listener, speaker) must be peeled."""
        assert (
            clean_translation("Xin chào em! [Quan hệ: ROMANCE]", source="Hello!") == "Xin chào em!"
        )
        assert (
            clean_translation("Xin chào em! [Cảm xúc: hesitant]", source="Hello!") == "Xin chào em!"
        )
        assert (
            clean_translation("Xin chào em! [Người nghe: Liana]", source="Hello!") == "Xin chào em!"
        )
        assert (
            clean_translation("Xin chào em! [Người nói: Ren]", source="Hello!") == "Xin chào em!"
        )
        assert (
            clean_translation("Suy nghĩ... [Độc thoại/Nội tâm: Ren]", source="(Thinking...)")
            == "Suy nghĩ..."
        )

    def test_solely_metadata_collapses_to_empty(self) -> None:
        """Target containing solely metadata frames must collapse to empty string."""
        assert clean_translation("[Người nói: Ren]", source="Hello!", speaker="ren") == ""
        assert (
            clean_translation(
                "[Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE]",
                source="Hello!",
                speaker="ren",
            )
            == ""
        )
        assert (
            clean_translation("[Độc thoại/Nội tâm: Ren]", source="Hello!", speaker="ren")
            == ""
        )
        assert clean_translation("[MC]", source="Hello!", speaker="mc") == ""

    def test_internal_whitespace_and_lowercase_variants(self) -> None:
        """Frames with internal whitespace and lowercase keywords must be peeled cleanly."""
        raw_ws = "[ Người nói : Ren -> Người nghe : Liana | Quan hệ : ROMANCE ] Xin chào!"
        assert clean_translation(raw_ws, speaker="ren", source="Hello!") == "Xin chào!"

        raw_lower = "[người nói: ren -> người nghe: liana | quan hệ: romance] Xin chào!"
        assert clean_translation(raw_lower, speaker="ren", source="Hello!") == "Xin chào!"

        raw_trailing_lower = "Xin chào! [quan hệ: romance]"
        assert clean_translation(raw_trailing_lower, speaker="ren", source="Hello!") == "Xin chào!"


class TestAdversarialGenuineBracketsPreservation:
    """Stress tests probing preservation of authentic Ren'Py tags and brackets."""

    def test_renpy_variable_interpolations(self) -> None:
        """Ren'Py variable interpolations at start, middle, or end must be preserved."""
        # Variable at start
        assert (
            clean_translation("[player_name] ơi, xin chào!", source="Hello, [player_name]!")
            == "[player_name] ơi, xin chào!"
        )
        # Variable in middle
        assert (
            clean_translation(
                "Anh hùng [player_name] đã đánh bại quái vật.",
                source="The hero [player_name] defeated the monster.",
            )
            == "Anh hùng [player_name] đã đánh bại quái vật."
        )
        # Variable at end
        assert (
            clean_translation("Coi chừng, [player_name]!", source="Look out, [player_name]!")
            == "Coi chừng, [player_name]!"
        )
        # Variable standalone
        assert clean_translation("[player_name]", source="[player_name]") == "[player_name]"
        # Property access and format flags
        assert (
            clean_translation(
                "[hero.name] ghi được [score!s] điểm.",
                source="[hero.name] got [score!s] pts.",
            )
            == "[hero.name] ghi được [score!s] điểm."
        )

    def test_sound_effects_in_brackets(self) -> None:
        """Genuine sound effects (English or translated Vietnamese) must be preserved."""
        # [Sigh] at start
        assert (
            clean_translation("[Thở dài] Thật là mệt mỏi.", source="[Sigh] It's tiring.")
            == "[Thở dài] Thật là mệt mỏi."
        )
        # [Laughs] at start
        assert (
            clean_translation("[Cười lớn] Buồn cười thật đấy!", source="[Laughs] That's hilarious!")
            == "[Cười lớn] Buồn cười thật đấy!"
        )
        # [Gasp] at start
        assert (
            clean_translation("[Kinh ngạc] Không thể nào!", source="[Gasp] No way!")
            == "[Kinh ngạc] Không thể nào!"
        )
        # [Chuckles] at start
        assert (
            clean_translation("[Cười khẽ] Đúng như mong đợi.", source="[Chuckles] As expected.")
            == "[Cười khẽ] Đúng như mong đợi."
        )
        # Sound effect alone on line
        assert clean_translation("[Thở dài]", source="[Sigh]") == "[Thở dài]"

    def test_pause_ellipsis_in_brackets(self) -> None:
        """Pause ellipsis [...] at start, end, or standalone must be preserved."""
        assert (
            clean_translation("[...] Không có ai ở đây cả.", source="[...] Nobody is here.")
            == "[...] Không có ai ở đây cả."
        )
        assert clean_translation("Khoan đã... [...]", source="Wait... [...]") == "Khoan đã... [...]"
        assert clean_translation("[...]", source="[...]") == "[...]"

    def test_numeric_choices_in_brackets(self) -> None:
        """Menu choice indices [1], [2] at start of line must be preserved."""
        assert (
            clean_translation("[1] Đi sang bên trái", source="[1] Go left")
            == "[1] Đi sang bên trái"
        )
        assert (
            clean_translation("[2] Đi sang bên phải", source="[2] Go right")
            == "[2] Đi sang bên phải"
        )

    def test_escaped_brackets_in_renpy(self) -> None:
        """Ren'Py escaped brackets [[...]] must be preserved intact."""
        assert (
            clean_translation("[[Thông báo]] Cập nhật hệ thống", source="[[Notice]] System update")
            == "[[Thông báo]] Cập nhật hệ thống"
        )
        assert (
            clean_translation("Nhấn [[Enter]] để tiếp tục", source="Press [[Enter]] to continue")
            == "Nhấn [[Enter]] để tiếp tục"
        )

    def test_renpy_text_tags(self) -> None:
        """Ren'Py text formatting tags {size}, {color}, {b}, {i}, {font} must remain intact."""
        assert (
            clean_translation("{size=+5}Cảnh báo!{/size}", source="{size=+5}Warning!{/size}")
            == "{size=+5}Cảnh báo!{/size}"
        )
        assert (
            clean_translation("{b}Chữ đậm{/b}", source="{b}Bold text{/b}")
            == "{b}Chữ đậm{/b}"
        )
        assert (
            clean_translation(
                "{color=#ff0000}Báo động đỏ{/color}",
                source="{color=#ff0000}Red alert{/color}",
            )
            == "{color=#ff0000}Báo động đỏ{/color}"
        )
        assert (
            clean_translation(
                "{font=gui/font.ttf}Phông chữ{/font}",
                source="{font=gui/font.ttf}Custom font{/font}",
            )
            == "{font=gui/font.ttf}Phông chữ{/font}"
        )

    def test_bracketed_source_speaker_preservation(self) -> None:
        """If source itself starts with bracketed character name, target preserves it."""
        assert (
            clean_translation("[Layla] Xin chào bạn!", source="[Layla] Hello!", speaker="Layla")
            == "[Layla] Xin chào bạn!"
        )


class TestAdversarialQAFatalSeverity:
    """Stress tests verifying fatal severity ('error') for all metadata leaks."""

    def test_speaker_metadata_leak_severity(self) -> None:
        """All variations of speaker metadata leaks must emit fatal 'error' issues."""
        checker = QAChecker()
        leak_variants = [
            "[Người nói: Ren] Xin chào!",
            "[Speaker: Ren] Xin chào!",
            "[Độc thoại/Nội tâm: Ren] Suy nghĩ...",
            "[Độc thoại: Ren] Suy nghĩ...",
            "[Nội tâm: Ren] Suy nghĩ...",
            "[Suy nghĩ: Ren] Suy nghĩ...",
            "[Monologue: Ren] Suy nghĩ...",
            "[Monologue/Internal: Ren] Suy nghĩ...",
            "[Monologue/Thought: Ren] Suy nghĩ...",
            "[Thought: Ren] Suy nghĩ...",
            "Xin chào em! [Người nói: Ren] Em khoẻ không?",
            "Suy nghĩ... [Độc thoại: Ren] Mình phải làm sao?",
        ]
        for variant in leak_variants:
            issues = checker.check_unit("test", "Hello!", None, variant)
            errs = [i for i in issues if i.code == "speaker_tag_leakage" and i.severity == "error"]
            assert len(errs) > 0, f"Expected fatal speaker_tag_leakage error for {variant!r}"

    def test_relation_metadata_leak_severity(self) -> None:
        """All variations of relation metadata leaks must emit fatal 'error' issues."""
        checker = QAChecker()
        leak_variants = [
            "[Quan hệ: ROMANCE] Xin chào em!",
            "[Relationship: ROMANCE] Xin chào em!",
            "[Relation: ROMANCE] Xin chào em!",
            "[Người nghe: Liana] Xin chào em!",
            "[Listener: Liana] Xin chào em!",
            "[Cảm xúc: hesitant] Xin chào em!",
            "[Emotion: hesitant] Xin chào em!",
            "[Ren xưng 'anh', gọi 'em'] Xin chào em!",
            '[Ren xưng "anh", gọi "em"] Xin chào em!',
            "Xin chào em! [Quan hệ: ROMANCE] Rất vui được gặp em.",
            "Xin chào em! [Cảm xúc: surprised] Thật bất ngờ!",
            "Xin chào em! [Người nghe: Liana] Em đây rồi.",
            "Xin chào em! [Ren xưng 'anh', gọi 'em'] Mau vào đi.",
        ]
        for variant in leak_variants:
            issues = checker.check_unit("test", "Hello!", None, variant)
            errs = [
                i for i in issues
                if i.code == "relation_metadata_leakage" and i.severity == "error"
            ]
            assert len(errs) > 0, f"Expected fatal relation_metadata_leakage error for {variant!r}"

    def test_standalone_speaker_and_empty_target_fatal_severity(self) -> None:
        """Standalone speaker tags and empty targets must emit fatal 'error' issues."""
        checker = QAChecker()

        # Standalone speaker bracket
        issues_spk = checker.check_unit("test", "Who is there?", None, "[MC]", speaker="MC")
        assert any(i.code == "speaker_tag_leakage" and i.severity == "error" for i in issues_spk)

        # Speaker name only without dialogue
        issues_name = checker.check_unit("test", "Who is there?", None, "MC:", speaker="MC")
        assert any(i.code == "speaker_tag_leakage" and i.severity == "error" for i in issues_name)

        # Empty target
        issues_empty = checker.check_unit("test", "Hello!", None, "")
        assert any(i.code == "empty_target" and i.severity == "error" for i in issues_empty)

        # Bad markup (unclosed bracket)
        issues_markup = checker.check_unit("test", "Hello!", None, "Xin chào [bạn!")
        assert any(i.code == "bad_markup" and i.severity == "error" for i in issues_markup)

    def test_clean_translations_pass_with_zero_fatal_errors(self) -> None:
        """Sanitized translations with genuine brackets pass QA with 0 fatal errors."""
        checker = QAChecker()
        valid_targets = [
            ("Hello, [player_name]!", "[player_name] ơi, xin chào!"),
            ("[Sigh] It's tiring.", "[Thở dài] Thật là mệt mỏi."),
            ("[...] Nobody is here.", "[...] Không có ai ở đây cả."),
            ("[1] Go left", "[1] Đi sang bên trái"),
            ("Press [[Enter]]", "Nhấn [[Enter]] để tiếp tục"),
            ("Good morning!", "Chào buổi sáng bạn nhé!"),
        ]
        for src, tgt in valid_targets:
            issues = checker.check_unit("test", src, None, tgt)
            fatal_issues = [i for i in issues if i.severity == "error"]
            assert len(fatal_issues) == 0, f"Unexpected fatal errors for {tgt!r}: {fatal_issues}"


class TestAdversarialQALatencyMicrobenchmark:
    """Latency micro-benchmarks ensuring O(1) performance."""

    def test_clean_translation_latency(self) -> None:
        """clean_translation must execute in under 15 microseconds per call."""
        raw = "[Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE] Xin chào em!"
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            clean_translation(raw, speaker="ren", source="Hello Liana!")
        elapsed = time.perf_counter() - start
        per_call_us = (elapsed / iterations) * 1e6
        assert per_call_us < 15.0, (
            f"clean_translation too slow: {per_call_us:.2f} us/call (limit: 15.0 us)"
        )

    def test_qa_checker_latency(self) -> None:
        """QAChecker.check_unit must execute in under 25 microseconds per call."""
        checker = QAChecker()
        cleaned = "Xin chào em!"
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            checker.check_unit("test", "Hello Liana!", None, cleaned, speaker="ren")
        elapsed = time.perf_counter() - start
        per_call_us = (elapsed / iterations) * 1e6
        assert per_call_us < 25.0, (
            f"QAChecker.check_unit too slow: {per_call_us:.2f} us/call (limit: 25.0 us)"
        )
