"""Tests for vitrans.qa."""

from __future__ import annotations

from vitrans.mask import mask
from vitrans.models import GlossaryEntry
from vitrans.qa import QAChecker, clean_translation


def test_clean_translation() -> None:
    # Model prepending speaker
    assert clean_translation('Eileen: "Xin chào!"', speaker="e") == "Xin chào!"
    assert clean_translation("Eileen: Xin chào bạn!", speaker="e") == "Xin chào bạn!"

    # Outer quotes removal
    assert clean_translation('"Bản dịch nằm trong ngoặc"') == "Bản dịch nằm trong ngoặc"

    # Preserves inner quotes
    assert clean_translation('Nói "cheese" đi!') == 'Nói "cheese" đi!'

    # Conversational prefix stripping
    assert clean_translation('Bản dịch: "Xin chào!"') == "Xin chào!"
    assert clean_translation("Translation: Hello!") == "Hello!"

    # Hallucinated bracketed speaker labels with or without colon
    assert (
        clean_translation('[Dyna] "Lực trường sẽ sớm bị phá vỡ."', speaker="Dyna")
        == "Lực trường sẽ sớm bị phá vỡ."
    )
    assert clean_translation('[MC] "Hả?"', speaker="Dyna") == "Hả?"
    assert clean_translation('[Female Deity] "..."', speaker="Dyna") == "..."
    assert (
        clean_translation('[Dyna] [Dyna] "Làm sao thú lại ở đây?"', speaker="Dyna")
        == "Làm sao thú lại ở đây?"
    )
    assert clean_translation('[Targon] "*Thở dài*, Mấy đứa trẻ…"') == "*Thở dài*, Mấy đứa trẻ…"
    assert clean_translation('(Jake): "Cẩn thận!"') == "Cẩn thận!"

    # Speaker metadata peeling (preventing [Người nói: ...] leaks)
    assert clean_translation("[Người nói: MC]", speaker="MC", source="Woah!") == ""
    assert clean_translation("[Người nói: Layla]", speaker="Layla", source="Exactly!") == ""
    assert clean_translation("[MC] [Người nói: MC]", speaker="MC", source="Woah!") == ""
    assert clean_translation("[MC]", speaker="MC", source="Woah!") == ""
    assert clean_translation("[Layla]", speaker=None, source="Woah!") == ""
    assert (
        clean_translation(
            "[Người nói: MC] Thật là tuyệt vời!",
            speaker="MC",
            source="Woah! This is great.",
        )
        == "Thật là tuyệt vời!"
    )

    # Preserving parenthesized and bracketed non-speaker content
    assert clean_translation("(statement)", source="(statement)") == "(statement)"
    assert clean_translation("(câu lệnh)", source="(statement)") == "(câu lệnh)"
    assert clean_translation("(tag)", source="(tag)") == "(tag)"
    assert clean_translation("[...]", source="[...]") == "[...]"
    assert (
        clean_translation("(Tôi nên làm gì?)", source="(What should I do?)")
        == "(Tôi nên làm gì?)"
    )
    assert clean_translation("(statement)") == "(statement)"

    # Unicode normalization to NFC
    import unicodedata

    nfd_text = unicodedata.normalize("NFD", "Tiếng Việt chuẩn xác")
    cleaned = clean_translation(nfd_text)
    assert cleaned == "Tiếng Việt chuẩn xác"
    assert unicodedata.is_normalized("NFC", cleaned)


def test_qa_empty_and_markup_errors() -> None:
    qa = QAChecker()
    # Empty
    issues = qa.check_unit("d:1", "Hello", None, "")
    assert any(i.code == "empty_target" for i in issues)

    # Bad markup
    issues = qa.check_unit("d:2", "Hello", None, "Xin [chào")
    assert any(i.code == "bad_markup" for i in issues)


def test_qa_placeholder_validation() -> None:
    qa = QAChecker()
    masked = mask("Hello [player_name]!")

    # Missing placeholder
    issues = qa.check_unit("d:1", "Hello [player_name]!", masked, "Xin chào bạn!")
    assert any("missing_placeholder" in i.code for i in issues)

    # Correct placeholder
    issues_ok = qa.check_unit("d:1", "Hello [player_name]!", masked, "Xin chào ⟦V1⟧!")
    assert not any(i.severity == "error" for i in issues_ok)


def test_qa_identical_verbatim() -> None:
    qa = QAChecker()
    issues = qa.check_unit("d:1", "Good morning to you", None, "Good morning to you")
    assert any(i.code == "identical_to_source" for i in issues)


def test_qa_language_leakage() -> None:
    qa = QAChecker()
    # CJK characters
    issues_cjk = qa.check_unit("d:1", "Test", None, "Xin chào 世界")
    assert any(i.code == "cjk_leakage" for i in issues_cjk)

    # Heavy English leakage
    leakage_text = "This is a sentence with only the and of you"
    issues_eng = qa.check_unit("d:2", "Some source", None, leakage_text)
    assert any(i.code == "english_leakage" for i in issues_eng)


def test_qa_glossary_compliance() -> None:
    qa = QAChecker()
    glossary = [GlossaryEntry("Excalibur", "Thánh Kiếm")]
    # Violated
    issues_violated = qa.check_unit(
        "d:1", "Draw the Excalibur!", None, "Hãy rút thanh kiếm ra!", glossary_matches=glossary
    )
    assert any(i.code == "glossary_violation" for i in issues_violated)

    # Complied
    issues_ok = qa.check_unit(
        "d:1", "Draw the Excalibur!", None, "Hãy rút Thánh Kiếm ra!", glossary_matches=glossary
    )
    assert not any(i.code == "glossary_violation" for i in issues_ok)


def test_qa_glossary_leakage() -> None:
    qa = QAChecker()
    glossary = [GlossaryEntry("Sect", "Tông môn", aliases=("sects",))]

    # Leaked English word
    issues = qa.check_unit(
        "d:1",
        "Near our sect.",
        None,
        "Gần Sect của chúng ta.",
        glossary_matches=glossary,
    )
    assert any(i.code == "untranslated_glossary_leakage" for i in issues)

    # Properly translated
    issues_clean = qa.check_unit(
        "d:1",
        "Near our sect.",
        None,
        "Gần Tông môn của chúng ta.",
        glossary_matches=glossary,
    )
    assert not any(i.code == "untranslated_glossary_leakage" for i in issues_clean)


def test_qa_pronoun_register_clash() -> None:
    qa = QAChecker()

    # 1st person clash: ta vs tôi
    issues_1st = qa.check_unit(
        "d:1", "Source text", None, "Ta nghĩ rằng tôi sẽ không đi cùng."
    )
    assert any(i.code == "pronoun_register_clash" for i in issues_1st)

    # 2nd person clash: ngươi vs cậu
    issues_2nd = qa.check_unit(
        "d:2", "Source text", None, "Ngươi nên cẩn thận nhé, cậu hiểu chưa?"
    )
    assert any(i.code == "pronoun_register_clash" for i in issues_2nd)

    # Coherent line: no clash
    issues_clean = qa.check_unit(
        "d:3", "Source text", None, "Ta sẽ dạy ngươi cách tu luyện."
    )
    assert not any(i.code == "pronoun_register_clash" for i in issues_clean)


def test_repair_glossary_leakage() -> None:
    from vitrans.qa import repair_glossary_leakage

    glossary = [
        GlossaryEntry("Sect", "Tông môn", aliases=("sects",)),
        GlossaryEntry("Inner Sanctum", "Nội Điện"),
    ]

    # Replaces leaked terms while preserving markup and placeholders
    text = "Làm sao lại vào Sect và Inner Sanctum, [player_name] hay ⟦V1⟧?"
    repaired = repair_glossary_leakage(text, glossary)
    assert repaired == "Làm sao lại vào Tông môn và Nội Điện, [player_name] hay ⟦V1⟧?"

    # Replaces plural aliases
    text_plural = "All sects must unite."
    repaired_plural = repair_glossary_leakage(text_plural, glossary)
    assert repaired_plural == "All Tông môn must unite."


def test_qa_speaker_tag_leakage() -> None:
    qa = QAChecker()

    # Explicit metadata leakage
    issues = qa.check_unit(
        "d:1",
        "Woah! This is great.",
        None,
        "[Người nói: MC] Thật tuyệt!",
        speaker="MC",
    )
    assert any(i.code == "speaker_tag_leakage" and i.severity == "error" for i in issues)

    # Standalone speaker bracket
    issues = qa.check_unit(
        "d:2",
        "Woah! This is great.",
        None,
        "[MC]",
        speaker="MC",
    )
    assert any(i.code == "speaker_tag_leakage" and i.severity == "error" for i in issues)

    # Speaker name without brackets echoing speaker
    issues = qa.check_unit(
        "d:3",
        "Woah! This is great.",
        None,
        "MC:",
        speaker="MC",
    )
    assert any(i.code == "speaker_tag_leakage" and i.severity == "error" for i in issues)

    # Valid dialogue with Ren'Py variable is NOT flagged
    issues_ok = qa.check_unit(
        "d:4",
        "[player_name]! Hello!",
        None,
        "[player_name]! Xin chào bạn!",
        speaker="MC",
    )
    assert not any(i.code == "speaker_tag_leakage" for i in issues_ok)


