"""Tests for vitrans.glossary."""

from __future__ import annotations

from pathlib import Path

from vitrans.glossary import (
    Glossary,
    mine_glossary_candidates,
    translate_glossary_with_llm,
)
from vitrans.models import GlossaryEntry


def test_glossary_matching_case_sensitive() -> None:
    g = Glossary(
        [
            GlossaryEntry("Apple", "Quả táo", case_sensitive=True),
            GlossaryEntry("apple", "công ty Apple", case_sensitive=True),
        ]
    )
    m1 = g.match_text("I eat an Apple today.")
    assert len(m1) == 1
    assert m1[0].translation == "Quả táo"

    m2 = g.match_text("He works at apple.")
    assert len(m2) == 1
    assert m2[0].translation == "công ty Apple"


def test_glossary_matching_case_insensitive_and_boundaries() -> None:
    g = Glossary(
        [
            GlossaryEntry("Mana", "Năng lượng", case_sensitive=False),
        ]
    )
    # Should match "mana" or "MANA"
    assert len(g.match_text("Restore 50 mana.")) == 1
    assert len(g.match_text("FULL MANA!")) == 1

    # Should NOT match inside word "Management"
    assert g.match_text("Management is hard.") == []


def test_longest_term_first_matching() -> None:
    g = Glossary(
        [
            GlossaryEntry("Council", "Hội đồng"),
            GlossaryEntry("Student Council", "Hội học sinh"),
        ]
    )
    matches = g.match_text("Meet the Student Council.")
    # Both terms match, but "Student Council" appears first in entries list
    assert matches[0].term == "Student Council"


def test_aliases_matching() -> None:
    g = Glossary(
        [
            GlossaryEntry(
                term="Magic Stone",
                translation="Ma Thạch",
                aliases=("Magic Stones", "Mana Stone"),
            )
        ]
    )
    assert len(g.match_text("Collect three Magic Stones.")) == 1
    assert len(g.match_text("Found a Mana Stone.")) == 1


def test_mine_glossary_candidates() -> None:
    texts = [
        "The Sacred Sword was forged in ancient fires.",
        "Take the Sacred Sword to the altar.",
        "We need the Sacred Sword to defeat the monster.",
        "Just a normal sentence with no special names.",
    ]
    candidates = mine_glossary_candidates(texts, min_count=2)
    terms = [c[0] for c in candidates]
    assert "Sacred Sword" in terms


def test_save_and_load_file(tmp_path: Path) -> None:
    g = Glossary(
        [
            GlossaryEntry("Hero", "Anh hùng", aliases=("Heroes",), notes="Nhân vật chính"),
        ]
    )
    path = tmp_path / "glossary.json"
    g.save_file(path)

    loaded = Glossary.load_file(path)
    assert len(loaded.entries) == 1
    assert loaded.entries[0].term == "Hero"
    assert loaded.entries[0].translation == "Anh hùng"
    assert loaded.entries[0].aliases == ("Heroes",)


def test_translate_glossary_with_llm() -> None:
    class MockLLM:
        def complete_json(self, system: str, user: str) -> tuple[dict[str, str], int, int]:
            return {"Sect": "Tông môn", "Elder": "Trưởng lão"}, 50, 20

    entries = [
        GlossaryEntry("Sect", ""),
        GlossaryEntry("Elder", ""),
        GlossaryEntry("Hero", "Anh hùng"),  # Already translated
    ]
    updated = translate_glossary_with_llm(entries, MockLLM())
    assert len(updated) == 3
    assert updated[0].translation == "Tông môn"
    assert updated[1].translation == "Trưởng lão"
    assert updated[2].translation == "Anh hùng"


def test_is_valid_glossary_term() -> None:
    from vitrans.glossary import is_valid_glossary_term

    # Rejects pronouns and clause starters
    assert not is_valid_glossary_term("Because I")
    assert not is_valid_glossary_term("If I")
    assert not is_valid_glossary_term("Should I")
    assert not is_valid_glossary_term("What")
    assert not is_valid_glossary_term("This")
    assert not is_valid_glossary_term("You")
    assert not is_valid_glossary_term("She")
    assert not is_valid_glossary_term("Damn")
    assert not is_valid_glossary_term("Fuck")
    assert not is_valid_glossary_term("ZCOOLQingKeHuangYou")
    assert not is_valid_glossary_term("custom_font.ttf")

    # Excluded character names
    assert not is_valid_glossary_term("Layla", exclude_terms={"layla", "liana"})

    # Valid terms
    assert is_valid_glossary_term("Sect")
    assert is_valid_glossary_term("Inner Sanctum")
    assert is_valid_glossary_term("Force Field")
    assert is_valid_glossary_term("Beast")


def test_clean_glossary_sanitizes_and_prioritizes() -> None:
    from vitrans.glossary import clean_glossary

    raw_entries = [
        GlossaryEntry("Because I", "", notes="xuất hiện 20 lần"),
        GlossaryEntry("What", "", notes="xuất hiện 50 lần"),
        GlossaryEntry("Layla", "Layla", notes="xuất hiện 99 lần"),
        GlossaryEntry("My Inner Sanctum", "", notes="xuất hiện 47 lần"),
        GlossaryEntry("Sect", "", case_sensitive=True, notes="xuất hiện 174 lần"),
        GlossaryEntry("Beast", "", case_sensitive=True, notes="xuất hiện 10 lần"),
        GlossaryEntry("ZCOOLQingKeHuangYou", "", notes="xuất hiện 5 lần"),
    ]

    cleaned = clean_glossary(raw_entries, exclude_terms={"layla"})
    terms = [e.term for e in cleaned]

    # Stopwords, pronouns, font names, character names purged
    assert "Because I" not in terms
    assert "What" not in terms
    assert "Layla" not in terms
    assert "ZCOOLQingKeHuangYou" not in terms

    # Prefix cleaned: "My Inner Sanctum" -> "Inner Sanctum"
    assert "Inner Sanctum" in terms
    assert "My Inner Sanctum" not in terms

    # Frequency descending order: Sect (174) -> Inner Sanctum (47) -> Beast (10)
    assert terms == ["Sect", "Inner Sanctum", "Beast"]

    # Invariants: case_sensitive=False and plural aliases added
    sect_entry = next(e for e in cleaned if e.term == "Sect")
    assert not sect_entry.case_sensitive
    assert "Sects" in sect_entry.aliases

