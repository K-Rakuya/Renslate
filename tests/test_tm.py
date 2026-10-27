"""Tests for vitrans.tm."""

from __future__ import annotations

from pathlib import Path

import pytest

from vitrans.db import Database
from vitrans.models import Origin
from vitrans.tm import TranslationMemory, normalize_source


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database.open(tmp_path / "test.db", create=True)


def test_normalize_source() -> None:
    assert normalize_source("  Hello   World!  ") == "Hello World!"
    assert normalize_source("tiếng   việt") == "tiếng việt"


def test_tm_exact_lookup(db: Database) -> None:
    tm = TranslationMemory(db)
    assert tm.lookup_exact("Hello") is None

    tm.record("Hello", "Xin chào", origin=Origin.MANUAL)
    assert tm.lookup_exact("Hello") == "Xin chào"
    assert tm.lookup_exact("  Hello  ") == "Xin chào"


def test_tm_fuzzy_lookup(db: Database) -> None:
    tm = TranslationMemory(db)
    tm.record("Good morning everyone!", "Chào buổi sáng mọi người!")
    tm.record("See you tomorrow evening.", "Hẹn gặp lại tối mai.")

    # High similarity query
    matches = tm.lookup_fuzzy("Good morning everyone", threshold=0.85)
    assert len(matches) == 1
    assert matches[0].source == "Good morning everyone!"
    assert matches[0].target == "Chào buổi sáng mọi người!"
    assert matches[0].score >= 0.90

    # Low similarity query -> should not match threshold
    assert tm.lookup_fuzzy("Random different sentence", threshold=0.85) == []


def test_tm_reload_persists(db: Database) -> None:
    tm = TranslationMemory(db)
    tm.record("Save Game", "Lưu game")

    # Re-instantiate from same DB
    tm2 = TranslationMemory(db)
    assert tm2.lookup_exact("Save Game") == "Lưu game"
