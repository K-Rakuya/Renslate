"""Tests for unattended_runner helper functions."""

from pathlib import Path
from unittest.mock import MagicMock

from scripts.unattended_runner import (
    detect_server_model,
    format_duration,
    query_db_status,
    render_progress_bar,
)


def test_format_duration() -> None:
    assert format_duration(0) == "--:--:--"
    assert format_duration(45) == "00m 45s"
    assert format_duration(125) == "02m 05s"
    assert format_duration(3665) == "01h 01m 05s"
    assert format_duration(float("nan")) == "--:--:--"


def test_render_progress_bar() -> None:
    bar_0 = render_progress_bar(0.0, width=10)
    assert "░░░░░░░░░░" in bar_0
    assert "  0.0%" in bar_0

    bar_50 = render_progress_bar(50.0, width=10)
    assert "█████░░░░░" in bar_50
    assert " 50.0%" in bar_50

    bar_100 = render_progress_bar(100.0, width=10)
    assert "██████████" in bar_100
    assert "100.0%" in bar_100


def test_query_db_status() -> None:
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchall.return_value = [
        ("pending", 500),
        ("translated", 250),
        ("approved", 50),
        ("failed", 10),
    ]

    status = query_db_status(mock_db)
    assert status == {
        "pending": 500,
        "translated": 250,
        "approved": 50,
        "failed": 10,
    }


def test_detect_server_model_fallback() -> None:
    # When endpoint cannot be reached, fallback to default name
    model = detect_server_model("http://127.0.0.1:9999", "default-fallback.gguf", timeout_s=0.1)
    assert model == "default-fallback.gguf"


def test_bootstrap_if_needed_auto_detect_and_idempotence(tmp_path: Path) -> None:
    from unittest.mock import patch

    from renslate.core.models import CharacterProfile, GlossaryEntry
    from renslate.core.project import Project
    from renslate.ui.runner import bootstrap_if_needed

    game_dir = tmp_path / "game"
    game_dir.mkdir()
    project = Project.create(tmp_path)
    db = project.open_db()

    # Pre-populate characters with empty role & speech_style and untranslated glossary
    db.save_character(CharacterProfile("c", "Charlotte", "female", role="", speech_style=""))
    db.save_glossary_entry(GlossaryEntry("Kingdom", "", notes="xuất hiện 5 lần"))

    with (
        patch("renslate.ui.runner.check_server_health", return_value=True),
        patch("renslate.ui.runner.LLMClient"),
        patch("renslate.ui.runner.bootstrap_characters_with_llm") as mock_boot_chars,
        patch("renslate.ui.runner.translate_glossary_with_llm") as mock_trans_gloss,
    ):
        mock_trans_gloss.return_value = [
            GlossaryEntry("Kingdom", "Vương quốc", notes="xuất hiện 5 lần")
        ]

        # Run 1: Auto-detection should trigger bootstrap when bootstrap=None
        bootstrap_if_needed(project, db, bootstrap=None)
        assert mock_boot_chars.call_count == 1
        assert mock_trans_gloss.call_count == 1
        assert db.get_meta("characters_bootstrapped") is not None
        assert db.get_meta("glossary_bootstrapped") is not None

        # Run 2: Subsequent run with same DB should NOT trigger LLM (0 calls)
        mock_boot_chars.reset_mock()
        mock_trans_gloss.reset_mock()
        bootstrap_if_needed(project, db, bootstrap=None)
        assert mock_boot_chars.call_count == 0
        assert mock_trans_gloss.call_count == 0

        # Run 3: Explicit skip (bootstrap=False) should NEVER trigger LLM
        db.execute("DELETE FROM meta WHERE key = ?", ("characters_bootstrapped",))
        bootstrap_if_needed(project, db, bootstrap=False)
        assert mock_boot_chars.call_count == 0
        assert mock_trans_gloss.call_count == 0

    db.close()
