"""Tests for unattended_runner helper functions."""

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
