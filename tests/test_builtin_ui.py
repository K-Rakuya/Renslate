"""Tests for renslate.builtin_ui."""

from __future__ import annotations

from renslate.builtin_ui import BUILTIN_UI_SEEDS, get_builtin_ui_translation


def test_known_seeds() -> None:
    assert get_builtin_ui_translation("Start") == "Bắt đầu"
    assert get_builtin_ui_translation("Quit") == "Thoát"
    assert get_builtin_ui_translation("Preferences") == "Cài đặt"
    assert get_builtin_ui_translation("Yes") == "Có"
    assert get_builtin_ui_translation("No") == "Không"
    assert (
        get_builtin_ui_translation("Are you sure you want to quit?")
        == "Bạn có chắc muốn thoát không?"
    )


def test_unknown_seed_returns_none() -> None:
    assert get_builtin_ui_translation("Unknown custom game string 12345") is None


def test_seed_count_is_substantial() -> None:
    assert len(BUILTIN_UI_SEEDS) >= 60
