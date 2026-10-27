"""Tests for Vietnamese font and language setup in writer.py."""

from __future__ import annotations

from pathlib import Path

from renslate.project import Project
from renslate.writer import setup_vietnamese_support


def test_setup_vietnamese_support(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)

    font_script = setup_vietnamese_support(project, font_file="CustomFont.ttf")
    assert font_script.is_file()
    assert font_script.name == "renslate_setup.rpy"

    content = font_script.read_text(encoding="utf-8")
    assert "translate vietnamese python:" in content
    assert "CustomFont.ttf" in content
    assert "gui.text_font" in content
