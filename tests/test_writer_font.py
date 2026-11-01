"""Tests for Vietnamese font discovery, compatibility checking, and Ren'Py setup."""

from __future__ import annotations

from pathlib import Path

from renslate.cli import main
from renslate.core.project import Project
from renslate.renpy.fonts import (
    check_font_vietnamese_support,
    classify_font_style,
    discover_game_fonts,
    setup_vietnamese_support,
)


def test_classify_font_style() -> None:
    assert classify_font_style("Antique_Soul.otf") == "serif"
    assert classify_font_style("OldBridge_Serif.ttf") == "serif"
    assert classify_font_style("TimesNewRoman.ttf") == "serif"
    assert classify_font_style("Lora-Regular.ttf") == "serif"
    assert classify_font_style("CuteDoodle_Chicken.ttf") == "display"
    assert classify_font_style("ComicHand_Script.ttf") == "display"
    assert classify_font_style("AnimeScript.otf") == "display"
    assert classify_font_style("Arial.ttf") == "sans"
    assert classify_font_style("BeVietnamPro-Regular.ttf") == "sans"
    assert classify_font_style("DejaVuSans.ttf") == "sans"


def test_check_font_vietnamese_support_missing_file(tmp_path: Path) -> None:
    non_existent = tmp_path / "ghost.ttf"
    supported, missing_count, missing_chars = check_font_vietnamese_support(non_existent)
    assert not supported
    assert missing_count > 0
    assert len(missing_chars) == missing_count


def test_check_font_vietnamese_support_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.ttf"
    corrupt.write_bytes(b"not a real true type font data")
    supported, missing_count, _missing_chars = check_font_vietnamese_support(corrupt)
    assert not supported
    assert missing_count > 0


def test_discover_game_fonts(tmp_path: Path) -> None:
    root = tmp_path / "test_game"
    game_dir = root / "game"
    fonts_dir = game_dir / "fonts"
    fonts_dir.mkdir(parents=True)
    project = Project.create(root)

    # Dummy font files
    (fonts_dir / "Soul_Serif.otf").write_bytes(b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    (fonts_dir / "Dialogue_Sans.ttf").write_bytes(b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

    # Script referencing a font literal
    (game_dir / "script.rpy").write_text(
        'define e = Character("Eileen", what_font="fonts/InlineCustom.ttf")\n',
        encoding="utf-8",
    )

    detected = discover_game_fonts(project)
    rel_paths = {d.rel_path for d in detected}
    assert "fonts/Soul_Serif.otf" in rel_paths
    assert "fonts/Dialogue_Sans.ttf" in rel_paths
    assert "fonts/InlineCustom.ttf" in rel_paths


def test_setup_vietnamese_support_generation(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    fonts_dir = root / "game" / "fonts"
    fonts_dir.mkdir(parents=True)
    project = Project.create(root)

    # Add a dummy serif font and a dummy sans font lacking Vietnamese
    (fonts_dir / "Fantasy_Soul.otf").write_bytes(b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    (fonts_dir / "Modern_Sans.ttf").write_bytes(b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

    font_script = setup_vietnamese_support(
        project,
        font_file="CustomSans.ttf",
        serif_font="CustomSerif.ttf",
    )
    assert font_script.is_file()
    assert font_script.name == "renslate_setup.rpy"

    content = font_script.read_text(encoding="utf-8")
    assert "translate vietnamese python:" in content
    assert '"CustomSans.ttf",' in content
    assert '"CustomSerif.ttf",' in content
    assert "gui.text_font = _vi_sans" in content
    assert "config.font_replacement_map" in content
    assert "Fantasy_Soul.otf" in content
    assert "Modern_Sans.ttf" in content


def test_cli_font_command(tmp_path: Path) -> None:
    root = tmp_path / "game_cli"
    (root / "game").mkdir(parents=True)
    Project.create(root)

    ret = main(["font", "-p", str(root)])
    assert ret == 0

    setup_file = root / "game" / "tl" / "vietnamese" / "renslate_setup.rpy"
    assert setup_file.is_file()

    # Test --inspect flag
    ret_inspect = main(["font", "-p", str(root), "--inspect"])
    assert ret_inspect == 0
