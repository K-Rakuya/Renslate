"""Unit tests for UI and special screen text extraction in Renslate."""

from __future__ import annotations

from pathlib import Path

from renslate.core.models import Origin, Status, UnitKind
from renslate.core.project import Project
from renslate.renpy.builtin_ui import get_builtin_ui_translation
from renslate.renpy.extract import (
    _has_human_text,
    _strip_outer_tags,
    extract,
    sync_project_ui_strings,
)
from renslate.renpy.parser import TranslationScanner
from renslate.renpy.writer import apply_translations


def test_has_human_text() -> None:
    assert _has_human_text("People")
    assert _has_human_text("{size=60}CHAPTER I{/size}")
    assert _has_human_text("Good Work!")
    assert _has_human_text("Hit 'SPACE' to unlock")
    assert not _has_human_text("")
    assert not _has_human_text("   ")
    assert not _has_human_text("[player_name]")
    assert not _has_human_text("[config.version!t]")
    assert not _has_human_text("gui/button/idle.png")
    assert not _has_human_text("sound/click.ogg")
    assert not _has_human_text("{size=50}")


def test_strip_outer_tags() -> None:
    assert _strip_outer_tags("People") == "People"
    assert _strip_outer_tags("{size=90}{font=fonts/TulpenOne-Regular.ttf}RETURN{/font}") == "RETURN"
    assert (
        _strip_outer_tags("{size=60}{font=f.otf}CHAPTER I: FALL{/font}{/size}") == "CHAPTER I: FALL"
    )
    assert _strip_outer_tags("{color=#cb0000}{size=150}DEAD{/size}{/color}") == "DEAD"


def test_builtin_ui_seeds_enhancements() -> None:
    assert get_builtin_ui_translation("Return") == "Quay lại"
    assert get_builtin_ui_translation("RETURN") == "QUAY LẠI"
    assert get_builtin_ui_translation("End Replay") == "Kết thúc xem lại"
    assert get_builtin_ui_translation("END REPLAY") == "KẾT THÚC XEM LẠI"
    assert get_builtin_ui_translation("Gallery") == "Thư viện ảnh"
    assert get_builtin_ui_translation("GALLERY") == "THƯ VIỆN ẢNH"

    # Preserving outer tags
    tagged = "{size=90}{font=fonts/TulpenOne-Regular.ttf}RETURN{/font}"
    expected = "{size=90}{font=fonts/TulpenOne-Regular.ttf}QUAY LẠI{/font}"
    assert get_builtin_ui_translation(tagged) == expected


def test_sync_project_ui_strings_and_extraction(tmp_path: Path) -> None:
    root = tmp_path / "game_project"
    game_dir = root / "game"
    tl_dir = game_dir / "tl" / "vietnamese"
    tl_dir.mkdir(parents=True)

    # Create dummy base script
    (game_dir / "script.rpy").write_text(
        """\
label start:
    show text "{size=60}{font=f.otf}CHAPTER I: AWAKENING{/font}{/size}"
    e "Hello world"
    return
""",
        encoding="utf-8",
    )

    # Create custom screen
    (game_dir / "screens.rpy").write_text(
        """\
screen custom_menu():
    vbox:
        textbutton "People" action NullAction()
        textbutton "Locations" action NullAction()
        textbutton "{size=90}{font=f.ttf}RETURN{/font}" action Return()
        text "Unlocked Pin"
        text "[var_only]"
""",
        encoding="utf-8",
    )

    # Create dummy tl file
    (tl_dir / "script.rpy").write_text(
        """\
translate vietnamese start_12345678:
    # e "Hello world"
    e "Hello world"
""",
        encoding="utf-8",
    )

    project = Project.create(root, target_lang="vietnamese")
    scanner = TranslationScanner("vietnamese")
    script_lines = (tl_dir / "script.rpy").read_text(encoding="utf-8").splitlines()
    scanner.scan_file("tl/vietnamese/script.rpy", script_lines)

    # Sync UI strings
    added = sync_project_ui_strings(project, scanner)
    assert added > 0

    ui_strings_file = tl_dir / "ui_strings.rpy"
    assert ui_strings_file.is_file()
    content = ui_strings_file.read_text(encoding="utf-8")
    assert 'old "People"' in content
    assert 'old "Locations"' in content
    assert 'old "RETURN"' in content
    assert 'old "Unlocked Pin"' in content
    assert 'old "CHAPTER I: AWAKENING"' in content
    assert "[var_only]" not in content

    # Test idempotency
    added_second = sync_project_ui_strings(project, scanner)
    assert added_second == 0

    # Test full extract
    with project.open_db() as db:
        report = extract(project, db)
        assert report.created > 0
        units = db.units()
        ui_units = [u for u in units.values() if u.kind == UnitKind.UI]
        assert len(ui_units) >= 5

        # Translate one UI unit and verify apply
        return_unit = next(u for u in ui_units if u.source == "RETURN")
        db.update_translation(
            return_unit.uid, target="QUAY LẠI", origin=Origin.MANUAL, status=Status.TRANSLATED
        )

        apply_report = apply_translations(project, db)
        assert apply_report.files_changed > 0
        assert apply_report.lines_written > 0

        # Verify translation in file
        updated_content = ui_strings_file.read_text(encoding="utf-8")
        assert 'new "QUAY LẠI"' in updated_content
