"""Tests for StageTracker and Ren'Py AST stage/sprite tracking."""

from pathlib import Path

from renslate.core.models import CharacterProfile, Status, Unit, UnitKind
from renslate.domain.models import SpriteState, StageSnapshot
from renslate.engine.context import ResolvedInterlocutor
from renslate.engine.context.prompt_frame import format_rich_dialogue_line
from renslate.renpy.extract import _scan_raw_script
from renslate.renpy.parser import TranslationScanner
from renslate.renpy.stage import StageTracker, build_stage_index


def test_stage_tracker_scene_and_show() -> None:
    tracker = StageTracker()
    assert tracker.scene_bg == ""
    assert len(tracker.active_sprites) == 0

    assert tracker.process_line("scene bg classroom with dissolve")
    assert tracker.scene_bg == "bg classroom"
    assert len(tracker.active_sprites) == 0

    assert tracker.process_line("show eileen happy at left with dissolve")
    sprites = tracker.active_sprites
    assert "eileen" in sprites
    assert sprites["eileen"].tag == "eileen"
    assert sprites["eileen"].expression == "happy"
    assert sprites["eileen"].position == "left"


def test_stage_tracker_expression_and_position_preservation() -> None:
    tracker = StageTracker()
    tracker.process_line("show eileen happy at left")

    # Update expression only
    tracker.process_line("show eileen blush")
    eileen = tracker.active_sprites["eileen"]
    assert eileen.expression == "blush"
    assert eileen.position == "left"

    # Update position only
    tracker.process_line("show eileen at right")
    eileen = tracker.active_sprites["eileen"]
    assert eileen.expression == "blush"
    assert eileen.position == "right"


def test_stage_tracker_multiple_sprites_and_hide() -> None:
    tracker = StageTracker()
    tracker.process_line("show alice happy at left")
    tracker.process_line("show bob serious at right")

    snap = tracker.capture()
    assert snap.active_tags == {"alice", "bob"}
    assert len(snap.active_sprites) == 2

    # Hide alice
    tracker.process_line("hide alice")
    snap2 = tracker.capture()
    assert snap2.active_tags == {"bob"}
    assert len(snap2.active_sprites) == 1
    assert snap2.active_sprites[0].tag == "bob"

    # Scene reset
    tracker.process_line("scene bg night")
    snap3 = tracker.capture()
    assert snap3.scene_bg == "bg night"
    assert len(snap3.active_sprites) == 0


def test_stage_tracker_ignores_non_sprites() -> None:
    tracker = StageTracker()
    assert not tracker.process_line('show text "Chapter One"')
    assert not tracker.process_line("show screen main_menu")
    assert not tracker.process_line("show layer master")
    assert not tracker.process_line("regular python = 1")
    assert len(tracker.active_sprites) == 0


def test_stage_prompt_frame_emotion_integration() -> None:
    stage_snap = StageSnapshot(
        scene_bg="bg park",
        active_sprites=(
            SpriteState(tag="alice", expression="blush", position="left"),
            SpriteState(tag="bob", expression="smirk", position="right"),
        ),
    )

    unit = Unit(
        uid="d:1",
        kind=UnitKind.DIALOGUE,
        seq=1,
        source="I really like you.",
        status=Status.PENDING,
        speaker="alice",
        stage=stage_snap,
    )

    resolved = ResolvedInterlocutor(
        speaker="alice",
        listener="bob",
        resolution_method="direct_vocative",
    )

    char_map = {
        "alice": CharacterProfile(tag="alice", display_name="Alice", gender="female"),
        "bob": CharacterProfile(tag="bob", display_name="Bob", gender="male"),
    }

    formatted = format_rich_dialogue_line(unit, resolved, char_map, None)
    assert "Alice" in formatted
    assert "Bob" in formatted
    assert "Biểu cảm: e thẹn (blush)" in formatted


def test_build_stage_index(tmp_path: Path) -> None:
    script_file = tmp_path / "script.rpy"
    script_file.write_text(
        "label start:\n"
        "    scene bg meadow\n"
        "    show clara happy at center\n"
        '    clara "The sun is warm."\n'
        "    show clara blush\n"
        '    clara "Do you want to walk together?"\n',
        encoding="utf-8",
    )

    stage_index = build_stage_index(tmp_path)
    assert "script.rpy:4" in stage_index
    assert "script.rpy:6" in stage_index

    snap4 = stage_index["script.rpy:4"]
    assert snap4.scene_bg == "bg meadow"
    assert snap4.active_tags == {"clara"}
    assert snap4.active_sprites[0].expression == "happy"

    snap6 = stage_index["script.rpy:6"]
    assert snap6.active_sprites[0].expression == "blush"


def test_scan_raw_script_attaches_stage() -> None:
    lines = [
        "label chapter1:",
        "    scene bg library",
        "    show elena smile at left",
        '    elena "Books are wonderful."',
    ]
    scanner = TranslationScanner("vi")
    _scan_raw_script("story.rpy", lines, scanner, {}, {})

    assert len(scanner.units) == 1
    unit = scanner.units[0]
    assert unit.stage is not None
    assert unit.stage.scene_bg == "bg library"
    assert unit.stage.active_tags == {"elena"}
    assert unit.stage.active_sprites[0].expression == "smile"
