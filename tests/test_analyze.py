"""Tests for vitrans.analyze."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vitrans.analyze import (
    bootstrap_characters_with_llm,
    extract_character_definitions,
    infer_character_gender_nlp,
    infer_character_role_nlp,
    run_analysis,
    segment_scenes,
)
from vitrans.db import Database
from vitrans.models import Status, Unit, UnitKind
from vitrans.project import Project


def test_extract_character_definitions(tmp_path: Path) -> None:
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    script = (
        'define e = Character("Eileen")\n'
        'define m = DynamicCharacter("mc_name")\n'
        'define s = Character(_("Sylvie"))\n'
    )
    (game_dir / "script.rpy").write_text(script, encoding="utf-8")

    chars = extract_character_definitions(tmp_path)
    assert chars.get("e") == "Eileen"
    assert chars.get("m") == "mc_name"
    assert chars.get("s") == "Sylvie"


def test_segment_scenes_groups_by_label_and_gap(tmp_path: Path) -> None:
    db = Database.open(tmp_path / "test.db", create=True)
    db.save_unit(Unit("d:1", UnitKind.DIALOGUE, 1, "Hello", Status.PENDING, label="start"))
    db.save_unit(Unit("d:2", UnitKind.DIALOGUE, 2, "World", Status.PENDING, label="start"))
    db.save_unit(Unit("d:3", UnitKind.DIALOGUE, 3, "Chapter 1", Status.PENDING, label="ch1"))

    scenes = segment_scenes(db)
    assert len(scenes) == 2
    assert scenes[0].label == "start"
    assert (scenes[0].first_seq, scenes[0].last_seq) == (1, 2)
    assert scenes[1].label == "ch1"


def test_run_analysis_pipeline(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    (root / "game" / "script.rpy").write_text('define e = Character("Eileen")', encoding="utf-8")

    project = Project.create(root)
    with project.open_db() as db:
        db.save_unit(
            Unit(
                "d:1",
                UnitKind.DIALOGUE,
                1,
                "Welcome to the Grand Castle!",
                Status.PENDING,
                speaker="e",
                label="intro",
            )
        )
        db.save_unit(
            Unit(
                "d:2",
                UnitKind.DIALOGUE,
                2,
                "The Grand Castle is ancient.",
                Status.PENDING,
                speaker="e",
                label="intro",
            )
        )

        stats = run_analysis(project, db)

    assert stats["characters"] >= 1
    assert stats["scenes"] >= 1
    assert (project.work_dir / "characters.json").is_file()
    assert (project.work_dir / "glossary.json").is_file()


def test_infer_character_role_and_gender_nlp() -> None:
    # 1. Role inference
    assert infer_character_role_nlp("mc", 50, 100) == "Nhân vật chính (Protagonist)"
    assert infer_character_role_nlp("narrator", 1, 100) == "Nhân vật chính (Protagonist)"
    assert infer_character_role_nlp("layla", 10, 100) == "Nhân vật chính tuyến (Main Cast)"
    assert infer_character_role_nlp("guard", 1, 100) == "Nhân vật phụ (Supporting Cast)"
    assert infer_character_role_nlp("villager", 0, 100) == ""

    # 2. Gender inference
    sources_fem = [
        "Liana is a talented mage. She cast her spell gracefully.",
        "I looked at her, and she smiled back at me.",
    ]
    assert infer_character_gender_nlp("Liana", sources_fem) == "female"

    sources_masc = [
        "Targon stood his ground. He grabbed his battleaxe firmly.",
        "The boy nodded to him with deep respect.",
    ]
    assert infer_character_gender_nlp("Targon", sources_masc) == "male"

    # Inherent keyword gender
    assert infer_character_gender_nlp("Female Deity", []) == "female"
    assert infer_character_gender_nlp("King Arthur", []) == "male"


def test_bootstrap_characters_with_llm(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)

    class MockLLM:
        def complete_json(
            self, system: str, user: str, *args: Any, **kwargs: Any
        ) -> tuple[dict[str, Any], int, int]:
            return (
                {
                    "layla": {
                        "gender": "female",
                        "role": "Sư phụ / Tông chủ",
                        "speech_style": "Uy nghiêm, điềm đạm",
                        "pronoun_map": {"self": "ta", "target": "con"},
                    }
                },
                60,
                30,
            )

    with project.open_db() as db:
        db.save_unit(
            Unit(
                "d:1",
                UnitKind.DIALOGUE,
                1,
                "I must protect the sect at all costs.",
                Status.PENDING,
                speaker="layla",
            )
        )
        chars = bootstrap_characters_with_llm(project, db, MockLLM())

    assert "layla" in chars
    p = chars["layla"]
    assert p.gender == "female"
    assert p.role == "Sư phụ / Tông chủ"
    assert p.speech_style == "Uy nghiêm, điềm đạm"
    assert p.pronoun_map.get("self") == "ta"
