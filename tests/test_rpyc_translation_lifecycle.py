"""End-to-end verification test for bytecode-only (.rpyc) translation lifecycle.

Validates the complete pipeline from pure .rpyc files (with no .rpy source files):
1. Direct .rpyc character definition extraction from Slot 1 AST statements.
2. Translation skeleton synthesis from Slot 2 TranslateSay statements.
3. Unit extraction into the SQLite WAL database.
4. Autonomous analysis: statistical NLP gender inference, speaker interaction network,
   reciprocal pairwise pronoun mapping (anh - em), scene segmentation, and glossary mining.
5. Idempotent bootstrap verification: ensure re-running does not duplicate.
6. Translation batch execution: verifies relational addressing prompts, scene context injection,
   tag/variable masking reconciliation.
7. Final application to disk (.rpy) with syntax preservation.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from unittest.mock import patch

from renslate.core.config import Config
from renslate.core.models import Status
from renslate.core.project import Project
from renslate.engine.analyze import extract_character_definitions, run_analysis
from renslate.engine.translate import TranslationPipeline
from renslate.renpy.extract import extract
from renslate.renpy.rpyc import (
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    Define,
    TranslateSay,
    VirtualPyCode,
    write_rpyc,
)
from renslate.renpy.sdk import generate_translations
from renslate.renpy.writer import apply_translations
from renslate.ui.runner import bootstrap_if_needed


def _create_bytecode_game(game_dir: Path) -> Path:
    game_dir.mkdir(parents=True, exist_ok=True)
    rpyc_file = game_dir / "story.rpyc"

    # Slot 1: Character definitions in Python AST
    char_e = Define(
        varname="e",
        code=VirtualPyCode('Character("Eileen", color="#c8ffc8")'),
    )
    char_mc = Define(
        varname="mc",
        code=VirtualPyCode('Character("Protagonist")'),
    )
    slot1_stmts = [char_e, char_mc]

    # Slot 2: Dialogue statements between Protagonist (mc) and Heroine (e)
    say1 = TranslateSay()
    say1.identifier = "story_01"
    say1.who = "mc"
    say1.what = "Are you ready for the mission, my lady Eileen?"
    say1.filename = "game/story.rpy"
    say1.linenumber = 10

    say2 = TranslateSay()
    say2.identifier = "story_02"
    say2.who = "e"
    say2.what = "Yes, but beware of the {color=#ff0000}Danger{/color} ahead!"
    say2.filename = "game/story.rpy"
    say2.linenumber = 12

    say3 = TranslateSay()
    say3.identifier = "story_03"
    say3.who = "mc"
    say3.what = "We have score [score] points already."
    say3.filename = "game/story.rpy"
    say3.linenumber = 14

    say4 = TranslateSay()
    say4.identifier = "story_04"
    say4.who = "e"
    say4.what = "The Shadow Guild will not stop us."
    say4.filename = "game/story.rpy"
    say4.linenumber = 16

    slot2_stmts = [say1, say2, say3, say4]

    slots: dict[int, bytes] = {
        SLOT_STATEMENTS_PRE: pickle.dumps((None, slot1_stmts)),
        SLOT_STATEMENTS_POST: pickle.dumps((None, slot2_stmts)),
    }
    write_rpyc(rpyc_file, slots)
    return rpyc_file


def test_pure_rpyc_e2e_translation_lifecycle(tmp_path: Path) -> None:
    project_root = tmp_path / "game_project"
    game_dir = project_root / "game"
    _create_bytecode_game(game_dir)

    # Confirm there are zero .rpy scripts in game_dir
    assert not list(game_dir.glob("*.rpy"))

    # Initialize project
    project = Project.create(project_root)

    # -------------------------------------------------------------------------
    # Step 1: Character Extraction directly from .rpyc bytecode
    # -------------------------------------------------------------------------
    chars = extract_character_definitions(project.root)
    assert "e" in chars
    assert chars["e"] == "Eileen"
    assert "mc" in chars
    assert chars["mc"] == "Protagonist"

    # -------------------------------------------------------------------------
    # Step 2: Skeleton synthesis & unit extraction
    # -------------------------------------------------------------------------
    gen_out = generate_translations(None, project.root, "vietnamese", fallback_rpyc=True)
    assert "bytecode .rpyc" in gen_out
    tl_story_rpy = project.root / "game" / "tl" / "vietnamese" / "story.rpy"
    assert tl_story_rpy.is_file()

    with project.open_db() as db:
        report = extract(project, db)
        assert report.files == 1
        assert report.units == 4
        assert len(db.pending_units()) == 4

        units = db.all_units()
        uids = [u.uid for u in units]
        assert "d:story_01" in uids
        assert "d:story_02" in uids
        assert "d:story_03" in uids
        assert "d:story_04" in uids

    # -------------------------------------------------------------------------
    # Step 3: Analysis: Characters, Reciprocal Addressing, Scenes, and Glossary
    # -------------------------------------------------------------------------
    with project.open_db() as db:
        stats = run_analysis(project, db, mine_glossary=True, enrich_llm=False)
        assert stats["characters"] == 2
        assert stats["scenes"] >= 1

        char_profiles = db.get_characters()
        p_e = char_profiles.get("e")
        p_mc = char_profiles.get("mc")
        assert p_e is not None
        assert p_mc is not None

        # Statistical NLP inference identifies Eileen as female
        assert p_e.gender == "female"
        assert p_e.display_name == "Eileen"

        # Protagonist identification
        assert "Protagonist" in p_mc.role

        # Relational pronoun matrix inferred from reciprocal dialogue interactions
        assert p_mc.pronoun_map.get("e") == "anh - em"
        assert p_e.pronoun_map.get("mc") == "em - anh"

        # Scene segmentation
        scenes = db.get_scenes()
        assert len(scenes) >= 1
        assert "prologue" in scenes[0].label or "Cảnh" in scenes[0].summary

        # Meta key set
        assert db.get_meta("analysis_completed") is not None

    # -------------------------------------------------------------------------
    # Step 4: Idempotence of bootstrap_if_needed
    # -------------------------------------------------------------------------
    with project.open_db() as db:
        # First call: already completed, should be a fast no-op
        bootstrap_if_needed(project, db, bootstrap=False)
        assert len(db.get_characters()) == 2
        assert len(db.get_scenes()) >= 1

        # Second call: still exact same state, no duplicate entries or exceptions
        bootstrap_if_needed(project, db, bootstrap=False)
        assert len(db.get_characters()) == 2

    # -------------------------------------------------------------------------
    # Step 5: Translation with LLM mock & relational prompt validation
    # -------------------------------------------------------------------------
    captured_prompts: list[str] = []

    def mock_complete_json(
        system: str,
        prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        captured_prompts.append(prompt)
        # Return mock translations with preserved masks and Vietnamese text
        response = {
            "d:story_01": "Em đã sẵn sàng cho nhiệm vụ chưa, Eileen?",
            "d:story_02": "Vâng, nhưng hãy cẩn thận với ⟦T1⟧Nguy hiểm⟦/T1⟧ phía trước!",
            "d:story_03": "Chúng ta đã ghi được ⟦V1⟧ điểm rồi.",
            "d:story_04": "Hội Bóng Đêm sẽ không thể ngăn cản chúng ta.",
        }
        return response, 120, 80

    cfg = Config()
    with (
        project.open_db() as db,
        patch("renslate.engine.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        trans_stats = pipeline.run()

        assert trans_stats.llm_translated == 4
        assert trans_stats.failed == 0

        # Verify that relational pronoun addressing was injected into prompt
        assert len(captured_prompts) >= 1
        active_prompt = captured_prompts[0]
        assert "QUAN HỆ XƯNG HÔ ĐỐI THOẠI TRONG PHÂN ĐOẠN" in active_prompt
        assert (
            "Protagonist và Eileen đang đối thoại trực tiếp" in active_prompt
            or "Eileen và Protagonist đang đối thoại trực tiếp" in active_prompt
        )
        assert "xưng 'anh', gọi 'em'" in active_prompt
        assert "xưng 'em', gọi 'anh'" in active_prompt
        assert "TÓM TẮT CẢNH HIỆN TẠI" in active_prompt

        # Verify DB units are marked TRANSLATED
        u2 = db.get_unit("d:story_02")
        assert u2 is not None
        assert u2.status is Status.TRANSLATED
        # Verify tag unmasking
        assert "{color=#ff0000}Nguy hiểm{/color}" in (u2.target or "")

        u3 = db.get_unit("d:story_03")
        assert u3 is not None
        assert "[score]" in (u3.target or "")

    # -------------------------------------------------------------------------
    # Step 6: Application to disk (.rpy)
    # -------------------------------------------------------------------------
    with project.open_db() as db:
        apply_report = apply_translations(project, db)
        assert apply_report.lines_written == 4
        assert apply_report.files_changed == 1

    # Verify target file contents on disk
    tl_content = tl_story_rpy.read_text(encoding="utf-8")
    expected_danger = (
        'e "Vâng, nhưng hãy cẩn thận với {color=#ff0000}Nguy hiểm{/color} phía trước!"'
    )
    assert expected_danger in tl_content
    assert 'mc "Chúng ta đã ghi được [score] điểm rồi."' in tl_content
    assert 'e "Hội Bóng Đêm sẽ không thể ngăn cản chúng ta."' in tl_content
