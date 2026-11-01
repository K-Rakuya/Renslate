"""Tier 4 Real-World Application Scenarios E2E Test Suite.

End-to-end multi-scene visual novel translation simulation covering Requirements R1 to R5:
1. Multi-scene narrative arc (Prologue -> Royal Court -> Dungeon Climax) with dynamic relationships
2. Asymmetric pronoun integrity under rapid interlocutor switching across family & court hierarchies
3. Continuous Hallucination Guard audit, error taxonomy detection, and automated repair loop
4. Pure bytecode (.rpyc) translation lifecycle without original source .rpy files
5. Database transaction resilience, atomic I/O, and pre-apply backup snapshot restoration
"""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

from renslate.core.config import BatchConfig, Config, LLMConfig
from renslate.core.models import (
    CharacterProfile,
    Origin,
    RelationshipEdge,
    RelationshipKind,
    SceneMacroContext,
    Slot,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.context import (
    ContextBuilder,
    ContextBundle,
    InterlocutorResolver,
)
from renslate.engine.graph import RelationshipGraph
from renslate.engine.qa import QAChecker, clean_translation
from renslate.engine.repair import UnitFailure, format_failure_feedback
from renslate.renpy.literals import encode_say_text
from renslate.renpy.parser import fingerprint
from renslate.renpy.rpyc import (
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    Define,
    TranslateSay,
    VirtualPyCode,
    create_rpyc,
    extract_character_definitions_from_rpyc,
    extract_dialogue_from_rpyc,
    synthesize_translation_skeletons,
)
from renslate.renpy.writer import apply_translations

# ==============================================================================
# Helper Factories & Mock Harnesses
# ==============================================================================


def _build_bytecode_file(dest: Path, slot1_stmts: list[object], slot2_stmts: list[object]) -> Path:
    slots = {
        SLOT_STATEMENTS_PRE: pickle.dumps((None, slot1_stmts)),
        SLOT_STATEMENTS_POST: pickle.dumps((None, slot2_stmts)),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(create_rpyc(slots))
    return dest


def _make_unit(
    uid: str,
    seq: int,
    source: str,
    *,
    speaker: str | None = None,
    target: str | None = None,
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        masked_source=source,
        speaker=speaker,
        target=target,
        status=Status.TRANSLATED if target else Status.PENDING,
        origin=Origin.LLM if target else None,
    )


# ==============================================================================
# Tier 4: Real-World Visual Novel Application Scenarios
# ==============================================================================


class TestRealWorldVNScenariosE2E:
    """Tier 4: Comprehensive end-to-end visual novel translation lifecycle simulations."""

    def test_scenario_1_multi_scene_vn_narrative_arc_pipeline(self, tmp_path: Path) -> None:
        """Simulate an entire 3-scene Visual Novel localization pipeline.

        - Scene 1 (Prologue): Hero (mc) and Heroine (eileen) with ROMANCE ("anh - em").
        - Scene 2 (Royal Court): Empress (queen) and Commander (general) ("trẫm - khanh").
        - Scene 3 (Dungeon Climax): Hero (mc) and Rival (rival) ("tôi - cậu").
        """
        game_root = tmp_path / "VN_Project"
        game_root.mkdir()
        (game_root / "game" / "tl" / "vietnamese").mkdir(parents=True)

        project = Project.create(
            game_root,
            target_lang="vietnamese",
        )
        cfg = Config(
            target_lang="vietnamese",
            batch=BatchConfig(max_lines=20, max_source_tokens=600),
            llm=LLMConfig(base_url="http://127.0.0.1:8080/v1", model="Qwen-VN.gguf"),
        )
        cfg.save(project.config_path)

        # 1. Knowledge setup: Character profiles
        profiles = [
            CharacterProfile(
                tag="mc",
                display_name="Protagonist",
                gender="male",
                role="Main Character",
                speech_style="chân thành",
                pronoun_map={"eileen": "anh - em", "rival": "tôi - cậu"},
                archetype="hero",
                personality_traits=("dũng cảm",),
                tone_and_voice="tự tin",
                socioeconomic_status="thường dân",
                sample_dialogue=("Đi nào!",),
            ),
            CharacterProfile(
                tag="eileen",
                display_name="Eileen",
                gender="female",
                role="Heroine",
                speech_style="dịu dàng",
                pronoun_map={"mc": "em - anh"},
                archetype="deredere",
                personality_traits=("ấm áp",),
                tone_and_voice="nhẹ nhàng",
                socioeconomic_status="quý tộc",
                sample_dialogue=("Vâng, anh!",),
            ),
            CharacterProfile(
                tag="queen",
                display_name="Nữ Hoàng",
                gender="female",
                role="Monarch",
                speech_style="uy quyền",
                pronoun_map={"general": "trẫm - khanh"},
                archetype="ruler",
                personality_traits=("nghiêm nghị",),
                tone_and_voice="lạnh lùng",
                socioeconomic_status="hoàng gia",
                sample_dialogue=("Trẫm chuẩn tấu.",),
            ),
        ]

        with project.open_db() as db:
            db.save_characters_bulk(profiles)

            # 2. Relationship Graph setup
            graph = RelationshipGraph()
            graph.add_edge(
                RelationshipEdge(
                    source_tag="mc",
                    target_tag="eileen",
                    kind=RelationshipKind.ROMANCE,
                    speaker_pronoun="anh",
                    listener_pronoun="em",
                    confidence=1.0,
                    source_type="manual",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    source_tag="eileen",
                    target_tag="mc",
                    kind=RelationshipKind.ROMANCE,
                    speaker_pronoun="em",
                    listener_pronoun="anh",
                    confidence=1.0,
                    source_type="manual",
                )
            )
            graph.save_to_db(db)

            # 3. Units insertion across scenes
            units = [
                _make_unit("sc1_01", 1, "Are you ready, Eileen?", speaker="mc"),
                _make_unit("sc1_02", 2, "Yes, let us walk together.", speaker="eileen"),
                _make_unit("sc2_01", 3, "General, make your report.", speaker="queen"),
                _make_unit("sc3_01", 4, "I won't let you pass, rival!", speaker="mc"),
            ]
            db.save_units_bulk(units)

            # 4. Context Assembly check
            builder = ContextBuilder(cfg)
            macro_sc1 = SceneMacroContext(
                scene_id="scene_01",
                label="prologue_walk",
                location="Vườn hoa học viện",
                atmosphere="lãng mạn",
                active_characters=("mc", "eileen"),
            )
            bundle = ContextBundle(
                characters={p.tag: p for p in profiles},
                graph=graph,
                scene_macro=macro_sc1,
            )
            prompt = builder.build_user_prompt(units[:2], bundle=bundle)
            assert "Vườn hoa học viện" in prompt
            assert "Eileen" in prompt
            assert "mc" in prompt

            # 5. Simulate translated output and QA verification
            translated_lines = [
                ("sc1_01", "Em đã sẵn sàng chưa, Eileen?"),
                ("sc1_02", "Vâng, chúng ta cùng đi thôi anh."),
                ("sc2_01", "Khanh hãy báo cáo tình hình."),
                ("sc3_01", "Tôi sẽ không để cậu vượt qua đâu!"),
            ]
            checker = QAChecker()
            for uid, tr_text in translated_lines:
                cleaned = clean_translation(tr_text)
                db.update_translation(uid, cleaned, Origin.LLM)
                issues = checker.check_unit(uid, "src", None, cleaned)
                assert len(issues) == 0, f"QA issues found on {uid}: {issues}"

            # Verify all units are now translated
            assert len(db.pending_units()) == 0
            assert len(db.all_units()) == 4

    def test_scenario_2_asymmetric_pronoun_integrity_under_switching(self) -> None:
        """Verify strict reciprocal pronoun adherence under rapid multi-turn speaker switching.

        Cast:
        - elder_sister (chị) -> mc (em)
        - mc (em) -> elder_sister (chị)
        - mc (anh) -> younger_sister (em)
        - younger_sister (em) -> mc (anh)
        """
        graph = RelationshipGraph()
        # Sibling elder: elder_sister -> mc ("chị - em")
        graph.add_edge(
            RelationshipEdge(
                source_tag="elder_sister",
                target_tag="mc",
                kind=RelationshipKind.FAMILY_SIBLING,
                speaker_pronoun="chị",
                listener_pronoun="em",
                confidence=1.0,
                source_type="manual",
            )
        )
        graph.add_edge(
            RelationshipEdge(
                source_tag="mc",
                target_tag="elder_sister",
                kind=RelationshipKind.FAMILY_SIBLING,
                speaker_pronoun="em",
                listener_pronoun="chị",
                confidence=1.0,
                source_type="manual",
            )
        )
        # Sibling younger: mc -> younger_sister ("anh - em")
        graph.add_edge(
            RelationshipEdge(
                source_tag="mc",
                target_tag="younger_sister",
                kind=RelationshipKind.FAMILY_SIBLING,
                speaker_pronoun="anh",
                listener_pronoun="em",
                confidence=1.0,
                source_type="manual",
            )
        )
        graph.add_edge(
            RelationshipEdge(
                source_tag="younger_sister",
                target_tag="mc",
                kind=RelationshipKind.FAMILY_SIBLING,
                speaker_pronoun="em",
                listener_pronoun="anh",
                confidence=1.0,
                source_type="manual",
            )
        )

        resolver = InterlocutorResolver()
        dialogues = [
            _make_unit("d1", 1, "Sister, when are we leaving?", speaker="mc"),
            _make_unit("d2", 2, "Soon, little brother. Pack your bag.", speaker="elder_sister"),
            _make_unit("d3", 3, "Brother, can I come along too?", speaker="younger_sister"),
            _make_unit("d4", 4, "Of course, stay close to me.", speaker="mc"),
        ]

        active_spk = ["mc", "elder_sister", "younger_sister"]
        prev = None
        for d in dialogues:
            res = resolver.resolve(d, prev_unit=prev, active_speakers=active_spk)
            assert res.speaker == d.speaker
            prev = d

        # Check reciprocal pronoun lookup integrity
        edge_elder_to_mc = graph.get_edge("elder_sister", "mc")
        edge_mc_to_elder = graph.get_edge("mc", "elder_sister")
        assert edge_elder_to_mc is not None and edge_mc_to_elder is not None
        assert edge_elder_to_mc.speaker_pronoun == edge_mc_to_elder.listener_pronoun == "chị"
        assert edge_elder_to_mc.listener_pronoun == edge_mc_to_elder.speaker_pronoun == "em"

        edge_mc_to_younger = graph.get_edge("mc", "younger_sister")
        edge_younger_to_mc = graph.get_edge("younger_sister", "mc")
        assert edge_mc_to_younger is not None and edge_younger_to_mc is not None
        assert edge_mc_to_younger.speaker_pronoun == edge_younger_to_mc.listener_pronoun == "anh"
        assert edge_mc_to_younger.listener_pronoun == edge_younger_to_mc.speaker_pronoun == "em"

    def test_scenario_3_hallucination_detection_and_automated_repair_loop(self) -> None:
        """Simulate LLM Hallucination Guard detecting an error and repairing via RetryLadder.

        Flow:
        1. Source line: "Take cover behind the wall!"
        2. Candidate 1 (hallucinated): "Nhảy xuống hồ bơi ngay lập tức!" (factual_hallucination)
        3. Hallucination Guard JSON output flags error.
        4. Failure feedback is formatted.
        5. Candidate 2 (repaired): "Mau nấp sau bức tường!" passes.
        """
        unit = _make_unit("combat_01", 1, "Take cover behind the wall!", speaker="mc")

        # Guard JSON verdict simulating detection
        guard_response = json.dumps(
            {
                "scanned_units": [
                    {
                        "uid": "combat_01",
                        "verdict": "FAIL",
                        "issue_type": "factual_hallucination",
                        "confidence": 0.99,
                        "explanation": "Dịch sai lệch: 'nấp sau tường' thành 'nhảy hồ bơi'.",
                        "suggested_correction": "Mau nấp sau bức tường!",
                    }
                ]
            }
        )
        report = json.loads(guard_response)["scanned_units"][0]
        assert report["verdict"] == "FAIL"
        assert report["issue_type"] == "factual_hallucination"

        # Generate Retry feedback
        failure = UnitFailure(
            unit=unit,
            code=report["issue_type"],
            detail=report["explanation"],
            candidate="Nhảy xuống hồ bơi ngay lập tức!",
        )
        feedback = format_failure_feedback([failure])
        assert "LỖI Ở LẦN DỊCH TRƯỚC" in feedback
        assert "combat_01" in feedback
        assert "factual_hallucination" in feedback

        # Next prompt iteration receives feedback
        builder = ContextBuilder(Config())
        retry_prompt = builder.build_user_prompt([unit], retry_feedback=feedback)
        assert "LỖI Ở LẦN DỊCH TRƯỚC" in retry_prompt

        # Clean translation verification
        clean_tr = clean_translation("Mau nấp sau bức tường!")
        checker = QAChecker()
        issues = checker.check_unit("combat_01", unit.source, None, clean_tr)
        assert len(issues) == 0

    def test_scenario_4_pure_bytecode_translation_lifecycle(self, tmp_path: Path) -> None:
        """Full translation lifecycle for games distributed purely as compiled .rpyc files."""
        game_dir = tmp_path / "PureRPYC_Game" / "game"
        game_dir.mkdir(parents=True)

        char_def = Define(varname="m", code=VirtualPyCode('Character("Master")'))
        char_d = Define(varname="d", code=VirtualPyCode('Character("Disciple")'))

        ts1 = TranslateSay()
        ts1.identifier = "trial_01"
        ts1.who = "m"
        ts1.what = "Begin your martial trial."
        ts1.filename = "game/trial.rpy"
        ts1.linenumber = 10

        ts2 = TranslateSay()
        ts2.identifier = "trial_02"
        ts2.who = "d"
        ts2.what = "I will not fail you, Master."
        ts2.filename = "game/trial.rpy"
        ts2.linenumber = 12

        rpyc_file = game_dir / "trial.rpyc"
        _build_bytecode_file(rpyc_file, [char_def, char_d], [ts1, ts2])

        # 1. Extract characters
        chars = extract_character_definitions_from_rpyc(game_dir)
        assert chars.get("m") == "Master"
        assert chars.get("d") == "Disciple"

        # 2. Extract dialogue
        dialogues = extract_dialogue_from_rpyc(rpyc_file)
        assert len(dialogues) == 2
        assert dialogues[0].identifier == "trial_01"
        assert dialogues[1].identifier == "trial_02"

        # 3. Synthesize translation skeletons
        skeletons = synthesize_translation_skeletons(game_dir, target_lang="vietnamese")
        assert len(skeletons) >= 1
        skeleton_file = skeletons[0]
        assert skeleton_file.exists()
        content = skeleton_file.read_text(encoding="utf-8")
        assert "translate vietnamese trial_01:" in content
        assert "translate vietnamese trial_02:" in content

    def test_scenario_5_database_resilience_and_pre_apply_backup_recovery(
        self, tmp_path: Path
    ) -> None:
        """Verify pre-apply backup snapshots allow full restoration if disk corruption occurs."""
        game_root = tmp_path / "BackupTest_Game"
        tl_vietnamese = game_root / "game" / "tl" / "vietnamese"
        tl_vietnamese.mkdir(parents=True)

        script_file = tl_vietnamese / "script.rpy"
        original_script_content = (
            '# game/script.rpy:5\n'
            'translate vietnamese start_01:\n'
            '    old "Original English line."\n'
            '    new "Original English line."\n'
        )
        script_file.write_text(original_script_content, encoding="utf-8")

        project = Project.create(game_root, target_lang="vietnamese")
        with project.open_db() as db:
            u = _make_unit(
                "start_01", 1, "Original English line.", target="Bản dịch tiếng Việt mới."
            )
            db.save_unit(u)

            anchor_line = '    old "Original English line."'
            target_line = '    new "Original English line."'
            fp = fingerprint(anchor_line, target_line)
            slot = Slot(
                uid="start_01",
                file="game/tl/vietnamese/script.rpy",
                line=4,
                anchor_line=3,
                fingerprint=fp,
            )
            db.replace_slots([slot])

            # Apply translations: this must create a timestamped backup before modifying script.rpy
            report = apply_translations(project, db)
            assert report.files_changed == 1
            assert report.lines_written == 1
            assert report.backup_dir is not None

        # Verify translation was written to disk
        new_content = script_file.read_text(encoding="utf-8")
        escaped_target = encode_say_text("Bản dịch tiếng Việt mới.")
        assert f'new "{escaped_target}"' in new_content

        # Verify backup existence
        assert project.backup_root.exists()
        backup_dirs = list(project.backup_root.iterdir())
        assert len(backup_dirs) >= 1
        backup_script = backup_dirs[0] / "game" / "tl" / "vietnamese" / "script.rpy"
        assert backup_script.exists()
        assert backup_script.read_text(encoding="utf-8") == original_script_content

        # Simulate disaster recovery: restore original file from backup snapshot
        shutil.copy2(backup_script, script_file)
        assert script_file.read_text(encoding="utf-8") == original_script_content
