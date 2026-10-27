"""Adversarial stress-test suite for Milestone 4: Frame peeling, QA leak prevention,
Ren'Py bracket preservation, and runner integration.

Probes and stress-tests:
1. Adversarial LLM outputs:
   - Tripled metadata frames (relational, monologue, mixed, suffix, nested quoted).
   - Interleaved metadata tags inside Vietnamese text.
   - Empty outputs after frame stripping (collapsing to empty string, flagged as empty_target).
2. Ren'Py bracket preservation:
   - Authentic brackets: [Sigh], [player_name], [...], [[Notice]], [[Warning]], [1].
   - Relocated variable interpolations at sentence-start.
   - Guarantee that QAChecker never flags authentic brackets as speaker_tag_leakage.
   - End-to-end pipeline execution ensuring brackets survive cleanly into SQLite and TM.
3. Fatal QA rejection & retry isolation:
   - Units with leaked metadata or pronoun inversion emit severity='error'.
   - Unpeeled or invalid units are strictly excluded from DB and TM commits.
   - Multi-unit batches isolate corrupted units while cleanly committing valid units.
   - Single-unit retry routing and healing verification.
4. Runner bootstrap & CLI integration:
   - Triple-run bootstrap idempotency across characters, relationships, and scenes.
   - Preservation of all 11 CharacterProfile persona attributes from characters.json.
   - CLI flags: --bootstrap, --no-bootstrap, and default None.
   - Startup ANSI banner relationship edge count rendering and empty-state omission.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from renslate.core.config import BatchConfig, Config
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.graph import RelationshipGraph
from renslate.engine.qa import QAChecker, clean_translation
from renslate.engine.translate import TranslationPipeline
from renslate.ui.runner import bootstrap_if_needed, run_unattended_pipeline
from renslate.ui.runner import main as runner_main


def _create_adversarial_project(tmp_path: Path, name: str) -> Project:
    root = tmp_path / name
    (root / "game").mkdir(parents=True, exist_ok=True)
    proj = Project.create(root, target_lang="vietnamese")
    proj.tl_root.mkdir(parents=True, exist_ok=True)
    return proj


class TestAdversarialFramePeelingAndEmptyOutputs:
    """Stress-tests multi-pass frame peeling under hostile, duplicated,
    and degenerate LLM echoes.
    """

    @pytest.mark.parametrize(
        ("raw_target", "expected_clean"),
        [
            (
                "[Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE] "
                "[Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE] "
                "[Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE] "
                "Anh yêu em nhiều lắm.",
                "Anh yêu em nhiều lắm.",
            ),
            (
                "[Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE] "
                "[Người nói: Ren -> Người nghe: Eileen] "
                "[Người nói: Ren] "
                "Chào em Eileen!",
                "Chào em Eileen!",
            ),
            (
                "[Độc thoại/Nội tâm: Ren] "
                "[Suy nghĩ: Ren] "
                "[Monologue/Thought: Ren] "
                "Mình phải vượt qua chuyện này.",
                "Mình phải vượt qua chuyện này.",
            ),
            (
                "d:101 | [Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE] "
                "[Speaker: Ren] "
                "Ren: \"Bản dịch: 'Chào em, ngày mới tốt lành!'\"",
                "Chào em, ngày mới tốt lành!",
            ),
            (
                '"Chào em Eileen!" [Người nói: Ren] [Người nghe: Eileen] [Quan hệ: ROMANCE]',
                "Chào em Eileen!",
            ),
            (
                '[Người nói: Ren] [Quan hệ: ROMANCE] "[Người nghe: Eileen] Em có đó không?"',
                "Em có đó không?",
            ),
            (
                "   [  NGƯỜI NÓI: Ren -> NGƯỜI NGHE: Eileen | QUAN HỆ: ROMANCE  ]   "
                '[Speaker: Ren] "Xin chào!"',
                "Xin chào!",
            ),
        ],
    )
    def test_tripled_and_adversarial_metadata_frames_peeling(
        self, raw_target: str, expected_clean: str
    ) -> None:
        """Verify that multi-layered framing echoes are peeled to pure Vietnamese text."""
        cleaned = clean_translation(raw_target, speaker="Ren", source="Hello Eileen!")
        assert cleaned == expected_clean

    def test_pipeline_tripled_frame_peeling_end_to_end(self, tmp_path: Path) -> None:
        """Verify that TranslationPipeline strips tripled frames and commits clean text to DB."""
        proj = _create_adversarial_project(tmp_path, "pipe_tripled")
        with proj.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            db.save_character(CharacterProfile("eileen", "Eileen", "female"))
            u = Unit(
                "d:adv1",
                UnitKind.DIALOGUE,
                1,
                "Good morning Eileen.",
                Status.PENDING,
                speaker="Ren",
            )
            db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(proj, db, cfg)

            tripled_output = (
                "[Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE] "
                "[Người nói: Ren -> Người nghe: Eileen] "
                '[Người nói: Ren] "Chào buổi sáng Eileen."'
            )

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=({"d:adv1": tripled_output}, 80, 25),
            ):
                stats = pipeline.run()

            assert stats.llm_translated == 1
            saved = db.get_unit("d:adv1")
            assert saved is not None
            assert saved.status == Status.TRANSLATED
            assert saved.target == "Chào buổi sáng Eileen."
            assert not saved.qa_flags

    @pytest.mark.parametrize(
        "empty_candidate",
        [
            "[Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE]",
            "[Người nói: Ren] [Người nghe: Eileen]",
            "[Ren]:",
            "[Ren]: ''",
            'd:999 | [Người nói: Ren] ""',
            "[Độc thoại/Nội tâm: Ren]",
            "   [Người nói: Ren]   ",
            "[Ren]",
            "",
            "   ",
        ],
    )
    def test_empty_outputs_after_peeling_collapse_to_empty(self, empty_candidate: str) -> None:
        """Verify that outputs consisting solely of metadata frames peel to an empty string."""
        cleaned = clean_translation(empty_candidate, speaker="Ren", source="Hello")
        assert cleaned == ""

    def test_empty_outputs_flagged_as_fatal_empty_target_by_qa(self) -> None:
        """Verify that empty strings emit a fatal empty_target QA error."""
        qc = QAChecker()
        issues = qc.check_unit("u_empty", "Hello", None, "")
        error_codes = [i.code for i in issues if i.severity == "error"]
        assert "empty_target" in error_codes

    def test_pipeline_rejects_empty_outputs_after_peeling(self, tmp_path: Path) -> None:
        """Verify that units whose frames peel to empty are marked FAILED and not committed."""
        proj = _create_adversarial_project(tmp_path, "pipe_empty")
        with proj.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            u = Unit("d:empty", UnitKind.DIALOGUE, 1, "Hello.", Status.PENDING, speaker="Ren")
            db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(proj, db, cfg)

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=(
                    {"d:empty": "[Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE]"},
                    50,
                    15,
                ),
            ):
                stats = pipeline.run()

            assert stats.failed == 1
            assert stats.llm_translated == 0
            saved = db.get_unit("d:empty")
            assert saved is not None
            assert saved.status == Status.FAILED
            assert saved.target is None or saved.target == ""
            assert pipeline.tm.lookup_exact("Hello.") is None


class TestAdversarialInterleavedMetadataAndFatalQARejection:
    """Stress-tests detection and rejection of metadata leaked inside dialogue bodies."""

    @pytest.mark.parametrize(
        ("corrupted_text", "expected_fatal_code"),
        [
            (
                "Chào em [Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE], em ăn chưa?",
                "speaker_tag_leakage",
            ),
            (
                "Chào em [Quan hệ: ROMANCE], hôm nay thế nào?",
                "relation_metadata_leakage",
            ),
            (
                "Tôi không chắc [Người nghe: Eileen] điều đó có đúng không.",
                "relation_metadata_leakage",
            ),
            (
                "Lùi lại mau [Cảm xúc: Lo lắng], nguy hiểm lắm!",
                "relation_metadata_leakage",
            ),
            (
                "Này [xưng 'anh', gọi 'em'] đừng đi nữa.",
                "relation_metadata_leakage",
            ),
            (
                "Tôi phải làm gì [Độc thoại/Nội tâm: Ren] bây giờ đây?",
                "speaker_tag_leakage",
            ),
            (
                "Này [Suy nghĩ: Ren] ổn không đấy?",
                "speaker_tag_leakage",
            ),
            (
                "Này [Speaker: Ren: Cẩn thận]!",
                "speaker_tag_leakage",
            ),
        ],
    )
    def test_interleaved_metadata_detected_as_fatal_error(
        self, corrupted_text: str, expected_fatal_code: str
    ) -> None:
        """Verify that interleaved metadata tags trigger fatal severity='error' in QAChecker."""
        qc = QAChecker()
        issues = qc.check_unit("u_leak", "Hello", None, corrupted_text)
        fatal_issues = [i for i in issues if i.severity == "error"]
        assert any(i.code == expected_fatal_code for i in fatal_issues)

    def test_pipeline_isolates_and_rejects_interleaved_leakage_in_batch(
        self, tmp_path: Path
    ) -> None:
        """Multi-unit batch: corrupted unit is rejected and retried while valid units succeed."""
        proj = _create_adversarial_project(tmp_path, "pipe_interleaved")
        with proj.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            u1 = Unit("d:1", UnitKind.DIALOGUE, 1, "Hello.", Status.PENDING, speaker="Ren")
            u2 = Unit("d:2", UnitKind.DIALOGUE, 2, "Watch out!", Status.PENDING, speaker="Ren")
            u3 = Unit(
                "d:3", UnitKind.DIALOGUE, 3, "See you tomorrow.", Status.PENDING, speaker="Ren"
            )
            db.save_unit(u1)
            db.save_unit(u2)
            db.save_unit(u3)

            cfg = Config(batch=BatchConfig(max_lines=10))
            pipeline = TranslationPipeline(proj, db, cfg)

            call_count = 0

            def mock_llm_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                nonlocal call_count
                call_count += 1
                if "d:2" in user_prompt and "d:1" in user_prompt:
                    # Initial batch: unit 2 has leaked relation metadata
                    return (
                        {
                            "d:1": "Xin chào.",
                            "d:2": "Cẩn thận đấy [Quan hệ: ROMANCE] nhé!",
                            "d:3": "Hẹn gặp lại ngày mai.",
                        },
                        120,
                        40,
                    )
                # Single-unit retry for d:2: persists in returning leaked metadata
                return ({"d:2": "Cẩn thận đấy [Người nghe: Eileen] nhé!"}, 40, 15)

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_llm_complete):
                stats = pipeline.run()

            # Valid units committed to DB and TM
            saved1 = db.get_unit("d:1")
            saved3 = db.get_unit("d:3")
            assert saved1 is not None and saved1.status == Status.TRANSLATED
            assert saved1.target == "Xin chào."
            assert saved3 is not None and saved3.status == Status.TRANSLATED
            assert saved3.target == "Hẹn gặp lại ngày mai."
            assert pipeline.tm.lookup_exact("Hello.") == "Xin chào."
            assert pipeline.tm.lookup_exact("See you tomorrow.") == "Hẹn gặp lại ngày mai."

            # Corrupted unit rejected from DB and TM
            saved2 = db.get_unit("d:2")
            assert saved2 is not None and saved2.status == Status.FAILED
            assert saved2.target is None or "[Quan hệ" not in saved2.target
            assert pipeline.tm.lookup_exact("Watch out!") is None

            assert stats.llm_translated == 2
            assert stats.failed == 1
            assert call_count >= 2

    def test_pronoun_inversion_2nd_to_1st_fatal_rejection_in_pipeline(
        self, tmp_path: Path
    ) -> None:
        """2nd-person (you) mistranslated as 1st-person pronoun must fail QA and be rejected."""
        proj = _create_adversarial_project(tmp_path, "pipe_inv_2to1")
        with proj.open_db() as db:
            db.save_character(CharacterProfile("eileen", "Eileen", "female"))
            db.save_character(CharacterProfile("ren", "Ren", "male"))

            graph = RelationshipGraph()
            graph.add_edge(
                RelationshipEdge(
                    source_tag="eileen",
                    target_tag="ren",
                    kind=RelationshipKind.ROMANCE,
                    speaker_pronoun="em",
                    listener_pronoun="anh",
                )
            )

            u = Unit(
                "d:inv_2",
                UnitKind.DIALOGUE,
                1,
                "Ren, why did you do that?",
                Status.PENDING,
                speaker="eileen",
            )
            db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(proj, db, cfg, graph=graph)

            # Eileen uses 'em' (self) instead of 'anh' (other) for 'you'
            inverted_output = "Ren, tại sao em lại làm như vậy?"

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=({"d:inv_2": inverted_output}, 50, 20),
            ):
                stats = pipeline.run()

            assert stats.failed == 1
            assert stats.llm_translated == 0
            saved = db.get_unit("d:inv_2")
            assert saved is not None
            assert saved.status == Status.FAILED
            assert pipeline.tm.lookup_exact("Ren, why did you do that?") is None

    def test_pronoun_inversion_1st_to_2nd_fatal_rejection_in_pipeline(self, tmp_path: Path) -> None:
        """1st-person (I) mistranslated as 2nd-person other-pronoun must fail QA and be rejected."""
        proj = _create_adversarial_project(tmp_path, "pipe_inv_1to2")
        with proj.open_db() as db:
            db.save_character(CharacterProfile("eileen", "Eileen", "female"))
            db.save_character(CharacterProfile("ren", "Ren", "male"))

            graph = RelationshipGraph()
            graph.add_edge(
                RelationshipEdge(
                    source_tag="eileen",
                    target_tag="ren",
                    kind=RelationshipKind.ROMANCE,
                    speaker_pronoun="em",
                    listener_pronoun="anh",
                )
            )

            u = Unit(
                "d:inv_1",
                UnitKind.DIALOGUE,
                1,
                "Ren, I must protect the castle.",
                Status.PENDING,
                speaker="eileen",
            )
            db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(proj, db, cfg, graph=graph)

            # Eileen uses 'anh' (other) instead of 'em' (self) for 'I'
            inverted_output = "Ren, anh phải bảo vệ lâu đài."

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=({"d:inv_1": inverted_output}, 50, 20),
            ):
                stats = pipeline.run()

            assert stats.failed == 1
            assert stats.llm_translated == 0
            saved = db.get_unit("d:inv_1")
            assert saved is not None
            assert saved.status == Status.FAILED
            assert pipeline.tm.lookup_exact("Ren, I must protect the castle.") is None

    def test_pronoun_inversion_heals_on_clean_retry(self, tmp_path: Path) -> None:
        """Verify that when a unit fails batch translation with inverted pronouns,
        clean retry heals it.
        """
        proj = _create_adversarial_project(tmp_path, "pipe_inv_heal")
        with proj.open_db() as db:
            db.save_character(CharacterProfile("eileen", "Eileen", "female"))
            db.save_character(CharacterProfile("ren", "Ren", "male"))

            graph = RelationshipGraph()
            graph.add_edge(
                RelationshipEdge(
                    source_tag="eileen",
                    target_tag="ren",
                    kind=RelationshipKind.ROMANCE,
                    speaker_pronoun="em",
                    listener_pronoun="anh",
                )
            )

            u1 = Unit(
                "d:ok", UnitKind.DIALOGUE, 1, "Good morning.", Status.PENDING, speaker="eileen"
            )
            u2 = Unit(
                "d:heal",
                UnitKind.DIALOGUE,
                2,
                "Ren, can you wait for a moment?",
                Status.PENDING,
                speaker="eileen",
            )
            db.save_unit(u1)
            db.save_unit(u2)

            cfg = Config(batch=BatchConfig(max_lines=10))
            pipeline = TranslationPipeline(proj, db, cfg, graph=graph)

            call_count = 0

            def mock_llm_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                nonlocal call_count
                call_count += 1
                if "d:ok" in user_prompt:
                    # Initial batch: u2 is inverted
                    return (
                        {
                            "d:ok": "Chào buổi sáng.",
                            "d:heal": "Ren, em có thể đợi một chút không?",
                        },
                        100,
                        30,
                    )
                # Single-unit retry: LLM heals and returns correct pronoun 'anh'
                return ({"d:heal": "Ren, anh có thể đợi một chút không?"}, 50, 20)

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_llm_complete):
                stats = pipeline.run()

            assert stats.llm_translated == 2
            assert stats.failed == 0
            saved2 = db.get_unit("d:heal")
            assert saved2 is not None
            assert saved2.status == Status.TRANSLATED
            assert saved2.target == "Ren, anh có thể đợi một chút không?"
            assert (
                pipeline.tm.lookup_exact("Ren, can you wait for a moment?")
                == "Ren, anh có thể đợi một chút không?"
            )


class TestRenpyBracketPreservation:
    """Stress-tests preservation of genuine Ren'Py square brackets and variable interpolations."""

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("[player_name] arrived safely.", "[player_name] đã đến nơi an toàn."),
            ("Welcome, [player_name]!", "Chào mừng, [player_name]!"),
            ("[Sigh] What a long day.", "[Sigh] Thật là một ngày dài."),
            ("[Sigh] What a long day.", "[Thở dài] Thật là một ngày dài."),
            ("[...]", "[...]"),
            ("...", "..."),
            ("[Laughs] That was brilliant!", "[Cười lớn] Xuất sắc thật đấy!"),
            ("[Gasp] Who is there?", "[Kinh ngạc] Ai ở đó thế?"),
            ("[Chuckles] You are funny.", "[Cười khẽ] Cậu hài hước thật."),
            ("[hero.name]'s level is [score!s].", "Cấp độ của [hero.name] là [score!s]."),
            ("Choice: [1] Attack [2] Defend", "Lựa chọn: [1] Tấn công [2] Phòng thủ"),
            ("[[Notice]] Server will restart.", "[[Thông báo]] Máy chủ sẽ khởi động lại."),
            ("[[Warning]] Do not enter!", "[[Cảnh báo]] Không được vào!"),
            ("Press [[SPACE]] to skip.", "Nhấn [[SPACE]] để bỏ qua."),
            ("Press [[ENTER]] or [[ESC]].", "Nhấn [[ENTER]] hoặc [[ESC]]."),
            ("[[...]]", "[[...]]"),
        ],
    )
    def test_authentic_brackets_preserved_by_clean_translation(
        self, source: str, target: str
    ) -> None:
        """Verify that genuine brackets and interpolations are strictly preserved."""
        cleaned = clean_translation(target, speaker="Ren", source=source)
        assert cleaned == target

    def test_bracket_relocated_to_sentence_start_preserved(self) -> None:
        """When an interpolation appears in the middle of source but at start of target,
        preserve it.
        """
        source = "Will [player_name] attend the royal festival?"
        target = "[player_name] có tham gia lễ hội hoàng gia không?"
        cleaned = clean_translation(target, speaker="Ren", source=source)
        assert cleaned == target

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("[player_name]", "[player_name]"),
            ("[Sigh]", "[Sigh]"),
            ("[Sigh]", "[Thở dài]"),
            ("[...]", "[...]"),
            ("[[Notice]]", "[[Notice]]"),
            ("[[Notice]]", "[[Thông báo]]"),
        ],
    )
    def test_qa_never_emits_false_speaker_tag_leakage_on_valid_brackets(
        self, source: str, target: str
    ) -> None:
        """Verify standalone authentic brackets are not misclassified as speaker tag leakages."""
        qc = QAChecker()
        issues = qc.check_unit("u_bracket", source, None, target, speaker="Ren")
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0, f"False QA errors emitted on authentic bracket: {errors}"

    def test_pipeline_bracket_preservation_end_to_end(self, tmp_path: Path) -> None:
        """Verify end-to-end pipeline preserves [player_name], [Sigh], [...], and [[Notice]]."""
        from renslate.renpy.mask import mask

        proj = _create_adversarial_project(tmp_path, "pipe_brackets")
        with proj.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            units = [
                Unit(
                    "d:b1",
                    UnitKind.DIALOGUE,
                    1,
                    "[player_name] arrived.",
                    Status.PENDING,
                    speaker="Ren",
                    masked_source=mask("[player_name] arrived.").text,
                ),
                Unit(
                    "d:b2",
                    UnitKind.DIALOGUE,
                    2,
                    "[Sigh] I'm exhausted.",
                    Status.PENDING,
                    speaker="Ren",
                    masked_source=mask("[Sigh] I'm exhausted.").text,
                ),
                Unit(
                    "d:b3",
                    UnitKind.DIALOGUE,
                    3,
                    "[...]",
                    Status.PENDING,
                    speaker="Ren",
                    masked_source=mask("[...]").text,
                ),
                Unit(
                    "d:b4",
                    UnitKind.DIALOGUE,
                    4,
                    "[[Notice]] Maintenance tonight.",
                    Status.PENDING,
                    speaker="Ren",
                    masked_source=mask("[[Notice]] Maintenance tonight.").text,
                ),
            ]
            for u in units:
                db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(proj, db, cfg)

            llm_responses = {
                "d:b1": "⟦V1⟧ đã đến.",
                "d:b2": "⟦V1⟧ Tôi kiệt sức rồi.",
                "d:b3": "⟦V1⟧",
                "d:b4": "⟦E1⟧Thông báo]] Bảo trì đêm nay.",
            }

            expected_targets = {
                "d:b1": "[player_name] đã đến.",
                "d:b2": "[Sigh] Tôi kiệt sức rồi.",
                "d:b3": "[...]",
                "d:b4": "[[Thông báo]] Bảo trì đêm nay.",
            }

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=(llm_responses, 150, 60),
            ):
                stats = pipeline.run()

            assert stats.llm_translated == 4
            assert stats.failed == 0

            for uid, expected in expected_targets.items():
                saved = db.get_unit(uid)
                assert saved is not None
                assert saved.status == Status.TRANSLATED
                assert saved.target == expected
                assert not saved.qa_flags


class TestRunnerBootstrapAndBannerDisplay:
    """Stress-tests runner bootstrap idempotency, persona attributes, CLI flags,
    and banner display.
    """

    def test_bootstrap_if_needed_idempotency_triple_run(self, tmp_path: Path) -> None:
        """Repeated bootstrap_if_needed executions must be 100% idempotent and conflict-free."""
        proj = _create_adversarial_project(tmp_path, "boot_idempotent")

        char_data = {
            "liana": {
                "display_name": "Liana",
                "gender": "female",
                "role": "Heroine",
                "speech_style": "formal",
            },
            "ren": {
                "display_name": "Ren",
                "gender": "male",
                "role": "Hero",
                "speech_style": "casual",
            },
        }
        (proj.work_dir / "characters.json").write_text(json.dumps(char_data), encoding="utf-8")

        rel_graph = RelationshipGraph()
        rel_graph.add_edge(
            RelationshipEdge(
                source_tag="liana",
                target_tag="ren",
                kind=RelationshipKind.ROMANCE,
                speaker_pronoun="em",
                listener_pronoun="anh",
            )
        )
        rel_graph.save_to_json_file(proj.work_dir / "relationships.json")

        with proj.open_db() as db:
            # Run 1
            bootstrap_if_needed(proj, db, bootstrap=False)
            chars1 = db.get_characters()
            graph1 = RelationshipGraph.from_db(db)

            # Run 2
            bootstrap_if_needed(proj, db, bootstrap=False)
            chars2 = db.get_characters()
            graph2 = RelationshipGraph.from_db(db)

            # Run 3
            bootstrap_if_needed(proj, db, bootstrap=False)
            chars3 = db.get_characters()
            graph3 = RelationshipGraph.from_db(db)

            assert len(chars1) == len(chars2) == len(chars3) == 2
            assert graph1.edge_count == graph2.edge_count == graph3.edge_count == 1
            assert graph3.has_edge("liana", "ren")

    def test_bootstrap_preserves_all_11_persona_fields(self, tmp_path: Path) -> None:
        """Verify that bootstrap_if_needed loads all 11 persona fields from characters.json."""
        proj = _create_adversarial_project(tmp_path, "boot_persona")

        full_persona = {
            "valkyrie": {
                "display_name": "Sigrun the Valkyrie",
                "gender": "female",
                "role": "Divine Protector",
                "speech_style": "archaic, solemn, commanding",
                "archetype": "battle_maiden",
                "personality_traits": ["stoic", "loyal", "resolute"],
                "tone_and_voice": "deep resonant contralto with noble poise",
                "socioeconomic_status": "divine celestial nobility",
                "sample_dialogue": [
                    "Kẻ nào dám xâm phạm thánh địa này sẽ phải trả giá bằng linh hồn.",
                ],
                "pronoun_map": {"mortal": "ta - ngươi", "odin": "con - phụ thần"},
            }
        }
        (proj.work_dir / "characters.json").write_text(json.dumps(full_persona), encoding="utf-8")

        with proj.open_db() as db:
            bootstrap_if_needed(proj, db, bootstrap=False)
            chars = db.get_characters()
            assert "valkyrie" in chars
            p = chars["valkyrie"]

            assert p.tag == "valkyrie"
            assert p.display_name == "Sigrun the Valkyrie"
            assert p.gender == "female"
            assert p.role == "Divine Protector"
            assert p.speech_style == "archaic, solemn, commanding"
            assert p.archetype == "battle_maiden"
            assert p.personality_traits == ("stoic", "loyal", "resolute")
            assert p.tone_and_voice == "deep resonant contralto with noble poise"
            assert p.socioeconomic_status == "divine celestial nobility"
            assert p.sample_dialogue == (
                "Kẻ nào dám xâm phạm thánh địa này sẽ phải trả giá bằng linh hồn.",
            )
            assert p.pronoun_map == {"mortal": "ta - ngươi", "odin": "con - phụ thần"}

    def test_cli_flags_bootstrap_and_no_bootstrap_parsing(self, tmp_path: Path) -> None:
        """Verify runner CLI correctly maps --bootstrap and --no-bootstrap to boolean parameters."""
        proj = _create_adversarial_project(tmp_path, "cli_flags")

        with patch("renslate.ui.runner.run_unattended_pipeline") as mock_pipeline:
            mock_pipeline.return_value = 0

            # 1. --bootstrap
            runner_main(["-p", str(proj.root), "--bootstrap"])
            assert mock_pipeline.call_args[1]["bootstrap"] is True

            # 2. --no-bootstrap
            runner_main(["-p", str(proj.root), "--no-bootstrap"])
            assert mock_pipeline.call_args[1]["bootstrap"] is False

            # 3. Default (omitted)
            runner_main(["-p", str(proj.root)])
            assert mock_pipeline.call_args[1]["bootstrap"] is None

    def test_runner_banner_displays_character_and_relationship_counts(self, tmp_path: Path) -> None:
        """Verify ANSI startup banner renders exact counts for characters and relationship edges."""
        proj = _create_adversarial_project(tmp_path, "banner_with_rel")

        with proj.open_db() as db:
            db.save_character(
                CharacterProfile("c1", "Hero", "male", role="Hero", speech_style="brave")
            )
            db.save_character(
                CharacterProfile("c2", "Heroine", "female", role="Heroine", speech_style="kind")
            )
            db.save_character(
                CharacterProfile("c3", "Villain", "male", role="Boss", speech_style="sinister")
            )
            # Pre-mark unit as translated so unattended runner loop exits immediately
            db.save_unit(
                Unit("d:dummy", UnitKind.DIALOGUE, 1, "Hi", Status.TRANSLATED, target="Chào")
            )

            graph = RelationshipGraph()
            graph.add_edge(RelationshipEdge("c1", "c2", RelationshipKind.ROMANCE, "anh", "em"))
            graph.add_edge(RelationshipEdge("c2", "c1", RelationshipKind.ROMANCE, "em", "anh"))
            graph.add_edge(
                RelationshipEdge("c1", "c3", RelationshipKind.RIVAL_HOSTILE, "ta", "ngươi")
            )
            graph.save_to_db(db)

        buf = io.StringIO()
        with (
            patch("sys.stdout", buf),
            patch("renslate.ui.runner.wait_for_server", return_value=True),
            patch("renslate.ui.runner.detect_server_model", return_value="deepseek-v3.gguf"),
            patch("renslate.ui.runner.check_server_health", return_value=True),
            patch("renslate.ui.runner.TranslationPipeline") as mock_pipe_cls,
        ):
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                total_pending=0, failed=0, llm_translated=0, seeded=0, tm_reused=0
            )
            mock_pipe_cls.return_value = mock_instance

            exit_code = run_unattended_pipeline(proj.root, plain=True, bootstrap=False)

        assert exit_code == 0
        banner_output = buf.getvalue()
        assert "Nhân vật & Quan hệ" in banner_output
        assert "3 nhân vật, 3 quan hệ" in banner_output

    def test_runner_banner_omits_field_when_characters_and_relationships_empty(
        self, tmp_path: Path
    ) -> None:
        """Verify ANSI startup banner cleanly omits 'Nhân vật & Quan hệ' when both are empty."""
        proj = _create_adversarial_project(tmp_path, "banner_empty_rel")

        with proj.open_db() as db:
            db.save_unit(
                Unit("d:dummy", UnitKind.DIALOGUE, 1, "Hi", Status.TRANSLATED, target="Chào")
            )

        buf = io.StringIO()
        with (
            patch("sys.stdout", buf),
            patch("renslate.ui.runner.wait_for_server", return_value=True),
            patch("renslate.ui.runner.detect_server_model", return_value="deepseek-v3.gguf"),
            patch("renslate.ui.runner.check_server_health", return_value=True),
            patch("renslate.ui.runner.TranslationPipeline") as mock_pipe_cls,
        ):
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock(
                total_pending=0, failed=0, llm_translated=0, seeded=0, tm_reused=0
            )
            mock_pipe_cls.return_value = mock_instance

            exit_code = run_unattended_pipeline(proj.root, plain=True, bootstrap=False)

        assert exit_code == 0
        banner_output = buf.getvalue()
        assert "Nhân vật & Quan hệ" not in banner_output
