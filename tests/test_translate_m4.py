"""Comprehensive Milestone 4 tests for TranslationPipeline and runner relationship integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

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
from renslate.engine.context import InterlocutorResolver
from renslate.engine.graph import RelationshipGraph
from renslate.engine.translate import TranslationPipeline
from renslate.ui.runner import bootstrap_if_needed
from renslate.ui.runner import main as runner_main


def _setup_test_project(tmp_path: Path, name: str = "m4_proj") -> Project:
    root = tmp_path / name
    (root / "game").mkdir(parents=True, exist_ok=True)
    return Project.create(root, target_lang="vietnamese")


class TestPipelineInitialization:
    """Verifies pipeline initialization with explicit vs auto-hydrated graphs and resolvers."""

    def test_pipeline_explicit_injection(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "explicit_init")
        with project.open_db() as db:
            custom_graph = RelationshipGraph()
            custom_graph.add_edge(
                RelationshipEdge(
                    source_tag="hero",
                    target_tag="heroine",
                    kind=RelationshipKind.ROMANCE,
                    speaker_pronoun="anh",
                    listener_pronoun="em",
                )
            )
            custom_resolver = InterlocutorResolver()
            cfg = Config()

            pipeline = TranslationPipeline(
                project, db, cfg, graph=custom_graph, resolver=custom_resolver
            )
            assert pipeline.graph is custom_graph
            assert pipeline.resolver is custom_resolver
            assert pipeline.graph.has_edge("hero", "heroine")

    def test_pipeline_auto_hydrate_from_relationships_json(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "json_init")
        rel_graph = RelationshipGraph()
        rel_graph.add_edge(
            RelationshipEdge(
                source_tag="alice",
                target_tag="bob",
                kind=RelationshipKind.ROMANCE,
                speaker_pronoun="em",
                listener_pronoun="anh",
                confidence=0.95,
                source_type="manual",
                notes="Lover",
            )
        )
        rel_file = project.work_dir / "relationships.json"
        rel_graph.save_to_json_file(rel_file)

        with project.open_db() as db:
            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)

            assert pipeline.graph is not None
            assert pipeline.graph.has_edge("alice", "bob")
            edge = pipeline.graph.get_edge("alice", "bob")
            assert edge is not None
            assert edge.speaker_pronoun == "em"
            assert edge.listener_pronoun == "anh"

    def test_pipeline_auto_hydrate_from_db(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "db_init")
        with project.open_db() as db:
            db_graph = RelationshipGraph()
            db_graph.add_edge(
                RelationshipEdge(
                    source_tag="master",
                    target_tag="disciple",
                    kind=RelationshipKind.MENTOR_DISCIPLE,
                    speaker_pronoun="ta",
                    listener_pronoun="con",
                )
            )
            db_graph.save_to_db(db)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)
            assert pipeline.graph.has_edge("master", "disciple")
            edge = pipeline.graph.get_edge("master", "disciple")
            assert edge is not None
            assert edge.kind == RelationshipKind.MENTOR_DISCIPLE

    def test_pipeline_default_empty_fallback(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "empty_init")
        with project.open_db() as db:
            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)
            assert isinstance(pipeline.graph, RelationshipGraph)
            assert pipeline.graph.edge_count == 0
            assert isinstance(pipeline.resolver, InterlocutorResolver)


class TestPromptAssemblyAndInterlocutorFraming:
    """Verifies build_user_prompt receives graph and resolver and creates rich frames."""

    def test_pipeline_build_user_prompt_rich_frames(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "rich_frames")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male", role="Protagonist"))
            db.save_character(CharacterProfile("eileen", "Eileen", "female", role="Heroine"))

            u1 = Unit(
                "d:1",
                UnitKind.DIALOGUE,
                1,
                "Eileen, are you ready?",
                Status.PENDING,
                speaker="Ren",
            )
            u2 = Unit(
                "d:2",
                UnitKind.DIALOGUE,
                2,
                "Yes, I am ready.",
                Status.PENDING,
                speaker="Eileen",
            )
            db.save_unit(u1)
            db.save_unit(u2)

            graph = RelationshipGraph()
            graph.add_edge(
                RelationshipEdge(
                    source_tag="ren",
                    target_tag="eileen",
                    kind=RelationshipKind.ROMANCE,
                    speaker_pronoun="anh",
                    listener_pronoun="em",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    source_tag="eileen",
                    target_tag="ren",
                    kind=RelationshipKind.ROMANCE,
                    speaker_pronoun="em",
                    listener_pronoun="anh",
                )
            )

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg, graph=graph)

            captured_prompts: list[str] = []

            def mock_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                captured_prompts.append(user_prompt)
                return (
                    {
                        "d:1": "Eileen, em đã sẵn sàng chưa?",
                        "d:2": "Vâng, em đã sẵn sàng rồi.",
                    },
                    100,
                    50,
                )

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_complete):
                stats = pipeline.run()
                assert stats.llm_translated == 2

            assert len(captured_prompts) > 0
            prompt = captured_prompts[0]
            # Verify rich framing with interlocutors and reciprocal pronouns
            assert "[Người nói: Ren -> Người nghe: Eileen" in prompt or "[Người nói: Ren" in prompt
            assert "ROMANCE" in prompt
            assert "QUAN HỆ XƯNG HÔ ĐỐI THOẠI TRONG PHÂN ĐOẠN" in prompt

    def test_pipeline_monologue_framing(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "monologue_frames")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            # Monologue / internal thought in parentheses
            u = Unit(
                "d:mono",
                UnitKind.DIALOGUE,
                1,
                "(What should I do now...?)",
                Status.PENDING,
                speaker="Ren",
            )
            db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)

            captured_prompts: list[str] = []

            def mock_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                captured_prompts.append(user_prompt)
                return ({"d:mono": "(Mình nên làm gì bây giờ...?)"}, 50, 20)

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_complete):
                stats = pipeline.run()
                assert stats.llm_translated == 1

            assert len(captured_prompts) > 0
            prompt = captured_prompts[0]
            assert "[Độc thoại/Nội tâm: Ren" in prompt

    def test_scene_transition_resets_resolver(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "scene_reset")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            db.save_character(CharacterProfile("vane", "Lord Vane", "male"))

            u1 = Unit(
                "d:1",
                UnitKind.DIALOGUE,
                1,
                "Father, be careful!",
                Status.PENDING,
                speaker="Ren",
                label="scene_morning",
            )
            u2 = Unit(
                "d:2",
                UnitKind.DIALOGUE,
                2,
                "Night has fallen.",
                Status.PENDING,
                speaker="Ren",
                label="scene_night",
            )
            db.save_unit(u1)
            db.save_unit(u2)

            cfg = Config(batch=BatchConfig(max_lines=1))  # Force separate batches per unit
            pipeline = TranslationPipeline(project, db, cfg)

            with (
                patch.object(
                    pipeline.resolver, "reset", wraps=pipeline.resolver.reset
                ) as mock_reset,
                patch.object(
                    pipeline.llm,
                    "complete_json",
                    return_value=({"d:1": "Cha cẩn thận!", "d:2": "Màn đêm buông xuống."}, 50, 20),
                ),
            ):
                pipeline.run()

            # reset() called on initial batch (label change from None -> scene_morning)
            # and on second batch (label change from scene_morning -> scene_night)
            assert mock_reset.call_count >= 2


class TestSanitizationAndAntiLeakGates:
    """Verifies clean_translation peels echoes and QA catches leaked metadata."""

    def test_pipeline_peels_echoed_metadata_frames(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "echo_peeling")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            u = Unit(
                "d:echo",
                UnitKind.DIALOGUE,
                1,
                "Hello Eileen!",
                Status.PENDING,
                speaker="Ren",
            )
            db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)

            # LLM echoes the prompt framing tags
            raw_llm_output = (
                '[Người nói: Ren -> Người nghe: Eileen | Quan hệ: ROMANCE] "Xin chào Eileen!"'
            )

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=({"d:echo": raw_llm_output}, 60, 25),
            ):
                stats = pipeline.run()
                assert stats.llm_translated == 1

            saved = db.get_unit("d:echo")
            assert saved is not None
            assert saved.target == "Xin chào Eileen!"
            assert saved.status == Status.TRANSLATED
            assert not saved.qa_flags

    def test_pipeline_catches_internal_leaked_metadata_and_retries(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "qa_leak")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            u = Unit(
                "d:leak",
                UnitKind.DIALOGUE,
                1,
                "Good morning.",
                Status.PENDING,
                speaker="Ren",
            )
            db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)

            # Metadata leaked inside the body of text (cannot be cleanly peeled from edges)
            leaked_output = "Chào buổi sáng [Quan hệ: ROMANCE] nhé em."

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=({"d:leak": leaked_output}, 50, 20),
            ):
                stats = pipeline.run()
                # Single-unit batch fails QA check with fatal error
                assert stats.failed == 1
                assert stats.llm_translated == 0

            saved = db.get_unit("d:leak")
            assert saved is not None
            assert saved.status == Status.FAILED
            # Target was not committed to DB as translated
            assert saved.target != leaked_output

    def test_pipeline_preserves_authentic_renpy_brackets(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "bracket_preservation")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            u1 = Unit(
                "d:b1",
                UnitKind.DIALOGUE,
                1,
                "[player_name] arrived safely.",
                Status.PENDING,
                speaker="Ren",
            )
            u2 = Unit(
                "d:b2",
                UnitKind.DIALOGUE,
                2,
                "[Sigh] What a day.",
                Status.PENDING,
                speaker="Ren",
            )
            db.save_unit(u1)
            db.save_unit(u2)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=(
                    {
                        "d:b1": "[player_name] đã đến nơi an toàn.",
                        "d:b2": "[Thở dài] Thật là một ngày mệt mỏi.",
                    },
                    70,
                    30,
                ),
            ):
                stats = pipeline.run()
                assert stats.llm_translated == 2

            saved1 = db.get_unit("d:b1")
            assert saved1 is not None
            assert saved1.target is not None and "[player_name]" in saved1.target

            saved2 = db.get_unit("d:b2")
            assert saved2 is not None
            assert saved2.target is not None and "[Thở dài]" in saved2.target

    def test_pipeline_pronoun_inversion_qa_gate(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "pronoun_inversion")
        with project.open_db() as db:
            db.save_character(CharacterProfile("eileen", "Eileen", "female"))
            db.save_character(CharacterProfile("ren", "Ren", "male"))

            # Eileen speaks to Ren with vocative: 2nd-person "Do you..."
            u = Unit(
                "d:inv",
                UnitKind.DIALOGUE,
                1,
                "Ren, do you want to die?",
                Status.PENDING,
                speaker="eileen",
            )
            db.save_unit(u)

            graph = RelationshipGraph()
            # Eileen calls Ren 'anh' and calls herself 'em'
            graph.add_edge(
                RelationshipEdge(
                    source_tag="eileen",
                    target_tag="ren",
                    kind=RelationshipKind.ROMANCE,
                    speaker_pronoun="em",
                    listener_pronoun="anh",
                )
            )

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg, graph=graph)

            # Inverted pronoun: Eileen translates "Do you" as "Ren, em muốn chết sao?"
            inverted_target = "Ren, em muốn chết sao?"

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=({"d:inv": inverted_target}, 40, 15),
            ):
                stats = pipeline.run()
                # Should fail QA with pronoun_inversion error
                assert stats.failed == 1
                assert stats.llm_translated == 0


class TestRunnerAndCLIIntegration:
    """Verifies runner.py relationship graph synchronization and banner display."""

    def test_bootstrap_if_needed_syncs_relationships(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "runner_bootstrap")
        rel_graph = RelationshipGraph()
        rel_graph.add_edge(
            RelationshipEdge(
                source_tag="charlotte",
                target_tag="mc",
                kind=RelationshipKind.ROMANCE,
                speaker_pronoun="em",
                listener_pronoun="anh",
                confidence=1.0,
                source_type="manual",
                notes="Main heroine",
            )
        )
        rel_file = project.work_dir / "relationships.json"
        rel_graph.save_to_json_file(rel_file)

        with project.open_db() as db:
            bootstrap_if_needed(project, db, bootstrap=False)

            # Verify relationship was synced to DB
            graph = RelationshipGraph.from_db(db)
            assert graph.has_edge("charlotte", "mc")
            edge = graph.get_edge("charlotte", "mc")
            assert edge is not None
            assert edge.speaker_pronoun == "em"
            assert edge.listener_pronoun == "anh"

    def test_runner_main_bootstrap_flags(self, tmp_path: Path) -> None:
        project = _setup_test_project(tmp_path, "cli_bootstrap")

        with patch("renslate.ui.runner.run_unattended_pipeline") as mock_run:
            mock_run.return_value = 0

            # Test --bootstrap
            runner_main(["-p", str(project.root), "--bootstrap"])
            assert mock_run.call_args[1]["bootstrap"] is True

            # Test --no-bootstrap
            runner_main(["-p", str(project.root), "--no-bootstrap"])
            assert mock_run.call_args[1]["bootstrap"] is False

            # Test default
            runner_main(["-p", str(project.root)])
            assert mock_run.call_args[1]["bootstrap"] is None
