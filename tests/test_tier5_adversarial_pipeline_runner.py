"""Tier 5 White-Box Adversarial Coverage Hardening for Context, Translate & Runner.

Empirical verification harness targeting:
1. Nested parentheses, stage directions mixed with Japanese corner brackets
   (`「(Thinking...) Hello, Liana!」`).
2. Consecutive monologues by multiple different characters without dialogue turns.
3. Scene macro context budget compaction when scene summary exceeds token limit.
4. TranslationPipeline batch processing boundary conditions: single-item batches,
   empty batches, all-monologue batches.
5. Runner error recovery and bootstrap idempotency under edge conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from renslate.core.config import Config, LLMConfig
from renslate.core.errors import LLMError
from renslate.core.models import (
    CharacterProfile,
    GlossaryEntry,
    Origin,
    SceneSummary,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.context import (
    ContextBuilder,
    InterlocutorResolver,
    _analyze_monologue_and_dialogue,
    _strip_outer_quotes,
    format_rich_dialogue_line,
    infer_scene_cues,
)
from renslate.engine.translate import TranslationPipeline, TranslationStats, partition_batches
from renslate.ui.runner import (
    bootstrap_if_needed,
    check_server_health,
    detect_server_model,
    query_db_status,
    run_unattended_pipeline,
    wait_for_server,
)


def _setup_test_project(tmp_path: Path, name: str = "t5_proj") -> Project:
    root = tmp_path / name
    (root / "game").mkdir(parents=True, exist_ok=True)
    proj = Project.create(root, target_lang="vietnamese")
    proj.tl_root.mkdir(parents=True, exist_ok=True)
    return proj


def _make_unit(
    uid: str,
    source: str,
    speaker: str | None = None,
    seq: int = 1,
    label: str | None = None,
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        status=Status.PENDING,
        speaker=speaker,
        label=label,
    )


# ==============================================================================
# 1. Nested Parentheses, Stage Directions & Japanese Corner Brackets
# ==============================================================================


class TestTier5AdversarialNestedParenthesesAndJapaneseBrackets:
    """Stress-test strip quotes, stage direction analysis, and vocative extraction."""

    def test_strip_outer_quotes_nested_combinations(self) -> None:
        """Verify peeling of heterogeneous nested outer quote brackets."""
        assert _strip_outer_quotes('「"『Hello』"」') == "『Hello』"
        assert _strip_outer_quotes("“‘Hello’“") == "“‘Hello’“"  # noqa: RUF001
        assert _strip_outer_quotes("«“Hello”»") == "Hello"
        assert _strip_outer_quotes("「Hello」") == "Hello"
        assert _strip_outer_quotes("   「  Hello world  」  ") == "Hello world"

    def test_stage_direction_in_japanese_corner_brackets_with_dialogue(self) -> None:
        """Verify 「(Thinking...) Hello, Liana!」 extracts dialogue and detects vocative."""
        text = "「(Thinking...) Hello, Liana!」"
        is_mono, clean_dial = _analyze_monologue_and_dialogue(text)
        assert not is_mono
        assert clean_dial == "Hello, Liana!"

        resolver = InterlocutorResolver()
        unit = _make_unit("u1", text, speaker="Ren")
        resolved = resolver.resolve(unit, known_names={"liana": "Liana"})

        assert not resolved.is_monologue
        assert resolved.speaker == "Ren"
        assert resolved.listener == "liana"
        assert resolved.resolution_method == "vocative"
        assert resolved.detected_emotion == "hesitant"

        char_map = {
            "ren": CharacterProfile("ren", "Ren", "male"),
            "liana": CharacterProfile("liana", "Liana", "female"),
        }
        rich_line = format_rich_dialogue_line(unit, resolved, char_map)
        assert "[Người nói: Ren -> Người nghe: Liana" in rich_line
        assert text in rich_line

    def test_stage_direction_in_japanese_brackets_without_dialogue(self) -> None:
        """Verify 「(Thinking deeply...)」 collapses to monologue."""
        text = "「(Thinking deeply...)」"
        is_mono, clean_dial = _analyze_monologue_and_dialogue(text)
        assert is_mono
        assert clean_dial == ""

        resolver = InterlocutorResolver()
        unit = _make_unit("u1", text, speaker="Ren")
        resolved = resolver.resolve(unit, known_names={"liana": "Liana"})

        assert resolved.is_monologue
        assert resolved.listener is None
        assert resolved.resolution_method == "monologue"

    def test_nested_parentheses_monologue(self) -> None:
        """Verify ((Deep inside nested parentheses)) is classified as monologue."""
        text = "((Deep inside my mind, I wondered if this was real.))"
        is_mono, clean_dial = _analyze_monologue_and_dialogue(text)
        assert is_mono
        assert clean_dial == ""

        resolver = InterlocutorResolver()
        unit = _make_unit("u1", text, speaker="Ren")
        resolved = resolver.resolve(unit)
        assert resolved.is_monologue
        assert resolved.listener is None

    @pytest.mark.parametrize(
        ("text", "expected_mono", "expected_listener"),
        [
            ("『(Whispers to self...)』", True, None),
            ("“‘(Laughs) Hey, Liana!’”", False, "liana"),  # noqa: RUF001
            ("「（sighs） Liana, are you ready?」", False, "liana"),  # noqa: RUF001
            ("«(Clears throat) Dear Arthur, please listen.»", False, "arthur"),
            ("„(Chuckles) Good morning, Eileen!”", False, "eileen"),
            ("（Thinking...）", True, None),  # noqa: RUF001
        ],
    )
    def test_mixed_quotes_and_brackets_variations(
        self,
        text: str,
        expected_mono: bool,
        expected_listener: str | None,
    ) -> None:
        """Stress-test diverse bracket/quote combinations across Japanese and Western styles."""
        resolver = InterlocutorResolver()
        known = {
            "liana": "Liana",
            "arthur": "Arthur",
            "eileen": "Eileen",
        }
        unit = _make_unit("u1", text, speaker="Protagonist")
        resolved = resolver.resolve(unit, known_names=known)
        assert resolved.is_monologue == expected_mono
        assert resolved.listener == expected_listener

    def test_stage_direction_only_punctuation_or_symbols(self) -> None:
        """Verify stage direction followed only by punctuation is treated as reaction/monologue."""
        text = "「(Gasps) ...!?」"
        is_mono, clean_dial = _analyze_monologue_and_dialogue(text)
        assert is_mono
        assert clean_dial == ""

        resolver = InterlocutorResolver()
        unit = _make_unit("u1", text, speaker="Hero")
        resolved = resolver.resolve(unit)
        assert resolved.is_monologue
        assert resolved.listener is None

    def test_kinship_address_following_stage_direction(self) -> None:
        """Verify stage directions preceding kinship vocatives resolve correctly."""
        text = "「(Smiling gently) Good morning, Father!」"
        resolver = InterlocutorResolver()
        active = {"hero", "king_father"}
        known = {"king_father": "King Father"}
        unit = _make_unit("u1", text, speaker="hero")
        resolved = resolver.resolve(unit, active_speakers=active, known_names=known)
        assert not resolved.is_monologue
        assert resolved.listener == "king_father"
        assert resolved.resolution_method == "vocative"


# ==============================================================================
# 2. Consecutive Monologues by Multiple Characters
# ==============================================================================


class TestTier5AdversarialConsecutiveMonologues:
    """Stress-test sequences of thoughts without interactive turns."""

    def test_consecutive_monologues_multiple_distinct_speakers(self) -> None:
        """Verify consecutive monologues by 3 distinct characters remain isolated."""
        resolver = InterlocutorResolver()
        known = {"alice": "Alice", "bob": "Bob", "clara": "Clara"}

        units = [
            _make_unit("u1", "(I need to find a way out...)", speaker="alice", seq=1),
            _make_unit("u2", "(He looks troubled, but I won't pry.)", speaker="bob", seq=2),
            _make_unit("u3", "「(What a quiet room...)」", speaker="clara", seq=3),
        ]

        prev: Unit | None = None
        for u in units:
            resolved = resolver.resolve(u, prev_unit=prev, known_names=known)
            assert resolved.is_monologue, f"Unit {u.uid} should be monologue"
            assert resolved.listener is None, f"Unit {u.uid} should have no listener"
            assert resolved.resolution_method == "monologue"
            prev = u

    def test_consecutive_monologue_does_not_pollute_later_dialogue(self) -> None:
        """Verify unaddressed dialogue after thoughts does not falsely turn-take."""
        resolver = InterlocutorResolver()
        known = {"alice": "Alice", "bob": "Bob"}

        u1 = _make_unit("u1", "(I should say something...)", speaker="alice", seq=1)
        r1 = resolver.resolve(u1, known_names=known)
        assert r1.is_monologue

        u2 = _make_unit("u2", "(She is trembling...)", speaker="bob", seq=2)
        r2 = resolver.resolve(u2, prev_unit=u1, known_names=known)
        assert r2.is_monologue

        u3 = _make_unit("u3", "The storm is getting heavier.", speaker="alice", seq=3)
        r3 = resolver.resolve(u3, prev_unit=u2, known_names=known)
        assert not r3.is_monologue
        # Turn-taking must NOT bind to bob because bob's line was a monologue!
        assert r3.listener is None
        assert r3.resolution_method == "default"

    def test_monologue_interleaved_in_two_person_dialogue(self) -> None:
        """Verify interlocutor flow when an inner thought separates two dialogue turns."""
        resolver = InterlocutorResolver()
        known = {"ren": "Ren", "eileen": "Eileen"}

        # 1. Ren speaks to Eileen with vocative
        u1 = _make_unit("u1", "Eileen, are you listening?", speaker="ren", seq=1)
        r1 = resolver.resolve(u1, known_names=known)
        assert r1.listener == "eileen"

        # 2. Eileen has an inner thought
        u2 = _make_unit("u2", "(Why is he looking at me like that?)", speaker="eileen", seq=2)
        r2 = resolver.resolve(u2, prev_unit=u1, known_names=known)
        assert r2.is_monologue
        assert r2.listener is None

        # 3. Eileen speaks aloud to Ren
        u3 = _make_unit("u3", "Ren, I am fine, don't worry.", speaker="eileen", seq=3)
        r3 = resolver.resolve(u3, prev_unit=u2, known_names=known)
        assert not r3.is_monologue
        assert r3.listener == "ren"
        assert r3.resolution_method == "vocative"

    def test_all_monologue_batch_prompt_formatting(self) -> None:
        """Verify prompt builder formats all lines with [Độc thoại/Nội tâm] when active."""
        cfg = Config()
        cb = ContextBuilder(cfg)
        chars = {
            "ren": CharacterProfile("ren", "Ren", "male"),
            "eileen": CharacterProfile("eileen", "Eileen", "female"),
        }
        units = [
            _make_unit("d:1", "(Thinking alone...)", speaker="ren", seq=1),
            _make_unit("d:2", "『Another inner contemplation...』", speaker="eileen", seq=2),
        ]
        prompt = cb.build_user_prompt(units, characters=chars, resolver=InterlocutorResolver())
        assert "d:1 | [Độc thoại/Nội tâm: Ren" in prompt
        assert "d:2 | [Độc thoại/Nội tâm: Eileen" in prompt
        # Ensure no reciprocal dialogue rule is asserted
        assert "đang đối thoại trực tiếp" not in prompt


# ==============================================================================
# 3. Scene Macro Context Budget Compaction
# ==============================================================================


class TestTier5AdversarialSceneMacroContextBudgetCompaction:
    """Stress-test token budgeting and pruning of scene macro summaries."""

    def test_scene_macro_context_fits_when_budget_sufficient(self) -> None:
        """Verify macro context is included when context capacity is ample."""
        cfg = Config(llm=LLMConfig(n_ctx=6144))
        cb = ContextBuilder(cfg)
        units = [
            _make_unit(f"d:{i}", f"Line content {i}", speaker="hero", seq=i) for i in range(5)
        ]
        summary = SceneSummary(
            scene_id="s1",
            first_seq=1,
            last_seq=5,
            label="scene_castle_night_tense",
            summary="Hero infiltrates the ancient castle.",
        )
        prompt = cb.build_user_prompt(units, scene_summary=summary)
        assert "BỐI CẢNH PHÂN CẢNH & TÓM TẮT CẢNH HIỆN TẠI (SCENE MACRO CONTEXT):" in prompt
        assert "Hero infiltrates the ancient castle." in prompt
        assert "scene_castle_night_tense" in prompt
        assert "Địa điểm (Location): Lâu đài" in prompt
        assert "Không khí / Cảm xúc (Atmosphere): Đêm khuya" in prompt

    def test_scene_macro_context_dropped_when_budget_exceeded(self) -> None:
        """Verify huge scene summary exceeding available budget is safely excluded."""
        cfg = Config(llm=LLMConfig(n_ctx=2048))
        cb = ContextBuilder(cfg)
        units = [
            _make_unit(f"d:{i}", f"Line content {i}", speaker="hero", seq=i) for i in range(10)
        ]
        huge_summary = "Long narrative description of historical events... " * 800
        summary = SceneSummary(
            scene_id="s2",
            first_seq=1,
            last_seq=10,
            label="scene_castle_night_tense",
            summary=huge_summary,
        )
        prompt = cb.build_user_prompt(units, scene_summary=summary)
        assert "SCENE MACRO CONTEXT" not in prompt
        for u in units:
            assert u.uid in prompt

    def test_scene_macro_context_inferred_cues_fallback(self) -> None:
        """Verify location and atmosphere inference from diverse label keywords."""
        loc, atmos = infer_scene_cues("scene_school_classroom_sunset_rain_romance")
        assert "Trường học" in loc or "Lớp học" in loc
        assert "Hoàng hôn" in atmos
        assert "Trời mưa" in atmos
        assert "Lãng mạn" in atmos

    def test_token_budget_starvation_prioritizes_dialogue_units(self) -> None:
        """Verify that under severe token starvation, batch units are strictly preserved."""
        cfg = Config(llm=LLMConfig(n_ctx=1024))
        cb = ContextBuilder(cfg)
        units = [
            _make_unit(f"d:{i}", "A short line." * 5, speaker="hero", seq=i) for i in range(10)
        ]
        summary = SceneSummary(
            scene_id="s3",
            first_seq=1,
            last_seq=10,
            label="scene_dungeon_dark",
            summary="A short summary.",
        )
        glossary = [GlossaryEntry("sword", "kiếm", "weapon")]
        prompt = cb.build_user_prompt(units, glossary_matches=glossary, scene_summary=summary)
        for u in units:
            assert u.uid in prompt


# ==============================================================================
# 4. TranslationPipeline Batch Processing Boundary Conditions
# ==============================================================================


class TestTier5AdversarialPipelineBatchBoundaries:
    """Stress-test single-item, empty, and all-monologue batches."""

    def test_partition_batches_boundary_conditions(self) -> None:
        """Verify partition_batches with empty, single-element, and label jumps."""
        assert partition_batches([]) == []

        u1 = _make_unit("u1", "Hello")
        assert partition_batches([u1]) == [[u1]]

        u_big = _make_unit("u_big", "Big payload " * 100)
        u2 = _make_unit("u2", "Short")
        batches = partition_batches([u_big, u2], max_tokens=100)
        assert len(batches) == 2

    def test_single_item_batch_successful_translation(self, tmp_path: Path) -> None:
        """Verify a single-item batch completes cleanly with LLM single key response."""
        project = _setup_test_project(tmp_path, "single_item_success")
        with project.open_db() as db:
            u = _make_unit("d:single", "Hello world!", speaker="Hero")
            db.save_units_bulk([u])

            pipe = TranslationPipeline(project, db, Config())
            stats = TranslationStats()
            mock_res = ({"d:single": "Xin chào thế giới!"}, 20, 10)
            with patch.object(pipe.llm, "complete_json", return_value=mock_res):
                failed = pipe._translate_batch([u], [], stats)
                assert failed == []
                assert stats.llm_translated == 1

            saved = db.get_unit("d:single")
            assert saved is not None
            assert saved.target == "Xin chào thế giới!"
            assert saved.status == Status.TRANSLATED

    def test_single_item_batch_failure_marks_failed_immediately(self, tmp_path: Path) -> None:
        """Verify a failing single-item batch is marked failed without infinite retry."""
        project = _setup_test_project(tmp_path, "single_item_fail")
        with project.open_db() as db:
            u = _make_unit("d:single_fail", "A test line.", speaker="Hero")
            db.save_units_bulk([u])

            pipe = TranslationPipeline(project, db, Config())
            stats = TranslationStats()
            with patch.object(pipe.llm, "complete_json", side_effect=LLMError("Server down")):
                failed = pipe._translate_batch([u], [], stats)
                assert failed == []
                assert stats.failed == 1

            saved = db.get_unit("d:single_fail")
            assert saved is not None
            assert saved.status == Status.FAILED
            assert any("LLM error" in str(flag) for flag in saved.qa_flags)

    def test_empty_batch_handling(self, tmp_path: Path) -> None:
        """Verify _translate_batch([]) and pipeline.run() handle empty inputs cleanly."""
        project = _setup_test_project(tmp_path, "empty_batch_proj")
        with project.open_db() as db:
            pipe = TranslationPipeline(project, db, Config())
            stats = TranslationStats()
            with patch.object(pipe.llm, "complete_json", return_value=({}, 0, 0)):
                failed = pipe._translate_batch([], [], stats)
                assert failed == []
                assert stats.llm_translated == 0

            run_stats = pipe.run()
            assert run_stats.total_pending == 0

    def test_all_monologue_batch_execution_and_qa_clearing(self, tmp_path: Path) -> None:
        """Verify all-monologue batch is translated, frames peeled, and cleanly committed."""
        project = _setup_test_project(tmp_path, "all_monologue_proj")
        with project.open_db() as db:
            units = [
                _make_unit("d:1", "(I need to run.)", speaker="Ren", seq=1),
                _make_unit("d:2", "『Is this the end?』", speaker="Eileen", seq=2),
                _make_unit("d:3", "（Silence fills the room.）", speaker="Liana", seq=3),  # noqa: RUF001
                _make_unit("d:4", "「(Sighs) Whatever.」", speaker="Ren", seq=4),
            ]
            db.save_units_bulk(units)

            pipe = TranslationPipeline(project, db, Config())
            stats = TranslationStats()

            mock_llm_response = {
                "d:1": "[Độc thoại/Nội tâm: Ren] Tôi cần phải chạy.",
                "d:2": "[Độc thoại/Nội tâm: Eileen] Đây là kết thúc sao?",
                "d:3": "[Độc thoại/Nội tâm: Liana] Sự im lặng bao trùm căn phòng.",
                "d:4": "[Độc thoại/Nội tâm: Ren] Thở dài Sao cũng được.",
            }
            with patch.object(pipe.llm, "complete_json", return_value=(mock_llm_response, 50, 40)):
                failed = pipe._translate_batch(units, [], stats)
                assert failed == []
                assert stats.llm_translated == 4

            u1_saved = db.get_unit("d:1")
            assert u1_saved is not None
            assert u1_saved.target == "Tôi cần phải chạy."
            assert u1_saved.status == Status.TRANSLATED

            u2_saved = db.get_unit("d:2")
            assert u2_saved is not None
            assert u2_saved.target == "Đây là kết thúc sao?"
            assert u2_saved.status == Status.TRANSLATED

    def test_mixed_batch_retry_split_to_singles(self, tmp_path: Path) -> None:
        """Verify pipeline isolates corrupted lines in multi-item batch and commits valid lines."""
        project = _setup_test_project(tmp_path, "mixed_batch_split")
        with project.open_db() as db:
            units = [
                _make_unit("d:good1", "Good line one.", speaker="Hero", seq=1),
                _make_unit("d:bad", "Bad line that fails.", speaker="Hero", seq=2),
                _make_unit("d:good2", "Good line two.", speaker="Hero", seq=3),
            ]
            db.save_units_bulk(units)

            pipe = TranslationPipeline(project, db, Config())
            call_count = 0

            def mock_complete(
                sys_p: str, user_p: str, **kwargs: object
            ) -> tuple[dict[str, str], int, int]:
                nonlocal call_count
                call_count += 1
                if "d:bad" in user_p and ("d:good1" in user_p or "d:good2" in user_p):
                    # Multi-unit batch: omit d:bad to trigger retry
                    return {"d:good1": "Dòng tốt một.", "d:good2": "Dòng tốt hai."}, 30, 20
                if "d:bad" in user_p:
                    # Single unit retry: raise LLMError
                    raise LLMError("Bad line permanently fails")
                if "d:good1" in user_p:
                    return {"d:good1": "Dòng tốt một."}, 10, 10
                if "d:good2" in user_p:
                    return {"d:good2": "Dòng tốt hai."}, 10, 10
                return {}, 0, 0

            with patch.object(pipe.llm, "complete_json", side_effect=mock_complete):
                stats = pipe.run()
                assert stats.llm_translated == 2
                assert stats.failed == 1

            saved_g1 = db.get_unit("d:good1")
            saved_g2 = db.get_unit("d:good2")
            saved_bad = db.get_unit("d:bad")

            assert saved_g1 is not None and saved_g1.status == Status.TRANSLATED
            assert saved_g2 is not None and saved_g2.status == Status.TRANSLATED
            assert saved_bad is not None and saved_bad.status == Status.FAILED


# ==============================================================================
# 5. Runner Error Recovery & Bootstrap Idempotency
# ==============================================================================


class TestTier5AdversarialRunnerErrorRecoveryAndBootstrap:
    """Stress-test runner bootstrap, server health monitor, and loop control."""

    def test_bootstrap_if_needed_idempotency_five_iterations(self, tmp_path: Path) -> None:
        """Verify calling bootstrap_if_needed 5 consecutive times produces identical state."""
        project = _setup_test_project(tmp_path, "bootstrap_idempotent")
        chars_file = project.work_dir / "characters.json"
        chars_file.write_text(
            json.dumps({
                "ren": {"display_name": "Ren", "gender": "male", "role": "Protagonist"},
                "eileen": {"display_name": "Eileen", "gender": "female", "role": "Heroine"},
            }),
            encoding="utf-8",
        )
        glossary_file = project.work_dir / "glossary.json"
        glossary_file.write_text(
            json.dumps([{"term": "Sect", "translation": "Tông môn", "notes": ""}]),
            encoding="utf-8",
        )

        with project.open_db() as db:
            for _ in range(5):
                bootstrap_if_needed(project, db)

            chars = db.get_characters()
            assert len(chars) == 2
            assert "ren" in chars
            assert "eileen" in chars

            cur_glossary = db.get_glossary()
            assert len(cur_glossary) == 1
            assert cur_glossary[0].term == "Sect"

    def test_bootstrap_with_corrupted_characters_json(self, tmp_path: Path) -> None:
        """Verify bootstrap_if_needed handles corrupted characters.json without crashing."""
        project = _setup_test_project(tmp_path, "corrupted_chars")
        chars_file = project.work_dir / "characters.json"
        chars_file.write_text("{broken json syntax: [", encoding="utf-8")

        with project.open_db() as db:
            bootstrap_if_needed(project, db)
            assert len(db.get_characters()) == 0

    def test_bootstrap_llm_server_offline_behavior(self, tmp_path: Path) -> None:
        """Verify bootstrap skips LLM enrichment gracefully when server is offline."""
        project = _setup_test_project(tmp_path, "bootstrap_offline")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male", role=""))

            with (
                patch("renslate.ui.runner.check_server_health", return_value=False),
                patch("renslate.ui.runner.bootstrap_characters_with_llm") as mock_boot,
            ):
                bootstrap_if_needed(project, db)
                mock_boot.assert_not_called()

    def test_server_health_check_handling(self) -> None:
        """Verify check_server_health and detect_server_model under connection failures."""
        assert not check_server_health("http://127.0.0.1:99999/v1", timeout_s=0.1)
        model = detect_server_model(
            "http://127.0.0.1:99999/v1", default_name="fallback-model", timeout_s=0.1
        )
        assert model == "fallback-model"

    def test_runner_stop_requested_before_first_batch(self, tmp_path: Path) -> None:
        """Verify runner exits with returncode 0 when stop is requested immediately."""
        project = _setup_test_project(tmp_path, "stop_req_proj")
        with (
            patch("renslate.ui.runner.check_server_health", return_value=True),
            patch("renslate.ui.runner._is_stop_requested", return_value=True),
        ):
            rc = run_unattended_pipeline(project.root)
            assert rc == 0

    def test_runner_server_offline_waits_and_aborts_on_stop(self, tmp_path: Path) -> None:
        """Verify wait_for_server returns False if stop requested while server offline."""
        with (
            patch("renslate.ui.runner.check_server_health", return_value=False),
            patch("renslate.ui.runner._is_stop_requested", side_effect=[False, True]),
        ):
            ok = wait_for_server("http://127.0.0.1:8080/v1", check_interval=0.01)
            assert not ok

    def test_runner_retry_flagged_resets_qa_flagged_units(self, tmp_path: Path) -> None:
        """Verify --retry-flagged resets qa_flagged status to pending."""
        project = _setup_test_project(tmp_path, "retry_flagged_proj")
        with project.open_db() as db:
            u = _make_unit("d:flagged", "Hello", speaker="Hero")
            db.save_units_bulk([u])
            db.update_translation(
                "d:flagged",
                "Xin chào 你好",
                Origin.LLM,
                Status.QA_FLAGGED,
                qa_flags=("cjk_leakage",),
            )

            counts = query_db_status(db)
            assert counts.get("qa_flagged") == 1

            with (
                patch("renslate.ui.runner.wait_for_server", return_value=True),
                patch("renslate.ui.runner.check_server_health", return_value=True),
                patch("renslate.ui.runner._is_stop_requested", side_effect=[False, True]),
                patch("renslate.ui.runner.TranslationPipeline.run") as mock_run,
            ):
                mock_run.return_value = TranslationStats()
                run_unattended_pipeline(project.root, retry_flagged=True)

            counts_after = query_db_status(db)
            assert counts_after.get("qa_flagged", 0) == 0
            assert counts_after.get("pending") == 1
