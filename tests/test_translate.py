"""Tests for renslate.translate."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from renslate.config import Config
from renslate.errors import LLMError
from renslate.models import Origin, Status, TMEntry, Unit, UnitKind
from renslate.project import Project
from renslate.translate import TranslationPipeline, partition_batches


def test_partition_batches() -> None:
    units = [
        Unit(f"d:{i}", UnitKind.DIALOGUE, i, f"Line {i}", Status.PENDING, label=f"lbl_{i // 20}")
        for i in range(50)
    ]
    batches = partition_batches(units, max_lines=20, max_tokens=1000)
    assert len(batches) >= 3
    assert sum(len(b) for b in batches) == 50


def test_translation_pipeline_seeds_tm_and_llm(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    with project.open_db() as db:
        # 1. UI seed item ("Start")
        db.save_unit(Unit("s:start", UnitKind.UI, 1, "Start", Status.PENDING))
        # 2. Item already in TM
        db.save_tm(TMEntry(src_norm="Known in TM", target="Đã có trong TM", uses=1))
        db.save_unit(Unit("s:known", UnitKind.STRING, 2, "Known in TM", Status.PENDING))
        # 3. Requires LLM
        db.save_unit(Unit("d:need_llm", UnitKind.DIALOGUE, 3, "Hello, friend!", Status.PENDING))

    # Mock the LLM client call
    with (
        project.open_db() as db,
        patch(
            "renslate.llm.LLMClient.complete_json",
            return_value=({"d:need_llm": "Xin chào, người bạn!"}, 40, 15),
        ),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        assert stats.total_pending == 3
        assert stats.seeded == 1
        assert stats.tm_reused == 1
        assert stats.llm_translated == 1

        # Check DB updates
        u_seed = db.get_unit("s:start")
        assert u_seed is not None
        assert u_seed.target == "Bắt đầu"
        assert u_seed.origin is Origin.SEED

        u_tm = db.get_unit("s:known")
        assert u_tm is not None
        assert u_tm.target == "Đã có trong TM"
        assert u_tm.origin is Origin.TM

        u_llm = db.get_unit("d:need_llm")
        assert u_llm is not None
        assert u_llm.target == "Xin chào, người bạn!"
        assert u_llm.origin is Origin.LLM


def test_translation_failure_marks_failed_unit(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    with project.open_db() as db:
        db.save_unit(Unit("d:fail_line", UnitKind.DIALOGUE, 1, "Unusual line.", Status.PENDING))

    with (
        project.open_db() as db,
        patch(
            "renslate.llm.LLMClient.complete_json",
            side_effect=LLMError("Server timeout"),
        ),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()
        assert stats.failed == 1

        unit = db.get_unit("d:fail_line")
        assert unit is not None
        assert unit.status is Status.FAILED

        # Re-running normally without include_failed finds 0 pending units
        stats_rerun = pipeline.run()
        assert stats_rerun.total_pending == 0

        # Running with include_failed finds the failed unit
        stats_retry = pipeline.run(include_failed=True)
        assert stats_retry.total_pending == 1


def test_partial_batch_recovery_commits_valid_units(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    with project.open_db() as db:
        db.save_unit(Unit("d:u1", UnitKind.DIALOGUE, 1, "Line one.", Status.PENDING))
        db.save_unit(Unit("d:u2", UnitKind.DIALOGUE, 2, "Line two {b}bold{/b}.", Status.PENDING))
        db.save_unit(Unit("d:u3", UnitKind.DIALOGUE, 3, "Line three.", Status.PENDING))

    calls: list[str] = []

    def mock_complete_json(
        system_prompt: str, user_prompt: str, **kwargs: object
    ) -> tuple[dict[str, str], int, int]:
        if "d:u1" in user_prompt and "d:u3" in user_prompt:
            calls.append("batch")
            # d:u2 has inverted tags ({/b} before {b}), which triggers tag_order error
            return (
                {
                    "d:u1": "Dòng một.",
                    "d:u2": "Dòng hai {/b}lỗi{b}.",
                    "d:u3": "Dòng ba.",
                },
                50,
                30,
            )
        if "d:u2" in user_prompt:
            calls.append("retry_u2")
            # Retry also fails
            return ({"d:u2": "Dòng hai {/b}vẫn lỗi{b}."}, 20, 10)
        return ({}, 0, 0)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        # Both calls occurred: 1 batch + 1 single line retry for u2
        assert calls == ["batch", "retry_u2"]
        assert stats.llm_translated == 2
        assert stats.failed == 1

        u1 = db.get_unit("d:u1")
        assert u1 is not None and u1.status is Status.TRANSLATED and u1.target == "Dòng một."

        u3 = db.get_unit("d:u3")
        assert u3 is not None and u3.status is Status.TRANSLATED and u3.target == "Dòng ba."

        u2 = db.get_unit("d:u2")
        assert u2 is not None and u2.status is Status.FAILED


def test_cross_chunk_lookback_retention(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    with project.open_db() as db:
        db.save_unit(Unit("d:1", UnitKind.DIALOGUE, 0, "Hello chunk 1", Status.PENDING))
        db.save_unit(Unit("d:2", UnitKind.DIALOGUE, 1, "World chunk 1", Status.PENDING))
        db.save_unit(Unit("d:3", UnitKind.DIALOGUE, 2, "Hello chunk 2", Status.PENDING))
        db.save_unit(Unit("d:4", UnitKind.DIALOGUE, 3, "World chunk 2", Status.PENDING))

    # Chunk 1: translates d:1 and d:2
    with (
        project.open_db() as db,
        patch(
            "renslate.llm.LLMClient.complete_json",
            return_value=(
                {"d:1": "Chào đoạn 1", "d:2": "Thế giới đoạn 1"},
                30,
                15,
            ),
        ),
    ):
        p1 = TranslationPipeline(project, db, cfg)
        stats1 = p1.run(limit=2)
        assert stats1.llm_translated == 2

    # Chunk 2: translates d:3 and d:4 in a separate pipeline instance
    captured_prompts: list[str] = []

    def mock_chunk2(
        system_prompt: str, user_prompt: str, **kwargs: object
    ) -> tuple[dict[str, str], int, int]:
        captured_prompts.append(user_prompt)
        return ({"d:3": "Chào đoạn 2", "d:4": "Thế giới đoạn 2"}, 30, 15)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_chunk2),
    ):
        p2 = TranslationPipeline(project, db, cfg)
        stats2 = p2.run(limit=2)
        assert stats2.llm_translated == 2
        assert len(captured_prompts) >= 1
        prompt = captured_prompts[0]
        assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" in prompt
        assert "Chào đoạn 1" in prompt
        assert "Thế giới đoạn 1" in prompt


def test_bounded_lookback_deque(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    # lookback = 2 -> deque maxlen = 8
    cfg = Config.from_dict({"batch": {"lookback": 2, "max_lines": 2}})

    with project.open_db() as db:
        for i in range(12):
            db.save_unit(Unit(f"d:{i}", UnitKind.DIALOGUE, i, f"Source line {i}", Status.PENDING))

    with (
        project.open_db() as db,
        patch(
            "renslate.llm.LLMClient.complete_json",
            side_effect=lambda sys, usr, **kwargs: (
                {f"d:{i}": f"Dịch {i}" for i in range(12)},
                20,
                10,
            ),
        ),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()
        assert stats.llm_translated == 12

    # Verify that recent_translated_units bounds query
    with project.open_db() as db:
        recent = db.recent_translated_units(limit=cfg.batch.lookback * 4)
        assert len(recent) == 8
        assert [u.uid for u in recent] == [f"d:{i}" for i in range(4, 12)]


def test_natural_dialogue_expansion_not_rejected(tmp_path: Path) -> None:
    """Short dialogue lines naturally expanding beyond 20 chars must NOT be rejected."""
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    with project.open_db() as db:
        # "What?" is 5 chars, translation is 26 chars (ratio > 5.0, len > 20)
        db.save_unit(Unit("d:short1", UnitKind.DIALOGUE, 1, "What?", Status.PENDING))
        # "What happened?" is 14 chars, translation is 28 chars (ratio 2.0, len > 20)
        db.save_unit(Unit("d:short2", UnitKind.DIALOGUE, 2, "What happened?", Status.PENDING))

    with (
        project.open_db() as db,
        patch(
            "renslate.llm.LLMClient.complete_json",
            return_value=(
                {
                    "d:short1": "Có chuyện gì thế hả bạn?",
                    "d:short2": "Đã có chuyện gì xảy ra vậy?",
                },
                30,
                20,
            ),
        ),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        # Both units translated successfully without single-line fallback retry
        assert stats.llm_translated == 2
        assert stats.failed == 0

        u1 = db.get_unit("d:short1")
        assert u1 is not None and u1.status in (Status.TRANSLATED, Status.QA_FLAGGED)
        assert u1.target == "Có chuyện gì thế hả bạn?"

        u2 = db.get_unit("d:short2")
        assert u2 is not None and u2.status in (Status.TRANSLATED, Status.QA_FLAGGED)
        assert u2.target == "Đã có chuyện gì xảy ra vậy?"


def test_batch_schema_cjk_ban_pattern(tmp_path: Path) -> None:
    """Batch schema must include CJK ban pattern when source has no CJK."""
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    with project.open_db() as db:
        db.save_unit(Unit("d:1", UnitKind.DIALOGUE, 1, "Hello world", Status.PENDING))

    captured_schema: dict[str, Any] = {}

    def mock_complete_json(
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        nonlocal captured_schema
        if schema:
            captured_schema = schema
        return ({"d:1": "Xin chào thế giới"}, 20, 10)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        pipeline.run()

        assert "d:1" in captured_schema.get("properties", {})
        pattern = captured_schema["properties"]["d:1"].get("pattern")
        assert pattern is not None
        assert "\\u4e00-\\u9fff" in pattern


def test_partition_batches_splits_on_kind_change() -> None:
    units = [
        Unit("d:1", UnitKind.DIALOGUE, 1, "Dialogue 1", Status.PENDING),
        Unit("d:2", UnitKind.DIALOGUE, 2, "Dialogue 2", Status.PENDING),
        Unit("s:1", UnitKind.STRING, 3, "Choice 1", Status.PENDING),
        Unit("s:2", UnitKind.STRING, 4, "Choice 2", Status.PENDING),
    ]
    batches = partition_batches(units, max_lines=20, max_tokens=1000)
    assert len(batches) == 2
    assert [u.uid for u in batches[0]] == ["d:1", "d:2"]
    assert [u.uid for u in batches[1]] == ["s:1", "s:2"]


def test_outer_tag_wrapper_auto_healing_in_pipeline(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    # Source has symmetric outer tag {i}...{i}
    with project.open_db() as db:
        db.save_unit(Unit("d:burst", UnitKind.DIALOGUE, 1, "{i}It'll burst!{i}", Status.PENDING))

    # Model returns raw text without placeholders or tags
    def mock_complete_json(
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        return ({"d:burst": "Nó sẽ nổ tung!"}, 15, 10)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        assert stats.failed == 0
        u = db.get_unit("d:burst")
        assert u is not None
        assert u.status == Status.TRANSLATED
        assert u.target == "{i}Nó sẽ nổ tung!{i}"


def test_prefix_tolerant_uid_resolution(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    with project.open_db() as db:
        # Units with s: and d: prefixes
        db.save_unit(Unit("s:choice_1", UnitKind.STRING, 1, "Refuse to pay", Status.PENDING))
        db.save_unit(Unit("d:speech_1", UnitKind.DIALOGUE, 2, "Hello there!", Status.PENDING))

    # Model stripped the s: and d: prefixes in JSON response keys
    def mock_complete_json(
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        return ({"choice_1": "Từ chối trả tiền", "speech_1": "Xin chào!"}, 20, 15)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        assert stats.failed == 0
        u1 = db.get_unit("s:choice_1")
        u2 = db.get_unit("d:speech_1")
        assert u1 is not None and u1.target == "Từ chối trả tiền"
        assert u2 is not None and u2.target == "Xin chào!"


def test_realign_outer_tags_mismatched_size(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    src_text = "{size=50}{font=fonts/chinese-rocks.otf}Later, at night.{/font}{/size}"
    with project.open_db() as db:
        db.save_unit(Unit("s:ui_1", UnitKind.STRING, 1, src_text, Status.PENDING))

    # Model hallucinated size=60 instead of size=50 in raw tag output
    def mock_complete_json(
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], int, int]:
        return (
            {"s:ui_1": "{size=60}{font=fonts/chinese-rocks.otf}Sau đó, vào ban đêm.{/font}{/size}"},
            20,
            15,
        )

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        assert stats.failed == 0
        u1 = db.get_unit("s:ui_1")
        assert u1 is not None
        assert u1.status == Status.TRANSLATED
        expected = "{size=50}{font=fonts/chinese-rocks.otf}Sau đó, vào ban đêm.{/font}{/size}"
        assert u1.target == expected
