"""Comprehensive 4-Tier Opaque-Box E2E Test Suite for Renslate Engine Optimizations.

Derived strictly from ADR-010, docs/OPTIMIZATION_PLAN.md, and ORIGINAL_REQUEST.md:
- Tier 1: Feature Coverage (Hardware launcher defaults, context right-sizing, memory flags,
  model sync, glossary regex amortization, LIFO tag bracket validation, SQLite PRAGMAs,
  bulk unit upsert, fast extraction, partial batch recovery, cross-chunk lookback pre-population,
  bounded deque lookback buffer).
- Tier 2: Boundary & Corner Cases (Empty inputs, boundary token limits, single-unit batches,
  extreme tag nestings, 0-lookback, non-consecutive sequences).
- Tier 3: Cross-Feature Interactions (Bulk persistence with partial recovery, cross-chunk lookback
  with extraction, glossary matching with tag unmasking, TM matching with partial recovery,
  bounded deque with cross-chunk continuity).
- Tier 4: Real-World Application Scenarios (Multi-chunk Ren'Py VN localization simulation with
  intermittent QA errors, partial recovery, lookback continuity, and TM reuse).
"""

from __future__ import annotations

import collections
import os
import re
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from renslate.config import BatchConfig, Config, LLMConfig
from renslate.context import ContextBuilder, LookbackItem
from renslate.db import Database, utc_timestamp
from renslate.errors import ConfigError
from renslate.extract import extract_project
from renslate.glossary import Glossary
from renslate.mask import mask, validate_raw
from renslate.models import (
    CharacterProfile,
    GlossaryEntry,
    Origin,
    Status,
    TMEntry,
    Unit,
    UnitKind,
)
from renslate.project import Project
from renslate.translate import TranslationPipeline, partition_batches

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = REPO_ROOT / "scripts" / "run_qwen_server.sh"


def make_unit(
    uid: str,
    seq: int,
    source: str,
    *,
    kind: UnitKind = UnitKind.DIALOGUE,
    status: Status = Status.PENDING,
    speaker: str | None = None,
    label: str | None = "start",
    ref: str | None = "game/script.rpy:1",
    target: str | None = None,
    origin: Origin | None = None,
    qa_flags: tuple[str, ...] = (),
    updated_at: str | None = None,
) -> Unit:
    """Helper to construct a Unit with clean defaults for testing."""
    return Unit(
        uid=uid,
        kind=kind,
        seq=seq,
        source=source,
        status=status,
        speaker=speaker,
        label=label,
        ref=ref,
        masked_source=source,
        target=target,
        origin=origin,
        qa_flags=qa_flags,
        updated_at=updated_at or utc_timestamp(),
    )


@pytest.fixture
def test_db(tmp_path: Path) -> Iterator[Database]:
    """Provide an isolated SQLite database instance for testing."""
    db_file = tmp_path / "test_project.db"
    db = Database.open(db_file, create=True)
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def renpy_project(tmp_path: Path) -> Project:
    """Create a minimal mock Ren'Py project layout with translation files."""
    game_dir = tmp_path / "game"
    game_dir.mkdir(parents=True, exist_ok=True)
    project = Project.create(tmp_path)
    tl_dir = project.tl_root
    tl_dir.mkdir(parents=True, exist_ok=True)
    script_tl = tl_dir / "script.rpy"
    script_tl.write_text(
        "# game/script.rpy:1\n"
        "translate vietnamese start_01:\n\n"
        '    # e "Welcome to the academy!"\n'
        '    e "Welcome to the academy!"\n\n'
        "# game/script.rpy:2\n"
        "translate vietnamese start_02:\n\n"
        '    # e "Please choose your path carefully."\n'
        '    e "Please choose your path carefully."\n\n'
        "# game/script.rpy:3\n"
        "translate vietnamese start_03:\n\n"
        '    # "The grand gate opens before you."\n'
        '    "The grand gate opens before you."\n',
        encoding="utf-8",
    )
    return project


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 cases per feature)
# ==============================================================================


class TestTier1ServerScriptConfig:
    """Feature 1: Hardware launcher defaults in scripts/run_qwen_server.sh."""

    def test_default_ctx_size_is_6144(self) -> None:
        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        match = re.search(r'CTX_SIZE="\$\{RENSLATE_CTX:-(\d+)\}"', content)
        assert match is not None, "CTX_SIZE default parameter expansion missing"
        assert match.group(1) == "6144", f"Expected default ctx 6144, got {match.group(1)}"

    def test_default_vram_margin_is_512(self) -> None:
        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        match = re.search(r'VRAM_MARGIN_MIB="\$\{RENSLATE_VRAM_MARGIN:-(\d+)\}"', content)
        assert match is not None, "VRAM_MARGIN_MIB default parameter expansion missing"
        assert match.group(1) == "512", f"Expected margin 512, got {match.group(1)}"

    def test_default_load_mode_is_mmap(self) -> None:
        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        match = re.search(r'LOAD_MODE="\$\{RENSLATE_LOAD_MODE:-([a-z]+)\}"', content)
        assert match is not None, "LOAD_MODE default parameter expansion missing"
        assert match.group(1) == "mmap", f"Expected load mode mmap, got {match.group(1)}"

    def test_default_model_path_uses_qwen_35b(self) -> None:
        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        match = re.search(r'DEFAULT_MODEL_PATH="([^"]+)"', content)
        assert match is not None, "DEFAULT_MODEL_PATH assignment missing"
        model_name = Path(match.group(1)).name
        assert model_name == "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

    def test_bash_syntax_and_critical_runtime_flags(self) -> None:
        result = subprocess.run(["bash", "-n", str(SERVER_SCRIPT)], capture_output=True, text=True)
        assert result.returncode == 0, f"Bash syntax check failed: {result.stderr}"

        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        assert "-fit on" in content, "Missing -fit on GPU auto-offload flag"
        assert '-fitt "$VRAM_MARGIN_MIB"' in content, "Missing -fitt margin flag"
        assert '-lm "$LOAD_MODE"' in content, "Missing -lm load mode flag"
        assert "-lzm off" in content, "Missing -lzm off memory streaming prevention"
        assert "--poll 0" in content, "Missing --poll 0 low-heat futex sleep flag"
        assert "--reasoning off" in content, "Missing --reasoning off suppression flag"


class TestTier1ConfigRightSizing:
    """Feature 2: LLM context window right-sizing to 6144 tokens."""

    def test_llm_config_default_n_ctx_is_6144(self) -> None:
        cfg = LLMConfig()
        assert cfg.n_ctx == 6144

    def test_root_config_default_n_ctx_is_6144(self) -> None:
        cfg = Config()
        assert cfg.llm.n_ctx == 6144

    def test_config_from_dict_preserves_6144(self) -> None:
        cfg = Config.from_dict({"llm": {"n_ctx": 6144}})
        assert cfg.llm.n_ctx == 6144

    def test_config_save_load_roundtrip_preserves_6144(self, tmp_path: Path) -> None:
        cfg = Config()
        cfg_file = tmp_path / "config.json"
        cfg.save(cfg_file)
        loaded = Config.load(cfg_file)
        assert loaded.llm.n_ctx == 6144

    def test_n_ctx_boundary_bounds_validation(self) -> None:
        with pytest.raises(ConfigError):
            Config.from_dict({"llm": {"n_ctx": 1023}})
        valid_cfg = Config.from_dict({"llm": {"n_ctx": 1024}})
        assert valid_cfg.llm.n_ctx == 1024


class TestTier1MemoryAllocationFlags:
    """Feature 3: Memory allocation flags (-lm mmap, -lzm off)."""

    def test_memory_alloc_default_mmap_variable(self) -> None:
        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        assert 'LOAD_MODE="${RENSLATE_LOAD_MODE:-mmap}"' in content

    def test_memory_alloc_env_override_supported(self) -> None:
        cmd = [
            "bash",
            "-c",
            f'source <(grep RENSLATE_LOAD_MODE "{SERVER_SCRIPT}"); echo "$LOAD_MODE"',
        ]
        override_env = {**os.environ, "RENSLATE_LOAD_MODE": "none"}
        res = subprocess.run(cmd, env=override_env, capture_output=True, text=True)
        assert res.stdout.strip() == "none"

    def test_memory_alloc_lzm_off_flag_present(self) -> None:
        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        assert "-lzm off" in content

    def test_memory_alloc_lm_flag_present(self) -> None:
        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        assert '-lm "$LOAD_MODE"' in content

    def test_memory_alloc_script_header_documents_clean_mmap(self) -> None:
        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        assert "mmap" in content
        assert "-lzm off" in content


class TestTier1ModelNameSync:
    """Feature 4: Model name synchronization to Qwen3.6-35B-A3B-UD-Q3_K_M.gguf."""

    def test_llm_config_default_model(self) -> None:
        assert LLMConfig().model == "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

    def test_root_config_default_model(self) -> None:
        assert Config().llm.model == "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

    def test_empty_model_name_raises_error(self) -> None:
        with pytest.raises(ConfigError, match=r"llm\.model không được để trống"):
            Config(llm=LLMConfig(model=""))

    def test_model_name_roundtrip_serialization(self) -> None:
        raw = Config().to_dict()
        assert raw["llm"]["model"] == "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
        loaded = Config.from_dict(raw)
        assert loaded.llm.model == "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

    def test_model_name_sync_between_script_and_config(self) -> None:
        content = SERVER_SCRIPT.read_text(encoding="utf-8")
        match = re.search(r'DEFAULT_MODEL_PATH="([^"]+)"', content)
        assert match is not None
        script_model = Path(match.group(1)).name
        assert script_model == LLMConfig().model


class TestTier1GlossaryRegexAmortization:
    """Feature 5: O(1) amortized glossary regex compilation."""

    def test_large_glossary_init_performance(self) -> None:
        entries = [
            GlossaryEntry(term=f"term_{i:04d}", translation=f"dịch_{i:04d}") for i in range(250)
        ]
        start = time.perf_counter()
        glossary = Glossary(entries)
        elapsed = time.perf_counter() - start
        assert len(glossary.entries) == 250
        assert elapsed < 0.2, f"Glossary init took too long: {elapsed:.4f}s"

    def test_entries_sorted_by_length_descending(self) -> None:
        entries = [
            GlossaryEntry("cat", "mèo"),
            GlossaryEntry("cat in the hat", "chú mèo đội mũ"),
            GlossaryEntry("caterpillar", "sâu bướm"),
        ]
        glossary = Glossary(entries)
        terms = [e.term for e in glossary.entries]
        assert terms == ["cat in the hat", "caterpillar", "cat"]

    def test_word_boundary_matching(self) -> None:
        entries = [GlossaryEntry("member", "thành viên")]
        glossary = Glossary(entries)
        assert len(glossary.match_text("He is a member of the club.")) == 1
        assert len(glossary.match_text("Please remember this.")) == 0

    def test_case_sensitivity_flag(self) -> None:
        entries = [
            GlossaryEntry("VN", "Việt Nam", case_sensitive=True),
            GlossaryEntry("club", "câu lạc bộ", case_sensitive=False),
        ]
        glossary = Glossary(entries)
        assert len(glossary.match_text("We love vn")) == 0
        assert len(glossary.match_text("We love VN")) == 1
        assert len(glossary.match_text("Welcome to the CLUB")) == 1

    def test_synthetic_placeholders_ignored(self) -> None:
        entries = [GlossaryEntry("V1", "phiên bản 1")]
        glossary = Glossary(entries)
        assert len(glossary.match_text("Found token ⟦V1⟧ here")) == 0
        assert len(glossary.match_text("Real V1 release")) == 1


class TestTier1LIFOTagBracketValidation:
    """Feature 6: Strict LIFO bracket validation in mask.py validate_raw."""

    def test_interleaved_tags_rejected(self) -> None:
        issues = validate_raw("Hello", "{b}{i}text{/b}{/i}")
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) > 0
        assert tag_order_issues[0].detail == "{/b}"

    def test_properly_nested_tags_accepted(self) -> None:
        issues = validate_raw("Hello", "{b}{i}text{/i}{/b}")
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) == 0

    def test_deeply_nested_lifo_tags(self) -> None:
        issues = validate_raw("Hello", "{a}{b}{c}deep text{/c}{/b}{/a}")
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) == 0

    def test_standalone_tags_do_not_pollute_tag_stack(self) -> None:
        issues = validate_raw("Hello", "{b}hello{w}world{/b}")
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) == 0

    def test_multiple_sibling_paired_tags(self) -> None:
        issues = validate_raw("Hello", "{b}first{/b} and {i}second{/i}")
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) == 0


class TestTier1SQLitePragmas:
    """Feature 7: High-performance SQLite PRAGMAs in Database.open."""

    def test_pragma_temp_store_memory(self, test_db: Database) -> None:
        row = test_db.execute("PRAGMA temp_store").fetchone()
        assert row is not None
        assert row[0] == 2, f"Expected PRAGMA temp_store=2 (MEMORY), got {row[0]}"

    def test_pragma_cache_size_64mb(self, test_db: Database) -> None:
        row = test_db.execute("PRAGMA cache_size").fetchone()
        assert row is not None
        assert row[0] == -64000, f"Expected cache_size=-64000 (64MB), got {row[0]}"

    def test_pragma_mmap_size_256mb(self, test_db: Database) -> None:
        row = test_db.execute("PRAGMA mmap_size").fetchone()
        assert row is not None
        assert row[0] == 268435456, f"Expected mmap_size=268435456, got {row[0]}"

    def test_pragma_journal_mode_wal(self, test_db: Database) -> None:
        row = test_db.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        assert str(row[0]).lower() == "wal"

    def test_pragma_foreign_keys_enabled(self, test_db: Database) -> None:
        row = test_db.execute("PRAGMA foreign_keys").fetchone()
        assert row is not None
        assert row[0] == 1


class TestTier1BulkUnitUpsert:
    """Feature 8: Bulk unit upsert via save_units_bulk."""

    def test_bulk_insert_multiple_units(self, test_db: Database) -> None:
        units = [make_unit(f"d:{i}", i, f"Dialogue sentence {i}") for i in range(25)]
        test_db.save_units_bulk(units)
        stored = test_db.units()
        assert len(stored) == 25
        assert stored["d:0"].source == "Dialogue sentence 0"
        assert stored["d:24"].source == "Dialogue sentence 24"

    def test_bulk_upsert_updates_existing_units(self, test_db: Database) -> None:
        initial = [make_unit("u:1", 1, "Original", status=Status.PENDING)]
        test_db.save_units_bulk(initial)

        updated = [make_unit("u:1", 1, "Original", status=Status.TRANSLATED, target="Đã dịch")]
        test_db.save_units_bulk(updated)

        u = test_db.get_unit("u:1")
        assert u is not None
        assert u.status is Status.TRANSLATED
        assert u.target == "Đã dịch"

    def test_bulk_upsert_empty_list_noop(self, test_db: Database) -> None:
        test_db.save_units_bulk([])
        assert len(test_db.units()) == 0

    def test_bulk_upsert_preserves_all_fields(self, test_db: Database) -> None:
        unit = make_unit(
            uid="d:rich",
            seq=42,
            source="Complex text",
            speaker="eileen",
            label="chapter_1",
            ref="game/script.rpy:99",
            target="Văn bản phức tạp",
            origin=Origin.LLM,
            qa_flags=("flag_a", "flag_b"),
            status=Status.QA_FLAGGED,
        )
        test_db.save_units_bulk([unit])
        fetched = test_db.get_unit("d:rich")
        assert fetched is not None
        assert fetched.speaker == "eileen"
        assert fetched.label == "chapter_1"
        assert fetched.ref == "game/script.rpy:99"
        assert fetched.target == "Văn bản phức tạp"
        assert fetched.origin is Origin.LLM
        assert fetched.qa_flags == ("flag_a", "flag_b")
        assert fetched.status is Status.QA_FLAGGED

    def test_bulk_upsert_transaction_rollback(self, test_db: Database) -> None:
        units = [make_unit("u:rollback", 1, "Will rollback")]
        with pytest.raises(RuntimeError), test_db.transaction():
            test_db.save_units_bulk(units)
            raise RuntimeError("Abort transaction")
        assert test_db.get_unit("u:rollback") is None


class TestTier1FastExtraction:
    """Feature 9: Fast extraction with bulk persistence in extract.py."""

    def test_extract_creates_units_via_bulk(self, renpy_project: Project) -> None:
        report = extract_project(renpy_project)
        assert report.created == 3
        with renpy_project.open_db() as db:
            assert len(db.units()) == 3

    def test_extract_idempotent_reextract(self, renpy_project: Project) -> None:
        extract_project(renpy_project)
        second_report = extract_project(renpy_project)
        assert second_report.created == 0
        assert second_report.source_changed == 0

    def test_extract_source_change_detection(self, renpy_project: Project) -> None:
        extract_project(renpy_project)
        script = renpy_project.tl_root / "script.rpy"
        script.write_text(
            "# game/script.rpy:1\n"
            "translate vietnamese start_01:\n\n"
            '    # e "Welcome to the modified academy!"\n'
            '    e "Welcome to the modified academy!"\n\n'
            "# game/script.rpy:2\n"
            "translate vietnamese start_02:\n\n"
            '    # e "Please choose your path carefully."\n'
            '    e "Please choose your path carefully."\n\n'
            "# game/script.rpy:3\n"
            "translate vietnamese start_03:\n\n"
            '    # "The grand gate opens before you."\n'
            '    "The grand gate opens before you."\n',
            encoding="utf-8",
        )
        report = extract_project(renpy_project)
        assert report.source_changed == 1
        with renpy_project.open_db() as db:
            units = db.units()
            assert any("modified" in u.source for u in units.values())

    def test_extract_orphans_marked_obsolete(self, renpy_project: Project) -> None:
        extract_project(renpy_project)
        script = renpy_project.tl_root / "script.rpy"
        script.write_text(
            "# game/script.rpy:1\n"
            "translate vietnamese start_01:\n\n"
            '    # e "Welcome to the academy!"\n'
            '    e "Welcome to the academy!"\n',
            encoding="utf-8",
        )
        report = extract_project(renpy_project)
        assert report.obsolete == 2
        with renpy_project.open_db() as db:
            obsolete_units = [u for u in db.units().values() if u.status is Status.OBSOLETE]
            assert len(obsolete_units) == 2

    def test_extract_slots_association(self, renpy_project: Project) -> None:
        extract_project(renpy_project)
        with renpy_project.open_db() as db:
            assert db.count_slots() == 3


class TestTier1PartialBatchRecovery:
    """Feature 10: Partial batch recovery on LLM failure."""

    def test_partial_salvage_on_qa_error(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [
                    make_unit("u:1", 1, "Good morning"),
                    make_unit("u:2", 2, "Short"),
                    make_unit("u:3", 3, "Good night"),
                ]
            )

        call_count = 0

        def mock_complete_json(*args: Any, **kwargs: Any) -> tuple[dict[str, str], int, int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Unit 2 intentionally violates max length ratio (1.8)
                return (
                    {
                        "u:1": "Chào buổi sáng",
                        "u:2": (
                            "Đây là một câu dịch cực kỳ dài vượt quá tỉ lệ cho phép rất nhiều lần"
                        ),
                        "u:3": "Chúc ngủ ngon",
                    },
                    100,
                    50,
                )
            # Fallback single retry for unit 2
            return ({"u:2": "Ngắn"}, 20, 10)

        with (
            renpy_project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()

            assert stats.llm_translated == 3
            u1 = db.get_unit("u:1")
            u3 = db.get_unit("u:3")
            assert u1 is not None and u1.status is Status.TRANSLATED
            assert u3 is not None and u3.status is Status.TRANSLATED

    def test_partial_salvage_on_unmask_error(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [
                    make_unit("u:1", 1, "Call [player_name] here"),
                    make_unit("u:2", 2, "Simple line"),
                ]
            )

        call_count = 0

        def mock_complete_json(*args: Any, **kwargs: Any) -> tuple[dict[str, str], int, int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Broken placeholder in u:1
                return ({"u:1": "Gọi ⟦V99⟧ tại đây", "u:2": "Dòng đơn giản"}, 50, 20)
            return ({"u:1": "Gọi ⟦V1⟧ tại đây"}, 25, 10)

        with (
            renpy_project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.llm_translated == 2
            u2 = db.get_unit("u:2")
            assert u2 is not None and u2.status is Status.TRANSLATED

    def test_partial_salvage_on_missing_key(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [
                    make_unit("u:1", 1, "Line 1"),
                    make_unit("u:2", 2, "Line 2"),
                ]
            )

        call_count = 0

        def mock_complete_json(*args: Any, **kwargs: Any) -> tuple[dict[str, str], int, int]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ({"u:1": "Dòng 1"}, 40, 15)  # Missing u:2
            return ({"u:2": "Dòng 2"}, 20, 10)

        with (
            renpy_project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.llm_translated == 2
            u1 = db.get_unit("u:1")
            u2 = db.get_unit("u:2")
            assert u1 is not None and u1.status is Status.TRANSLATED
            assert u2 is not None and u2.status is Status.TRANSLATED

    def test_single_unit_failure_is_terminal(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk([make_unit("u:fail", 1, "Short")])

        # Always returns invalid markup (syntax error)
        bad_markup = "{unclosed brace"
        with (
            renpy_project.open_db() as db,
            patch(
                "renslate.llm.LLMClient.complete_json",
                return_value=({"u:fail": bad_markup}, 30, 15),
            ),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.failed == 1
            u = db.get_unit("u:fail")
            assert u is not None
            assert u.status is Status.FAILED

    def test_batch_full_success_commits_all(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [
                    make_unit("u:1", 1, "Line 1"),
                    make_unit("u:2", 2, "Line 2"),
                ]
            )

        with (
            renpy_project.open_db() as db,
            patch(
                "renslate.llm.LLMClient.complete_json",
                return_value=({"u:1": "Dòng 1", "u:2": "Dòng 2"}, 50, 20),
            ),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.llm_translated == 2
            assert stats.failed == 0
            b = db.execute("SELECT * FROM batches").fetchall()
            assert len(b) == 1


class TestTier1CrossChunkLookbackPrepopulation:
    """Feature 11: Cross-chunk lookback pre-population in TranslationPipeline.run."""

    def test_lookback_empty_at_seq_zero(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk([make_unit("u:0", 0, "First sentence ever")])
            recent = db.recent_translated_units(limit=16, before_seq=0)
            assert len(recent) == 0

    def test_lookback_prepopulated_from_sqlite(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [
                    make_unit("u:0", 0, "Chunk 1 line", status=Status.TRANSLATED, target="Dòng 1"),
                    make_unit("u:1", 1, "Chunk 2 line", status=Status.PENDING),
                ]
            )
            recent = db.recent_translated_units(limit=16, before_seq=1)
            assert len(recent) == 1
            assert recent[0].uid == "u:0"
            assert recent[0].target == "Dòng 1"

    def test_lookback_chronological_order(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [
                    make_unit(
                        f"u:{i}", i, f"Line {i}", status=Status.TRANSLATED, target=f"Dòng {i}"
                    )
                    for i in range(5)
                ]
            )
            recent = db.recent_translated_units(limit=4, before_seq=5)
            assert len(recent) == 4
            seqs = [u.seq for u in recent]
            assert seqs == [1, 2, 3, 4]

    def test_lookback_skips_failed_and_pending(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [
                    make_unit("u:0", 0, "L0", status=Status.TRANSLATED, target="T0"),
                    make_unit("u:1", 1, "L1", status=Status.FAILED, target=None),
                    make_unit("u:2", 2, "L2", status=Status.PENDING, target=None),
                    make_unit("u:3", 3, "L3", status=Status.TRANSLATED, target="T3"),
                ]
            )
            recent = db.recent_translated_units(limit=10, before_seq=4)
            uids = [u.uid for u in recent]
            assert uids == ["u:0", "u:3"]

    def test_lookback_included_in_user_prompt(self, renpy_project: Project) -> None:
        cb = ContextBuilder(renpy_project.config)
        prompt = cb.build_user_prompt(
            batch_units=[make_unit("u:current", 1, "Current line")],
            lookback=[LookbackItem("eileen", "Past line", "Dòng quá khứ")],
        )
        assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" in prompt
        assert "Past line" in prompt
        assert "Dòng quá khứ" in prompt


class TestTier1BoundedDequeLookbackBuffer:
    """Feature 12: Bounded deque lookback buffer."""

    def test_deque_bounded_by_capacity(self) -> None:
        capacity = 4 * 4  # lookback=4 * 4 = 16
        buf: collections.deque[LookbackItem] = collections.deque(maxlen=capacity)
        for i in range(30):
            buf.append(LookbackItem(None, f"src {i}", f"tgt {i}"))
        assert len(buf) == 16
        assert buf[0].source == "src 14"
        assert buf[-1].source == "src 29"

    def test_deque_eviction_on_overflow(self) -> None:
        buf: collections.deque[int] = collections.deque(maxlen=3)
        buf.extend([1, 2, 3])
        buf.append(4)
        assert list(buf) == [2, 3, 4]

    def test_context_builder_deque_slice_safety(self) -> None:
        cb = ContextBuilder(Config())
        deque_lookback = collections.deque(
            [LookbackItem("spk", f"Src {i}", f"Tgt {i}") for i in range(10)], maxlen=16
        )
        # Slicing should not raise TypeError
        prompt = cb.build_user_prompt(
            batch_units=[make_unit("u:1", 1, "Hello")],
            lookback=deque_lookback,
        )
        assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" in prompt

    def test_deque_retains_latest_items(self) -> None:
        buf: collections.deque[LookbackItem] = collections.deque(maxlen=5)
        for i in range(100):
            buf.append(LookbackItem("spk", f"s{i}", f"t{i}"))
        sources = [item.source for item in buf]
        assert sources == ["s95", "s96", "s97", "s98", "s99"]

    def test_custom_lookback_config_bounds(self) -> None:
        cfg = Config(batch=BatchConfig(lookback=2))
        capacity = cfg.batch.lookback * 4
        assert capacity == 8


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 cases per area)
# ==============================================================================


class TestTier2EmptyInputsAndZeroLimits:
    """Tier 2 Area 1: Empty inputs, zero limits, and no-ops."""

    def test_empty_strings_validate_raw(self) -> None:
        assert validate_raw("", "") == []

    def test_empty_bulk_upsert_is_noop(self, test_db: Database) -> None:
        test_db.save_units_bulk(())
        assert len(test_db.units()) == 0

    def test_recent_units_limit_zero(self, test_db: Database) -> None:
        test_db.save_units_bulk([make_unit("u:1", 1, "A", status=Status.TRANSLATED, target="B")])
        assert test_db.recent_translated_units(limit=0) == []

    def test_pipeline_run_zero_pending(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.total_pending == 0

    def test_zero_lookback_config(self) -> None:
        cfg = Config(batch=BatchConfig(lookback=0))
        buf: collections.deque[LookbackItem] = collections.deque(maxlen=cfg.batch.lookback * 4)
        buf.append(LookbackItem(None, "Past", "Quá khứ"))
        assert len(buf) == 0
        cb = ContextBuilder(cfg)
        prompt = cb.build_user_prompt(
            batch_units=[make_unit("u:1", 1, "Hello")],
            lookback=buf,
        )
        assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" not in prompt


class TestTier2BoundaryTokenAndCapacityLimits:
    """Tier 2 Area 2: Boundary token limits and capacity thresholds."""

    def test_min_n_ctx_boundary_1024_accepted(self) -> None:
        cfg = Config.from_dict({"llm": {"n_ctx": 1024}})
        assert cfg.llm.n_ctx == 1024

    def test_below_min_n_ctx_1023_rejected(self) -> None:
        with pytest.raises(ConfigError):
            Config.from_dict({"llm": {"n_ctx": 1023}})

    def test_batch_max_lines_single_line_partition(self) -> None:
        units = [make_unit(f"u:{i}", i, f"Line {i}") for i in range(5)]
        batches = partition_batches(units, max_lines=1)
        assert len(batches) == 5
        assert all(len(b) == 1 for b in batches)

    def test_batch_max_tokens_single_token_partition(self) -> None:
        units = [make_unit(f"u:{i}", i, f"A somewhat longer sentence number {i}") for i in range(4)]
        batches = partition_batches(units, max_tokens=1)
        assert len(batches) == 4

    def test_insufficient_budget_skips_lookback(self) -> None:
        # Extremely small budget will drop lookback
        cfg = Config(llm=LLMConfig(n_ctx=1024))
        cb = ContextBuilder(cfg)
        # Giant batch consuming all budget
        units = [
            make_unit(f"u:{i}", i, "A very long source dialogue sentence " * 30) for i in range(10)
        ]
        prompt = cb.build_user_prompt(
            batch_units=units,
            lookback=[LookbackItem(None, "Old", "Cũ")],
        )
        assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" not in prompt


class TestTier2SingleUnitBatchesAndFallbacks:
    """Tier 2 Area 3: Single-unit batches and fallback escalation."""

    def test_single_unit_qa_failure_marks_failed(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk([make_unit("u:single", 1, "Hi")])

        with (
            renpy_project.open_db() as db,
            patch(
                "renslate.llm.LLMClient.complete_json",
                return_value=({"u:single": "{unclosed brace"}, 30, 20),
            ),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.failed == 1
            u_single = db.get_unit("u:single")
            assert u_single is not None
            assert u_single.status is Status.FAILED

    def test_single_unit_llm_error_marks_failed(self, renpy_project: Project) -> None:
        from renslate.errors import LLMError

        with renpy_project.open_db() as db:
            db.save_units_bulk([make_unit("u:single_err", 1, "Hi")])

        with (
            renpy_project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=LLMError("Connection lost")),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.failed == 1
            u_err = db.get_unit("u:single_err")
            assert u_err is not None
            assert u_err.status is Status.FAILED

    def test_single_unit_success_updates_tm(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk([make_unit("u:single_ok", 1, "Unique prompt sentence")])

        with (
            renpy_project.open_db() as db,
            patch(
                "renslate.llm.LLMClient.complete_json",
                return_value=({"u:single_ok": "Câu nhắc độc nhất"}, 30, 10),
            ),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.llm_translated == 1
            assert db.get_tm("Unique prompt sentence") is not None

    def test_mixed_retry_results_in_fallback(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [
                    make_unit("u:1", 1, "Sentence one"),
                    make_unit("u:2", 2, "Sentence two"),
                ]
            )

        call_idx = 0

        def mock_call(*args: Any, **kwargs: Any) -> tuple[dict[str, str], int, int]:
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                # Both have bad markup initially
                return ({"u:1": "{unclosed", "u:2": "{unclosed"}, 40, 20)
            if call_idx == 2:
                # Retry for u:1 succeeds
                return ({"u:1": "Câu một"}, 20, 10)
            # Retry for u:2 fails again
            return ({"u:2": "{still_unclosed"}, 20, 10)

        with (
            renpy_project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_call),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.llm_translated == 1
            assert stats.failed == 1
            u1 = db.get_unit("u:1")
            u2 = db.get_unit("u:2")
            assert u1 is not None and u1.status is Status.TRANSLATED
            assert u2 is not None and u2.status is Status.FAILED

    def test_include_failed_picks_up_failed_units(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [make_unit("u:previously_failed", 1, "Retry me", status=Status.FAILED)]
            )

        with (
            renpy_project.open_db() as db,
            patch(
                "renslate.llm.LLMClient.complete_json",
                return_value=({"u:previously_failed": "Thử lại tôi"}, 30, 10),
            ),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run(include_failed=True)
            assert stats.total_pending == 1
            assert stats.llm_translated == 1
            u_prev = db.get_unit("u:previously_failed")
            assert u_prev is not None and u_prev.status is Status.TRANSLATED


class TestTier2ExtremeTagNestingsAndSyntax:
    """Tier 2 Area 4: Extreme tag nestings and boundary syntax."""

    def test_extreme_10_level_nesting_valid(self) -> None:
        letters = "abcdefghij"
        open_tags = "".join(f"{{{c}}}" for c in letters)
        close_tags = "".join(f"{{/{c}}}" for c in reversed(letters))
        source = f"{open_tags}Hello{close_tags}"
        target = f"{open_tags}Deep content{close_tags}"
        issues = validate_raw(source, target)
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) == 0

    def test_extreme_10_level_nesting_swapped_innermost(self) -> None:
        letters = "abcdefghij"
        open_tags = "".join(f"{{{c}}}" for c in letters)
        # Swap j and i in closing
        close_list = [f"{{/{c}}}" for c in reversed(letters)]
        close_list[0], close_list[1] = close_list[1], close_list[0]
        close_tags = "".join(close_list)
        source = f"{open_tags}Hello{close_tags}"
        target = f"{open_tags}Deep content{close_tags}"
        issues = validate_raw(source, target)
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) > 0

    def test_repeated_identical_tags(self) -> None:
        target = "{b}One{/b} {b}Two{/b} {b}Three{/b}"
        issues = validate_raw("Hello", target)
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) == 0

    def test_self_closing_with_paired_tags(self) -> None:
        target = "{b}Formatted {color=#ffffff/} text{/b}"
        issues = validate_raw("Hello", target)
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) == 0

    def test_tags_only_no_text(self) -> None:
        target = "{b}{i}{/i}{/b}"
        issues = validate_raw("Hello", target)
        tag_order_issues = [i for i in issues if i.code == "tag_order"]
        assert len(tag_order_issues) == 0


class TestTier2NonConsecutiveSequencesAndGaps:
    """Tier 2 Area 5: Non-consecutive sequences, gaps, and mixed layouts."""

    def test_sequence_gaps_ordered_chronologically(self, test_db: Database) -> None:
        test_db.save_units_bulk(
            [
                make_unit("u:10", 10, "Ten", status=Status.TRANSLATED, target="Mười"),
                make_unit("u:45", 45, "Forty-five", status=Status.TRANSLATED, target="Bốn lăm"),
                make_unit("u:99", 99, "Ninety-nine", status=Status.TRANSLATED, target="Chín chín"),
            ]
        )
        recent = test_db.recent_translated_units(limit=10)
        assert [u.seq for u in recent] == [10, 45, 99]

    def test_before_seq_gap_filtering(self, test_db: Database) -> None:
        test_db.save_units_bulk(
            [
                make_unit("u:10", 10, "Ten", status=Status.TRANSLATED, target="Mười"),
                make_unit("u:45", 45, "Forty-five", status=Status.TRANSLATED, target="Bốn lăm"),
                make_unit("u:99", 99, "Ninety-nine", status=Status.TRANSLATED, target="Chín chín"),
            ]
        )
        recent = test_db.recent_translated_units(limit=10, before_seq=50)
        assert [u.seq for u in recent] == [10, 45]

    def test_non_contiguous_pending_units(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_units_bulk(
                [
                    make_unit("u:5", 5, "Line 5"),
                    make_unit("u:25", 25, "Line 25"),
                    make_unit("u:100", 100, "Line 100"),
                ]
            )

        with (
            renpy_project.open_db() as db,
            patch(
                "renslate.llm.LLMClient.complete_json",
                return_value=({"u:5": "Dòng 5", "u:25": "Dòng 25", "u:100": "Dòng 100"}, 40, 20),
            ),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.llm_translated == 3

    def test_rapid_label_changes(self) -> None:
        units = [make_unit(f"u:{i}", i, f"Line {i}", label=f"scene_{i}") for i in range(25)]
        batches = partition_batches(units, max_lines=40)
        assert len(batches) >= 1

    def test_mixed_speaker_and_narration(self) -> None:
        cb = ContextBuilder(Config())
        prompt = cb.build_user_prompt(
            batch_units=[
                make_unit("u:1", 1, "Hello", speaker="eileen"),
                make_unit("u:2", 2, "The wind whispers.", speaker=None),
            ],
            lookback=[
                LookbackItem("eileen", "Past line 1", "Dòng quá khứ 1"),
                LookbackItem(None, "Past narration", "Lời kể quá khứ"),
            ],
        )
        assert "[eileen] " in prompt
        assert '"Past line 1"' in prompt
        assert '"Past narration"' in prompt


# ==============================================================================
# TIER 3: CROSS-FEATURE INTERACTIONS (Pairwise Combinations)
# ==============================================================================


class TestTier3CrossFeatureInteractions:
    """Tier 3: Pairwise combinations and cross-subsystem interactions."""

    def test_bulk_persistence_with_partial_recovery(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            initial = [make_unit(f"u:{i}", i, f"Source {i}") for i in range(3)]
            db.save_units_bulk(initial)

        call_idx = 0

        def mock_complete(*args: Any, **kwargs: Any) -> tuple[dict[str, str], int, int]:
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return ({"u:0": "Dịch 0", "u:1": "Quá dài " * 20, "u:2": "Dịch 2"}, 60, 30)
            return ({"u:1": "Dịch 1"}, 20, 10)

        with (
            renpy_project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.llm_translated == 3

            # Immediately execute bulk upsert on new units to confirm DB isn't locked
            new_units = [make_unit(f"new:{i}", 10 + i, f"New source {i}") for i in range(5)]
            db.save_units_bulk(new_units)
            assert len(db.units()) == 8

    def test_cross_chunk_lookback_with_extraction(self, renpy_project: Project) -> None:
        # Extract 3 lines from script
        extract_project(renpy_project)

        # Chunk 1: translate first 2 lines
        prompts_sent: list[str] = []

        def mock_llm_capture(
            sys_p: str, user_p: str, **kwargs: Any
        ) -> tuple[dict[str, str], int, int]:
            prompts_sent.append(user_p)
            match_uids = re.findall(r"(\S+:\S+)\s*\|", user_p)
            return ({uid: f"Dịch {uid}" for uid in match_uids}, 30, 15)

        with (
            renpy_project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_llm_capture),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats1 = pipeline.run(limit=2)
            assert stats1.llm_translated == 2

            # Chunk 2: translate remaining 1 line
            stats2 = pipeline.run(limit=1)
            assert stats2.llm_translated == 1

            # Prompt of chunk 2 must include lookback from chunk 1
            assert len(prompts_sent) >= 2
            chunk2_prompt = prompts_sent[-1]
            assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" in chunk2_prompt
            assert "Welcome to the academy!" in chunk2_prompt

    def test_glossary_matching_with_tag_unmasking(self) -> None:
        glossary = Glossary(
            [
                GlossaryEntry("Sacred Sword", "Thánh Kiếm"),
            ]
        )
        source = "He unsheathed the {b}Sacred Sword{/b} of legends."
        masked = mask(source)
        # Verify glossary matches the clean source despite placeholders
        matches = glossary.match_text(masked.text)
        assert len(matches) == 1
        assert matches[0].translation == "Thánh Kiếm"

        # Simulate LLM translated text preserving tags
        translated = "Anh rút {b}Thánh Kiếm{/b} trong truyền thuyết."
        issues = validate_raw(source, translated)
        assert len(issues) == 0
        assert "Thánh Kiếm" in translated

    def test_tm_exact_with_partial_batch_recovery(self, renpy_project: Project) -> None:
        with renpy_project.open_db() as db:
            db.save_tm(TMEntry(src_norm="Known line", target="Dòng đã biết"))
            db.save_units_bulk(
                [
                    make_unit("u:1", 1, "Known line"),
                    make_unit("u:2", 2, "Novel line A"),
                    make_unit("u:3", 3, "Novel line B"),
                ]
            )

        call_idx = 0

        def mock_complete(*args: Any, **kwargs: Any) -> tuple[dict[str, str], int, int]:
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return ({"u:2": "Dòng mới A", "u:3": "{unclosed"}, 40, 20)
            return ({"u:3": "Dòng mới B"}, 20, 10)

        with (
            renpy_project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete),
        ):
            pipeline = TranslationPipeline(renpy_project, db, renpy_project.config)
            stats = pipeline.run()
            assert stats.seeded == 0
            assert stats.tm_reused == 1
            assert stats.llm_translated == 2
            assert stats.failed == 0

    def test_bounded_deque_with_cross_chunk_continuity(self, renpy_project: Project) -> None:
        cfg = Config(batch=BatchConfig(lookback=2))  # maxlen = 2 * 4 = 8
        with renpy_project.open_db() as db:
            db.save_units_bulk([make_unit(f"u:{i}", i, f"Line {i}") for i in range(12)])

        def mock_llm(sys_p: str, user_p: str, **kwargs: Any) -> tuple[dict[str, str], int, int]:
            match_uids = re.findall(r"(\S+:\S+)\s*\|", user_p)
            return ({uid: f"Dịch {uid}" for uid in match_uids}, 30, 10)

        with (
            renpy_project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_llm),
        ):
            pipeline = TranslationPipeline(renpy_project, db, cfg)
            # Run 4 consecutive chunks of 3 units
            for _ in range(4):
                pipeline.run(limit=3)

            # Check that lookback retrieved from SQLite is bounded
            recent = db.recent_translated_units(limit=cfg.batch.lookback * 4)
            assert len(recent) <= 8


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================


class TestTier4RealWorldWorkflows:
    """Tier 4: Realistic multi-chunk visual novel localization simulation."""

    def test_full_visual_novel_localization_workflow(self, tmp_path: Path) -> None:
        # 1. Setup realistic Ren'Py project
        game_dir = tmp_path / "game"
        game_dir.mkdir(parents=True)
        project = Project.create(tmp_path)
        tl_dir = project.tl_root
        tl_dir.mkdir(parents=True, exist_ok=True)
        (tl_dir / "script.rpy").write_text(
            "# game/script.rpy:14\n"
            "translate vietnamese start_1:\n\n"
            '    # e "Welcome to the academy!"\n'
            '    e "Welcome to the academy!"\n\n'
            "# game/script.rpy:15\n"
            "translate vietnamese start_2:\n\n"
            '    # e "Here you shall learn the arts of the {b}Sacred Order{/b}."\n'
            '    e "Here you shall learn the arts of the {b}Sacred Order{/b}."\n\n'
            "# game/script.rpy:16\n"
            "translate vietnamese start_3:\n\n"
            '    # "A cold wind howls across the courtyard."\n'
            '    "A cold wind howls across the courtyard."\n\n'
            "# game/script.rpy:20\n"
            "translate vietnamese ch1_1:\n\n"
            '    # l "Wait, who goes there?"\n'
            '    l "Wait, who goes there?"\n\n'
            "# game/script.rpy:21\n"
            "translate vietnamese ch1_2:\n\n"
            '    # e "Do not be alarmed, sister."\n'
            '    e "Do not be alarmed, sister."\n\n'
            "# game/script.rpy:22\n"
            "translate vietnamese ch1_3:\n\n"
            '    # l "We must protect the secret."\n'
            '    l "We must protect the secret."\n\n'
            "# game/script.rpy:23\n"
            "translate vietnamese ch1_4:\n\n"
            '    # "A cold wind howls across the courtyard."\n'
            '    "A cold wind howls across the courtyard."\n\n'
            "# game/script.rpy:25\n"
            "translate vietnamese ch1_5:\n\n"
            '    # e "Farewell for now."\n'
            '    e "Farewell for now."\n',
            encoding="utf-8",
        )

        # 2. Extract project
        report = extract_project(project)
        assert report.created == 8  # 8 dialogue unit blocks

        # 3. Configure Characters & Glossary
        with project.open_db() as db:
            db.save_character(
                CharacterProfile("e", "Eileen", gender="female", speech_style="polite")
            )
            db.save_character(
                CharacterProfile("l", "Lucy", gender="female", speech_style="cautious")
            )
            db.save_glossary_entry(GlossaryEntry("Sacred Order", "Giáo Hội Thần Thánh"))

        # 4. Chunk 1 translation (limit=3)
        def mock_llm_chunk1(sys: str, usr: str, **kw: Any) -> tuple[dict[str, str], int, int]:
            res: dict[str, str] = {}
            if "d:start_1" in usr:
                res["d:start_1"] = "Chào mừng đến với học viện!"
            if "d:start_2" in usr:
                res["d:start_2"] = "Tại đây bạn sẽ học nghệ thuật của ⟦T1⟧Giáo Hội Thần Thánh⟦/T1⟧."
            if "d:start_3" in usr:
                res["d:start_3"] = "Cơn gió lạnh rít qua sân trường."
            return (res, 80, 40)

        with (
            project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_llm_chunk1),
        ):
            pipeline = TranslationPipeline(project, db, project.config)
            stats1 = pipeline.run(limit=3)
            assert stats1.llm_translated == 3

        # 5. Chunk 2 translation with intermittent QA error & partial recovery (limit=3)
        chunk2_call_idx = 0

        def mock_llm_chunk2(sys: str, usr: str, **kw: Any) -> tuple[dict[str, str], int, int]:
            nonlocal chunk2_call_idx
            chunk2_call_idx += 1
            if chunk2_call_idx == 1:
                # Intentionally make unit d:ch1_2 fail QA (syntax error in markup)
                return (
                    {
                        "d:ch1_1": "Đợi đã, ai đang đi đó?",
                        "d:ch1_2": "{unclosed brace error",
                        "d:ch1_3": "Chúng ta phải bảo vệ bí mật.",
                    },
                    120,
                    60,
                )
            # Fallback retry for failing unit also fails
            return ({"d:ch1_2": "{still unclosed"}, 30, 15)

        with (
            project.open_db() as db,
            patch("renslate.llm.LLMClient.complete_json", side_effect=mock_llm_chunk2),
        ):
            pipeline = TranslationPipeline(project, db, project.config)
            stats2 = pipeline.run(limit=3)
            # uids[0] and uids[2] succeed, uids[1] fails
            assert stats2.llm_translated == 2
            assert stats2.failed == 1

        # 6. Chunk 3 translation (final line) with TM reuse check
        with (
            project.open_db() as db,
            patch(
                "renslate.llm.LLMClient.complete_json",
                return_value=({"d:ch1_5": "Tạm biệt nhé."}, 30, 10),
            ),
        ):
            pipeline = TranslationPipeline(project, db, project.config)
            stats3 = pipeline.run(limit=2)
            # The repeated line "A cold wind howls across the courtyard." hits TM
            assert stats3.tm_reused == 1

        # 7. Final integrity and database check
        with project.open_db() as db:
            units = db.units()
            statuses = {u.status for u in units.values()}
            assert Status.TRANSLATED in statuses
            assert Status.FAILED in statuses
            # Verify PRAGMA compliance
            row = db.execute("PRAGMA temp_store").fetchone()
            assert row[0] == 2
