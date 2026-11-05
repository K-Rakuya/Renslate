"""Comprehensive verification of complete glossary eradication (Milestone 1).

Covers:
- Database schema migration v6 (clean drop of legacy table, fresh DB omission)
- Complete eradication of renslate.glossary and renslate.engine.glossary modules
- Removal of GlossaryEntry from models and __all__
- Removal of glossary methods from Database and repositories
- Removal of glossary checks and repair_glossary_leakage from QA
- Removal of glossary section and parameters from ContextBuilder
- Removal of --no-glossary CLI argument and glossary analyze steps
- DashboardSnapshot omission of glossary_count
- Programmatic source tree cleanliness check (0 matches for 'glossary' in src/)
"""

from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import fields
from pathlib import Path

import pytest

import renslate
import renslate.core.db as core_db
import renslate.core.models as core_models
import renslate.engine.qa as engine_qa
import renslate.models as root_models
import renslate.qa as root_qa
from renslate.cli import _build_parser as build_parser
from renslate.core.config import Config
from renslate.core.db import Database
from renslate.core.db.schema import MIGRATIONS, get_schema_version, run_migrations
from renslate.core.models import Status, Unit, UnitKind
from renslate.core.project import Project
from renslate.engine.analyze import run_analysis
from renslate.engine.context import ContextBuilder
from renslate.engine.context.builder import ContextBundle
from renslate.engine.qa import QAChecker
from renslate.ui.dashboard import DashboardSnapshot


def test_schema_v6_drops_existing_glossary_table(tmp_path: Path) -> None:
    """Migrating legacy DB with glossary table drops it cleanly at v6."""
    db_file = tmp_path / "legacy_v5.db"
    conn = sqlite3.connect(str(db_file))

    # Manually execute v1..v4 migrations and insert legacy glossary table
    for script in MIGRATIONS[:4]:
        conn.executescript(script)
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS glossary (term TEXT PRIMARY KEY, translation TEXT NOT NULL, notes TEXT);"
    )
    conn.execute("INSERT INTO glossary (term, translation, notes) VALUES ('Sect', 'Tông môn', 'note');")
    conn.execute("PRAGMA user_version = 4;")
    conn.commit()

    # Verify table exists in legacy state
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='glossary';").fetchall()
    assert len(rows) == 1
    conn.close()

    # Now open with Database to trigger migrations through v6
    with Database.open(db_file) as db:
        assert db.schema_version >= 6
        tables = {
            r["name"]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        }
        assert "glossary" not in tables


def test_fresh_database_has_no_glossary_table(tmp_path: Path) -> None:
    """Fresh database initialization creates all tables except glossary."""
    db_file = tmp_path / "fresh.db"
    with Database.open(db_file, create=True) as db:
        tables = {
            r["name"]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        }
        assert "glossary" not in tables
        assert "meta" in tables
        assert "units" in tables
        assert "characters" in tables
        assert "scenes" in tables
        assert db.schema_version >= 6


def test_glossary_module_cannot_be_imported() -> None:
    """Importing renslate.glossary must raise ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        import renslate.glossary  # noqa: F401


def test_engine_glossary_module_cannot_be_imported() -> None:
    """Importing renslate.engine.glossary must raise ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        import renslate.engine.glossary  # noqa: F401


def test_renslate_top_level_does_not_export_glossary() -> None:
    """renslate top level does not export Glossary or glossary symbols."""
    assert not hasattr(renslate, "Glossary")
    assert not hasattr(renslate, "clean_glossary")
    assert not hasattr(renslate, "is_valid_glossary_term")
    assert not hasattr(renslate, "mine_glossary_candidates")
    assert not hasattr(renslate, "translate_glossary_with_llm")
    assert not hasattr(renslate, "repair_glossary_leakage")
    assert "Glossary" not in getattr(renslate, "__all__", [])


def test_models_do_not_export_glossary_entry() -> None:
    """GlossaryEntry must not be present in core.models or root models."""
    assert not hasattr(core_models, "GlossaryEntry")
    assert not hasattr(root_models, "GlossaryEntry")
    assert "GlossaryEntry" not in core_models.__all__
    assert "GlossaryEntry" not in root_models.__all__


def test_project_has_no_glossary_path(tmp_path: Path) -> None:
    """Project instance must not have glossary_path property."""
    proj = Project(tmp_path, Config())
    assert not hasattr(proj, "glossary_path")


def test_database_and_repositories_have_no_glossary_methods(tmp_path: Path) -> None:
    """Database must not have any glossary CRUD methods or GlossaryRepository."""
    db_file = tmp_path / "no_glossary_db.db"
    with Database.open(db_file, create=True) as db:
        assert not hasattr(db, "get_glossary")
        assert not hasattr(db, "save_glossary")
        assert not hasattr(db, "save_glossary_entry")
        assert not hasattr(db, "save_glossary_bulk")
        assert not hasattr(core_db, "GlossaryRepository")


def test_qa_checker_has_no_glossary_rules_or_params() -> None:
    """QAChecker does not check glossary violations or accept glossary_matches."""
    from renslate.mask import mask

    checker = QAChecker()
    source = "This is a test of ancient kingdom."
    target = "Đây là thử nghiệm về vương quốc cổ đại."
    issues = checker.check_unit("u_test", source, mask(source), target)
    assert not any(i.code in ("glossary_violation", "untranslated_glossary_leakage") for i in issues)

    # repair_glossary_leakage must be completely removed
    assert not hasattr(engine_qa, "repair_glossary_leakage")
    assert not hasattr(root_qa, "repair_glossary_leakage")


def test_context_builder_has_no_glossary_matches_parameter() -> None:
    """ContextBuilder.build_user_prompt does not take glossary_matches parameter."""
    builder = ContextBuilder(Config())
    u = Unit(
        uid="u_test",
        kind=UnitKind.DIALOGUE,
        seq=1,
        source="Welcome hero to the realm.",
        status=Status.PENDING,
    )
    # Calling with glossary_matches should raise TypeError
    with pytest.raises(TypeError):
        builder.build_user_prompt([u], glossary_matches=[])  # type: ignore[call-arg]

    # Normal build without glossary_matches succeeds and contains zero glossary headings
    prompt = builder.build_user_prompt([u])
    assert "THUẬT NGỮ BẮT BUỘC" not in prompt
    assert "GLOSSARY" not in prompt.upper() or "GLOSSARY" not in prompt


def test_context_bundle_dataclass_fields_exclude_glossary() -> None:
    """ContextBundle dataclass has no glossary_matches field."""
    field_names = {f.name for f in fields(ContextBundle)}
    assert "glossary_matches" not in field_names


def test_pipeline_run_analysis_omits_glossary(tmp_path: Path) -> None:
    """run_analysis runs character and scene analysis without glossary mining."""
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    (game_dir / "script.rpy").write_text('define e = Character("Eileen")\nlabel start:\n    e "Hello world"\n')

    proj = Project.create(tmp_path)
    with proj.open_db() as db:
        stats = run_analysis(proj, db, enrich_llm=False)
        assert "glossary_candidates" not in stats
        assert "characters" in stats
        assert "scenes" in stats
        # Assert no glossary.json file was written
        assert not (proj.work_dir / "glossary.json").exists()


def test_cli_parser_rejects_no_glossary_flag() -> None:
    """CLI argument parser does not recognize --no-glossary."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["translate", "/some/path", "--no-glossary"])


def test_dashboard_snapshot_has_no_glossary_count() -> None:
    """DashboardSnapshot dataclass has no glossary_count field."""
    field_names = {f.name for f in fields(DashboardSnapshot)}
    assert "glossary_count" not in field_names


def test_schema_migration_v6_idempotent(tmp_path: Path) -> None:
    """Running migrations repeatedly via run_migrations is safe and idempotent."""
    db_file = tmp_path / "idempotent.db"
    conn = sqlite3.connect(str(db_file))
    run_migrations(conn)
    assert get_schema_version(conn) == len(MIGRATIONS)
    run_migrations(conn)
    assert get_schema_version(conn) == len(MIGRATIONS)
    conn.close()


def test_source_tree_contains_zero_glossary_occurrences() -> None:
    """git grep -i 'glossary' src/ returns 0 matches in code."""
    res = subprocess.run(
        ["git", "grep", "-i", "glossary", "src/"],
        capture_output=True,
        text=True,
    )
    # git grep returns exit code 1 when no matches are found
    assert res.returncode == 1, f"Found unexpected glossary matches in src/:\n{res.stdout}"
    assert res.stdout.strip() == ""
