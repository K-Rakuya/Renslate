"""Tests for Milestone 1: Subpackage Architecture Reorganization.

Verifies:
- Domain subpackages (renslate.core, renslate.renpy, renslate.engine, renslate.ui)
- Top-level re-exports and __all__ completeness
- sys.modules aliasing for historical paths
- unittest.mock.patch stability on legacy paths
- Strict Acyclic Directed Graph (DAG) dependencies
- Decoupled Database.sanitize_stored_translations() execution
- Forwarding shim identity with subpackage targets
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import patch

import renslate
import renslate.core
import renslate.core.config
import renslate.core.db
import renslate.core.errors
import renslate.core.models
import renslate.core.project
import renslate.core.textio
import renslate.engine
import renslate.engine.analyze
import renslate.engine.context
import renslate.engine.glossary
import renslate.engine.llm
import renslate.engine.qa
import renslate.engine.tm
import renslate.engine.translate
import renslate.engine.tsvio
import renslate.renpy
import renslate.renpy.builtin_ui
import renslate.renpy.extract
import renslate.renpy.literals
import renslate.renpy.mask
import renslate.renpy.parser
import renslate.renpy.sdk
import renslate.renpy.writer
import renslate.ui
import renslate.ui.dashboard
import renslate.ui.progress
import renslate.ui.runner
import renslate.ui.tables
import renslate.ui.theme
import renslate.ui.wizard


def test_subpackage_core_exports() -> None:
    """Verify renslate.core exports all core domain primitives and modules."""
    assert hasattr(renslate.core, "Unit")
    assert hasattr(renslate.core, "Database")
    assert hasattr(renslate.core, "Config")
    assert hasattr(renslate.core, "Project")
    assert hasattr(renslate.core, "TextFile")
    assert hasattr(renslate.core, "RenslateError")
    assert hasattr(renslate.core, "ConfigError")
    assert hasattr(renslate.core, "ProjectError")
    assert hasattr(renslate.core, "MIGRATIONS")
    assert hasattr(renslate.core, "utc_timestamp")


def test_subpackage_renpy_exports() -> None:
    """Verify renslate.renpy exports Ren'Py domain modules and symbols."""
    assert hasattr(renslate.renpy.literals, "decode_say_text")
    assert hasattr(renslate.renpy.parser, "TranslationScanner")
    assert hasattr(renslate.renpy.mask, "mask")
    assert hasattr(renslate.renpy.mask, "unmask")
    assert hasattr(renslate.renpy.mask, "validate_raw")
    assert hasattr(renslate.renpy.extract, "extract_project")
    assert hasattr(renslate.renpy.writer, "apply_translations")
    assert hasattr(renslate.renpy.sdk, "generate_translations")
    assert hasattr(renslate.renpy.builtin_ui, "BUILTIN_UI_SEEDS")


def test_subpackage_engine_exports() -> None:
    """Verify renslate.engine exports pipeline and analysis symbols."""
    assert hasattr(renslate.engine.translate, "TranslationPipeline")
    assert hasattr(renslate.engine.llm, "LLMClient")
    assert hasattr(renslate.engine.context, "ContextBuilder")
    assert hasattr(renslate.engine.qa, "QAChecker")
    assert hasattr(renslate.engine.glossary, "Glossary")
    assert hasattr(renslate.engine.tm, "TranslationMemory")
    assert hasattr(renslate.engine.analyze, "run_analysis")
    assert hasattr(renslate.engine.tsvio, "export_tsv")


def test_subpackage_ui_exports() -> None:
    """Verify renslate.ui exports UI formatting and runner utilities."""
    assert hasattr(renslate.ui.theme, "IS_TTY")
    assert hasattr(renslate.ui.progress, "format_duration")
    assert hasattr(renslate.ui.progress, "render_progress_bar")
    assert hasattr(renslate.ui.tables, "format_table")
    assert hasattr(renslate.ui.dashboard, "LiveDashboard")
    assert hasattr(renslate.ui.wizard, "run_interactive_wizard")
    assert hasattr(renslate.ui.runner, "check_server_health")
    assert hasattr(renslate.ui.runner, "detect_server_model")
    assert hasattr(renslate.ui.runner, "query_db_status")


def test_top_level_reexports_and_all() -> None:
    """Verify top-level renslate re-exports all public symbols declared in __all__."""
    assert hasattr(renslate, "__all__")
    assert len(renslate.__all__) >= 50
    for symbol in renslate.__all__:
        assert hasattr(renslate, symbol), f"renslate is missing symbol '{symbol}' from __all__"


def test_sys_modules_legacy_mapping() -> None:
    """Verify sys.modules maps all 21 historical paths to subpackage modules."""
    legacy_keys = [
        "renslate.models",
        "renslate.errors",
        "renslate.config",
        "renslate.db",
        "renslate.project",
        "renslate.textio",
        "renslate.literals",
        "renslate.parser",
        "renslate.mask",
        "renslate.extract",
        "renslate.writer",
        "renslate.sdk",
        "renslate.builtin_ui",
        "renslate.translate",
        "renslate.llm",
        "renslate.context",
        "renslate.qa",
        "renslate.glossary",
        "renslate.tm",
        "renslate.analyze",
        "renslate.tsvio",
    ]
    for key in legacy_keys:
        assert key in sys.modules, f"Legacy module path '{key}' not registered in sys.modules"
        mod = sys.modules[key]
        assert mod is not None, f"Module '{key}' is None in sys.modules"


def test_mock_patch_on_legacy_paths() -> None:
    """Verify unittest.mock.patch stably patches attributes through legacy paths."""
    with patch("renslate.llm.LLMClient.complete_json", return_value=({"test": 1}, 10, 5)):
        from renslate.engine.llm import LLMClient

        client = LLMClient(renslate.core.config.LLMConfig())
        result = client.complete_json("sys", "user")
        assert result == ({"test": 1}, 10, 5)

    with patch("renslate.writer.apply_translations", return_value="mocked_report"):
        from renslate.renpy.writer import apply_translations

        assert apply_translations(None, None) == "mocked_report"  # type: ignore[arg-type,comparison-overlap]


def test_strict_acyclic_dag_hierarchy() -> None:
    """Verify that source AST imports strictly respect the subpackage DAG hierarchy."""
    src_dir = Path(renslate.__file__).parent
    core_dir = src_dir / "core"
    renpy_dir = src_dir / "renpy"
    engine_dir = src_dir / "engine"

    # Core must not import renpy, engine, or ui
    for py_file in core_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("renslate.renpy"), (
                    f"{py_file.name} in core violates DAG by importing {node.module}"
                )
                assert not node.module.startswith("renslate.engine"), (
                    f"{py_file.name} in core violates DAG by importing {node.module}"
                )
                assert not node.module.startswith("renslate.ui"), (
                    f"{py_file.name} in core violates DAG by importing {node.module}"
                )

    # Ren'Py must not import engine or ui
    for py_file in renpy_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("renslate.engine"), (
                    f"{py_file.name} in renpy violates DAG by importing {node.module}"
                )
                assert not node.module.startswith("renslate.ui"), (
                    f"{py_file.name} in renpy violates DAG by importing {node.module}"
                )

    # Engine must not import ui
    for py_file in engine_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("renslate.ui"), (
                    f"{py_file.name} in engine violates DAG by importing {node.module}"
                )


def test_forwarding_shim_identities() -> None:
    """Verify symbols imported via legacy shims are identical objects to subpackage symbols."""
    import renslate.db as legacy_db
    import renslate.models as legacy_models
    import renslate.translate as legacy_translate
    from renslate.core.db import Database as CoreDatabase
    from renslate.core.models import Unit as CoreUnit
    from renslate.engine.translate import TranslationPipeline as EnginePipeline

    assert legacy_models.Unit is CoreUnit
    assert legacy_db.Database is CoreDatabase
    assert legacy_translate.TranslationPipeline is EnginePipeline


def test_sanitize_stored_translations_decoupled_execution(tmp_path: Path) -> None:
    """Verify Database.sanitize_stored_translations executes cleanly with decoupled imports."""
    db_path = tmp_path / "test_sanitize.db"
    db = renslate.core.db.Database.open(db_path, create=True)
    try:
        # Insert a sample translated unit with a speaker tag hallucination
        raw_target = 'Alice: "Xin chào thế giới!"'
        db.save_unit(
            renslate.core.models.Unit(
                uid="u_001",
                kind=renslate.core.models.UnitKind.DIALOGUE,
                seq=1,
                source="Hello world!",
                status=renslate.core.models.Status.TRANSLATED,
                speaker="Alice",
                target=raw_target,
            )
        )
        cleaned_units, cleaned_tm = db.sanitize_stored_translations()
        assert cleaned_units == 1
        assert cleaned_tm == 0

        # Verify unit target was sanitized
        updated_unit = db.get_unit("u_001")
        assert updated_unit is not None
        assert updated_unit.target == "Xin chào thế giới!"
    finally:
        db.close()
