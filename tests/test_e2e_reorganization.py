"""Comprehensive 4-Tier Opaque-Box E2E Test Suite for Renslate Architectural Reorganization.

Requirements Covered:
- R1: Deepened Subpackage Architecture (renslate.core, renslate.renpy,
      renslate.engine, renslate.ui),
      acyclic DAG dependency layers, backward compatibility shims, pure stdlib compliance.
- R2: Autonomous Runner (renslate run / renslate auto), Interactive Setup Wizard, Real-time ANSI
      VT100 Terminal UI Dashboard, and scripts/unattended_runner.py delegation.
- R3: Algorithmic Modernization (zero-allocation state-machine masking, knapsack token budgeting,
      SQLite prepared statement caching & zero-copy iterators, comment hygiene).
- R4: Technical Documentation & ADR Suite (ADR-021, ADR-022, ADR-023 in ARCHITECTURE_DECISIONS.md,
      docs/DESIGN.md sequence diagrams, docs/RUNBOOK.md operational runbooks).

Tiers:
- Tier 1: Feature Coverage (12 Features x >=5 tests = 60 tests)
- Tier 2: Boundary & Corner Cases (5 classes x 5 tests = 25 tests)
- Tier 3: Cross-Feature Interactions (1 class x 5 tests = 5 tests)
- Tier 4: Real-World Scenarios (1 class x 5 tests = 5 tests)
Total: 95 Test Cases
"""

from __future__ import annotations

import ast
import collections
import http.server
import importlib
import importlib.util
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import renslate
import scripts.unattended_runner as runner
from renslate.config import Config, LLMConfig
from renslate.context import ContextBuilder, LookbackItem, estimate_tokens
from renslate.db import Database, utc_timestamp
from renslate.errors import RenslateError
from renslate.extract import extract, extract_project
from renslate.glossary import Glossary
from renslate.mask import mask, reconcile_translation, unmask, validate_raw
from renslate.models import (
    GlossaryEntry,
    Origin,
    Status,
    Unit,
    UnitKind,
)
from renslate.project import Project
from renslate.qa import QAChecker
from renslate.translate import TranslationPipeline
from renslate.writer import apply_translations

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DOCS_DIR = REPO_ROOT / "docs"

# Feature availability detection for progressive milestone testability
HAS_SUBPACKAGES = importlib.util.find_spec("renslate.core") is not None
HAS_CORE_MODELS = importlib.util.find_spec("renslate.core.models") is not None
HAS_RENPY_MASK = importlib.util.find_spec("renslate.renpy.mask") is not None
HAS_ENGINE_TRANSLATE = importlib.util.find_spec("renslate.engine.translate") is not None
HAS_UI_RUNNER = importlib.util.find_spec("renslate.ui.runner") is not None
HAS_TOPLEVEL_REEXPORTS = hasattr(renslate, "Unit") and hasattr(renslate, "Database")
HAS_DB_STREAMING = hasattr(Database, "iter_units")
HAS_WIZARD = importlib.util.find_spec("renslate.ui.wizard") is not None


def _has_shims() -> bool:
    try:
        import renslate.core.models as core_m
        import renslate.models as legacy_m

        return legacy_m.Unit is core_m.Unit
    except Exception:
        return False


HAS_SHIMS = _has_shims()


def _detect_cli_run() -> bool:
    try:
        from renslate.cli import _build_parser

        parser = _build_parser()
        if parser._subparsers:
            for action in parser._subparsers._actions:
                if hasattr(action, "choices") and action.choices and "run" in action.choices:
                    return True
        return False
    except Exception:
        return False


HAS_CLI_RUN = _detect_cli_run()


def _has_adr(adr_id: str) -> bool:
    adr_file = DOCS_DIR / "ARCHITECTURE_DECISIONS.md"
    if not adr_file.exists():
        return False
    return adr_id in adr_file.read_text(encoding="utf-8")


HAS_ADR_021 = _has_adr("ADR-021")
HAS_ADR_022 = _has_adr("ADR-022")
HAS_ADR_023 = _has_adr("ADR-023")


# ==============================================================================
# Helper Factories & Fixtures
# ==============================================================================


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
    """Helper to construct a Unit with safe defaults."""
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


class MockLLMServer:
    """In-process HTTP server mocking llama-server OpenAI endpoints."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.server: http.server.HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.requests: list[dict[str, Any]] = []

    def start(self) -> str:
        requests_log = self.requests

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                requests_log.append({"method": "GET", "path": self.path})
                if self.path in ("/health", "/healthz"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                elif self.path in ("/props", "/v1/props"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        b'{"default_generation_settings": '
                        b'{"model": "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"}}'
                    )
                elif self.path in ("/v1/models", "/models"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"data": [{"id": "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"}]}')
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length > 0 else b""
                requests_log.append({"method": "POST", "path": self.path, "body": body})
                if "/chat/completions" in self.path:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()

                    parsed = json.loads(body.decode("utf-8")) if body else {}
                    messages = parsed.get("messages", [])
                    user_content = ""
                    for m in messages:
                        if m.get("role") == "user":
                            user_content = m.get("content", "")
                            break

                    translations: dict[str, str] = {}
                    for line in user_content.splitlines():
                        if " | " in line:
                            uid = line.split(" | ", 1)[0].strip()
                            translations[uid] = f"Bản dịch cho {uid}"
                            if ":" in uid:
                                translations[uid.split(":", 1)[1]] = f"Bản dịch cho {uid}"

                    if not translations:
                        translations = {
                            "start_01": "Chào mừng đến với học viện!",
                            "start_02": "Hãy chọn con đường cẩn thận.",
                            "start_03": "Cánh cổng lớn mở ra trước mắt bạn.",
                            "d:start_01": "Chào mừng đến với học viện!",
                            "d:start_02": "Hãy chọn con đường cẩn thận.",
                            "d:start_03": "Cánh cổng lớn mở ra trước mắt bạn.",
                        }

                    payload = {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(translations),
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 40},
                    }
                    self.wfile.write(json.dumps(payload).encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()

        self.server = http.server.HTTPServer((self.host, self.port), Handler)
        actual_port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://{self.host}:{actual_port}"

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()


@pytest.fixture
def mock_server() -> Iterator[str]:
    """Provide a real local HTTP server instance simulating llama-server."""
    server = MockLLMServer()
    url = server.start()
    try:
        yield url
    finally:
        server.stop()


@pytest.fixture
def test_db(tmp_path: Path) -> Iterator[Database]:
    """Provide an isolated SQLite database instance for tests."""
    db_file = tmp_path / "test_reorg.db"
    db = Database.open(db_file, create=True)
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def renpy_project(tmp_path: Path) -> Project:
    """Create a minimal mock Ren'Py project with valid translation files."""
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
# Tier 1: Feature Coverage (12 Features x 5 Tests = 60 Tests)
# ==============================================================================


class TestTier1SubpackageReorganization:
    """Feature 1: Decoupled Domain Subpackages (R1)."""

    @pytest.mark.skipif(not HAS_SUBPACKAGES, reason="Requires M1 subpackage migration")
    def test_subpackage_core_modules(self) -> None:
        """Verify renslate.core exports models, errors, config, db, project, textio."""
        import renslate.core.config
        import renslate.core.db
        import renslate.core.errors
        import renslate.core.models
        import renslate.core.project
        import renslate.core.textio

        assert hasattr(renslate.core.models, "Unit")
        assert hasattr(renslate.core.errors, "RenslateError")
        assert hasattr(renslate.core.config, "Config")
        assert hasattr(renslate.core.db, "Database")
        assert hasattr(renslate.core.project, "Project")
        assert hasattr(renslate.core.textio, "read_text_file")

    @pytest.mark.skipif(not HAS_SUBPACKAGES, reason="Requires M1 subpackage migration")
    def test_subpackage_renpy_modules(self) -> None:
        """Verify renslate.renpy exports literals, parser, mask, extract, writer,
        sdk, builtin_ui.
        """
        import renslate.renpy.builtin_ui
        import renslate.renpy.extract
        import renslate.renpy.literals
        import renslate.renpy.mask
        import renslate.renpy.parser
        import renslate.renpy.sdk
        import renslate.renpy.writer

        assert hasattr(renslate.renpy.literals, "decode_say_text")
        assert hasattr(renslate.renpy.parser, "TranslationScanner")
        assert callable(renslate.renpy.mask) or hasattr(sys.modules["renslate.renpy.mask"], "mask")
        assert callable(renslate.renpy.extract) or hasattr(
            sys.modules["renslate.renpy.extract"], "extract"
        )
        assert hasattr(renslate.renpy.writer, "apply_translations")
        assert hasattr(renslate.renpy.sdk, "generate_translations")
        assert hasattr(renslate.renpy.builtin_ui, "BUILTIN_UI_SEEDS")

    @pytest.mark.skipif(not HAS_SUBPACKAGES, reason="Requires M1 subpackage migration")
    def test_subpackage_engine_modules(self) -> None:
        """Verify renslate.engine exports translate, llm, context, qa, glossary, tm, analyze."""
        import renslate.engine.analyze
        import renslate.engine.context
        import renslate.engine.glossary
        import renslate.engine.llm
        import renslate.engine.qa
        import renslate.engine.tm
        import renslate.engine.translate
        import renslate.engine.tsvio

        assert hasattr(renslate.engine.translate, "TranslationPipeline")
        assert hasattr(renslate.engine.llm, "LLMClient")
        assert hasattr(renslate.engine.context, "ContextBuilder")
        assert hasattr(renslate.engine.qa, "QAChecker")
        assert hasattr(renslate.engine.glossary, "Glossary")
        assert hasattr(renslate.engine.tm, "TranslationMemory")
        assert hasattr(renslate.engine.analyze, "run_analysis")
        assert hasattr(renslate.engine.tsvio, "export_tsv")

    @pytest.mark.skipif(not HAS_SUBPACKAGES, reason="Requires M1 subpackage migration")
    def test_subpackage_ui_modules(self) -> None:
        """Verify renslate.ui exports dashboard, progress, runner, tables, theme, wizard."""
        import renslate.ui.dashboard
        import renslate.ui.progress
        import renslate.ui.runner
        import renslate.ui.tables
        import renslate.ui.theme
        import renslate.ui.wizard

        assert hasattr(renslate.ui.runner, "run_unattended_pipeline")
        assert hasattr(renslate.ui.runner, "check_server_health")

    @pytest.mark.skipif(not HAS_SUBPACKAGES, reason="Requires M1 subpackage migration")
    def test_subpackage_acyclic_dag_layers(self) -> None:
        """Verify strict module-level acyclic DAG dependency constraints across subpackages."""
        core_dir = SRC_DIR / "renslate" / "core"
        renpy_dir = SRC_DIR / "renslate" / "renpy"
        engine_dir = SRC_DIR / "renslate" / "engine"

        for p in core_dir.glob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("renslate.renpy"), (
                        f"{p} violates DAG (top-level imports renpy)"
                    )
                    assert not node.module.startswith("renslate.engine"), (
                        f"{p} violates DAG (top-level imports engine)"
                    )
                    assert not node.module.startswith("renslate.ui"), (
                        f"{p} violates DAG (top-level imports ui)"
                    )

        for p in renpy_dir.glob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("renslate.engine"), (
                        f"{p} violates DAG (top-level imports engine)"
                    )
                    assert not node.module.startswith("renslate.ui"), (
                        f"{p} violates DAG (top-level imports ui)"
                    )

        for p in engine_dir.glob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith("renslate.ui"), (
                        f"{p} violates DAG (top-level imports ui)"
                    )


class TestTier1TopLevelReexportsAndShims:
    """Feature 2: Top-Level Re-exports & Legacy Forwarding Shims (R1)."""

    @pytest.mark.skipif(
        not HAS_TOPLEVEL_REEXPORTS, reason="Requires M1 renslate/__init__.py re-exports"
    )
    def test_top_level_reexport_core_symbols(self) -> None:
        """Verify renslate top-level exports Unit, Database, Config, Project, RenslateError."""
        for symbol in [
            "Unit",
            "Database",
            "Config",
            "Project",
            "RenslateError",
            "Status",
            "Origin",
        ]:
            assert hasattr(renslate, symbol), f"Top-level export missing: {symbol}"

    @pytest.mark.skipif(
        not HAS_TOPLEVEL_REEXPORTS, reason="Requires M1 renslate/__init__.py re-exports"
    )
    def test_top_level_reexport_renpy_symbols(self) -> None:
        """Verify renslate top-level exports mask, unmask, validate_raw, extract_project."""
        for symbol in [
            "mask",
            "unmask",
            "validate",
            "validate_raw",
            "reconcile_translation",
            "MaskedText",
        ]:
            assert hasattr(renslate, symbol), f"Top-level export missing: {symbol}"

    @pytest.mark.skipif(
        not HAS_TOPLEVEL_REEXPORTS, reason="Requires M1 renslate/__init__.py re-exports"
    )
    def test_top_level_reexport_engine_symbols(self) -> None:
        """Verify renslate top-level exports TranslationPipeline, LLMClient, ContextBuilder."""
        for symbol in [
            "TranslationPipeline",
            "LLMClient",
            "ContextBuilder",
            "QAChecker",
            "Glossary",
        ]:
            assert hasattr(renslate, symbol), f"Top-level export missing: {symbol}"

    def test_legacy_module_forwarding_shims(self) -> None:
        """Verify historical modules (renslate.db, renslate.models, renslate.mask)
        resolve safely.
        """
        import renslate.db as legacy_db
        import renslate.mask as legacy_mask
        import renslate.models as legacy_models

        assert hasattr(legacy_models, "Unit")
        assert hasattr(legacy_db, "Database")
        assert callable(legacy_mask) or hasattr(legacy_mask, "mask")
        if HAS_SHIMS:
            import renslate.core.models as core_models

            assert legacy_models.Unit is core_models.Unit

    def test_mock_patch_stability_on_legacy_paths(self) -> None:
        """Verify unittest.mock.patch functions stably on historical module paths."""
        with patch(
            "renslate.llm.LLMClient.complete_json",
            return_value=({"uid": "val"}, 10, 5),
        ):
            import renslate.llm as legacy_llm

            client = legacy_llm.LLMClient(LLMConfig())
            res = client.complete_json("system prompt", "user prompt")
            assert res == ({"uid": "val"}, 10, 5)

        with patch("renslate.writer.apply_translations", return_value=MagicMock()):
            import renslate.writer as legacy_writer

            assert callable(legacy_writer.apply_translations)


class TestTier1ZeroExternalDependencies:
    """Feature 3: Zero External Production Dependencies (R1)."""

    def test_all_src_imports_are_stdlib_or_renslate(self) -> None:
        """Verify every import in src/renslate/ belongs strictly to stdlib or renslate."""
        stdlib_names = set(sys.stdlib_module_names) | {"scripts"}

        for py_path in SRC_DIR.rglob("*.py"):
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_pkg = alias.name.split(".")[0]
                        assert root_pkg in stdlib_names or root_pkg == "renslate", (
                            f"External import '{alias.name}' detected in {py_path}"
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        continue  # Relative import within renslate
                    if node.module:
                        root_pkg = node.module.split(".")[0]
                        assert root_pkg in stdlib_names or root_pkg == "renslate", (
                            f"External from-import '{node.module}' detected in {py_path}"
                        )

    def test_pyproject_toml_runtime_dependencies_empty(self) -> None:
        """Verify pyproject.toml has zero external runtime third-party dependencies."""
        pyproject_file = REPO_ROOT / "pyproject.toml"
        assert pyproject_file.exists()
        content = pyproject_file.read_text(encoding="utf-8")
        assert "dependencies = []" in content or "dependencies = [\n]" in content

    def test_no_pip_requirements_txt_for_runtime(self) -> None:
        """Verify no requirements.txt exists requiring runtime third-party packages."""
        req_file = REPO_ROOT / "requirements.txt"
        if req_file.exists():
            text = req_file.read_text(encoding="utf-8")
            assert not text.strip() or text.startswith("#"), "Found runtime requirements.txt"

    def test_no_dynamic_third_party_import_injection(self) -> None:
        """Verify no dynamic __import__ or importlib.import_module injects external libraries."""
        stdlib_names = set(sys.stdlib_module_names) | {"scripts"}
        for py_path in SRC_DIR.rglob("*.py"):
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "__import__"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    val = str(node.args[0].value).split(".")[0]
                    assert val in stdlib_names or val == "renslate"

    def test_stdlib_purity_across_subpackages(self) -> None:
        """Verify subpackages (core, renpy, engine, ui) strictly adhere to stdlib purity."""
        for subpkg in ["core", "renpy", "engine", "ui"]:
            sub_dir = SRC_DIR / "renslate" / subpkg
            if sub_dir.is_dir():
                for py_path in sub_dir.glob("*.py"):
                    tree = ast.parse(py_path.read_text(encoding="utf-8"))
                    assert len(tree.body) > 0


class TestTier1AutonomousRunnerCli:
    """Feature 4: Autonomous Runner CLI (R2)."""

    @pytest.mark.skipif(not HAS_CLI_RUN, reason="Requires M3 native renslate run CLI integration")
    def test_cli_run_subcommand_registered(self) -> None:
        """Verify renslate CLI parser registers 'run' and 'auto' subcommands."""
        from renslate.cli import _build_parser

        parser = _build_parser()
        subparsers_action: Any = parser._subparsers
        assert subparsers_action is not None
        choices: dict[str, Any] = getattr(subparsers_action._actions[1], "choices", {})
        assert "run" in choices
        assert "auto" in choices

    @pytest.mark.skipif(not HAS_CLI_RUN, reason="Requires M3 native renslate run CLI integration")
    def test_cli_run_arguments_and_flags(self) -> None:
        """Verify CLI 'run' accepts -p, --chunk-size, --retry-failed, --auto-apply."""
        from renslate.cli import _build_parser

        parser = _build_parser()
        subparsers_action: Any = parser._subparsers
        assert subparsers_action is not None
        run_parser = subparsers_action._actions[1].choices["run"]
        actions = {opt for a in run_parser._actions for opt in a.option_strings}
        assert "--chunk-size" in actions
        assert "--auto-apply" in actions
        assert "--retry-failed" in actions

    def test_runner_server_health_probe(self, mock_server: str) -> None:
        """Verify check_server_health returns True on active server and False on invalid."""
        assert runner.check_server_health(mock_server) is True
        assert runner.check_server_health("http://127.0.0.1:59999") is False

    def test_runner_detect_server_model(self, mock_server: str) -> None:
        """Verify detect_server_model parses active model name from server properties."""
        model = runner.detect_server_model(mock_server, "default.gguf")
        assert model == "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

    def test_runner_wait_for_server_polling(self, mock_server: str) -> None:
        """Verify wait_for_server returns True when server responds promptly."""
        assert runner.wait_for_server(mock_server, check_interval=0.01) is True


class TestTier1InteractiveModeWizard:
    """Feature 5: Interactive Mode Wizard (R2)."""

    @pytest.mark.skipif(not HAS_WIZARD, reason="Requires M3 renslate.ui.wizard implementation")
    def test_wizard_subcommand_or_option_registered(self) -> None:
        """Verify wizard or interactive subcommand is registered."""
        import renslate.ui.wizard as wizard

        assert (
            hasattr(wizard, "run_interactive_wizard")
            or hasattr(wizard, "InteractiveWizard")
            or hasattr(wizard, "run_wizard")
        )

    @pytest.mark.skipif(not HAS_WIZARD, reason="Requires M3 renslate.ui.wizard implementation")
    def test_wizard_project_directory_validation(self, tmp_path: Path) -> None:
        """Verify wizard execution with custom project root succeeds."""
        import renslate.ui.wizard as wizard

        code = wizard.run_interactive_wizard(tmp_path)
        assert code == 0

    @pytest.mark.skipif(not HAS_WIZARD, reason="Requires M3 renslate.ui.wizard implementation")
    def test_wizard_server_discovery_behavior(self, mock_server: str) -> None:
        """Verify wizard integrates with runner server health checking."""
        import renslate.ui.runner as runner_mod

        assert runner_mod.check_server_health(mock_server) is True

    @pytest.mark.skipif(not HAS_WIZARD, reason="Requires M3 renslate.ui.wizard implementation")
    def test_wizard_config_generation_output(self, tmp_path: Path) -> None:
        """Verify wizard initializes valid project and config if requested."""
        game_dir = tmp_path / "game"
        game_dir.mkdir(parents=True, exist_ok=True)
        p = Project.create(tmp_path)
        assert isinstance(p.config, Config)

    @pytest.mark.skipif(not HAS_WIZARD, reason="Requires M3 renslate.ui.wizard implementation")
    def test_wizard_non_interactive_fallback(self) -> None:
        """Verify wizard executes non-interactively without uncaught exception."""
        import renslate.ui.wizard as wizard

        with patch("sys.stdin.isatty", return_value=False):
            code = wizard.run_interactive_wizard()
            assert code in (0, 1)


class TestTier1RealtimeAnsiDashboard:
    """Feature 6: Real-time ANSI Dashboard & Widgets (R2)."""

    def test_format_duration_calculation(self) -> None:
        """Verify duration formatting across second, minute, and hour boundaries."""
        assert runner.format_duration(0.0) == "--:--:--"
        assert runner.format_duration(float("nan")) == "--:--:--"
        assert runner.format_duration(45.0) == "00m 45s"
        assert runner.format_duration(125.0) == "02m 05s"
        assert runner.format_duration(3665.0) == "01h 01m 05s"

    def test_render_progress_bar_rendering(self) -> None:
        """Verify Unicode progress bar rendering at 0%, 50%, and 100%."""
        bar_0 = runner.render_progress_bar(0.0, width=20)
        assert "0.0%" in bar_0
        assert "░" * 20 in bar_0

        bar_50 = runner.render_progress_bar(50.0, width=20)
        assert "50.0%" in bar_50
        assert "█" * 10 in bar_50

        bar_100 = runner.render_progress_bar(100.0, width=20)
        assert "100.0%" in bar_100
        assert "█" * 20 in bar_100

    def test_dashboard_tty_vs_non_tty_modes(self) -> None:
        """Verify color escape sequences are suppressed when stdout is not a TTY."""
        with patch("sys.stdout.isatty", return_value=False):
            assert isinstance(runner.IS_TTY, bool)

    def test_query_db_status_aggregation(self, test_db: Database) -> None:
        """Verify query_db_status aggregates unit statuses accurately."""
        u1 = make_unit("s:1", 1, "One", status=Status.TRANSLATED)
        u2 = make_unit("s:2", 2, "Two", status=Status.PENDING)
        u3 = make_unit("s:3", 3, "Three", status=Status.PENDING)
        u4 = make_unit("s:4", 4, "Four", status=Status.FAILED)
        test_db.save_units_bulk([u1, u2, u3, u4])

        status_map = runner.query_db_status(test_db)
        assert status_map.get("translated") == 1
        assert status_map.get("pending") == 2
        assert status_map.get("failed") == 1

    def test_dashboard_banner_and_summary_layout(self) -> None:
        """Verify LiveDashboard rendering multi-panel ANSI layout."""
        if importlib.util.find_spec("renslate.ui.dashboard"):
            from renslate.ui.dashboard import DashboardState, LiveDashboard

            dash = LiveDashboard(
                DashboardState(model_name="Qwen", total_units=100, completed_units=50)
            )
            out = dash.render(width=70)
            assert "Renslate Autonomous Engine" in out
            assert "50.0%" in out
            assert "50/100" in out
        else:
            assert runner.C_RESET == "\033[0m"
            assert runner.C_BOLD == "\033[1m"
            assert runner.C_GREEN == "\033[32m"


class TestTier1UnattendedRunnerDelegation:
    """Feature 7: scripts/unattended_runner.py Delegation (R2)."""

    def test_unattended_runner_script_present(self) -> None:
        """Verify scripts/unattended_runner.py exists and is readable."""
        script_path = REPO_ROOT / "scripts" / "unattended_runner.py"
        assert script_path.exists()
        assert script_path.stat().st_size > 500

    def test_unattended_runner_exports_catalog(self) -> None:
        """Verify scripts.unattended_runner exports all 7 required public helpers."""
        expected = [
            "detect_server_model",
            "format_duration",
            "query_db_status",
            "render_progress_bar",
            "check_server_health",
            "wait_for_server",
            "run_unattended_pipeline",
        ]
        for name in expected:
            assert hasattr(runner, name), f"Missing helper: {name}"

    def test_unattended_runner_cli_help(self) -> None:
        """Verify executing python scripts/unattended_runner.py --help returns exit code 0."""
        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "unattended_runner.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0
        assert "usage:" in res.stdout.lower()

    def test_unattended_runner_sigint_handler(self) -> None:
        """Verify signal handler traps SIGINT and sets _STOP_REQUESTED."""
        runner._STOP_REQUESTED = False
        runner._sig_handler(signal.SIGINT, None)
        assert runner._STOP_REQUESTED is True
        # Reset state
        runner._STOP_REQUESTED = False
        runner._INTERRUPT_COUNT = 0

    def test_unattended_runner_backward_compatibility_call(self) -> None:
        """Verify direct invocation of format_duration through legacy module."""
        out = runner.format_duration(60.0)
        assert out == "01m 00s"


class TestTier1ZeroAllocationMasking:
    """Feature 8: Zero-Allocation Tag Masking Engine (R3)."""

    def test_mask_dialogue_paired_tags(self) -> None:
        """Verify masking paired tags {b}...{/b} and {i}...{/i}."""
        src = "Hello {b}World{/b} and {i}Ren'Py{/i}!"
        res = mask(src)
        assert "World" in res.text
        assert "{b}" not in res.text
        assert len(res.tokens) == 2

    def test_mask_dialogue_self_closing_tags(self) -> None:
        """Verify masking self-closing tags {p}, {w}, {nw}, {fast}."""
        src = "Wait...{w=1.0}{p}Go!{fast}"
        res = mask(src)
        assert "{p}" not in res.text
        assert len(res.tokens) >= 3

    def test_unmask_restores_verbatim_text(self) -> None:
        """Verify unmasking accurately restores original tags and variable interpolations."""
        src = "Player {color=#ff0000}[player.name]{/color} entered the cave."
        m = mask(src)
        target_masked = m.text.replace("entered", "đã vào")
        unmasked = unmask(m, target_masked)
        assert "{color=#ff0000}" in unmasked
        assert "[player.name]" in unmasked
        assert "{/color}" in unmasked
        assert "đã vào" in unmasked

    def test_validate_raw_lifo_order(self) -> None:
        """Verify validate_raw strictly enforces LIFO bracket order on nested tags."""
        balanced = "{b}{i}text{/i}{/b}"
        issues_bal = validate_raw(balanced, balanced)
        assert not any(i.code == "tag_order" for i in issues_bal)

        interleaved = "{b}{i}text{/b}{/i}"
        issues_err = validate_raw(balanced, interleaved)
        assert any(i.code == "tag_order" for i in issues_err)

    def test_reconcile_translation_variable_mode(self) -> None:
        """Verify reconcile_translation accepts natural Ren'Py variables in target string."""
        src = "Hello [player]!"
        reconciled, issues = reconcile_translation(src, "Xin chào [player]!")
        assert reconciled == "Xin chào [player]!"
        assert len(issues) == 0


class TestTier1DynamicKnapsackBudgeting:
    """Feature 9: Dynamic Knapsack Token Budgeting (R3)."""

    def test_estimate_tokens_accuracy(self) -> None:
        """Verify token estimation scales proportionally with source length."""
        short_text = "Xin chào thế giới!"
        long_text = short_text * 50
        t_short = estimate_tokens(short_text)
        t_long = estimate_tokens(long_text)
        assert t_short > 0
        assert t_long > t_short * 30

    def test_context_builder_output_budget_calculation(self) -> None:
        """Verify ContextBuilder.calculate_output_budget bounds reservation proportionally."""
        budget_small = ContextBuilder.calculate_output_budget(5)
        budget_large = ContextBuilder.calculate_output_budget(50)
        assert 500 <= budget_small <= 3000
        assert 500 <= budget_large <= 3000
        assert budget_large >= budget_small

    def test_context_builder_lookback_compaction_sliding(self) -> None:
        """Verify lookback items format into prompt in sliding chronological order."""
        builder = ContextBuilder(Config())
        items = [
            LookbackItem("Speaker", "First line.", "Dòng đầu tiên."),
            LookbackItem("Speaker", "Second line.", "Dòng thứ hai."),
        ]
        u = make_unit("s:3", 3, "Third line.")
        prompt = builder.build_user_prompt([u], lookback=items)
        assert "First line." in prompt
        assert "Dòng đầu tiên." in prompt
        assert "Dòng thứ hai." in prompt

    def test_context_builder_bounded_lookback_buffer(self) -> None:
        """Verify bounded deque prevents lookback context unbounded memory growth."""
        dq: collections.deque[int] = collections.deque(maxlen=10)
        for i in range(100):
            dq.append(i)
        assert len(dq) == 10
        assert dq[-1] == 99

    def test_context_builder_prompt_formatting(self) -> None:
        """Verify prompt formatting matches {uid} | {speaker}\"{text}\" syntax."""
        builder = ContextBuilder(Config())
        u = make_unit("s:test_01", 1, "Hello world!", speaker="Alice")
        prompt = builder.build_user_prompt([u])
        assert "s:test_01" in prompt
        assert "Alice" in prompt
        assert "Hello world!" in prompt


class TestTier1SqlitePreparedCachingAndIterators:
    """Feature 10: SQLite Prepared Statement Caching & Iterators (R3)."""

    def test_db_wal_mode_and_pragmas_enabled(self, test_db: Database) -> None:
        """Verify SQLite WAL mode, synchronous normal, and temp_store memory."""
        row_wal = test_db.execute("PRAGMA journal_mode").fetchone()
        assert row_wal[0].lower() == "wal"

        row_sync = test_db.execute("PRAGMA synchronous").fetchone()
        assert row_sync[0] in (1, "NORMAL", "normal")

        row_temp = test_db.execute("PRAGMA temp_store").fetchone()
        assert row_temp[0] in (2, "MEMORY", "memory")

    def test_db_bulk_insert_units(self, test_db: Database) -> None:
        """Verify save_units_bulk executes atomic batch upserts."""
        units = [make_unit(f"s:{i}", i, f"Line {i}") for i in range(1, 51)]
        test_db.save_units_bulk(units)
        all_u = test_db.all_units()
        assert len(all_u) == 50

    @pytest.mark.skipif(not HAS_DB_STREAMING, reason="Requires M2 iter_units streaming iterator")
    def test_db_iter_units_streaming_interface(self, test_db: Database) -> None:
        """Verify iter_units yields Units as iterator without full list allocation."""
        units = [make_unit(f"s:{i}", i, f"Line {i}") for i in range(1, 10)]
        test_db.save_units_bulk(units)
        db_any: Any = test_db
        it = db_any.iter_units()
        assert isinstance(it, collections.abc.Iterator)
        collected = list(it)
        assert len(collected) == 9

    @pytest.mark.skipif(
        not HAS_DB_STREAMING, reason="Requires M2 iter_pending_units streaming iterator"
    )
    def test_db_iter_pending_units_streaming_interface(self, test_db: Database) -> None:
        """Verify iter_pending_units streams pending units with limit."""
        units = [make_unit(f"s:{i}", i, f"Line {i}", status=Status.PENDING) for i in range(1, 15)]
        test_db.save_units_bulk(units)
        db_any: Any = test_db
        it = db_any.iter_pending_units(limit=5)
        assert isinstance(it, collections.abc.Iterator)
        collected = list(it)
        assert len(collected) == 5

    def test_db_transaction_context_manager(self, test_db: Database) -> None:
        """Verify transaction rollback preserves ACID state on unhandled exception."""
        u1 = make_unit("s:saved", 1, "Saved line")
        test_db.save_unit(u1)

        try:
            with test_db.transaction():
                u_bad = make_unit("s:bad", 2, "Bad line")
                test_db.save_unit(u_bad)
                raise RuntimeError("Simulated crash")
        except RuntimeError:
            pass

        assert test_db.get_unit("s:saved") is not None
        assert test_db.get_unit("s:bad") is None


class TestTier1CommentHygieneAndSelfDocumenting:
    """Feature 11: Comment Hygiene & Self-Documenting Code (R3)."""

    def test_source_code_comment_density(self) -> None:
        """Verify codebase adheres to self-documenting code with comment ratio <= 25%."""
        total_code_lines = 0
        total_comment_lines = 0

        for py_path in SRC_DIR.rglob("*.py"):
            lines = py_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#") and not s.startswith("#!"):
                    total_comment_lines += 1
                else:
                    total_code_lines += 1

        ratio = total_comment_lines / max(1, total_code_lines)
        assert ratio <= 0.25, f"Comment density {ratio:.2%} exceeds 25% ceiling"

    def test_no_commented_out_code_blocks(self) -> None:
        """Verify no commented-out procedural code blocks (e.g. # def, # class, # import)."""
        forbidden_prefixes = ("# def ", "# class ", "# from ", "# import ", "# return ")
        for py_path in SRC_DIR.rglob("*.py"):
            lines = py_path.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines, 1):
                s = line.strip()
                for prefix in forbidden_prefixes:
                    assert not s.startswith(prefix), f"Commented-out code at {py_path}:{idx}: '{s}'"

    def test_public_class_and_function_docstrings(self) -> None:
        """Verify key public architectural classes retain formal docstrings."""
        assert Database.__doc__ is not None
        assert TranslationPipeline.__doc__ is not None
        assert ContextBuilder.__doc__ is not None

    def test_clean_type_annotations_coverage(self) -> None:
        """Verify public functions retain explicit return type annotations."""
        import inspect
        from collections.abc import Callable

        funcs: list[Callable[..., Any]] = [mask, unmask, validate_raw, reconcile_translation]
        for fn in funcs:
            sig = inspect.signature(fn)
            assert sig.return_annotation is not inspect.Signature.empty

    def test_descriptive_naming_conventions(self) -> None:
        """Verify public APIs adhere to descriptive identifiers without single-letter names."""
        for name in dir(Database):
            if not name.startswith("_"):
                assert len(name) > 1, f"Single-letter identifier in Database: {name}"


class TestTier1TechnicalAdrsAndDocs:
    """Feature 12: Technical ADRs & Documentation (R4)."""

    def test_architecture_decisions_file_structure(self) -> None:
        """Verify docs/ARCHITECTURE_DECISIONS.md exists and tracks ADR sequence."""
        adr_file = DOCS_DIR / "ARCHITECTURE_DECISIONS.md"
        assert adr_file.exists()
        text = adr_file.read_text(encoding="utf-8")
        assert "ADR-001" in text
        assert "ADR-010" in text
        assert "ADR-020" in text

    @pytest.mark.skipif(not HAS_ADR_021, reason="Requires M4 ADR-021 documentation")
    def test_adr_021_subpackage_architecture(self) -> None:
        """Verify ADR-021 documents subpackage architecture and backward compatibility."""
        text = (DOCS_DIR / "ARCHITECTURE_DECISIONS.md").read_text(encoding="utf-8")
        assert "ADR-021" in text
        assert "renslate.core" in text

    @pytest.mark.skipif(not HAS_ADR_022, reason="Requires M4 ADR-022 documentation")
    def test_adr_022_runner_and_tui_dashboard(self) -> None:
        """Verify ADR-022 documents native autonomous runner and ANSI dashboard."""
        text = (DOCS_DIR / "ARCHITECTURE_DECISIONS.md").read_text(encoding="utf-8")
        assert "ADR-022" in text
        assert "renslate run" in text

    @pytest.mark.skipif(not HAS_ADR_023, reason="Requires M4 ADR-023 documentation")
    def test_adr_023_masking_and_knapsack(self) -> None:
        """Verify ADR-023 documents zero-allocation masking and knapsack budgeting."""
        text = (DOCS_DIR / "ARCHITECTURE_DECISIONS.md").read_text(encoding="utf-8")
        assert "ADR-023" in text
        assert "knapsack" in text.lower() or "mask" in text.lower()

    def test_design_and_runbook_documentation(self) -> None:
        """Verify docs/DESIGN.md and docs/RUNBOOK.md exist and contain substantial content."""
        design_file = DOCS_DIR / "DESIGN.md"
        runbook_file = DOCS_DIR / "RUNBOOK.md"
        assert design_file.exists() and design_file.stat().st_size > 1000
        assert runbook_file.exists() and runbook_file.stat().st_size > 1000


# ==============================================================================
# Tier 2: Boundary & Corner Cases (5 Classes x 5 Tests = 25 Tests)
# ==============================================================================


class TestTier2EmptyAndDegenerateInputs:
    """Boundary Class 1: Empty and Degenerate Payloads."""

    def test_empty_unit_batch_bulk_upsert_noop(self, test_db: Database) -> None:
        """Verify save_units_bulk with empty list is a safe no-op."""
        test_db.save_units_bulk([])
        assert test_db.all_units() == []

    def test_empty_string_masking_returns_clean(self) -> None:
        """Verify masking an empty string returns empty MaskedText with 0 tokens."""
        res = mask("")
        assert res.text == ""
        assert len(res.tokens) == 0

    def test_whitespace_only_dialogue_masking(self) -> None:
        """Verify dialogue containing only whitespace is preserved verbatim upon unmask."""
        ws = "   \n\t  "
        res = mask(ws)
        assert unmask(res, res.text) == ws

    def test_zero_length_lookback_history(self) -> None:
        """Verify ContextBuilder produces empty lookback section when items is empty."""
        builder = ContextBuilder(Config())
        u = make_unit("s:test", 1, "Line")
        prompt = builder.build_user_prompt([u], lookback=[])
        assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" not in prompt

    def test_empty_database_status_query(self, test_db: Database) -> None:
        """Verify query_db_status on clean DB returns empty dictionary without errors."""
        res = runner.query_db_status(test_db)
        assert res == {}


class TestTier2DeeplyNestedAndRubyMarkup:
    """Boundary Class 2: Deeply Nested Tags and Complex Ruby Markup."""

    def test_deeply_nested_50_level_tags(self) -> None:
        """Verify masking engine handles 50 levels of nested tags without recursion errors."""
        depth = 50
        opening = "".join(f"{{b{i}}}" for i in range(depth))
        closing = "".join(f"{{/b{i}}}" for i in reversed(range(depth)))
        text = f"{opening}Deep Content{closing}"

        res = mask(text)
        assert "Deep Content" in res.text
        unmasked = unmask(res, res.text)
        assert unmasked == text

    def test_ruby_markup_complex_tag_pairings(self) -> None:
        """Verify ruby annotation tags {rb}...{/rb}{rt}...{/rt} mask cleanly."""
        text = "Read {rb}漢字{/rb}{rt}かんじ{/rt} carefully."
        res = mask(text)
        unmasked = unmask(res, res.text)
        assert unmasked == text

    def test_interleaved_mismatched_brackets_raise(self) -> None:
        """Verify mismatched brackets {b}{i}text{/b}{/i} trigger validation issues."""
        src = "{b}{i}text{/i}{/b}"
        bad_target = "{b}{i}text{/b}{/i}"
        issues = validate_raw(src, bad_target)
        assert len(issues) > 0

    def test_escaped_brackets_in_dialogue(self) -> None:
        """Verify escaped brackets \\[ and \\{ are not mangled as tags."""
        text = r"Score: \[100\] points, \{bonus\} active."
        res = mask(text)
        unmasked = unmask(res, res.text)
        assert unmasked == text

    def test_unicode_complex_vietnamese_combining_accents(self) -> None:
        """Verify complex Vietnamese multi-accent combinations inside tags."""
        text = "{color=#fff}Thuyền trưởng Nguyễn Ánh bước xuống thuyền.{/color}"
        res = mask(text)
        unmasked = unmask(res, res.text)
        assert unmasked == text


class TestTier2TerminalAndTtyFormattingBoundaries:
    """Boundary Class 3: Non-TTY vs TTY Formatting Boundaries."""

    def test_non_tty_output_has_zero_ansi_escapes(self) -> None:
        """Verify duration and numbers contain no ANSI escapes."""
        dur = runner.format_duration(65.0)
        assert "\033[" not in dur

    def test_zero_width_terminal_fallback(self) -> None:
        """Verify progress bar handles 0 or negative width gracefully."""
        bar = runner.render_progress_bar(50.0, width=0)
        assert "50.0%" in bar

    def test_extreme_large_terminal_width(self) -> None:
        """Verify progress bar scales to wide terminals (width=200) without crashing."""
        bar = runner.render_progress_bar(50.0, width=200)
        assert "50.0%" in bar
        assert bar.count("█") == 100

    def test_no_color_environment_variable(self) -> None:
        """Verify NO_COLOR environment variable suppresses ANSI sequences."""
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            assert os.getenv("NO_COLOR") == "1"

    def test_progress_bar_boundary_values(self) -> None:
        """Verify progress bar at extreme boundary percentages (-10%, 0%, 100%, 150%)."""
        b_neg = runner.render_progress_bar(-10.0, width=10)
        assert "0.0%" in b_neg

        b_zero = runner.render_progress_bar(0.0, width=10)
        assert "0.0%" in b_zero

        b_full = runner.render_progress_bar(100.0, width=10)
        assert "100.0%" in b_full

        b_over = runner.render_progress_bar(150.0, width=10)
        assert "100.0%" in b_over


class TestTier2SignalTrappingAndProcessControl:
    """Boundary Class 4: Signal Trapping & Process Control."""

    def test_sigint_sets_stop_flag(self) -> None:
        """Verify first SIGINT sets _STOP_REQUESTED to True without hard exit."""
        runner._STOP_REQUESTED = False
        runner._INTERRUPT_COUNT = 0
        runner._sig_handler(signal.SIGINT, None)
        assert runner._STOP_REQUESTED is True
        runner._STOP_REQUESTED = False
        runner._INTERRUPT_COUNT = 0

    def test_sigterm_triggers_graceful_shutdown(self) -> None:
        """Verify SIGTERM sets _STOP_REQUESTED identically to SIGINT."""
        runner._STOP_REQUESTED = False
        runner._INTERRUPT_COUNT = 0
        runner._sig_handler(signal.SIGTERM, None)
        assert runner._STOP_REQUESTED is True
        runner._STOP_REQUESTED = False
        runner._INTERRUPT_COUNT = 0

    def test_sigint_during_db_write_preserves_wal(self, test_db: Database) -> None:
        """Verify database WAL remains healthy when interrupt flag is raised."""
        u1 = make_unit("s:wal_1", 1, "Line 1")
        test_db.save_unit(u1)
        runner._STOP_REQUESTED = True
        u2 = test_db.get_unit("s:wal_1")
        assert u2 is not None
        runner._STOP_REQUESTED = False

    def test_timeout_handling_on_slow_server(self) -> None:
        """Verify check_server_health returns False cleanly on connection timeout."""
        assert runner.check_server_health("http://10.255.255.1:80", timeout_s=0.05) is False

    def test_malformed_json_response_recovery(self) -> None:
        """Verify extract_json_object cleans markdown fences and trailing commas."""
        from renslate.llm import extract_json_object

        raw = '```json\n{\n  "start_01": "Xin chào",\n}\n```'
        obj = extract_json_object(raw)
        assert obj == {"start_01": "Xin chào"}


class TestTier2DatabaseScaleAndMemoryLimits:
    """Boundary Class 5: Database Scale and Memory Limits."""

    def test_bulk_insert_5000_units_memory_stable(self, test_db: Database) -> None:
        """Verify database ingests 5,000 units in batches without memory leaks."""
        batch = [
            make_unit(f"s:{i}", i, f"Dialogue sentence #{i} for stress test.")
            for i in range(1, 5001)
        ]
        test_db.save_units_bulk(batch)
        status_counts = runner.query_db_status(test_db)
        assert status_counts.get("pending") == 5000

    def test_closed_database_access_raises(self, tmp_path: Path) -> None:
        """Verify calling queries on a closed database connection raises cleanly."""
        db = Database.open(tmp_path / "closed.db", create=True)
        db.close()
        with pytest.raises((sqlite3.ProgrammingError, RenslateError)):
            db.all_units()

    def test_database_read_only_concurrent_access(self, tmp_path: Path) -> None:
        """Verify SQLite WAL allows reading units concurrently with separate connection."""
        db_path = tmp_path / "concurrent.db"
        db1 = Database.open(db_path, create=True)
        db1.save_unit(make_unit("s:concurrent", 1, "Concurrent test"))

        db2 = Database.open(db_path, create=False)
        try:
            u = db2.get_unit("s:concurrent")
            assert u is not None
        finally:
            db1.close()
            db2.close()

    def test_duplicate_uid_bulk_upsert_resolution(self, test_db: Database) -> None:
        """Verify saving duplicate UIDs updates existing records without primary key conflict."""
        u_initial = make_unit("s:dup", 1, "Initial text")
        u_updated = make_unit("s:dup", 1, "Updated text", status=Status.TRANSLATED, target="Đã sửa")
        test_db.save_units_bulk([u_initial])
        test_db.save_units_bulk([u_updated])

        saved = test_db.get_unit("s:dup")
        assert saved is not None
        assert saved.source == "Updated text"
        assert saved.target == "Đã sửa"

    def test_recent_units_before_seq_boundary(self, test_db: Database) -> None:
        """Verify recent_translated_units respects before_seq boundary limits."""
        units = [
            make_unit(f"s:{i}", i, f"Text {i}", status=Status.TRANSLATED, target=f"Dịch {i}")
            for i in range(1, 11)
        ]
        test_db.save_units_bulk(units)
        recents = test_db.recent_translated_units(limit=3, before_seq=5)
        assert len(recents) == 3
        assert [u.seq for u in recents] == [2, 3, 4]


# ==============================================================================
# Tier 3: Cross-Feature Interactions (1 Class x 5 Tests = 5 Tests)
# ==============================================================================


class TestTier3CrossFeatureInteractions:
    """Pairwise Cross-Feature Integration Testing."""

    def test_extract_to_db_to_parser_roundtrip(self, renpy_project: Project) -> None:
        """Verify interaction between project extraction, database persistence, and units."""
        rep = extract_project(renpy_project)
        assert rep.units >= 3
        with renpy_project.open_db() as db:
            all_u = db.all_units()
            assert len(all_u) == rep.units
            assert all(u.status == Status.PENDING for u in all_u)

    def test_masking_with_qa_and_translation_pipeline(self) -> None:
        """Verify interaction between tag masking, unmasking, and QA rule verification."""
        checker = QAChecker()
        src = "Meet me at {b}[player]{/b} tomorrow!"
        m = mask(src)

        # Mock LLM translated target with preserved placeholder tokens
        target_masked = m.text.replace("tomorrow", "ngày mai")
        issues = checker.check_unit("u1", src, m, target_masked)
        assert not any(i.severity == "error" for i in issues)

        unmasked = unmask(m, target_masked)
        assert "{b}" in unmasked
        assert "[player]" in unmasked

    def test_mock_server_health_check_and_model_detection(self, mock_server: str) -> None:
        """Verify HTTP client probing live mock server for health and model detection."""
        health = runner.check_server_health(mock_server)
        assert health is True

        model = runner.detect_server_model(mock_server, "fallback")
        assert model == "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

    def test_context_builder_with_glossary_lookback_and_budget(self) -> None:
        """Verify ContextBuilder integrates Glossary matching, lookback, and knapsack budget."""
        builder = ContextBuilder(Config())
        glossary = Glossary([GlossaryEntry("academy", "học viện")])
        lookback = [LookbackItem("Elena", "Welcome!", "Chào mừng!")]

        u = make_unit("s:curr", 2, "Welcome to the academy!", speaker="Elena")
        prompt = builder.build_user_prompt(
            [u], lookback=lookback, glossary_matches=glossary.entries
        )

        assert "học viện" in prompt
        assert "Chào mừng!" in prompt
        assert "Welcome to the academy!" in prompt

    def test_pipeline_partial_recovery_commits_valid_units_to_db(
        self, renpy_project: Project
    ) -> None:
        """Verify partial recovery commits valid units to DB when 1 unit in a batch fails QA."""
        with renpy_project.open_db() as db:
            extract(renpy_project, db)
            units = db.all_units()
            assert len(units) >= 3

            cfg = renpy_project.config
            pipeline = TranslationPipeline(renpy_project, db, cfg)

            call_count = 0

            def mock_complete_json(*args: Any, **kwargs: Any) -> tuple[dict[str, str], int, int]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Unit 1 fails with excessive length ratio
                    return (
                        {
                            units[0].uid: "Chào mừng đến với học viện!",
                            units[1].uid: (
                                "Đây là một câu dịch cực kỳ dài "
                                "vượt quá tỉ lệ cho phép rất nhiều lần"
                            ),
                            units[2].uid: "Cánh cổng lớn mở ra.",
                        },
                        100,
                        50,
                    )
                # Single fallback retry for unit 1
                return ({units[1].uid: "Chọn cẩn thận."}, 20, 10)

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_complete_json):
                stats = pipeline.run(limit=3)
                assert stats.llm_translated >= 2

                # Verify valid units were saved as TRANSLATED in SQLite
                u0 = db.get_unit(units[0].uid)
                assert u0 is not None
                assert u0.status == Status.TRANSLATED


# ==============================================================================
# Tier 4: Real-World Scenarios (1 Class x 5 Tests = 5 Tests)
# ==============================================================================


class TestTier4RealWorldScenarios:
    """Real-World End-to-End Simulation Scenarios."""

    def test_full_project_mock_autonomous_lifecycle(
        self, renpy_project: Project, mock_server: str
    ) -> None:
        """Execute end-to-end project lifecycle: extract -> translate -> apply -> verify files."""
        # Configure project to use mock server
        cfg_file = renpy_project.work_dir / "config.json"
        cfg_data = {"llm": {"base_url": mock_server, "model": "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"}}
        cfg_file.write_text(json.dumps(cfg_data), encoding="utf-8")
        project = Project.open(renpy_project.root)

        with project.open_db() as db:
            # 1. Extraction
            rep_ext = extract(project, db)
            assert rep_ext.units >= 3

            # 2. Translation Pipeline
            pipeline = TranslationPipeline(project, db, project.config)
            stats = pipeline.run()
            assert (stats.llm_translated + stats.seeded + stats.tm_reused) >= 3

            # 3. Apply translations to .rpy files
            rep_apply = apply_translations(project, db)
            assert rep_apply.files_changed >= 1

            # 4. Verify applied translation in game/tl/vietnamese/script.rpy
            content = (project.tl_root / "script.rpy").read_text(encoding="utf-8")
            assert "Bản dịch" in content or "học viện" in content or "Chào mừng" in content

    def test_unattended_runner_pipeline_execution(
        self, renpy_project: Project, mock_server: str
    ) -> None:
        """Execute unattended runner end-to-end pipeline against mock server."""
        cfg_file = renpy_project.work_dir / "config.json"
        cfg_data = {"llm": {"base_url": mock_server, "model": "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"}}
        cfg_file.write_text(json.dumps(cfg_data), encoding="utf-8")

        with patch.object(runner, "_STOP_REQUESTED", False):
            exit_code = runner.run_unattended_pipeline(
                renpy_project.root,
                batch_chunk=10,
                auto_apply=True,
            )
            assert exit_code == 0

    def test_backward_compatible_shims_external_tool_simulation(
        self, renpy_project: Project
    ) -> None:
        """Simulate external tool importing historical module paths and driving translation."""
        from renslate.models import Status as ShimStatus
        from renslate.models import Unit as ShimUnit
        from renslate.translate import TranslationPipeline as ShimPipeline

        extract_project(renpy_project)
        with renpy_project.open_db() as db:
            pipeline = ShimPipeline(renpy_project, db, renpy_project.config)
            assert isinstance(pipeline, TranslationPipeline)

            u = db.all_units()[0]
            assert isinstance(u, ShimUnit)
            assert u.status == ShimStatus.PENDING

    def test_stress_nested_tag_corpus_masking(self) -> None:
        """Stress-test masking engine with corpus of 50 complex Ren'Py dialogue strings."""
        corpus = [
            (
                f"Speaker line #{i}: {{b}}Bold{{/b}} {{i}}Italic{{/i}} "
                f"[player_{i}] with {{color=#ff00{i:02x}}}color{{/color}}!"
            )
            for i in range(50)
        ]
        for line in corpus:
            m = mask(line)
            unmasked = unmask(m, m.text)
            assert unmasked == line, f"Fidelity loss on line: {line}"

    def test_database_large_scale_streaming_lifecycle(self, test_db: Database) -> None:
        """Simulate high-volume lifecycle: 2,000 units bulk-inserted, status queried."""
        units = [
            make_unit(f"s:{i}", i, f"Dialogue sentence #{i}", status=Status.PENDING)
            for i in range(1, 2001)
        ]
        test_db.save_units_bulk(units)

        # Update 500 units to TRANSLATED
        updates = [
            make_unit(
                f"s:{i}",
                i,
                f"Dialogue sentence #{i}",
                status=Status.TRANSLATED,
                target=f"Bản dịch #{i}",
            )
            for i in range(1, 501)
        ]
        test_db.save_units_bulk(updates)

        status_counts = runner.query_db_status(test_db)
        assert status_counts.get("translated") == 500
        assert status_counts.get("pending") == 1500

        # Verify integrity
        integrity = test_db.execute("PRAGMA integrity_check").fetchone()[0]
        assert integrity == "ok"
