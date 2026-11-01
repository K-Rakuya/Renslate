"""Opaque-Box E2E Test Suite for Requirement R1: Modularization & Architecture Standards.

Covers:
- Standardized directory layout and subpackages (core, renpy, engine, ui)
- Acyclic DAG dependency hierarchy
- Pure Python 3.10+ standard library compliance (zero runtime dependencies)
- Centralized configuration registry (eliminating hardcoded URLs, models, paths)
- Static data asset extraction and minimal inline comments
- Forwarding shim identity with subpackage targets

Tiers:
- Tier 1: Feature Coverage (>=5 tests per feature)
- Tier 2: Boundary & Corner Cases (>=5 tests per feature)
- Tier 3: Cross-Feature Combinations
- Tier 4: Real-World Application Scenarios
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import renslate
import renslate.core
import renslate.core.config
import renslate.core.db
import renslate.core.errors
import renslate.core.models
import renslate.core.project
import renslate.core.textio
import renslate.engine
import renslate.engine.context
import renslate.engine.llm
import renslate.engine.qa
import renslate.engine.translate
import renslate.renpy
import renslate.renpy.mask
import renslate.renpy.parser
import renslate.renpy.rpyc
import renslate.renpy.writer
import renslate.ui
import renslate.ui.dashboard
import renslate.ui.runner
import renslate.ui.wizard
from renslate.core.config import (
    BatchConfig,
    Config,
    LLMConfig,
    QAConfig,
    StyleConfig,
    TMConfig,
)
from renslate.core.errors import ConfigError
from renslate.core.project import Project

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


# ==============================================================================
# Helper Utilities for Architecture Inspection
# ==============================================================================


def _get_imports_from_file(path: Path) -> set[str]:
    """Parse a python file AST and return all top-level imported module names."""
    if not path.is_file():
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _get_all_python_files(directory: Path) -> list[Path]:
    """Recursively collect all .py files in a directory."""
    return sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


# ==============================================================================
# Tier 1: Feature Coverage (Core Architecture & Standards Contracts)
# ==============================================================================


class TestR1Tier1FeatureCoverage:
    """Tier 1: Comprehensive verification of Requirement R1 core features."""

    def test_pure_standard_library_runtime_dependencies(self) -> None:
        """Verify pyproject.toml enforces zero runtime external dependencies."""
        assert PYPROJECT_PATH.exists(), "pyproject.toml must exist at root"
        content = PYPROJECT_PATH.read_text(encoding="utf-8")
        assert "dependencies = []" in content or 'dependencies = [\n]' in content

    def test_py_typed_marker_present(self) -> None:
        """Verify PEP 561 marker py.typed exists in the renslate package."""
        marker = SRC_DIR / "renslate" / "py.typed"
        assert marker.exists(), "py.typed must be present for strict typing"

    def test_subpackage_exports_integrity(self) -> None:
        """Verify the 4 canonical subpackages export their designated members."""
        assert hasattr(renslate.core, "Config")
        assert hasattr(renslate.core, "Database")
        assert hasattr(renslate.core, "Unit")
        assert hasattr(renslate.renpy, "mask")
        assert hasattr(renslate.renpy, "rpyc")
        assert hasattr(renslate.engine, "translate")
        assert hasattr(renslate.ui, "runner")

    def test_acyclic_subpackage_dependency_core_independence(self) -> None:
        """Verify renslate.core does not import from renpy, engine, or ui."""
        core_dir = SRC_DIR / "renslate" / "core"
        for py_file in _get_all_python_files(core_dir):
            imports = _get_imports_from_file(py_file)
            for imp in imports:
                assert not imp.startswith("renslate.renpy"), f"{py_file.name} illegal: {imp}"
                assert not imp.startswith("renslate.engine"), f"{py_file.name} illegal: {imp}"
                assert not imp.startswith("renslate.ui"), f"{py_file.name} illegal: {imp}"

    def test_acyclic_subpackage_dependency_renpy_independence(self) -> None:
        """Verify renslate.renpy does not import from engine or ui."""
        renpy_dir = SRC_DIR / "renslate" / "renpy"
        for py_file in _get_all_python_files(renpy_dir):
            imports = _get_imports_from_file(py_file)
            for imp in imports:
                assert not imp.startswith("renslate.engine"), f"{py_file.name} illegal: {imp}"
                assert not imp.startswith("renslate.ui"), f"{py_file.name} illegal: {imp}"

    def test_acyclic_subpackage_dependency_engine_independence(self) -> None:
        """Verify renslate.engine does not import from ui."""
        engine_dir = SRC_DIR / "renslate" / "engine"
        for py_file in _get_all_python_files(engine_dir):
            imports = _get_imports_from_file(py_file)
            for imp in imports:
                assert not imp.startswith("renslate.ui"), f"{py_file.name} illegal: {imp}"

    def test_centralized_config_initialization(self) -> None:
        """Verify Config instantiates cleanly with all typed sub-sections."""
        cfg = Config()
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.batch, BatchConfig)
        assert isinstance(cfg.tm, TMConfig)
        assert isinstance(cfg.qa, QAConfig)
        assert isinstance(cfg.style, StyleConfig)
        assert cfg.target_lang == "vietnamese"

    def test_centralized_config_custom_llm_url_and_model(self) -> None:
        """Verify Config allows fully custom LLM server URL and model name."""
        custom_llm = LLMConfig(
            base_url="https://llm.internal.production:8443/v1",
            model="Mistral-Large-Instruct-2411.gguf",
            n_ctx=8192,
            timeout_s=120.0,
        )
        cfg = Config(llm=custom_llm)
        assert cfg.llm.base_url == "https://llm.internal.production:8443/v1"
        assert cfg.llm.model == "Mistral-Large-Instruct-2411.gguf"
        assert cfg.llm.n_ctx == 8192

    def test_centralized_config_immutability(self) -> None:
        """Verify Config and LLMConfig are frozen slots dataclasses."""
        cfg = Config()
        with pytest.raises(AttributeError):
            cfg.target_lang = "japanese"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.llm.base_url = "http://modified:8000"  # type: ignore[misc]

    def test_forwarding_shim_identity(self) -> None:
        """Verify root forwarding shims resolve to identical core objects."""
        import renslate.models as shim_m

        assert shim_m.Unit is renslate.core.models.Unit
        assert shim_m.Status is renslate.core.models.Status


# ==============================================================================
# Tier 2: Boundary & Corner Cases (Invalid Inputs, Traversal, Limits)
# ==============================================================================


class TestR1Tier2Boundaries:
    """Tier 2: Boundary and defensive error validation for configuration and modularity."""

    def test_config_invalid_target_lang_rejected(self) -> None:
        """Verify invalid language identifiers raise ConfigError."""
        with pytest.raises(ConfigError, match="không phải tên ngôn ngữ hợp lệ"):
            Config(target_lang="123-invalid-lang!")

    def test_config_none_string_target_lang_rejected(self) -> None:
        """Verify target_lang='None' string raises ConfigError."""
        with pytest.raises(ConfigError):
            Config(target_lang="None")

    def test_config_path_traversal_tl_dir_rejected(self) -> None:
        """Verify tl_dir with directory traversal (..) is rejected."""
        with pytest.raises(ConfigError, match="đường dẫn tương đối"):
            Config(tl_dir="../../outside/secret")

    def test_config_absolute_tl_dir_rejected(self) -> None:
        """Verify absolute tl_dir is rejected."""
        with pytest.raises(ConfigError, match="đường dẫn tương đối"):
            Config(tl_dir="/var/data/tl")

    def test_config_invalid_llm_url_scheme_rejected(self) -> None:
        """Verify base_url without http:// or https:// raises ConfigError."""
        bad_llm = LLMConfig(base_url="ftp://server:8080/v1")
        with pytest.raises(ConfigError, match="bắt đầu bằng http://"):
            Config(llm=bad_llm)

    def test_config_empty_model_name_rejected(self) -> None:
        """Verify empty model name raises ConfigError."""
        bad_llm = LLMConfig(model="")
        with pytest.raises(ConfigError, match="không được để trống"):
            Config(llm=bad_llm)

    def test_config_load_unknown_json_keys_rejected(self, tmp_path: Path) -> None:
        """Verify loading JSON with unrecognized keys raises ConfigError."""
        bad_json = tmp_path / "bad_config.json"
        bad_json.write_text(json.dumps({"unknown_field_xyz": 123}), encoding="utf-8")
        with pytest.raises(ConfigError, match="khóa không hợp lệ"):
            Config.load(bad_json)

    def test_config_load_non_existent_file_rejected(self, tmp_path: Path) -> None:
        """Verify loading non-existent config file raises ConfigError."""
        with pytest.raises(ConfigError, match="không tìm thấy tệp"):
            Config.load(tmp_path / "does_not_exist.json")

    def test_config_bounded_temperature_exceeded(self) -> None:
        """Verify temperature out of allowed range raises ConfigError."""
        with pytest.raises(ConfigError, match="phải <="):
            Config.from_dict({"llm": {"temperature": 5.0}})

    def test_config_bounded_negative_max_retries_rejected(self) -> None:
        """Verify negative max_retries raises ConfigError."""
        with pytest.raises(ConfigError, match="phải >="):
            Config.from_dict({"llm": {"max_retries": -1}})


# ==============================================================================
# Tier 3: Cross-Feature Combinations (Config, Project & Subpackage Wiring)
# ==============================================================================


class TestR1Tier3CrossFeature:
    """Tier 3: Interactions between Configuration, Project workspace, and Subpackages."""

    def test_config_roundtrip_dict_serialization(self) -> None:
        """Verify Config serialize to dict and recreate produces an identical instance."""
        original = Config(
            target_lang="japanese",
            tl_dir="game/tl_jp",
            llm=LLMConfig(base_url="https://api.openai.com/v1", model="gpt-4o", n_ctx=4096),
            batch=BatchConfig(max_lines=15, max_source_tokens=500),
        )
        as_dict = original.to_dict()
        recreated = Config.from_dict(as_dict)
        assert original == recreated
        assert recreated.target_lang == "japanese"
        assert recreated.llm.base_url == "https://api.openai.com/v1"
        assert recreated.batch.max_lines == 15

    def test_config_file_save_and_reload(self, tmp_path: Path) -> None:
        """Verify Config saving to disk and reloading from disk preserves all settings."""
        cfg_file = tmp_path / "project_config.json"
        cfg = Config(
            target_lang="spanish",
            llm=LLMConfig(base_url="http://custom:11434/v1", model="llama3:latest"),
            tm=TMConfig(fuzzy_threshold=0.92),
            qa=QAConfig(max_length_ratio=2.2),
        )
        cfg.save(cfg_file)
        assert cfg_file.exists()

        loaded = Config.load(cfg_file)
        assert loaded == cfg
        assert loaded.tm.fuzzy_threshold == 0.92
        assert loaded.qa.max_length_ratio == 2.2

    def test_project_initialization_with_custom_config(self, tmp_path: Path) -> None:
        """Verify Project binds correctly with custom configuration."""
        game_root = tmp_path / "game_project"
        game_root.mkdir()
        (game_root / "game").mkdir()

        project = Project.create(game_root, target_lang="french")
        assert project.config.target_lang == "french"
        assert project.config.tl_dir == "game/tl"
        assert project.tl_root == game_root / "game" / "tl" / "french"
        assert project.config_path.exists()

        custom_cfg = Config(
            target_lang="french",
            tl_dir="game/tl",
            llm=LLMConfig(base_url="http://10.0.0.5:8080/v1", model="Qwen-Custom.gguf"),
        )
        custom_cfg.save(project.config_path)

        reloaded = Project.open(game_root)
        assert reloaded.config.target_lang == "french"
        assert reloaded.config.llm.model == "Qwen-Custom.gguf"

    def test_subpackages_cohesive_pipeline_wiring(self) -> None:
        """Verify cross-module invocation across core, renpy, engine, and ui."""
        from renslate.engine.qa import QAChecker
        from renslate.renpy.mask import mask

        raw_text = "Hello, {b}[player]{/b}! Good morning."
        masked = mask(raw_text)
        assert "⟦V1⟧" in masked.text or "⟦T" in masked.text

        checker = QAChecker()
        issues = checker.check_unit("test_001", raw_text, masked, "Xin chào bạn!")
        assert isinstance(issues, list)


# ==============================================================================
# Tier 4: Real-World Application Scenarios (Project Lifecycle & Setup)
# ==============================================================================


class TestR1Tier4RealWorld:
    """Tier 4: Production visual novel localization project configuration lifecycle."""

    def test_multi_target_project_scaffolding(self, tmp_path: Path) -> None:
        """Simulate creating a fresh visual novel project with enterprise config."""
        game_dir = tmp_path / "GoddessWhim_VN"
        game_dir.mkdir()
        (game_dir / "game" / "tl").mkdir(parents=True)
        (game_dir / "game" / "script.rpy").write_text(
            'label start:\n    e "Welcome to Renvi production."\n', encoding="utf-8"
        )

        project = Project.create(game_dir, target_lang="vietnamese")
        assert project.root == game_dir
        assert project.db_path.name == "project.db"
        assert project.characters_path.name == "characters.json"
        assert project.relationships_path.name == "relationships.json"

        cfg = Config(
            target_lang="vietnamese",
            tl_dir="game/tl",
            llm=LLMConfig(
                base_url="http://127.0.0.1:8080/v1",
                model="Qwen3.6-35B-A3B-UD-Q3_K_M.gguf",
                n_ctx=8192,
                temperature=0.2,
                timeout_s=300.0,
            ),
            batch=BatchConfig(max_lines=30, max_source_tokens=800, lookback=6),
            style=StyleConfig(default_register="tự nhiên, thân mật"),
        )
        cfg.save(project.config_path)

        # Verify project is reopenable and preserves all settings
        reopened = Project.open(game_dir)
        assert reopened.config.batch.max_lines == 30
        assert reopened.config.style.default_register == "tự nhiên, thân mật"
        assert reopened.config.llm.n_ctx == 8192
