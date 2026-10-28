"""Renslate: Ren'Py Visual Novel Localization Engine.

High-cohesion decoupled subpackage architecture:
- renslate.core: Domain models, exceptions, configuration, SQLite database,
  and filesystem workspace.
- renslate.renpy: Ren'Py AST parsing, lexical codecs, dialogue tag masking,
  extraction, and code generation.
- renslate.engine: Translation pipeline, local LLM client, context budgeting, QA, TM, and analysis.
- renslate.ui: Rich terminal UI, ANSI live dashboard, progress meters, and autonomous runner.
"""

from __future__ import annotations

import importlib
import sys

# Subpackages
from renslate import core, engine, renpy, ui

# Re-exports from renslate.core
from renslate.core.config import (
    BatchConfig,
    Config,
    LLMConfig,
    QAConfig,
    StyleConfig,
    TMConfig,
)
from renslate.core.db import MIGRATIONS, Database, utc_timestamp
from renslate.core.errors import (
    ApplyError,
    ConfigError,
    LLMError,
    MarkupError,
    ProjectError,
    QAError,
    RenslateError,
    SdkError,
    TranslationError,
)
from renslate.core.models import (
    LOCKED_ORIGINS,
    ApplyTarget,
    BatchLog,
    CharacterProfile,
    GlossaryEntry,
    Origin,
    QAIssue,
    RelationshipEdge,
    RelationshipKind,
    SceneMacroContext,
    SceneSummary,
    Slot,
    Status,
    TMEntry,
    Unit,
    UnitKind,
)
from renslate.core.project import WORK_DIR_NAME, Project
from renslate.core.textio import TextFile, read_text_file, write_atomic

# Re-exports from renslate.engine
from renslate.engine.analyze import (
    bootstrap_characters_with_llm,
    extract_character_definitions,
    extract_scene_macro_contexts,
    infer_character_gender_nlp,
    infer_character_role_nlp,
    run_analysis,
    segment_scenes,
)
from renslate.engine.context import (
    QUOTED_RULES,
    ContextBuilder,
    ContextBundle,
    LookbackItem,
    estimate_tokens,
)
from renslate.engine.glossary import (
    Glossary,
    clean_glossary,
    is_valid_glossary_term,
    mine_glossary_candidates,
    translate_glossary_with_llm,
)
from renslate.engine.graph import RelationshipGraph
from renslate.engine.llm import LLMClient, extract_json_object
from renslate.engine.qa import (
    QAChecker,
    clean_translation,
    repair_glossary_leakage,
)
from renslate.engine.tm import FuzzyMatch, TranslationMemory, normalize_source
from renslate.engine.translate import (
    ProgressCallback,
    TranslationPipeline,
    TranslationStats,
    partition_batches,
)
from renslate.engine.tsvio import export_tsv, import_tsv

# Re-exports from renslate.renpy
from renslate.renpy.builtin_ui import (
    BUILTIN_UI_SEEDS,
    get_builtin_ui_translation,
)
from renslate.renpy.extract import (
    FLAG_MARKUP_INVALID,
    FLAG_SOURCE_CHANGED,
    ExtractReport,
    extract,
    extract_project,
    scan_project,
)
from renslate.renpy.literals import (
    decode_say_text,
    encode_say_text,
    parse_python_string,
    quote_python_text,
)
from renslate.renpy.mask import (
    CLOSE_MARK,
    OPEN_MARK,
    MaskedText,
    MaskToken,
    PlaceholderError,
    PlaceholderIssue,
    Token,
    TokenKind,
    auto_repair_placeholders,
    find_markup_error,
    mask,
    reconcile_translation,
    tokenize,
    unmask,
    validate,
    validate_raw,
)
from renslate.renpy.parser import (
    Issue,
    ParsedUnit,
    SayStatement,
    Severity,
    TranslationScanner,
    fingerprint,
    parse_say,
    parse_string_line,
    string_uid,
)
from renslate.renpy.sdk import (
    Runner,
    build_command,
    find_launcher,
    generate_translations,
)
from renslate.renpy.writer import (
    ApplyReport,
    apply_translations,
    setup_vietnamese_support,
)

# Re-exports from renslate.ui
from renslate.ui.dashboard import DashboardState, LiveDashboard
from renslate.ui.progress import format_duration, format_speed, render_progress_bar
from renslate.ui.runner import (
    check_server_health,
    detect_server_model,
    query_db_status,
    run_unattended_pipeline,
    wait_for_server,
)
from renslate.ui.tables import format_table
from renslate.ui.theme import (
    C_BLUE,
    C_BOLD,
    C_CYAN,
    C_DIM,
    C_GRAY,
    C_GREEN,
    C_MAGENTA,
    C_RED,
    C_RESET,
    C_YELLOW,
    IS_TTY,
    strip_ansi,
    visible_len,
)
from renslate.ui.wizard import run_interactive_wizard

__version__ = "0.1.0"

_LEGACY_MAP: dict[str, str] = {
    "renslate.models": "renslate.core.models",
    "renslate.errors": "renslate.core.errors",
    "renslate.config": "renslate.core.config",
    "renslate.db": "renslate.core.db",
    "renslate.project": "renslate.core.project",
    "renslate.textio": "renslate.core.textio",
    "renslate.literals": "renslate.renpy.literals",
    "renslate.parser": "renslate.renpy.parser",
    "renslate.mask": "renslate.renpy.mask",
    "renslate.extract": "renslate.renpy.extract",
    "renslate.writer": "renslate.renpy.writer",
    "renslate.sdk": "renslate.renpy.sdk",
    "renslate.builtin_ui": "renslate.renpy.builtin_ui",
    "renslate.translate": "renslate.engine.translate",
    "renslate.llm": "renslate.engine.llm",
    "renslate.context": "renslate.engine.context",
    "renslate.qa": "renslate.engine.qa",
    "renslate.glossary": "renslate.engine.glossary",
    "renslate.tm": "renslate.engine.tm",
    "renslate.analyze": "renslate.engine.analyze",
    "renslate.tsvio": "renslate.engine.tsvio",
}

_this_mod = sys.modules[__name__]
for legacy_name, target in _LEGACY_MAP.items():
    if target not in sys.modules:
        importlib.import_module(target)
    target_mod = sys.modules[target]
    sys.modules[legacy_name] = target_mod
    attr_name = legacy_name.split(".")[1]
    if attr_name != "mask":
        setattr(_this_mod, attr_name, target_mod)

__all__ = [
    # Built-in UI
    "BUILTIN_UI_SEEDS",
    "CLOSE_MARK",
    "C_BLUE",
    "C_BOLD",
    "C_CYAN",
    "C_DIM",
    "C_GRAY",
    "C_GREEN",
    "C_MAGENTA",
    "C_RED",
    "C_RESET",
    "C_YELLOW",
    "FLAG_MARKUP_INVALID",
    # Extract
    "FLAG_SOURCE_CHANGED",
    # UI
    "IS_TTY",
    "LOCKED_ORIGINS",
    "MIGRATIONS",
    # Mask
    "OPEN_MARK",
    "QUOTED_RULES",
    "WORK_DIR_NAME",
    "ApplyError",
    # Writer
    "ApplyReport",
    "ApplyTarget",
    "BatchConfig",
    "BatchLog",
    "CharacterProfile",
    # Config
    "Config",
    "ConfigError",
    # Context
    "ContextBuilder",
    "ContextBundle",
    "DashboardState",
    # Database
    "Database",
    "ExtractReport",
    "FuzzyMatch",
    # Glossary
    "Glossary",
    "GlossaryEntry",
    "Issue",
    # LLM
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "LiveDashboard",
    "LookbackItem",
    "MarkupError",
    "MaskToken",
    "MaskedText",
    "Origin",
    "ParsedUnit",
    "PlaceholderError",
    "PlaceholderIssue",
    "ProgressCallback",
    # Project
    "Project",
    "ProjectError",
    # QA
    "QAChecker",
    "QAConfig",
    "QAError",
    "QAIssue",
    "RelationshipEdge",
    "RelationshipGraph",
    "RelationshipKind",
    # Errors
    "RenslateError",
    # SDK
    "Runner",
    "SayStatement",
    "SceneMacroContext",
    "SceneSummary",
    "SdkError",
    # Parser
    "Severity",
    "Slot",
    "Status",
    "StyleConfig",
    "TMConfig",
    "TMEntry",
    # Text I/O
    "TextFile",
    "Token",
    "TokenKind",
    "TranslationError",
    # TM
    "TranslationMemory",
    # Translation pipeline
    "TranslationPipeline",
    "TranslationScanner",
    "TranslationStats",
    "Unit",
    # Models
    "UnitKind",
    "__version__",
    "apply_translations",
    "auto_repair_placeholders",
    "bootstrap_characters_with_llm",
    "build_command",
    "check_server_health",
    "clean_glossary",
    "clean_translation",
    # Subpackages
    "core",
    # Literals
    "decode_say_text",
    "detect_server_model",
    "encode_say_text",
    "engine",
    "estimate_tokens",
    # TSV I/O
    "export_tsv",
    "extract",
    "extract_character_definitions",
    "extract_json_object",
    "extract_project",
    "extract_scene_macro_contexts",
    "find_launcher",
    "find_markup_error",
    "fingerprint",
    "format_duration",
    "format_speed",
    "format_table",
    "generate_translations",
    "get_builtin_ui_translation",
    "import_tsv",
    "infer_character_gender_nlp",
    "infer_character_role_nlp",
    "is_valid_glossary_term",
    "mask",
    "mine_glossary_candidates",
    "normalize_source",
    "parse_python_string",
    "parse_say",
    "parse_string_line",
    "partition_batches",
    "query_db_status",
    "quote_python_text",
    "read_text_file",
    "reconcile_translation",
    "render_progress_bar",
    "renpy",
    "repair_glossary_leakage",
    # Analysis
    "run_analysis",
    "run_interactive_wizard",
    "run_unattended_pipeline",
    "scan_project",
    "segment_scenes",
    "setup_vietnamese_support",
    "string_uid",
    "strip_ansi",
    "tokenize",
    "translate_glossary_with_llm",
    "ui",
    "unmask",
    "utc_timestamp",
    "validate",
    "validate_raw",
    "visible_len",
    "wait_for_server",
    "write_atomic",
]
