"""Core domain models, exceptions, configuration, SQLite database, and filesystem workspace."""

from __future__ import annotations

from renslate.core.config import (
    BatchConfig,
    Config,
    LLMConfig,
    QAConfig,
    StyleConfig,
    TMConfig,
)
from renslate.core.db import (
    MIGRATIONS,
    Database,
    load_characters_from_json,
    save_characters_to_json,
    sync_characters,
    utc_timestamp,
)
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

__all__ = [
    "LOCKED_ORIGINS",
    "MIGRATIONS",
    "WORK_DIR_NAME",
    "ApplyError",
    "ApplyTarget",
    "BatchConfig",
    "BatchLog",
    "CharacterProfile",
    # Config
    "Config",
    "ConfigError",
    # Database
    "Database",
    "GlossaryEntry",
    "LLMConfig",
    "LLMError",
    "MarkupError",
    "Origin",
    # Project
    "Project",
    "ProjectError",
    "QAConfig",
    "QAError",
    "QAIssue",
    "RelationshipEdge",
    "RelationshipKind",
    # Errors
    "RenslateError",
    "SceneMacroContext",
    "SceneSummary",
    "SdkError",
    "Slot",
    "Status",
    "StyleConfig",
    "TMConfig",
    "TMEntry",
    # Text I/O
    "TextFile",
    "TranslationError",
    "Unit",
    # Models
    "UnitKind",
    "load_characters_from_json",
    "read_text_file",
    "save_characters_to_json",
    "sync_characters",
    "utc_timestamp",
    "write_atomic",
]
