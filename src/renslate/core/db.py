"""Backward compatibility shim forwarding to renslate.core.db package."""

from __future__ import annotations

from renslate.core.db import (
    MIGRATIONS,
    Database,
    backup_before_migration,
    configure_connection,
    create_connection,
    get_schema_version,
    load_characters_from_json,
    run_migrations,
    save_characters_to_json,
    sync_characters,
    utc_timestamp,
)

__all__ = [
    "MIGRATIONS",
    "Database",
    "backup_before_migration",
    "configure_connection",
    "create_connection",
    "get_schema_version",
    "load_characters_from_json",
    "run_migrations",
    "save_characters_to_json",
    "sync_characters",
    "utc_timestamp",
]
