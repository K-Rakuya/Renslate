"""SQLite connection factory with WAL mode, foreign keys, and pre-migration backup."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

_LOG = logging.getLogger("renslate.core.db.connection")

__all__ = [
    "backup_before_migration",
    "configure_connection",
    "create_connection",
]


def configure_connection(connection: sqlite3.Connection) -> None:
    """Configure SQLite pragmas for thread-safety, integrity, and performance."""
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -64000")
    connection.execute("PRAGMA mmap_size = 268435456")


def backup_before_migration(
    path: Path | str, current_version: int, target_version: int = 5
) -> Path | None:
    """Create an automated pre-migration backup snapshot if migrating to v5."""
    if current_version >= target_version:
        return None

    db_path = Path(path) if isinstance(path, str) else path
    if str(db_path) == ":memory:" or not db_path.is_file():
        return None

    bak_path = Path(f"{db_path}.v4.bak")
    try:
        shutil.copy2(db_path, bak_path)
        _LOG.info("Tạo bản sao lưu tiền di trú tại %s", bak_path)
        return bak_path
    except OSError as exc:
        _LOG.warning("Không thể tạo bản sao lưu tiền di trú: %s", exc)
        return None


def create_connection(path: Path | str, timeout: float = 30.0) -> sqlite3.Connection:
    """Create and configure a thread-safe SQLite connection."""
    db_path = Path(path) if isinstance(path, str) else path
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(db_path),
        timeout=timeout,
        isolation_level=None,
        check_same_thread=False,
    )
    configure_connection(connection)
    return connection
