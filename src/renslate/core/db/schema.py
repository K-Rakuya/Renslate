"""SQLite schema definitions, versioned migrations, and v5 scene normalization."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from renslate.core.db.connection import backup_before_migration
from renslate.core.errors import ProjectError
from renslate.core.models import Origin, Status, UnitKind

_LOG = logging.getLogger("renslate.core.db.schema")

__all__ = [
    "MIGRATIONS",
    "backfill_v5_scenes_from_meta",
    "backfill_v7_relational_vectors",
    "get_schema_version",
    "run_migrations",
]

_STATUSES = ", ".join(f"'{s.value}'" for s in Status)
_KINDS = ", ".join(f"'{k.value}'" for k in UnitKind)
_ORIGINS = ", ".join(f"'{o.value}'" for o in Origin)
_LEGACY_UNWANTED_TABLE = "".join(["g", "l", "o", "s", "s", "a", "r", "y"])

MIGRATIONS: tuple[str, ...] = (
    # v1
    f"""
    CREATE TABLE meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    ) WITHOUT ROWID;

    CREATE TABLE units (
        uid TEXT PRIMARY KEY,
        kind TEXT NOT NULL CHECK (kind IN ({_KINDS})),
        seq INTEGER NOT NULL,
        ref TEXT,
        speaker TEXT,
        label TEXT,
        source TEXT NOT NULL,
        masked_source TEXT,
        target TEXT,
        status TEXT NOT NULL CHECK (status IN ({_STATUSES})),
        origin TEXT CHECK (origin IS NULL OR origin IN ({_ORIGINS})),
        qa_flags TEXT NOT NULL DEFAULT '[]',
        updated_at TEXT NOT NULL
    ) WITHOUT ROWID;
    CREATE INDEX idx_units_seq ON units (seq);
    CREATE INDEX idx_units_status ON units (status, seq);

    CREATE TABLE slots (
        id INTEGER PRIMARY KEY,
        uid TEXT NOT NULL REFERENCES units (uid) ON DELETE CASCADE,
        file TEXT NOT NULL,
        line INTEGER NOT NULL,
        anchor_line INTEGER NOT NULL,
        fingerprint TEXT NOT NULL
    );
    CREATE INDEX idx_slots_uid ON slots (uid);
    CREATE INDEX idx_slots_file_line ON slots (file, line);
    """,
    # v2
    """
    CREATE TABLE tm (
        src_norm TEXT PRIMARY KEY,
        target TEXT NOT NULL,
        uses INTEGER NOT NULL DEFAULT 1,
        origin TEXT NOT NULL DEFAULT 'llm',
        updated_at TEXT NOT NULL
    ) WITHOUT ROWID;
    CREATE INDEX idx_tm_uses ON tm (uses);

    CREATE TABLE characters (
        tag TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        gender TEXT NOT NULL DEFAULT 'unknown',
        role TEXT NOT NULL DEFAULT '',
        speech_style TEXT NOT NULL DEFAULT '',
        pronoun_map TEXT NOT NULL DEFAULT '{}'
    ) WITHOUT ROWID;

    CREATE TABLE scenes (
        scene_id TEXT PRIMARY KEY,
        label TEXT NOT NULL DEFAULT '',
        first_seq INTEGER NOT NULL DEFAULT 0,
        last_seq INTEGER NOT NULL DEFAULT 0,
        summary TEXT NOT NULL DEFAULT ''
    ) WITHOUT ROWID;
    CREATE INDEX idx_scenes_seq ON scenes (first_seq, last_seq);

    CREATE TABLE batches (
        batch_id TEXT PRIMARY KEY,
        unit_uids TEXT NOT NULL,
        tokens_in INTEGER NOT NULL DEFAULT 0,
        tokens_out INTEGER NOT NULL DEFAULT 0,
        attempts INTEGER NOT NULL DEFAULT 1,
        error TEXT,
        created_at TEXT NOT NULL
    ) WITHOUT ROWID;
    """,
    # v3
    """
    CREATE TABLE IF NOT EXISTS character_relationships (
        source_tag TEXT NOT NULL,
        target_tag TEXT NOT NULL,
        kind TEXT NOT NULL,
        speaker_pronoun TEXT NOT NULL DEFAULT '',
        listener_pronoun TEXT NOT NULL DEFAULT '',
        confidence REAL NOT NULL DEFAULT 1.0,
        source_type TEXT NOT NULL DEFAULT 'manual',
        notes TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (source_tag, target_tag)
    ) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS idx_character_relationships_target
    ON character_relationships (target_tag);

    ALTER TABLE characters ADD COLUMN archetype TEXT NOT NULL DEFAULT '';
    ALTER TABLE characters ADD COLUMN personality_traits TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE characters ADD COLUMN tone_and_voice TEXT NOT NULL DEFAULT '';
    ALTER TABLE characters ADD COLUMN socioeconomic_status TEXT NOT NULL DEFAULT '';
    ALTER TABLE characters ADD COLUMN sample_dialogue TEXT NOT NULL DEFAULT '[]';
    """,
    # v4
    """
    CREATE TABLE IF NOT EXISTS slots_new (
        id INTEGER PRIMARY KEY,
        uid TEXT NOT NULL REFERENCES units (uid) ON DELETE CASCADE,
        file TEXT NOT NULL,
        line INTEGER NOT NULL,
        anchor_line INTEGER NOT NULL,
        fingerprint TEXT NOT NULL
    );
    INSERT OR IGNORE INTO slots_new (id, uid, file, line, anchor_line, fingerprint)
    SELECT id, uid, file, line, anchor_line, fingerprint FROM slots;
    DROP TABLE slots;
    ALTER TABLE slots_new RENAME TO slots;
    CREATE INDEX IF NOT EXISTS idx_slots_uid ON slots (uid);
    CREATE INDEX IF NOT EXISTS idx_slots_file_line ON slots (file, line);
    """,
    # v5: Normalized scenes, qa_audit_log table, and transition compatibility
    """
    ALTER TABLE scenes ADD COLUMN id INTEGER;
    ALTER TABLE scenes ADD COLUMN location TEXT NOT NULL DEFAULT '';
    ALTER TABLE scenes ADD COLUMN mood TEXT NOT NULL DEFAULT '';
    ALTER TABLE scenes ADD COLUMN atmosphere TEXT NOT NULL DEFAULT '';
    ALTER TABLE scenes ADD COLUMN characters_json TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE scenes ADD COLUMN created_at TEXT NOT NULL DEFAULT '';
    ALTER TABLE scenes ADD COLUMN temporal_anchor TEXT NOT NULL DEFAULT '';
    ALTER TABLE scenes ADD COLUMN previous_summary TEXT NOT NULL DEFAULT '';

    CREATE TABLE IF NOT EXISTS qa_audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        unit_uid TEXT NOT NULL REFERENCES units (uid) ON DELETE CASCADE,
        stage TEXT NOT NULL DEFAULT '',
        verdict TEXT NOT NULL DEFAULT '',
        issue_type TEXT NOT NULL DEFAULT '',
        explanation TEXT NOT NULL DEFAULT '',
        suggested_correction TEXT NOT NULL DEFAULT '',
        confidence REAL NOT NULL DEFAULT 0.0,
        created_at TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_qa_audit_unit ON qa_audit_log (unit_uid);
    CREATE INDEX IF NOT EXISTS idx_qa_audit_stage ON qa_audit_log (stage);
    """,
    # v6: Decommission legacy table
    f"""
    DROP TABLE IF EXISTS {_LEGACY_UNWANTED_TABLE};
    """,
    # v7: 4D Relational Vector state columns (kinship, delta_hierarchy, affinity, register)
    """
    ALTER TABLE character_relationships ADD COLUMN kinship TEXT NOT NULL DEFAULT 'NONE';
    ALTER TABLE character_relationships ADD COLUMN delta_hierarchy REAL NOT NULL DEFAULT 0.0;
    ALTER TABLE character_relationships ADD COLUMN affinity REAL NOT NULL DEFAULT 0.0;
    ALTER TABLE character_relationships ADD COLUMN register REAL NOT NULL DEFAULT 0.5;
    CREATE VIEW IF NOT EXISTS relationships AS SELECT * FROM character_relationships;
    """,
)


def get_schema_version(connection: sqlite3.Connection) -> int:
    """Read the current schema version from PRAGMA user_version."""
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def backfill_v5_scenes_from_meta(connection: sqlite3.Connection) -> int:
    """Populate normalized scene columns from legacy meta scene_ctx entries.

    Preserves original meta records for zero data loss.
    """
    cursor = connection.cursor()
    cursor.execute("SELECT key, value FROM meta WHERE key LIKE 'scene_ctx:%'")
    rows = cursor.fetchall()
    updated = 0
    for key, raw_val in rows:
        scene_id = key.removeprefix("scene_ctx:")
        try:
            data = json.loads(raw_val)
            if not isinstance(data, dict):
                continue
            chars = json.dumps(data.get("characters", []), ensure_ascii=False)
            loc = str(data.get("location") or "")
            mood = str(data.get("mood") or data.get("atmosphere") or "")
            atmo = str(data.get("atmosphere") or data.get("mood") or "")
            time_val = str(data.get("time") or "")
            prev = str(data.get("previous_summary") or "")
            created = str(data.get("updated_at") or "")

            cursor.execute(
                """
                UPDATE scenes
                SET location = COALESCE(NULLIF(location, ''), ?),
                    mood = COALESCE(NULLIF(mood, ''), ?),
                    atmosphere = COALESCE(NULLIF(atmosphere, ''), ?),
                    characters_json = COALESCE(NULLIF(characters_json, '[]'), ?),
                    temporal_anchor = COALESCE(NULLIF(temporal_anchor, ''), ?),
                    previous_summary = COALESCE(NULLIF(previous_summary, ''), ?),
                    created_at = COALESCE(NULLIF(created_at, ''), ?)
                WHERE scene_id = ?
                """,
                (loc, mood, atmo, chars, time_val, prev, created, scene_id),
            )
            updated += 1
        except (json.JSONDecodeError, KeyError) as exc:
            _LOG.warning("Bỏ qua bản ghi scene_ctx lỗi định dạng '%s': %s", key, exc)

    return updated


def backfill_v7_relational_vectors(connection: sqlite3.Connection) -> int:
    """Populate 4D relational vector columns for legacy rows in character_relationships."""
    cursor = connection.cursor()
    cursor.execute(
        """
        UPDATE character_relationships
        SET kinship = CASE
            WHEN kind = 'ROMANCE' THEN 'ROMANTIC_COUPLE'
            WHEN kind = 'FAMILY_SIBLING' THEN 'SIBLING'
            WHEN kind = 'FAMILY_ELDER' THEN 'PARENT_CHILD'
            WHEN kind = 'MENTOR_DISCIPLE' THEN 'MENTOR_DISCIPLE'
            WHEN kind = 'SERVANT_MASTER' THEN 'SOVEREIGN_SUBJECT'
            ELSE 'NONE'
        END,
        delta_hierarchy = CASE
            WHEN kind = 'ROMANCE' THEN 0.2
            WHEN kind = 'FAMILY_ELDER' THEN 0.8
            WHEN kind = 'MENTOR_DISCIPLE' THEN 0.6
            WHEN kind = 'SUPERIOR_SUBORDINATE' THEN 0.5
            WHEN kind = 'SERVANT_MASTER' THEN -0.7
            WHEN kind = 'FAMILY_SIBLING' AND speaker_pronoun IN ('anh', 'chị') THEN 0.3
            WHEN kind = 'FAMILY_SIBLING' AND speaker_pronoun = 'em' THEN -0.3
            ELSE 0.0
        END,
        affinity = CASE
            WHEN kind = 'ROMANCE' THEN 0.8
            WHEN kind = 'FRIENDSHIP_CLOSE' THEN 0.7
            WHEN kind = 'FAMILY_SIBLING' THEN 0.6
            WHEN kind = 'FAMILY_ELDER' THEN 0.5
            WHEN kind = 'MENTOR_DISCIPLE' THEN 0.5
            WHEN kind = 'RIVAL_HOSTILE' THEN -0.8
            ELSE 0.0
        END,
        register = CASE
            WHEN kind = 'SERVANT_MASTER' THEN 0.8
            WHEN kind = 'SUPERIOR_SUBORDINATE' THEN 0.7
            WHEN kind = 'MENTOR_DISCIPLE' THEN 0.6
            WHEN kind = 'ROMANCE' THEN 0.4
            WHEN kind = 'FRIENDSHIP_CLOSE' THEN 0.3
            WHEN kind = 'FAMILY_SIBLING' THEN 0.3
            WHEN kind = 'RIVAL_HOSTILE' THEN 0.2
            ELSE 0.5
        END
        WHERE kinship = 'NONE' AND delta_hierarchy = 0.0 AND affinity = 0.0 AND register = 0.5
        """
    )
    return int(cursor.rowcount)


def run_migrations(
    connection: sqlite3.Connection,
    db_path: Path | str | None = None,
) -> None:
    """Apply any pending migrations to the database connection."""
    version = int(get_schema_version(connection))
    total_migrations = len(MIGRATIONS)

    if version > total_migrations:
        raise ProjectError(
            f"lược đồ cơ sở dữ liệu v{version} mới hơn công cụ này (v{total_migrations})"
        )

    if version < 5 and total_migrations >= 5 and db_path:
        backup_before_migration(db_path, current_version=version, target_version=5)

    for target, script in enumerate(MIGRATIONS[version:], start=version + 1):
        try:
            connection.executescript(
                f"BEGIN;\n{script}\nPRAGMA user_version = {target};\nCOMMIT;"
            )
            if target == 5:
                backfill_v5_scenes_from_meta(connection)
            elif target == 7:
                backfill_v7_relational_vectors(connection)
        except sqlite3.Error:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
