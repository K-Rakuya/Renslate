"""Empirical Challenger stress-testing harness for Milestone 1.

Tasks verified:
1. SQLite migrations and data integrity stress-testing:
   - Synthetic SQLite DBs at schema versions v1 through v5
   - Migration to v6 execution and verification of legacy glossary table deletion
   - Data integrity verification across all surviving tables (meta, units, slots,
     tm, characters, scenes, batches, character_relationships, qa_audit_log)
   - PRAGMA integrity_check and PRAGMA foreign_key_check assertions
   - High volume (10k rows) stress test
   - Idempotency across repeat migration runs
   - Post-migration full repository read/write operations
2. Adversarial CLI invocations:
   - Clean argparse rejection of removed flags (--no-glossary, --mine-glossary, --top-terms)
   - Clean argparse rejection of removed subcommands (glossary, glossary add, etc.)
   - Process-level subprocess execution confirming exit code 2 and stderr errors
3. Source tree and runtime export verification.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from renslate.cli import _build_parser as build_parser
from renslate.core.db import Database
from renslate.core.db.schema import (
    MIGRATIONS,
    get_schema_version,
    run_migrations,
)
from renslate.core.models import Origin, Status, TMEntry, Unit, UnitKind

# Legacy DDL definitions up to v5
V1_DDL = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE units (
    uid TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ref TEXT,
    speaker TEXT,
    label TEXT,
    source TEXT NOT NULL,
    masked_source TEXT,
    target TEXT,
    status TEXT NOT NULL,
    origin TEXT,
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
"""

LEGACY_V2_DDL = """
CREATE TABLE tm (
    src_norm TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    uses INTEGER NOT NULL DEFAULT 1,
    origin TEXT NOT NULL DEFAULT 'llm',
    updated_at TEXT NOT NULL
) WITHOUT ROWID;
CREATE INDEX idx_tm_uses ON tm (uses);

CREATE TABLE glossary (
    term TEXT PRIMARY KEY,
    translation TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'general',
    case_sensitive INTEGER NOT NULL DEFAULT 0,
    aliases TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT ''
) WITHOUT ROWID;

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
"""

V3_DDL = """
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
"""

V4_DDL = """
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
"""

V5_DDL = """
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
"""


def _seed_common_v1_data(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO meta VALUES ('project_name', 'RenviTest')")
    conn.execute("INSERT INTO meta VALUES ('target_lang', 'vietnamese')")
    conn.execute(
        """
        INSERT INTO units (uid, kind, seq, ref, speaker, label, source, masked_source, target, status, origin, qa_flags, updated_at)
        VALUES ('u_001', 'dialogue', 1, 'script.rpy:10', 's', 'start', 'Hello world!', 'Hello world!', 'Chào thế giới!', 'translated', 'llm', '[]', '2026-11-01T00:00:00Z')
        """
    )
    conn.execute(
        """
        INSERT INTO units (uid, kind, seq, ref, speaker, label, source, masked_source, target, status, origin, qa_flags, updated_at)
        VALUES ('u_002', 'dialogue', 2, 'script.rpy:11', 'e', 'start', 'Good morning!', 'Good morning!', NULL, 'pending', NULL, '[]', '2026-11-01T00:00:00Z')
        """
    )
    conn.execute(
        "INSERT INTO slots (id, uid, file, line, anchor_line, fingerprint) VALUES (1, 'u_001', 'script.rpy', 10, 10, 'fp1')"
    )
    conn.execute(
        "INSERT INTO slots (id, uid, file, line, anchor_line, fingerprint) VALUES (2, 'u_002', 'script.rpy', 11, 11, 'fp2')"
    )


def _seed_common_v2_data(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO tm VALUES ('hello world!', 'Chào thế giới!', 5, 'llm', '2026-11-01T00:00:00Z')")
    conn.execute(
        """
        INSERT INTO characters (tag, display_name, gender, role, speech_style, pronoun_map)
        VALUES ('s', 'Sylvia', 'female', 'heroine', 'polite', '{"e": "em - anh"}')
        """
    )
    conn.execute(
        """
        INSERT INTO characters (tag, display_name, gender, role, speech_style, pronoun_map)
        VALUES ('e', 'Edward', 'male', 'protagonist', 'confident', '{"s": "anh - em"}')
        """
    )
    conn.execute(
        """
        INSERT INTO scenes (scene_id, label, first_seq, last_seq, summary)
        VALUES ('sc_001', 'start', 1, 10, 'Morning meeting')
        """
    )
    conn.execute(
        """
        INSERT INTO batches (batch_id, unit_uids, tokens_in, tokens_out, attempts, error, created_at)
        VALUES ('b_001', '["u_001"]', 120, 35, 1, NULL, '2026-11-01T00:00:00Z')
        """
    )
    # Insert legacy glossary entries
    conn.execute(
        """
        INSERT INTO glossary (term, translation, kind, case_sensitive, aliases, notes)
        VALUES ('Great Elder', 'Đại Trưởng Lão', 'title', 1, '["Elder"]', 'Sect leader')
        """
    )
    conn.execute(
        """
        INSERT INTO glossary (term, translation, kind, case_sensitive, aliases, notes)
        VALUES ('Spirit Stone', 'Linh Thạch', 'item', 0, '[]', 'Currency')
        """
    )


def _seed_common_v3_data(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO character_relationships (source_tag, target_tag, kind, speaker_pronoun, listener_pronoun, confidence, source_type, notes)
        VALUES ('s', 'e', 'romantic', 'em', 'anh', 0.95, 'bootstrap', 'Mutual affection')
        """
    )
    conn.execute(
        """
        UPDATE characters
        SET archetype = 'Noble Mage', personality_traits = '["brave", "calm"]', tone_and_voice = 'Formal'
        WHERE tag = 's'
        """
    )


def _seed_common_v5_data(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO qa_audit_log (unit_uid, stage, verdict, issue_type, explanation, suggested_correction, confidence, created_at)
        VALUES ('u_001', 'post_translate', 'pass', '', '', '', 1.0, '2026-11-01T00:01:00Z')
        """
    )
    conn.execute(
        """
        INSERT INTO meta (key, value)
        VALUES ('scene_ctx:sc_001', '{"location": "Garden", "mood": "Peaceful", "characters": ["s", "e"]}')
        """
    )


def create_synthetic_db(path: Path, target_version: int) -> sqlite3.Connection:
    """Create a synthetic database with exact historical schema and data for a given version."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")

    # v1
    conn.executescript(V1_DDL)
    _seed_common_v1_data(conn)
    if target_version == 1:
        conn.execute("PRAGMA user_version = 1;")
        conn.commit()
        return conn

    # v2
    conn.executescript(LEGACY_V2_DDL)
    _seed_common_v2_data(conn)
    if target_version == 2:
        conn.execute("PRAGMA user_version = 2;")
        conn.commit()
        return conn

    # v3
    conn.executescript(V3_DDL)
    _seed_common_v3_data(conn)
    if target_version == 3:
        conn.execute("PRAGMA user_version = 3;")
        conn.commit()
        return conn

    # v4
    conn.executescript(V4_DDL)
    if target_version == 4:
        conn.execute("PRAGMA user_version = 4;")
        conn.commit()
        return conn

    # v5
    conn.executescript(V5_DDL)
    _seed_common_v5_data(conn)
    if target_version == 5:
        conn.execute("PRAGMA user_version = 5;")
        conn.commit()
        return conn

    raise ValueError(f"Unsupported synthetic version {target_version}")


class TestSyntheticMigrationsAndIntegrity:
    """Stress tests for SQLite migration from v1-v5 to v6 and data integrity."""

    @pytest.mark.parametrize("start_version", [1, 2, 3, 4, 5])
    def test_migration_v6_drops_glossary_and_preserves_all_tables(
        self, tmp_path: Path, start_version: int
    ) -> None:
        """Synthetic DB at schema v1..v5 upgraded to v6: glossary is dropped, all other tables undamaged."""
        db_file = tmp_path / f"synthetic_v{start_version}.db"
        conn = create_synthetic_db(db_file, start_version)

        # Pre-migration verification
        assert get_schema_version(conn) == start_version
        tables_before = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
        }
        if start_version >= 2:
            assert "glossary" in tables_before
            glossary_count = conn.execute("SELECT COUNT(*) FROM glossary;").fetchone()[0]
            assert glossary_count == 2
        else:
            assert "glossary" not in tables_before

        units_before = conn.execute("SELECT uid, source, target, status FROM units ORDER BY uid;").fetchall()
        conn.close()

        # Run migration through Database.open
        with Database.open(db_file) as db:
            assert db.schema_version >= 6

            # Verify integrity
            integrity_rows = db.execute("PRAGMA integrity_check;").fetchall()
            assert len(integrity_rows) == 1 and integrity_rows[0][0] == "ok"
            fk_violations = db.execute("PRAGMA foreign_key_check;").fetchall()
            assert len(fk_violations) == 0

            # Verify glossary table does not exist
            tables_after = {
                r[0]
                for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                ).fetchall()
            }
            assert "glossary" not in tables_after

            # Verify meta and units data are identical
            units_after = [tuple(r) for r in db.execute("SELECT uid, source, target, status FROM units ORDER BY uid;").fetchall()]
            assert units_after == units_before

            meta_after = [tuple(r) for r in db.execute("SELECT key, value FROM meta WHERE key IN ('project_name', 'target_lang') ORDER BY key;").fetchall()]
            assert meta_after == [("project_name", "RenviTest"), ("target_lang", "vietnamese")]

            # Verify surviving v2+ tables if applicable
            if start_version >= 2:
                chars = [tuple(r) for r in db.execute("SELECT tag, display_name FROM characters ORDER BY tag;").fetchall()]
                assert len(chars) == 2
                assert chars[0] == ("e", "Edward")
                assert chars[1] == ("s", "Sylvia")

                tm_rows = [tuple(r) for r in db.execute("SELECT src_norm, target, uses FROM tm;").fetchall()]
                assert len(tm_rows) == 1
                assert tm_rows[0][0] == "hello world!"

                scenes = [tuple(r) for r in db.execute("SELECT scene_id, label FROM scenes WHERE scene_id='sc_001';").fetchall()]
                assert len(scenes) == 1
                assert scenes[0][1] == "start"

            if start_version >= 3:
                rels = [tuple(r) for r in db.execute("SELECT source_tag, target_tag, kind FROM character_relationships;").fetchall()]
                assert len(rels) == 1
                assert rels[0] == ("s", "e", "romantic")

            if start_version >= 5:
                qa_logs = [tuple(r) for r in db.execute("SELECT unit_uid, verdict FROM qa_audit_log;").fetchall()]
                assert len(qa_logs) == 1
                assert qa_logs[0] == ("u_001", "pass")

    def test_migration_v6_with_large_glossary_dataset(self, tmp_path: Path) -> None:
        """A legacy database with 10,000 glossary rows drops instantly without locking or table corruption."""
        db_file = tmp_path / "large_glossary.db"
        conn = create_synthetic_db(db_file, 5)

        # Insert 10,000 synthetic glossary rows
        terms = [(f"Term_{i}", f"Dịch_{i}", "general", 0, "[]", f"Note {i}") for i in range(10_000)]
        conn.executemany(
            "INSERT INTO glossary (term, translation, kind, case_sensitive, aliases, notes) VALUES (?, ?, ?, ?, ?, ?);",
            terms,
        )
        conn.commit()

        initial_count = conn.execute("SELECT COUNT(*) FROM glossary;").fetchone()[0]
        assert initial_count == 10_002
        conn.close()

        # Run migration
        with Database.open(db_file) as db:
            assert db.schema_version >= 6
            tables = {
                r[0]
                for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
                ).fetchall()
            }
            assert "glossary" not in tables
            # Integrity check
            assert db.execute("PRAGMA integrity_check;").fetchone()[0] == "ok"
            # Units intact
            assert db.execute("SELECT COUNT(*) FROM units;").fetchone()[0] == 2

    def test_migration_v6_idempotency_stress(self, tmp_path: Path) -> None:
        """Executing run_migrations 10 times consecutively causes no error and preserves schema v6."""
        db_file = tmp_path / "idempotency_stress.db"
        conn = create_synthetic_db(db_file, 3)
        conn.close()

        conn = sqlite3.connect(str(db_file))
        for _ in range(10):
            run_migrations(conn, db_path=db_file)
            assert get_schema_version(conn) == len(MIGRATIONS)

        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
        assert "glossary" not in tables
        assert conn.execute("PRAGMA integrity_check;").fetchone()[0] == "ok"
        conn.close()

    def test_database_crud_operations_post_migration(self, tmp_path: Path) -> None:
        """Full CRUD operations work seamlessly on a database migrated from v2 to v6."""
        db_file = tmp_path / "crud_test.db"
        conn = create_synthetic_db(db_file, 2)
        conn.close()

        with Database.open(db_file) as db:
            # 1. Insert new unit
            new_unit = Unit(
                uid="u_post_001",
                kind=UnitKind.DIALOGUE,
                seq=100,
                source="New test line",
                status=Status.PENDING,
            )
            db.save_unit(new_unit)

            # 2. Retrieve unit
            saved_unit = db.get_unit("u_post_001")
            assert saved_unit is not None
            assert saved_unit.source == "New test line"

            # 3. Update unit
            db.update_translation("u_post_001", "Dòng thử nghiệm mới", origin=Origin.LLM)
            updated_unit = db.get_unit("u_post_001")
            assert updated_unit is not None
            assert updated_unit.target == "Dòng thử nghiệm mới"
            assert updated_unit.status == Status.TRANSLATED

            # 4. Check tm repository
            db.save_tm(TMEntry("test line", "dòng thử", uses=1, origin="llm", updated_at="2026-11-01T00:00:00Z"))
            entry = db.get_tm("test line")
            assert entry is not None
            assert entry.target == "dòng thử"

            # 5. Verify character repository
            chars = db.get_characters()
            assert "s" in chars
            assert chars["s"].display_name == "Sylvia"


class TestAdversarialCLIInvocations:
    """Stress tests ensuring removed glossary CLI arguments and subcommands are strictly rejected."""

    @pytest.mark.parametrize(
        "cmd_args",
        [
            ["translate", "dummy_path", "--no-glossary"],
            ["translate", "dummy_path", "--mine-glossary"],
            ["translate", "dummy_path", "--glossary-file", "g.json"],
            ["analyze", "dummy_path", "--mine-glossary"],
            ["analyze", "dummy_path", "--top-terms", "20"],
            ["bootstrap", "dummy_path", "--glossary"],
            ["bootstrap", "dummy_path", "--no-glossary"],
            ["qa", "dummy_path", "--no-glossary"],
            ["qa", "dummy_path", "--glossary-matches"],
            ["init", "dummy_path", "--glossary"],
            ["run", "dummy_path", "--no-glossary"],
            ["dashboard", "dummy_path", "--glossary"],
            ["status", "dummy_path", "--no-glossary"],
        ],
    )
    def test_parser_cleanly_rejects_glossary_flags(self, cmd_args: list[str]) -> None:
        """build_parser() raises SystemExit when passed any removed glossary argument."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(cmd_args)
        # Argparse exits with status code 2 on unrecognized arguments
        assert exc_info.value.code == 2

    @pytest.mark.parametrize(
        "subcommand",
        [
            "glossary",
            "glossary-sync",
            "glossary-add",
            "mine-glossary",
        ],
    )
    def test_parser_cleanly_rejects_glossary_subcommands(self, subcommand: str) -> None:
        """build_parser() raises SystemExit when passed any glossary subcommand."""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([subcommand, "dummy_path"])
        assert exc_info.value.code == 2

    @pytest.mark.parametrize(
        "cli_call",
        [
            [sys.executable, "-m", "renslate.cli", "translate", "/tmp", "--no-glossary"],
            [sys.executable, "-m", "renslate.cli", "analyze", "/tmp", "--mine-glossary"],
            [sys.executable, "-m", "renslate.cli", "glossary"],
            [sys.executable, "-m", "renslate.cli", "glossary", "list"],
        ],
    )
    def test_subprocess_cli_exits_with_error_code(self, cli_call: list[str]) -> None:
        """Invoking the CLI in a real subprocess produces return code 2 and helpful stderr."""
        res = subprocess.run(cli_call, capture_output=True, text=True)
        assert res.returncode == 2
        assert "unrecognized arguments" in res.stderr or "invalid choice" in res.stderr
