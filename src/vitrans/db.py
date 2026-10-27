"""SQLite store: the single source of truth for units, their slots and project metadata."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from vitrans.errors import ProjectError
from vitrans.models import (
    ApplyTarget,
    BatchLog,
    CharacterProfile,
    GlossaryEntry,
    Origin,
    SceneSummary,
    Slot,
    Status,
    TMEntry,
    Unit,
    UnitKind,
)

_STATUSES = ", ".join(f"'{s.value}'" for s in Status)
_KINDS = ", ".join(f"'{k.value}'" for k in UnitKind)
_ORIGINS = ", ".join(f"'{o.value}'" for o in Origin)

MIGRATIONS: tuple[str, ...] = (
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
        fingerprint TEXT NOT NULL,
        UNIQUE (file, line)
    );
    CREATE INDEX idx_slots_uid ON slots (uid);
    """,
    """
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
    """,
)

_UNIT_COLUMNS = (
    "uid, kind, seq, ref, speaker, label, source, masked_source, target, status, origin, "
    "qa_flags, updated_at"
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unit_from_row(row: sqlite3.Row) -> Unit:
    origin = row["origin"]
    return Unit(
        uid=row["uid"],
        kind=UnitKind(row["kind"]),
        seq=row["seq"],
        ref=row["ref"],
        speaker=row["speaker"],
        label=row["label"],
        source=row["source"],
        masked_source=row["masked_source"],
        target=row["target"],
        status=Status(row["status"]),
        origin=Origin(origin) if origin is not None else None,
        qa_flags=tuple(json.loads(row["qa_flags"])),
        updated_at=row["updated_at"],
    )


class Database:
    """Thin wrapper over a SQLite connection exposing the operations the pipeline needs."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, path: Path, *, create: bool = False) -> Database:
        if not create and not path.is_file():
            raise ProjectError(f"không tìm thấy cơ sở dữ liệu: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA cache_size = -64000")
        connection.execute("PRAGMA mmap_size = 268435456")
        database = cls(connection)
        try:
            database._migrate()
        except BaseException:
            connection.close()
            raise
        return database

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    def _migrate(self) -> None:
        version = self.schema_version
        if version > len(MIGRATIONS):
            raise ProjectError(
                f"lược đồ cơ sở dữ liệu v{version} mới hơn công cụ này (v{len(MIGRATIONS)})"
            )
        for target, script in enumerate(MIGRATIONS[version:], start=version + 1):
            try:
                self._connection.executescript(
                    f"BEGIN;\n{script}\nPRAGMA user_version = {target};\nCOMMIT;"
                )
            except sqlite3.Error:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self._connection.execute(sql, params)

    def units(self) -> dict[str, Unit]:
        rows = self._connection.execute(f"SELECT {_UNIT_COLUMNS} FROM units ORDER BY seq")
        return {row["uid"]: _unit_from_row(row) for row in rows}

    def save_unit(self, unit: Unit) -> None:
        self._connection.execute(
            f"""
            INSERT INTO units ({_UNIT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (uid) DO UPDATE SET
                kind = excluded.kind, seq = excluded.seq, ref = excluded.ref,
                speaker = excluded.speaker, label = excluded.label, source = excluded.source,
                masked_source = excluded.masked_source, target = excluded.target,
                status = excluded.status, origin = excluded.origin,
                qa_flags = excluded.qa_flags, updated_at = excluded.updated_at
            """,
            (
                unit.uid,
                unit.kind.value,
                unit.seq,
                unit.ref,
                unit.speaker,
                unit.label,
                unit.source,
                unit.masked_source,
                unit.target,
                unit.status.value,
                unit.origin.value if unit.origin is not None else None,
                json.dumps(list(unit.qa_flags)),
                unit.updated_at,
            ),
        )

    def save_units_bulk(self, units: Sequence[Unit]) -> None:
        """Bulk upsert units in a single statement using executemany."""
        if not units:
            return
        self._connection.executemany(
            f"""
            INSERT INTO units ({_UNIT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (uid) DO UPDATE SET
                kind = excluded.kind, seq = excluded.seq, ref = excluded.ref,
                speaker = excluded.speaker, label = excluded.label, source = excluded.source,
                masked_source = excluded.masked_source, target = excluded.target,
                status = excluded.status, origin = excluded.origin,
                qa_flags = excluded.qa_flags, updated_at = excluded.updated_at
            """,
            [
                (
                    unit.uid,
                    unit.kind.value,
                    unit.seq,
                    unit.ref,
                    unit.speaker,
                    unit.label,
                    unit.source,
                    unit.masked_source,
                    unit.target,
                    unit.status.value,
                    unit.origin.value if unit.origin is not None else None,
                    json.dumps(list(unit.qa_flags)),
                    unit.updated_at,
                )
                for unit in units
            ],
        )

    def replace_slots(self, slots: Iterable[Slot]) -> None:
        self._connection.execute("DELETE FROM slots")
        self._connection.executemany(
            "INSERT INTO slots (uid, file, line, anchor_line, fingerprint) VALUES (?, ?, ?, ?, ?)",
            [(s.uid, s.file, s.line, s.anchor_line, s.fingerprint) for s in slots],
        )

    def mark_orphans_obsolete(self, timestamp: str) -> int:
        cursor = self._connection.execute(
            """
            UPDATE units SET status = 'obsolete', updated_at = ?
            WHERE status != 'obsolete' AND uid NOT IN (SELECT uid FROM slots)
            """,
            (timestamp,),
        )
        return cursor.rowcount

    def apply_targets(self) -> list[ApplyTarget]:
        rows = self._connection.execute(
            """
            SELECT s.uid, s.file, s.line, s.anchor_line, s.fingerprint, u.kind, u.target
            FROM slots s JOIN units u ON u.uid = s.uid
            WHERE u.target IS NOT NULL AND u.status != 'obsolete'
            ORDER BY s.file, s.line
            """
        )
        return [
            ApplyTarget(
                Slot(r["uid"], r["file"], r["line"], r["anchor_line"], r["fingerprint"]),
                UnitKind(r["kind"]),
                r["target"],
            )
            for r in rows
        ]

    def update_fingerprints(self, updates: Iterable[tuple[str, int, str]]) -> None:
        self._connection.executemany(
            "UPDATE slots SET fingerprint = ? WHERE file = ? AND line = ?",
            [(fingerprint, file, line) for file, line, fingerprint in updates],
        )

    def count_slots(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM slots").fetchone()
        return int(row[0])

    def stats(self) -> list[tuple[UnitKind, Status, int]]:
        rows = self._connection.execute(
            "SELECT kind, status, COUNT(*) AS total FROM units GROUP BY kind, status "
            "ORDER BY kind, status"
        )
        return [(UnitKind(r["kind"]), Status(r["status"]), int(r["total"])) for r in rows]

    def get_meta(self, key: str) -> str | None:
        row = self._connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_unit(self, uid: str) -> Unit | None:
        row = self._connection.execute(
            f"SELECT {_UNIT_COLUMNS} FROM units WHERE uid = ?", (uid,)
        ).fetchone()
        return None if row is None else _unit_from_row(row)

    def all_units(self) -> list[Unit]:
        rows = self._connection.execute(f"SELECT {_UNIT_COLUMNS} FROM units ORDER BY seq")
        return [_unit_from_row(row) for row in rows]

    def pending_units(
        self, limit: int | None = None, *, include_failed: bool = False
    ) -> list[Unit]:
        status_clause = (
            "status IN ('pending', 'failed')" if include_failed else "status = 'pending'"
        )
        query = f"SELECT {_UNIT_COLUMNS} FROM units WHERE {status_clause} ORDER BY seq"
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = self._connection.execute(query)
        return [_unit_from_row(row) for row in rows]

    def recent_translated_units(self, limit: int = 16, before_seq: int | None = None) -> list[Unit]:
        """Fetch recently translated units before before_seq in chronological order."""
        valid_statuses = "'translated', 'reviewed', 'qa_flagged'"
        if before_seq is not None:
            query = (
                f"SELECT {_UNIT_COLUMNS} FROM units "
                f"WHERE seq < ? AND target IS NOT NULL AND status IN ({valid_statuses}) "
                f"ORDER BY seq DESC LIMIT {int(limit)}"
            )
            rows = self._connection.execute(query, (before_seq,)).fetchall()
        else:
            query = (
                f"SELECT {_UNIT_COLUMNS} FROM units "
                f"WHERE target IS NOT NULL AND status IN ({valid_statuses}) "
                f"ORDER BY seq DESC LIMIT {int(limit)}"
            )
            rows = self._connection.execute(query).fetchall()
        return [_unit_from_row(row) for row in reversed(rows)]

    def mark_failed(self, uid: str, error_reason: str) -> None:
        stamp = utc_timestamp()
        self._connection.execute(
            "UPDATE units SET status = 'failed', qa_flags = ?, updated_at = ? WHERE uid = ?",
            (json.dumps([f"error:{error_reason}"]), stamp, uid),
        )

    def reset_failed(self) -> int:
        cursor = self._connection.execute(
            "UPDATE units SET status = 'pending', qa_flags = '[]' WHERE status = 'failed'"
        )
        return int(cursor.rowcount)

    def update_translation(
        self,
        uid: str,
        target: str,
        origin: Origin,
        status: Status = Status.TRANSLATED,
        qa_flags: tuple[str, ...] = (),
        updated_at: str | None = None,
    ) -> None:
        stamp = updated_at or utc_timestamp()
        self._connection.execute(
            "UPDATE units SET target = ?, origin = ?, status = ?, qa_flags = ?, updated_at = ? "
            "WHERE uid = ?",
            (target, origin.value, status.value, json.dumps(list(qa_flags)), stamp, uid),
        )

    # ── Translation Memory ──────────────────────────────────────────

    def save_tm(self, entry: TMEntry) -> None:
        stamp = entry.updated_at or utc_timestamp()
        self._connection.execute(
            """
            INSERT INTO tm (src_norm, target, uses, origin, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (src_norm) DO UPDATE SET
                target = excluded.target,
                uses = tm.uses + 1,
                origin = excluded.origin,
                updated_at = excluded.updated_at
            """,
            (entry.src_norm, entry.target, entry.uses, entry.origin.value, stamp),
        )

    def get_tm(self, src_norm: str) -> TMEntry | None:
        query = (
            "SELECT src_norm, target, uses, origin, updated_at FROM tm "
            "WHERE src_norm = ? OR lower(src_norm) = lower(?)"
        )
        row = self._connection.execute(query, (src_norm, src_norm)).fetchone()
        if row is None:
            return None
        return TMEntry(
            src_norm=row["src_norm"],
            target=row["target"],
            uses=row["uses"],
            origin=Origin(row["origin"]),
            updated_at=row["updated_at"],
        )

    def all_tm(self) -> list[TMEntry]:
        rows = self._connection.execute(
            "SELECT src_norm, target, uses, origin, updated_at FROM tm ORDER BY uses DESC"
        )
        return [
            TMEntry(
                src_norm=r["src_norm"],
                target=r["target"],
                uses=r["uses"],
                origin=Origin(r["origin"]),
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    # ── Glossary ────────────────────────────────────────────────────

    def save_glossary_entry(self, entry: GlossaryEntry) -> None:
        self._connection.execute(
            """
            INSERT INTO glossary (term, translation, kind, case_sensitive, aliases, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (term) DO UPDATE SET
                translation = excluded.translation,
                kind = excluded.kind,
                case_sensitive = excluded.case_sensitive,
                aliases = excluded.aliases,
                notes = excluded.notes
            """,
            (
                entry.term,
                entry.translation,
                entry.kind,
                1 if entry.case_sensitive else 0,
                json.dumps(list(entry.aliases)),
                entry.notes,
            ),
        )

    def get_glossary(self) -> list[GlossaryEntry]:
        rows = self._connection.execute(
            "SELECT term, translation, kind, case_sensitive, aliases, notes FROM glossary "
            "ORDER BY length(term) DESC"
        )
        return [
            GlossaryEntry(
                term=r["term"],
                translation=r["translation"],
                kind=r["kind"],
                case_sensitive=bool(r["case_sensitive"]),
                aliases=tuple(json.loads(r["aliases"])),
                notes=r["notes"],
            )
            for r in rows
        ]

    # ── Characters ──────────────────────────────────────────────────

    def save_character(self, profile: CharacterProfile) -> None:
        self._connection.execute(
            """
            INSERT INTO characters (tag, display_name, gender, role, speech_style, pronoun_map)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (tag) DO UPDATE SET
                display_name = excluded.display_name,
                gender = excluded.gender,
                role = excluded.role,
                speech_style = excluded.speech_style,
                pronoun_map = excluded.pronoun_map
            """,
            (
                profile.tag,
                profile.display_name,
                profile.gender,
                profile.role,
                profile.speech_style,
                json.dumps(profile.pronoun_map),
            ),
        )

    def get_characters(self) -> dict[str, CharacterProfile]:
        rows = self._connection.execute(
            "SELECT tag, display_name, gender, role, speech_style, pronoun_map FROM characters"
        )
        return {
            r["tag"]: CharacterProfile(
                tag=r["tag"],
                display_name=r["display_name"],
                gender=r["gender"],
                role=r["role"],
                speech_style=r["speech_style"],
                pronoun_map=dict(json.loads(r["pronoun_map"])),
            )
            for r in rows
        }

    # ── Scenes ──────────────────────────────────────────────────────

    def save_scene(self, scene: SceneSummary) -> None:
        self._connection.execute(
            """
            INSERT INTO scenes (scene_id, label, first_seq, last_seq, summary)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (scene_id) DO UPDATE SET
                label = excluded.label,
                first_seq = excluded.first_seq,
                last_seq = excluded.last_seq,
                summary = excluded.summary
            """,
            (scene.scene_id, scene.label, scene.first_seq, scene.last_seq, scene.summary),
        )

    def get_scenes(self) -> list[SceneSummary]:
        rows = self._connection.execute(
            "SELECT scene_id, label, first_seq, last_seq, summary FROM scenes ORDER BY first_seq"
        )
        return [
            SceneSummary(
                scene_id=r["scene_id"],
                label=r["label"],
                first_seq=r["first_seq"],
                last_seq=r["last_seq"],
                summary=r["summary"],
            )
            for r in rows
        ]

    # ── Batches ─────────────────────────────────────────────────────

    def save_batch(self, batch: BatchLog) -> None:
        stamp = batch.created_at or utc_timestamp()
        self._connection.execute(
            """
            INSERT INTO batches (
                batch_id, unit_uids, tokens_in, tokens_out, attempts, error, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (batch_id) DO UPDATE SET
                tokens_in = excluded.tokens_in,
                tokens_out = excluded.tokens_out,
                attempts = excluded.attempts,
                error = excluded.error,
                created_at = excluded.created_at
            """,
            (
                batch.batch_id,
                json.dumps(list(batch.unit_uids)),
                batch.tokens_in,
                batch.tokens_out,
                batch.attempts,
                batch.error,
                stamp,
            ),
        )

    def sanitize_stored_translations(self) -> tuple[int, int]:
        """Sanitize existing units and TM entries by stripping hallucinated speaker tags and quotes.

        Returns (cleaned_units_count, cleaned_tm_count).
        """
        from vitrans.qa import clean_translation

        cleaned_units = 0
        cleaned_tm = 0

        with self.transaction():
            # 1. Sanitize units table
            cur = self.execute(
                "SELECT uid, speaker, source, target FROM units WHERE target IS NOT NULL"
            )
            unit_updates = []
            for row in cur.fetchall():
                uid, spk, src, tgt = row["uid"], row["speaker"], row["source"], row["target"]
                cleaned = clean_translation(tgt, speaker=spk, source=src)
                if cleaned != tgt:
                    unit_updates.append((cleaned, uid))

            if unit_updates:
                self._connection.executemany(
                    "UPDATE units SET target = ? WHERE uid = ?",
                    unit_updates,
                )
                cleaned_units = len(unit_updates)

            # 2. Sanitize TM table
            cur_tm = self.execute("SELECT src_norm, target FROM tm WHERE target IS NOT NULL")
            tm_updates = []
            for row in cur_tm.fetchall():
                src_norm, target = row["src_norm"], row["target"]
                cleaned = clean_translation(target, source=src_norm)
                if cleaned != target:
                    tm_updates.append((cleaned, src_norm))

            if tm_updates:
                self._connection.executemany(
                    "UPDATE tm SET target = ? WHERE src_norm = ?",
                    tm_updates,
                )
                cleaned_tm = len(tm_updates)

            # 3. Re-evaluate qa_flagged units using updated QAChecker
            qa_cur = self.execute(
                "SELECT uid, speaker, source, target FROM units WHERE status = 'qa_flagged'"
            )
            qa_rows = qa_cur.fetchall()
            if qa_rows:
                from vitrans.glossary import Glossary
                from vitrans.mask import mask
                from vitrans.qa import QAChecker

                g_obj = Glossary(self.get_glossary())
                qa_checker = QAChecker()
                recheck_updates = []
                for row in qa_rows:
                    uid, spk, src, tgt = row["uid"], row["speaker"], row["source"], row["target"]
                    masked = mask(src) if src else None
                    unit_glossary = g_obj.match_text(src) if src else ()
                    new_issues = qa_checker.check_unit(
                        uid, src, masked, tgt, glossary_matches=unit_glossary, speaker=spk
                    )
                    new_codes = tuple(sorted({i.code for i in new_issues}))
                    new_status = Status.QA_FLAGGED if new_issues else Status.TRANSLATED
                    recheck_updates.append((new_status.value, json.dumps(list(new_codes)), uid))

                if recheck_updates:
                    self._connection.executemany(
                        "UPDATE units SET status = ?, qa_flags = ? WHERE uid = ?",
                        recheck_updates,
                    )

        return cleaned_units, cleaned_tm
