"""SQLite store: the single source of truth for units, their slots and project metadata."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from renslate.core.errors import ProjectError
from renslate.core.models import (
    ApplyTarget,
    BatchLog,
    CharacterProfile,
    GlossaryEntry,
    Origin,
    RelationshipEdge,
    RelationshipKind,
    SceneSummary,
    Slot,
    Status,
    TMEntry,
    Unit,
    UnitKind,
)
from renslate.core.textio import write_atomic

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
)

_UNIT_COLUMNS = (
    "uid, kind, seq, ref, speaker, label, source, masked_source, target, status, origin, "
    "qa_flags, updated_at"
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_SQL_INSERT_UNIT = f"""
INSERT INTO units ({_UNIT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (uid) DO UPDATE SET
    kind = excluded.kind, seq = excluded.seq, ref = excluded.ref,
    speaker = excluded.speaker, label = excluded.label, source = excluded.source,
    masked_source = excluded.masked_source, target = excluded.target,
    status = excluded.status, origin = excluded.origin,
    qa_flags = excluded.qa_flags, updated_at = excluded.updated_at
"""

_SQL_SELECT_BY_UID = f"SELECT {_UNIT_COLUMNS} FROM units WHERE uid = ?"
_SQL_SELECT_ALL_ORDERED = f"SELECT {_UNIT_COLUMNS} FROM units ORDER BY seq"
_SQL_SELECT_BY_STATUS_ORDERED = f"SELECT {_UNIT_COLUMNS} FROM units WHERE status = ? ORDER BY seq"
_SQL_SELECT_PENDING = f"SELECT {_UNIT_COLUMNS} FROM units WHERE status = 'pending' ORDER BY seq"
_SQL_SELECT_PENDING_OR_FAILED = (
    f"SELECT {_UNIT_COLUMNS} FROM units WHERE status IN ('pending', 'failed') ORDER BY seq"
)
_SQL_SELECT_PENDING_LIMIT = (
    f"SELECT {_UNIT_COLUMNS} FROM units WHERE status = 'pending' ORDER BY seq LIMIT ?"
)
_SQL_SELECT_PENDING_OR_FAILED_LIMIT = (
    f"SELECT {_UNIT_COLUMNS} FROM units WHERE status IN ('pending', 'failed') ORDER BY seq LIMIT ?"
)
_SQL_SELECT_RECENT_BEFORE_SEQ = (
    f"SELECT {_UNIT_COLUMNS} FROM units "
    "WHERE seq < ? AND target IS NOT NULL AND status IN ('translated', 'reviewed', 'qa_flagged') "
    "ORDER BY seq DESC LIMIT ?"
)
_SQL_SELECT_RECENT = (
    f"SELECT {_UNIT_COLUMNS} FROM units "
    "WHERE target IS NOT NULL AND status IN ('translated', 'reviewed', 'qa_flagged') "
    "ORDER BY seq DESC LIMIT ?"
)
_SQL_MARK_FAILED = "UPDATE units SET status = 'failed', qa_flags = ?, updated_at = ? WHERE uid = ?"
_SQL_RESET_FAILED = "UPDATE units SET status = 'pending', qa_flags = '[]' WHERE status = 'failed'"
_SQL_UPDATE_TRANSLATION = (
    "UPDATE units SET target = ?, origin = ?, status = ?, qa_flags = ?, updated_at = ? "
    "WHERE uid = ?"
)
_SQL_DELETE_SLOTS = "DELETE FROM slots"
_SQL_INSERT_SLOT = (
    "INSERT INTO slots (uid, file, line, anchor_line, fingerprint) VALUES (?, ?, ?, ?, ?)"
)
_SQL_UPDATE_SLOT_FINGERPRINT = "UPDATE slots SET fingerprint = ? WHERE file = ? AND line = ?"
_SQL_COUNT_SLOTS = "SELECT COUNT(*) FROM slots"
_SQL_MARK_ORPHANS_OBSOLETE = """
UPDATE units SET status = 'obsolete', updated_at = ?
WHERE status != 'obsolete' AND uid NOT IN (SELECT uid FROM slots)
"""
_SQL_SELECT_APPLY_TARGETS = """
SELECT s.uid, s.file, s.line, s.anchor_line, s.fingerprint, u.kind, u.target
FROM slots s JOIN units u ON u.uid = s.uid
WHERE u.target IS NOT NULL AND u.status != 'obsolete'
ORDER BY s.file, s.line
"""
_SQL_SELECT_STATS = (
    "SELECT kind, status, COUNT(*) AS total FROM units GROUP BY kind, status ORDER BY kind, status"
)
_SQL_SELECT_META = "SELECT value FROM meta WHERE key = ?"
_SQL_UPSERT_META = (
    "INSERT INTO meta (key, value) VALUES (?, ?) "
    "ON CONFLICT (key) DO UPDATE SET value = excluded.value"
)
_SQL_UPSERT_TM = """
INSERT INTO tm (src_norm, target, uses, origin, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (src_norm) DO UPDATE SET
    target = excluded.target,
    uses = tm.uses + 1,
    origin = excluded.origin,
    updated_at = excluded.updated_at
"""
_SQL_SELECT_TM = (
    "SELECT src_norm, target, uses, origin, updated_at FROM tm "
    "WHERE src_norm = ? OR lower(src_norm) = lower(?)"
)
_SQL_SELECT_ALL_TM = "SELECT src_norm, target, uses, origin, updated_at FROM tm ORDER BY uses DESC"
_SQL_UPSERT_GLOSSARY = """
INSERT INTO glossary (term, translation, kind, case_sensitive, aliases, notes)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (term) DO UPDATE SET
    translation = excluded.translation,
    kind = excluded.kind,
    case_sensitive = excluded.case_sensitive,
    aliases = excluded.aliases,
    notes = excluded.notes
"""
_SQL_SELECT_GLOSSARY = (
    "SELECT term, translation, kind, case_sensitive, aliases, notes FROM glossary "
    "ORDER BY length(term) DESC"
)
_SQL_UPSERT_CHARACTER = """
INSERT INTO characters (
    tag, display_name, gender, role, speech_style, pronoun_map,
    archetype, personality_traits, tone_and_voice, socioeconomic_status, sample_dialogue
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (tag) DO UPDATE SET
    display_name = excluded.display_name,
    gender = excluded.gender,
    role = excluded.role,
    speech_style = excluded.speech_style,
    pronoun_map = excluded.pronoun_map,
    archetype = excluded.archetype,
    personality_traits = excluded.personality_traits,
    tone_and_voice = excluded.tone_and_voice,
    socioeconomic_status = excluded.socioeconomic_status,
    sample_dialogue = excluded.sample_dialogue
"""
_SQL_SELECT_CHARACTERS = (
    "SELECT tag, display_name, gender, role, speech_style, pronoun_map, "
    "archetype, personality_traits, tone_and_voice, socioeconomic_status, sample_dialogue "
    "FROM characters"
)

_SQL_UPSERT_RELATIONSHIP = """
INSERT INTO character_relationships (
    source_tag, target_tag, kind, speaker_pronoun, listener_pronoun,
    confidence, source_type, notes
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (source_tag, target_tag) DO UPDATE SET
    kind = excluded.kind,
    speaker_pronoun = excluded.speaker_pronoun,
    listener_pronoun = excluded.listener_pronoun,
    confidence = excluded.confidence,
    source_type = excluded.source_type,
    notes = excluded.notes
"""
_RELATIONSHIP_COLUMNS = (
    "source_tag, target_tag, kind, speaker_pronoun, listener_pronoun, "
    "confidence, source_type, notes"
)
_SQL_SELECT_ALL_RELATIONSHIPS = (
    f"SELECT {_RELATIONSHIP_COLUMNS} FROM character_relationships "
    "ORDER BY source_tag, target_tag"
)
_SQL_SELECT_RELATIONSHIP = (
    f"SELECT {_RELATIONSHIP_COLUMNS} FROM character_relationships "
    "WHERE source_tag = ? AND target_tag = ?"
)
_SQL_SELECT_RELATIONSHIPS_BY_SOURCE = (
    f"SELECT {_RELATIONSHIP_COLUMNS} FROM character_relationships "
    "WHERE source_tag = ? ORDER BY target_tag"
)
_SQL_SELECT_RELATIONSHIPS_BY_TARGET = (
    f"SELECT {_RELATIONSHIP_COLUMNS} FROM character_relationships "
    "WHERE target_tag = ? ORDER BY source_tag"
)
_SQL_DELETE_RELATIONSHIP = (
    "DELETE FROM character_relationships WHERE source_tag = ? AND target_tag = ?"
)
_SQL_DELETE_ALL_RELATIONSHIPS = "DELETE FROM character_relationships"
_SQL_COUNT_RELATIONSHIPS = "SELECT COUNT(*) FROM character_relationships"
_SQL_UPSERT_SCENE = """
INSERT INTO scenes (scene_id, label, first_seq, last_seq, summary)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (scene_id) DO UPDATE SET
    label = excluded.label,
    first_seq = excluded.first_seq,
    last_seq = excluded.last_seq,
    summary = excluded.summary
"""
_SQL_SELECT_SCENES = (
    "SELECT scene_id, label, first_seq, last_seq, summary FROM scenes ORDER BY first_seq"
)
_SQL_UPSERT_BATCH = """
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
"""


def _unit_from_tuple(row: tuple[Any, ...] | sqlite3.Row) -> Unit:
    origin = row[10]
    qa_raw = row[11]
    qa_flags: tuple[str, ...]
    if isinstance(qa_raw, str):
        qa_flags = tuple(json.loads(qa_raw))
    elif isinstance(qa_raw, (tuple, list)):
        qa_flags = tuple(qa_raw)
    else:
        qa_flags = ()
    return Unit(
        uid=row[0],
        kind=UnitKind(row[1]),
        seq=row[2],
        ref=row[3],
        speaker=row[4],
        label=row[5],
        source=row[6],
        masked_source=row[7],
        target=row[8],
        status=Status(row[9]),
        origin=Origin(origin) if origin is not None else None,
        qa_flags=qa_flags,
        updated_at=row[12],
    )


def _unit_from_row(row: sqlite3.Row) -> Unit:
    return _unit_from_tuple(tuple(row))


def _relationship_from_row(row: sqlite3.Row | tuple[Any, ...]) -> RelationshipEdge:
    if isinstance(row, sqlite3.Row):
        src = str(row["source_tag"])
        tgt = str(row["target_tag"])
        kind_raw = str(row["kind"])
        spk_pro = str(row["speaker_pronoun"])
        lis_pro = str(row["listener_pronoun"])
        conf = float(row["confidence"])
        stype = str(row["source_type"])
        notes = str(row["notes"])
    else:
        src, tgt, kind_raw, spk_pro, lis_pro, conf, stype, notes = row[:8]
        conf = float(conf)

    try:
        kind = RelationshipKind(kind_raw)
    except ValueError:
        try:
            kind = RelationshipKind.from_str(str(kind_raw))
        except ValueError:
            kind = RelationshipKind.FORMAL_ACQUAINTANCE

    return RelationshipEdge(
        source_tag=src,
        target_tag=tgt,
        kind=kind,
        speaker_pronoun=spk_pro,
        listener_pronoun=lis_pro,
        confidence=conf,
        source_type=stype,
        notes=notes,
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
        return {u.uid: u for u in self.iter_units()}

    def iter_units(self, status: Status | None = None) -> Iterator[Unit]:
        """Stream units directly from SQLite without materializing full list in RAM."""
        cursor = self._connection.cursor()
        if status is not None:
            cursor.execute(_SQL_SELECT_BY_STATUS_ORDERED, (status.value,))
        else:
            cursor.execute(_SQL_SELECT_ALL_ORDERED)
        for row in cursor:
            yield _unit_from_tuple(tuple(row))

    def iter_pending_units(
        self, limit: int | None = None, *, include_failed: bool = False
    ) -> Iterator[Unit]:
        """Stream pending units with optional limit directly from SQLite cursor."""
        cursor = self._connection.cursor()
        if limit is not None:
            query = (
                _SQL_SELECT_PENDING_OR_FAILED_LIMIT if include_failed else _SQL_SELECT_PENDING_LIMIT
            )
            cursor.execute(query, (int(limit),))
        else:
            query = _SQL_SELECT_PENDING_OR_FAILED if include_failed else _SQL_SELECT_PENDING
            cursor.execute(query)
        for row in cursor:
            yield _unit_from_tuple(tuple(row))

    def save_unit(self, unit: Unit) -> None:
        self._connection.execute(
            _SQL_INSERT_UNIT,
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
        params = [
            (
                u.uid,
                u.kind.value,
                u.seq,
                u.ref,
                u.speaker,
                u.label,
                u.source,
                u.masked_source,
                u.target,
                u.status.value,
                u.origin.value if u.origin is not None else None,
                json.dumps(list(u.qa_flags)),
                u.updated_at,
            )
            for u in units
        ]
        if self._connection.in_transaction:
            self._connection.executemany(_SQL_INSERT_UNIT, params)
        else:
            with self.transaction():
                self._connection.executemany(_SQL_INSERT_UNIT, params)

    def replace_slots(self, slots: Iterable[Slot]) -> None:
        slot_params = [(s.uid, s.file, s.line, s.anchor_line, s.fingerprint) for s in slots]
        if self._connection.in_transaction:
            self._connection.execute(_SQL_DELETE_SLOTS)
            self._connection.executemany(_SQL_INSERT_SLOT, slot_params)
        else:
            with self.transaction():
                self._connection.execute(_SQL_DELETE_SLOTS)
                self._connection.executemany(_SQL_INSERT_SLOT, slot_params)

    def mark_orphans_obsolete(self, timestamp: str) -> int:
        cursor = self._connection.execute(_SQL_MARK_ORPHANS_OBSOLETE, (timestamp,))
        return cursor.rowcount

    def apply_targets(self) -> list[ApplyTarget]:
        rows = self._connection.execute(_SQL_SELECT_APPLY_TARGETS)
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
            _SQL_UPDATE_SLOT_FINGERPRINT,
            [(fingerprint, file, line) for file, line, fingerprint in updates],
        )

    def count_slots(self) -> int:
        row = self._connection.execute(_SQL_COUNT_SLOTS).fetchone()
        return int(row[0])

    def stats(self) -> list[tuple[UnitKind, Status, int]]:
        rows = self._connection.execute(_SQL_SELECT_STATS)
        return [(UnitKind(r["kind"]), Status(r["status"]), int(r["total"])) for r in rows]

    def get_meta(self, key: str) -> str | None:
        row = self._connection.execute(_SQL_SELECT_META, (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self._connection.execute(_SQL_UPSERT_META, (key, value))

    def get_unit(self, uid: str) -> Unit | None:
        row = self._connection.execute(_SQL_SELECT_BY_UID, (uid,)).fetchone()
        return None if row is None else _unit_from_tuple(tuple(row))

    def all_units(self) -> list[Unit]:
        return list(self.iter_units())

    def pending_units(
        self, limit: int | None = None, *, include_failed: bool = False
    ) -> list[Unit]:
        return list(self.iter_pending_units(limit=limit, include_failed=include_failed))

    def recent_translated_units(self, limit: int = 16, before_seq: int | None = None) -> list[Unit]:
        """Fetch recently translated units before before_seq in chronological order."""
        if before_seq is not None:
            rows = self._connection.execute(
                _SQL_SELECT_RECENT_BEFORE_SEQ, (before_seq, int(limit))
            ).fetchall()
        else:
            rows = self._connection.execute(_SQL_SELECT_RECENT, (int(limit),)).fetchall()
        return [_unit_from_tuple(tuple(row)) for row in reversed(rows)]

    def mark_failed(self, uid: str, error_reason: str) -> None:
        stamp = utc_timestamp()
        self._connection.execute(
            _SQL_MARK_FAILED,
            (json.dumps([f"error:{error_reason}"]), stamp, uid),
        )

    def reset_failed(self) -> int:
        cursor = self._connection.execute(_SQL_RESET_FAILED)
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
            _SQL_UPDATE_TRANSLATION,
            (target, origin.value, status.value, json.dumps(list(qa_flags)), stamp, uid),
        )

    # ── Translation Memory ──────────────────────────────────────────

    def save_tm(self, entry: TMEntry) -> None:
        stamp = entry.updated_at or utc_timestamp()
        self._connection.execute(
            _SQL_UPSERT_TM,
            (entry.src_norm, entry.target, entry.uses, entry.origin.value, stamp),
        )

    def get_tm(self, src_norm: str) -> TMEntry | None:
        row = self._connection.execute(_SQL_SELECT_TM, (src_norm, src_norm)).fetchone()
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
        rows = self._connection.execute(_SQL_SELECT_ALL_TM)
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
            _SQL_UPSERT_GLOSSARY,
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
        rows = self._connection.execute(_SQL_SELECT_GLOSSARY)
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
            _SQL_UPSERT_CHARACTER,
            (
                profile.tag,
                profile.display_name,
                profile.gender,
                profile.role,
                profile.speech_style,
                json.dumps(profile.pronoun_map),
                profile.archetype,
                json.dumps(list(profile.personality_traits)),
                profile.tone_and_voice,
                profile.socioeconomic_status,
                json.dumps(list(profile.sample_dialogue)),
            ),
        )

    def save_characters_bulk(self, profiles: Sequence[CharacterProfile]) -> None:
        """Bulk upsert character profiles in a single query execution."""
        if not profiles:
            return
        params = [
            (
                p.tag,
                p.display_name,
                p.gender,
                p.role,
                p.speech_style,
                json.dumps(p.pronoun_map),
                p.archetype,
                json.dumps(list(p.personality_traits)),
                p.tone_and_voice,
                p.socioeconomic_status,
                json.dumps(list(p.sample_dialogue)),
            )
            for p in profiles
        ]
        self._connection.executemany(_SQL_UPSERT_CHARACTER, params)

    def clear_characters(self) -> int:
        """Delete all characters from database. Returns count deleted."""
        cur = self._connection.execute("DELETE FROM characters")
        return cur.rowcount

    def get_characters(self) -> dict[str, CharacterProfile]:
        rows = self._connection.execute(_SQL_SELECT_CHARACTERS)
        result: dict[str, CharacterProfile] = {}
        for r in rows:
            tag = str(r["tag"])
            traits_raw = r["personality_traits"]
            traits = (
                tuple(json.loads(traits_raw))
                if isinstance(traits_raw, str)
                else tuple(traits_raw)
            )
            dlg_raw = r["sample_dialogue"]
            dlg = (
                tuple(json.loads(dlg_raw))
                if isinstance(dlg_raw, str)
                else tuple(dlg_raw)
            )
            result[tag] = CharacterProfile(
                tag=tag,
                display_name=str(r["display_name"]),
                gender=str(r["gender"]),
                role=str(r["role"]),
                speech_style=str(r["speech_style"]),
                pronoun_map=dict(json.loads(r["pronoun_map"])),
                archetype=str(r["archetype"]),
                personality_traits=traits,
                tone_and_voice=str(r["tone_and_voice"]),
                socioeconomic_status=str(r["socioeconomic_status"]),
                sample_dialogue=dlg,
            )
        return result

    # ── Character Relationships ─────────────────────────────────────

    def save_relationship(self, edge: RelationshipEdge) -> None:
        """Upsert a single character relationship edge."""
        kind_val = edge.kind.value if hasattr(edge.kind, "value") else str(edge.kind)
        self._connection.execute(
            _SQL_UPSERT_RELATIONSHIP,
            (
                edge.source_tag,
                edge.target_tag,
                kind_val,
                edge.speaker_pronoun,
                edge.listener_pronoun,
                float(edge.confidence),
                edge.source_type,
                edge.notes,
            ),
        )
        self.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())

    def save_relationships_bulk(self, edges: Sequence[RelationshipEdge]) -> None:
        """Bulk upsert relationship edges in a single atomic transaction."""
        if not edges:
            return
        params = [
            (
                e.source_tag,
                e.target_tag,
                e.kind.value if hasattr(e.kind, "value") else str(e.kind),
                e.speaker_pronoun,
                e.listener_pronoun,
                float(e.confidence),
                e.source_type,
                e.notes,
            )
            for e in edges
        ]
        if self._connection.in_transaction:
            self._connection.executemany(_SQL_UPSERT_RELATIONSHIP, params)
            self.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())
        else:
            with self.transaction():
                self._connection.executemany(_SQL_UPSERT_RELATIONSHIP, params)
                self.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())

    def get_relationship(self, src: str, tgt: str) -> RelationshipEdge | None:
        """Fetch a single relationship edge between source and target, or None."""
        row = self._connection.execute(_SQL_SELECT_RELATIONSHIP, (src, tgt)).fetchone()
        if row is None:
            return None
        return _relationship_from_row(row)

    def get_relationships(
        self, source_tag: str | None = None, *, target_tag: str | None = None
    ) -> list[RelationshipEdge]:
        """Fetch relationship edges, optionally filtered by source or target tag."""
        if source_tag is not None and target_tag is not None:
            edge = self.get_relationship(source_tag, target_tag)
            return [edge] if edge is not None else []
        if source_tag is not None:
            rows = self._connection.execute(
                _SQL_SELECT_RELATIONSHIPS_BY_SOURCE, (source_tag,)
            ).fetchall()
        elif target_tag is not None:
            rows = self._connection.execute(
                _SQL_SELECT_RELATIONSHIPS_BY_TARGET, (target_tag,)
            ).fetchall()
        else:
            rows = self._connection.execute(_SQL_SELECT_ALL_RELATIONSHIPS).fetchall()
        return [_relationship_from_row(r) for r in rows]

    def delete_relationship(self, src: str, tgt: str) -> bool:
        """Delete an edge from character_relationships. Returns True if deleted."""
        cursor = self._connection.execute(_SQL_DELETE_RELATIONSHIP, (src, tgt))
        if cursor.rowcount > 0:
            self.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())
            return True
        return False

    def clear_relationships(self) -> int:
        """Delete all edges from character_relationships. Returns count deleted."""
        cursor = self._connection.execute(_SQL_DELETE_ALL_RELATIONSHIPS)
        if cursor.rowcount > 0:
            self.set_meta("relationships_updated_at", datetime.now(timezone.utc).isoformat())
        return int(cursor.rowcount)

    def count_relationships(self) -> int:
        """Return total count of relationship edges in the database."""
        row = self._connection.execute(_SQL_COUNT_RELATIONSHIPS).fetchone()
        return int(row[0]) if row else 0

    # ── Scenes ──────────────────────────────────────────────────────

    def save_scene(self, scene: SceneSummary) -> None:
        self._connection.execute(
            _SQL_UPSERT_SCENE,
            (scene.scene_id, scene.label, scene.first_seq, scene.last_seq, scene.summary),
        )

    def get_scenes(self) -> list[SceneSummary]:
        rows = self._connection.execute(_SQL_SELECT_SCENES)
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
            _SQL_UPSERT_BATCH,
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
        import importlib

        qa_mod = importlib.import_module("renslate.engine.qa")
        clean_translation = qa_mod.clean_translation

        cleaned_units = 0
        cleaned_tm = 0

        with self.transaction():
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

            qa_cur = self.execute(
                "SELECT uid, speaker, source, target FROM units WHERE status = 'qa_flagged'"
            )
            qa_rows = qa_cur.fetchall()
            if qa_rows:
                glossary_mod = importlib.import_module("renslate.engine.glossary")
                Glossary = glossary_mod.Glossary
                QAChecker = qa_mod.QAChecker
                mask_mod = importlib.import_module("renslate.renpy.mask")
                mask = mask_mod.mask

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


# ── Characters Dual Persistence & Bidirectional Synchronization ─────────────


def load_characters_from_json(path: Path | str) -> dict[str, CharacterProfile]:
    """Load character profiles from a JSON file supporting dictionary or list schemas."""
    p = Path(path)
    if not p.is_file():
        return {}
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        return {}
    data = json.loads(content)
    result: dict[str, CharacterProfile] = {}
    if isinstance(data, dict):
        for tag, fields in data.items():
            if isinstance(fields, dict):
                result[str(tag)] = CharacterProfile.from_dict(fields, tag=str(tag))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "tag" in item:
                t = str(item["tag"])
                result[t] = CharacterProfile.from_dict(item, tag=t)
    else:
        msg = f"Expected dict or list for characters JSON, got {type(data).__name__}"
        raise TypeError(msg)
    return result


def save_characters_to_json(
    characters: Mapping[str, CharacterProfile] | Sequence[CharacterProfile],
    path: Path | str,
    *,
    indent: int = 2,
) -> None:
    """Atomically save character profiles to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(characters, Mapping):
        chars_list = sorted(characters.values(), key=lambda c: c.tag)
    else:
        chars_list = sorted(characters, key=lambda c: c.tag)

    data: dict[str, dict[str, Any]] = {}
    for prof in chars_list:
        data[prof.tag] = prof.to_dict(include_tag=False)

    payload = (json.dumps(data, ensure_ascii=False, indent=indent) + "\n").encode("utf-8")
    write_atomic(p, payload)


def sync_characters(
    db: Database,
    json_path: Path | str,
    *,
    direction: str = "auto",
) -> tuple[int, int]:
    """Bidirectionally synchronize character profiles between SQLite and characters.json.

    Supported directions:
    - "auto": Intelligent two-way merge using content hash and microsecond timestamps.
    - "db_to_json": Explicit unidirectional sync from SQLite DB to JSON file.
    - "json_to_db": Explicit unidirectional sync from JSON file to SQLite DB.

    Returns:
        (db_character_count, json_character_count)
    """
    p = Path(json_path)
    if direction not in ("auto", "db_to_json", "json_to_db"):
        msg = (
            f"Invalid direction: '{direction}'. "
            "Must be 'auto', 'db_to_json', or 'json_to_db'."
        )
        raise ValueError(msg)

    if direction == "db_to_json":
        db_chars = db.get_characters()
        save_characters_to_json(db_chars, p)
        content = p.read_bytes()
        db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
        db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())
        return len(db_chars), len(db_chars)

    if direction == "json_to_db":
        if not p.is_file():
            msg = f"Cannot sync json_to_db: JSON file not found at {p}"
            raise FileNotFoundError(msg)
        json_chars = load_characters_from_json(p)
        with db.transaction():
            db.clear_characters()
            db.save_characters_bulk(list(json_chars.values()))
        content = p.read_bytes()
        db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
        db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())
        return len(json_chars), len(json_chars)

    # direction == "auto"
    if not p.is_file():
        db_chars = db.get_characters()
        if len(db_chars) > 0:
            save_characters_to_json(db_chars, p)
            content = p.read_bytes()
            db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
            db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())
            return len(db_chars), len(db_chars)
        return len(db_chars), 0

    json_chars = load_characters_from_json(p)
    db_chars = db.get_characters()

    if len(db_chars) == 0 and len(json_chars) > 0:
        with db.transaction():
            db.save_characters_bulk(list(json_chars.values()))
        content = p.read_bytes()
        db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
        db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())
        return len(json_chars), len(json_chars)

    if len(json_chars) == 0 and len(db_chars) > 0:
        save_characters_to_json(db_chars, p)
        content = p.read_bytes()
        db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
        db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())
        return len(db_chars), len(db_chars)

    stored_hash = db.get_meta("characters_json_hash")
    current_content = p.read_bytes()
    current_hash = hashlib.sha256(current_content).hexdigest()

    db_updated_meta = db.get_meta("characters_updated_at")
    try:
        db_ts = (
            datetime.fromisoformat(db_updated_meta.replace("Z", "+00:00")).timestamp()
            if db_updated_meta
            else 0.0
        )
    except Exception:
        db_ts = 0.0
    json_ts = p.stat().st_mtime

    if stored_hash is not None:
        json_wins_tie = False if current_hash == stored_hash else db_ts <= json_ts + 0.005
    else:
        json_wins_tie = json_ts > db_ts + 0.005

    merged: dict[str, CharacterProfile] = {}
    all_tags = set(json_chars.keys()) | set(db_chars.keys())

    for tag in all_tags:
        j_prof = json_chars.get(tag)
        d_prof = db_chars.get(tag)

        if j_prof is not None and d_prof is None:
            merged[tag] = j_prof
        elif d_prof is not None and j_prof is None:
            merged[tag] = d_prof
        elif j_prof is not None and d_prof is not None:
            winner = j_prof if json_wins_tie else d_prof
            fallback = d_prof if json_wins_tie else j_prof
            merged[tag] = CharacterProfile(
                tag=winner.tag,
                display_name=winner.display_name or fallback.display_name,
                gender=winner.gender if winner.gender != "unknown" else fallback.gender,
                role=winner.role or fallback.role,
                speech_style=winner.speech_style or fallback.speech_style,
                pronoun_map=winner.pronoun_map or fallback.pronoun_map,
                archetype=winner.archetype or fallback.archetype,
                personality_traits=winner.personality_traits or fallback.personality_traits,
                tone_and_voice=winner.tone_and_voice or fallback.tone_and_voice,
                socioeconomic_status=winner.socioeconomic_status or fallback.socioeconomic_status,
                sample_dialogue=winner.sample_dialogue or fallback.sample_dialogue,
            )

    with db.transaction():
        db.save_characters_bulk(list(merged.values()))
    save_characters_to_json(merged, p)
    new_content = p.read_bytes()
    db.set_meta("characters_json_hash", hashlib.sha256(new_content).hexdigest())
    db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())
    return len(merged), len(merged)

