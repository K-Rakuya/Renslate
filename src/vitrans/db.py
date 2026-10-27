"""SQLite store: the single source of truth for units, their slots and project metadata."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType

from vitrans.errors import ProjectError
from vitrans.models import ApplyTarget, Origin, Slot, Status, Unit, UnitKind

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
        connection = sqlite3.connect(path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
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
