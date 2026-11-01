"""SQLite store: the single source of truth for units, slots, scenes, and metadata."""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from renslate.core.db.connection import (
    backup_before_migration,
    configure_connection,
    create_connection,
)
from renslate.core.db.repositories import (
    BatchRepository,
    CharacterRepository,
    GlossaryRepository,
    QAAuditRepository,
    RelationshipRepository,
    SceneRepository,
    TMRepository,
    UnitRepository,
    utc_timestamp,
)
from renslate.core.db.schema import (
    MIGRATIONS,
    get_schema_version,
    run_migrations,
)
from renslate.core.errors import ProjectError
from renslate.core.models import (
    ApplyTarget,
    BatchLog,
    CharacterProfile,
    GlossaryEntry,
    Origin,
    RelationshipEdge,
    SceneMacroContext,
    SceneSummary,
    Slot,
    Status,
    TMEntry,
    Unit,
    UnitKind,
)
from renslate.core.textio import write_atomic

_LOG = logging.getLogger("renslate.core.db")

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


class Database:
    """Thin wrapper over a SQLite connection exposing all pipeline operations."""

    def __init__(self, connection: sqlite3.Connection, path: Path | None = None) -> None:
        self._connection = connection
        self._path = path
        configure_connection(self._connection)
        self._units_repo = UnitRepository(self._connection)
        self._chars_repo = CharacterRepository(self._connection)
        self._rel_repo = RelationshipRepository(self._connection)
        self._scenes_repo = SceneRepository(self._connection)
        self._tm_repo = TMRepository(self._connection)
        self._batch_repo = BatchRepository(self._connection)
        self._glossary_repo = GlossaryRepository(self._connection)
        self._audit_repo = QAAuditRepository(self._connection)

    @classmethod
    def open(cls, path: Path, *, create: bool = False) -> Database:
        if not create and not path.is_file():
            raise ProjectError(f"không tìm thấy cơ sở dữ liệu: {path}")
        connection = create_connection(path)
        database = cls(connection, path=path)
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
        return get_schema_version(self._connection)

    def _migrate(self) -> None:
        run_migrations(self._connection, db_path=self._path)

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

    # ── Units & Slots ───────────────────────────────────────────────

    def units(self) -> dict[str, Unit]:
        return self._units_repo.units()

    def all_units(self) -> list[Unit]:
        return self._units_repo.all_units()

    def iter_units(self, status: Status | None = None) -> Iterator[Unit]:
        return self._units_repo.iter_units(status=status)

    def iter_pending_units(
        self, limit: int | None = None, *, include_failed: bool = False
    ) -> Iterator[Unit]:
        return self._units_repo.iter_pending_units(limit=limit, include_failed=include_failed)

    def get_unit(self, uid: str) -> Unit | None:
        return self._units_repo.get_unit(uid)

    def units_by_status(self, status: Status) -> list[Unit]:
        return self._units_repo.units_by_status(status)

    def pending_units(self, limit: int | None = None, include_failed: bool = False) -> list[Unit]:
        return self._units_repo.pending_units(limit=limit, include_failed=include_failed)

    def save_unit(self, unit: Unit) -> None:
        self._units_repo.save_unit(unit)

    def save_units(self, units: Sequence[Unit]) -> None:
        self._units_repo.save_units(units)

    def save_units_bulk(self, units: Sequence[Unit]) -> None:
        self._units_repo.save_units_bulk(units)

    def update_translation(
        self,
        uid: str,
        target: str,
        origin: Origin = Origin.LLM,
        status: Status = Status.TRANSLATED,
        qa_flags: Sequence[str] = (),
        updated_at: str | None = None,
    ) -> None:
        self._units_repo.update_translation(
            uid, target, origin, status, qa_flags, updated_at=updated_at
        )

    def mark_failed(self, uid: str, error_reason: str | Sequence[str]) -> None:
        self._units_repo.mark_failed(uid, error_reason)

    def reset_failed(self) -> int:
        return self._units_repo.reset_failed()

    def replace_slots(self, slots: Iterable[Slot]) -> None:
        self._units_repo.replace_slots(slots)

    def save_slots(self, slots: Sequence[Slot]) -> None:
        self._units_repo.save_slots(slots)

    def get_slots(self) -> list[Slot]:
        return self._units_repo.get_slots()

    def get_slots_for_unit(self, uid: str) -> list[Slot]:
        return self._units_repo.get_slots_for_unit(uid)

    def apply_targets(self) -> list[ApplyTarget]:
        return self._units_repo.apply_targets()

    def mark_orphans_obsolete(self, timestamp: str) -> int:
        return self._units_repo.mark_orphans_obsolete(timestamp)

    def update_fingerprints(self, updates: Iterable[tuple[str, int, str]]) -> None:
        self._units_repo.update_fingerprints(updates)

    def count_slots(self) -> int:
        return self._units_repo.count_slots()

    def stats(self) -> list[tuple[UnitKind, Status, int]]:
        return self._units_repo.stats()

    def recent_translated_units(
        self, limit: int = 16, before_seq: int | None = None
    ) -> list[Unit]:
        return self._units_repo.recent_translated_units(limit=limit, before_seq=before_seq)

    def recent_context(self, before_seq: int, limit: int = 4) -> list[Unit]:
        return self._units_repo.recent_context(before_seq, limit)

    # ── Meta ────────────────────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def delete_meta(self, key: str) -> None:
        self._connection.execute("DELETE FROM meta WHERE key = ?", (key,))

    # ── Translation Memory ──────────────────────────────────────────

    def get_tm(self, src_norm: str) -> TMEntry | None:
        return self._tm_repo.get(src_norm)

    def save_tm(self, entry: TMEntry) -> None:
        self._tm_repo.save(entry)

    def all_tm(self) -> list[TMEntry]:
        return self._tm_repo.all_entries()

    def lookup_tm_fuzzy(
        self,
        src_norm: str,
        threshold: float = 0.88,
        limit: int = 1,
    ) -> list[tuple[TMEntry, float]]:
        entries = self.all_tm()
        if not entries:
            return []
        scored: list[tuple[TMEntry, float]] = []
        for e in entries:
            ratio = difflib.SequenceMatcher(None, src_norm, e.src_norm).ratio()
            if ratio >= threshold:
                scored.append((e, ratio))
        scored.sort(key=lambda x: (x[1], x[0].uses), reverse=True)
        return scored[:limit]

    # ── Glossary (Legacy Compatibility) ─────────────────────────────

    def get_glossary(self) -> list[GlossaryEntry]:
        return self._glossary_repo.get_all()

    def save_glossary(self, entry: GlossaryEntry) -> None:
        self._glossary_repo.save(entry)

    def save_glossary_entry(self, entry: GlossaryEntry) -> None:
        self.save_glossary(entry)

    def save_glossary_bulk(self, entries: Sequence[GlossaryEntry]) -> None:
        self._glossary_repo.save_bulk(entries)

    # ── Characters ──────────────────────────────────────────────────

    def get_characters(self) -> dict[str, CharacterProfile]:
        return self._chars_repo.get_characters()

    def save_character(self, profile: CharacterProfile) -> None:
        self._chars_repo.save_character(profile)

    def save_characters_bulk(self, profiles: Sequence[CharacterProfile]) -> None:
        self._chars_repo.save_characters_bulk(profiles)

    def clear_characters(self) -> None:
        self._chars_repo.clear_characters()

    # ── Relationships ───────────────────────────────────────────────

    def get_relationship(self, source_tag: str, target_tag: str) -> RelationshipEdge | None:
        return self._rel_repo.get_relationship(source_tag, target_tag)

    def get_relationships(
        self,
        source_tag: str | None = None,
        target_tag: str | None = None,
    ) -> list[RelationshipEdge]:
        return self._rel_repo.get_relationships(source_tag=source_tag, target_tag=target_tag)

    def get_relationships_for_source(self, source_tag: str) -> list[RelationshipEdge]:
        return self._rel_repo.relationships_for_source(source_tag)

    def get_relationships_for_target(self, target_tag: str) -> list[RelationshipEdge]:
        return self._rel_repo.relationships_for_target(target_tag)

    def save_relationship(self, edge: RelationshipEdge) -> None:
        self._rel_repo.save_relationship(edge)

    def save_relationships_bulk(self, edges: Sequence[RelationshipEdge]) -> None:
        if self._connection.in_transaction:
            self._rel_repo.save_relationships_bulk(edges)
        else:
            with self.transaction():
                self._rel_repo.save_relationships_bulk(edges)

    def delete_relationship(self, source_tag: str, target_tag: str) -> bool:
        return self._rel_repo.delete_relationship(source_tag, target_tag)

    def clear_relationships(self) -> int:
        return self._rel_repo.clear_relationships()

    def count_relationships(self) -> int:
        return self._rel_repo.count_relationships()

    # ── Scenes ──────────────────────────────────────────────────────

    def save_scene(self, scene: SceneSummary) -> None:
        self._scenes_repo.save_scene(scene)

    def get_scenes(self) -> list[SceneSummary]:
        return self._scenes_repo.get_scenes()

    def get_scene_macro_context(self, scene_id: str) -> SceneMacroContext | None:
        return self._scenes_repo.get_scene_macro_context(scene_id)

    get_scene_context = get_scene_macro_context

    # ── QA Audit Log ────────────────────────────────────────────────

    def log_qa_audit(
        self,
        unit_uid: str,
        stage: str,
        verdict: str,
        issue_type: str = "",
        explanation: str = "",
        suggested_correction: str = "",
        confidence: float = 0.0,
    ) -> int:
        return self._audit_repo.log_audit(
            unit_uid=unit_uid,
            stage=stage,
            verdict=verdict,
            issue_type=issue_type,
            explanation=explanation,
            suggested_correction=suggested_correction,
            confidence=confidence,
        )

    def get_qa_audit_logs(self, unit_uid: str) -> list[dict[str, Any]]:
        return self._audit_repo.get_audit_logs_for_unit(unit_uid)

    # ── Batches ─────────────────────────────────────────────────────

    def save_batch(self, batch: BatchLog) -> None:
        self._batch_repo.save(batch)

    def sanitize_stored_translations(self) -> tuple[int, int]:
        """Sanitize existing units and TM entries by stripping hallucinated speaker tags and
        quotes."""
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
    """Bidirectionally synchronize character profiles between SQLite and characters.json."""
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
