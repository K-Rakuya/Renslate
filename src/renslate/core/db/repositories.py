"""Domain repositories for SQLite operations: units, characters, relationships, and scenes."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from typing import Any

from renslate.core.models import (
    ApplyTarget,
    BatchLog,
    CharacterProfile,
    GlossaryEntry,
    Origin,
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

_LOG = logging.getLogger("renslate.core.db.repositories")

__all__ = [
    "BatchRepository",
    "CharacterRepository",
    "GlossaryRepository",
    "QAAuditRepository",
    "RelationshipRepository",
    "SceneRepository",
    "TMRepository",
    "UnitRepository",
    "utc_timestamp",
]

_UNIT_COLUMNS = (
    "uid, kind, seq, ref, speaker, label, source, masked_source, target, status, origin, "
    "qa_flags, updated_at"
)

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
    "INSERT OR IGNORE INTO slots (uid, file, line, anchor_line, fingerprint) VALUES (?, ?, ?, ?, ?)"
)
_SQL_SELECT_SLOTS = (
    "SELECT id, uid, file, line, anchor_line, fingerprint FROM slots ORDER BY file, line"
)
_SQL_SELECT_SLOTS_BY_UID = (
    "SELECT id, uid, file, line, anchor_line, fingerprint FROM slots WHERE uid = ? ORDER BY line"
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

_SQL_SELECT_TM = (
    "SELECT src_norm, target, uses, origin, updated_at FROM tm "
    "WHERE src_norm = ? OR lower(src_norm) = lower(?)"
)
_SQL_SELECT_ALL_TM = "SELECT src_norm, target, uses, origin, updated_at FROM tm ORDER BY uses DESC"
_SQL_UPSERT_TM = """
INSERT INTO tm (src_norm, target, uses, origin, updated_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (src_norm) DO UPDATE SET
    target = excluded.target,
    uses = tm.uses + 1,
    origin = excluded.origin,
    updated_at = excluded.updated_at
"""
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

_RELATIONSHIP_COLUMNS = (
    "source_tag, target_tag, kind, speaker_pronoun, listener_pronoun, "
    "confidence, source_type, notes"
)
_SQL_UPSERT_RELATIONSHIP = f"""
INSERT INTO character_relationships ({_RELATIONSHIP_COLUMNS})
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (source_tag, target_tag) DO UPDATE SET
    kind = excluded.kind,
    speaker_pronoun = excluded.speaker_pronoun,
    listener_pronoun = excluded.listener_pronoun,
    confidence = excluded.confidence,
    source_type = excluded.source_type,
    notes = excluded.notes
"""
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


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unit_from_tuple(row: tuple[Any, ...] | sqlite3.Row) -> Unit:
    origin = row[10]
    qa_raw = row[11]
    qa_flags: tuple[str, ...]
    if isinstance(qa_raw, str):
        try:
            qa_flags = tuple(json.loads(qa_raw))
        except (json.JSONDecodeError, TypeError):
            qa_flags = ()
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


class UnitRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_unit(self, uid: str) -> Unit | None:
        row = self._conn.execute(_SQL_SELECT_BY_UID, (uid,)).fetchone()
        return _unit_from_tuple(tuple(row)) if row else None

    def all_units(self) -> list[Unit]:
        return list(self.iter_units())

    def units(self) -> dict[str, Unit]:
        return {u.uid: u for u in self.iter_units()}

    def units_by_status(self, status: Status) -> list[Unit]:
        return list(self.iter_units(status=status))

    def iter_units(self, status: Status | None = None) -> Iterator[Unit]:
        cursor = self._conn.cursor()
        if status is not None:
            cursor.execute(_SQL_SELECT_BY_STATUS_ORDERED, (status.value,))
        else:
            cursor.execute(_SQL_SELECT_ALL_ORDERED)
        for row in cursor:
            yield _unit_from_tuple(tuple(row))

    def iter_pending_units(
        self, limit: int | None = None, *, include_failed: bool = False
    ) -> Iterator[Unit]:
        cursor = self._conn.cursor()
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

    def pending_units(
        self, limit: int | None = None, *, include_failed: bool = False
    ) -> list[Unit]:
        return list(self.iter_pending_units(limit=limit, include_failed=include_failed))

    def save_unit(self, unit: Unit) -> None:
        self.save_units_bulk([unit])

    def save_units(self, units: Sequence[Unit]) -> None:
        self.save_units_bulk(units)

    def save_units_bulk(self, units: Sequence[Unit]) -> None:
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
                u.updated_at or utc_timestamp(),
            )
            for u in units
        ]
        self._conn.executemany(_SQL_INSERT_UNIT, params)

    def update_translation(
        self,
        uid: str,
        target: str,
        origin: Origin = Origin.LLM,
        status: Status = Status.TRANSLATED,
        qa_flags: tuple[str, ...] | Sequence[str] = (),
        updated_at: str | None = None,
    ) -> None:
        stamp = updated_at or utc_timestamp()
        self._conn.execute(
            _SQL_UPDATE_TRANSLATION,
            (target, origin.value, status.value, json.dumps(list(qa_flags)), stamp, uid),
        )

    def mark_failed(self, uid: str, error_reason: str | Sequence[str]) -> None:
        stamp = utc_timestamp()
        flags = [f"error:{error_reason}"] if isinstance(error_reason, str) else list(error_reason)
        self._conn.execute(
            _SQL_MARK_FAILED,
            (json.dumps(flags), stamp, uid),
        )

    def reset_failed(self) -> int:
        cur = self._conn.execute(_SQL_RESET_FAILED)
        return int(cur.rowcount)

    def replace_slots(self, slots: Iterable[Slot]) -> None:
        slot_params = [(s.uid, s.file, s.line, s.anchor_line, s.fingerprint) for s in slots]
        self._conn.execute(_SQL_DELETE_SLOTS)
        if slot_params:
            self._conn.executemany(_SQL_INSERT_SLOT, slot_params)

    def save_slots(self, slots: Sequence[Slot]) -> None:
        self.replace_slots(slots)

    def get_slots(self) -> list[Slot]:
        rows = self._conn.execute(_SQL_SELECT_SLOTS).fetchall()
        return [
            Slot(
                uid=r["uid"],
                file=r["file"],
                line=r["line"],
                anchor_line=r["anchor_line"],
                fingerprint=r["fingerprint"],
            )
            for r in rows
        ]

    def get_slots_for_unit(self, uid: str) -> list[Slot]:
        rows = self._conn.execute(_SQL_SELECT_SLOTS_BY_UID, (uid,)).fetchall()
        return [
            Slot(
                uid=r["uid"],
                file=r["file"],
                line=r["line"],
                anchor_line=r["anchor_line"],
                fingerprint=r["fingerprint"],
            )
            for r in rows
        ]

    def apply_targets(self) -> list[ApplyTarget]:
        rows = self._conn.execute(_SQL_SELECT_APPLY_TARGETS).fetchall()
        return [
            ApplyTarget(
                Slot(r["uid"], r["file"], r["line"], r["anchor_line"], r["fingerprint"]),
                UnitKind(r["kind"]),
                r["target"],
            )
            for r in rows
        ]

    def mark_orphans_obsolete(self, timestamp: str) -> int:
        cur = self._conn.execute(_SQL_MARK_ORPHANS_OBSOLETE, (timestamp,))
        return int(cur.rowcount)

    def update_fingerprints(self, updates: Iterable[tuple[str, int, str]]) -> None:
        self._conn.executemany(
            _SQL_UPDATE_SLOT_FINGERPRINT,
            [(fingerprint, file, line) for file, line, fingerprint in updates],
        )

    def count_slots(self) -> int:
        row = self._conn.execute(_SQL_COUNT_SLOTS).fetchone()
        return int(row[0]) if row else 0

    def stats(self) -> list[tuple[UnitKind, Status, int]]:
        rows = self._conn.execute(_SQL_SELECT_STATS).fetchall()
        return [(UnitKind(r["kind"]), Status(r["status"]), int(r["total"])) for r in rows]

    def recent_translated_units(
        self, limit: int = 16, before_seq: int | None = None
    ) -> list[Unit]:
        if before_seq is not None:
            rows = self._conn.execute(
                _SQL_SELECT_RECENT_BEFORE_SEQ, (before_seq, int(limit))
            ).fetchall()
        else:
            rows = self._conn.execute(_SQL_SELECT_RECENT, (int(limit),)).fetchall()
        return [_unit_from_tuple(tuple(r)) for r in reversed(rows)]

    def recent_context(self, before_seq: int, limit: int = 4) -> list[Unit]:
        return self.recent_translated_units(limit=limit, before_seq=before_seq)


class CharacterRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_characters(self) -> dict[str, CharacterProfile]:
        rows = self._conn.execute(_SQL_SELECT_CHARACTERS).fetchall()
        res: dict[str, CharacterProfile] = {}
        for r in rows:
            traits_raw = r["personality_traits"]
            traits = tuple(json.loads(traits_raw)) if traits_raw else ()
            dialogue_raw = r["sample_dialogue"]
            dialogue = tuple(json.loads(dialogue_raw)) if dialogue_raw else ()
            pronoun_raw = r["pronoun_map"]
            pronoun_map = json.loads(pronoun_raw) if pronoun_raw else {}

            res[r["tag"]] = CharacterProfile(
                tag=r["tag"],
                display_name=r["display_name"],
                gender=r["gender"],
                role=r["role"],
                speech_style=r["speech_style"],
                pronoun_map=pronoun_map,
                archetype=r["archetype"],
                personality_traits=traits,
                tone_and_voice=r["tone_and_voice"],
                socioeconomic_status=r["socioeconomic_status"],
                sample_dialogue=dialogue,
            )
        return res

    def save_character(self, profile: CharacterProfile) -> None:
        self.save_characters_bulk([profile])

    def save_characters_bulk(self, profiles: Sequence[CharacterProfile]) -> None:
        if not profiles:
            return
        payload = [
            (
                p.tag,
                p.display_name,
                p.gender,
                p.role,
                p.speech_style,
                json.dumps(p.pronoun_map, ensure_ascii=False),
                p.archetype,
                json.dumps(list(p.personality_traits), ensure_ascii=False),
                p.tone_and_voice,
                p.socioeconomic_status,
                json.dumps(list(p.sample_dialogue), ensure_ascii=False),
            )
            for p in profiles
        ]
        self._conn.executemany(_SQL_UPSERT_CHARACTER, payload)

    def clear_characters(self) -> int:
        cur = self._conn.execute("DELETE FROM characters")
        return int(cur.rowcount)


class RelationshipRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_relationship(self, source_tag: str, target_tag: str) -> RelationshipEdge | None:
        row = self._conn.execute(_SQL_SELECT_RELATIONSHIP, (source_tag, target_tag)).fetchone()
        return _relationship_from_row(row) if row else None

    def all_relationships(self) -> list[RelationshipEdge]:
        rows = self._conn.execute(_SQL_SELECT_ALL_RELATIONSHIPS).fetchall()
        return [_relationship_from_row(r) for r in rows]

    def get_relationships(
        self,
        source_tag: str | None = None,
        target_tag: str | None = None,
    ) -> list[RelationshipEdge]:
        if source_tag and target_tag:
            edge = self.get_relationship(source_tag, target_tag)
            return [edge] if edge is not None else []
        if source_tag:
            return self.relationships_for_source(source_tag)
        if target_tag:
            return self.relationships_for_target(target_tag)
        return self.all_relationships()

    def relationships_for_source(self, source_tag: str) -> list[RelationshipEdge]:
        rows = self._conn.execute(_SQL_SELECT_RELATIONSHIPS_BY_SOURCE, (source_tag,)).fetchall()
        return [_relationship_from_row(r) for r in rows]

    def relationships_for_target(self, target_tag: str) -> list[RelationshipEdge]:
        rows = self._conn.execute(_SQL_SELECT_RELATIONSHIPS_BY_TARGET, (target_tag,)).fetchall()
        return [_relationship_from_row(r) for r in rows]

    def save_relationship(self, edge: RelationshipEdge) -> None:
        self._conn.execute(
            _SQL_UPSERT_RELATIONSHIP,
            (
                edge.source_tag,
                edge.target_tag,
                edge.kind.value,
                edge.speaker_pronoun,
                edge.listener_pronoun,
                edge.confidence,
                edge.source_type,
                edge.notes,
            ),
        )

    def save_relationships_bulk(self, edges: Sequence[RelationshipEdge]) -> None:
        if not edges:
            return
        payload = [
            (
                e.source_tag,
                e.target_tag,
                e.kind.value,
                e.speaker_pronoun,
                e.listener_pronoun,
                e.confidence,
                e.source_type,
                e.notes,
            )
            for e in edges
        ]
        self._conn.executemany(_SQL_UPSERT_RELATIONSHIP, payload)

    def delete_relationship(self, source_tag: str, target_tag: str) -> bool:
        cur = self._conn.execute(_SQL_DELETE_RELATIONSHIP, (source_tag, target_tag))
        return cur.rowcount > 0

    def clear_relationships(self) -> int:
        cur = self._conn.execute(_SQL_DELETE_ALL_RELATIONSHIPS)
        return int(cur.rowcount)

    def count_relationships(self) -> int:
        row = self._conn.execute(_SQL_COUNT_RELATIONSHIPS).fetchone()
        return int(row[0]) if row else 0


class SceneRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save_scene(self, scene: SceneSummary) -> None:
        self._conn.execute(
            _SQL_UPSERT_SCENE,
            (scene.scene_id, scene.label, scene.first_seq, scene.last_seq, scene.summary),
        )

    def get_scenes(self) -> list[SceneSummary]:
        rows = self._conn.execute(_SQL_SELECT_SCENES).fetchall()
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

    def get_scene_macro_context(self, scene_id: str) -> SceneMacroContext | None:
        """Dual-read scene macro context: normalized columns first, then meta fallback."""
        row = self._conn.execute(
            """
            SELECT scene_id, label, first_seq, last_seq, summary, location, mood,
                   atmosphere, characters_json, temporal_anchor, previous_summary
            FROM scenes WHERE scene_id = ?
            """,
            (scene_id,),
        ).fetchone()

        if row and (row["location"] or row["atmosphere"] or row["mood"] or row["summary"]):
            chars_raw = row["characters_json"]
            chars: tuple[str, ...] = ()
            if chars_raw:
                try:
                    chars = tuple(json.loads(chars_raw))
                except Exception:
                    chars = ()
            metadata: dict[str, Any] = {}
            if row["previous_summary"]:
                metadata["previous_summary"] = row["previous_summary"]

            return SceneMacroContext(
                scene_id=row["scene_id"],
                label=row["label"],
                summary=row["summary"],
                location=row["location"],
                atmosphere=row["atmosphere"] or row["mood"],
                active_characters=chars,
                temporal_anchor=row["temporal_anchor"],
                first_seq=row["first_seq"],
                last_seq=row["last_seq"],
                metadata=metadata,
            )

        # Fallback to meta scene_ctx:<scene_id>
        meta_row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (f"scene_ctx:{scene_id}",)
        ).fetchone()
        if not meta_row:
            return None

        try:
            data = json.loads(meta_row["value"])
            chars = tuple(data.get("characters", []))
            metadata = {}
            if data.get("previous_summary"):
                metadata["previous_summary"] = data["previous_summary"]
            return SceneMacroContext(
                scene_id=scene_id,
                label=str(data.get("label", "")),
                summary=str(data.get("summary", "")),
                location=str(data.get("location", "")),
                atmosphere=str(data.get("atmosphere", "") or data.get("mood", "")),
                active_characters=chars,
                temporal_anchor=str(data.get("time", "")),
                first_seq=int(data.get("first_seq", 0)),
                last_seq=int(data.get("last_seq", 0)),
                metadata=metadata,
            )
        except Exception:
            return None

    get_scene_context = get_scene_macro_context


class QAAuditRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def log_audit(
        self,
        unit_uid: str,
        stage: str,
        verdict: str,
        issue_type: str = "",
        explanation: str = "",
        suggested_correction: str = "",
        confidence: float = 0.0,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO qa_audit_log (
                unit_uid, stage, verdict, issue_type, explanation,
                suggested_correction, confidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                unit_uid,
                stage,
                verdict,
                issue_type,
                explanation,
                suggested_correction,
                confidence,
                utc_timestamp(),
            ),
        )
        return int(cur.lastrowid or 0)

    def get_audit_logs_for_unit(self, unit_uid: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, unit_uid, stage, verdict, issue_type, explanation,
                   suggested_correction, confidence, created_at
            FROM qa_audit_log WHERE unit_uid = ? ORDER BY id
            """,
            (unit_uid,),
        ).fetchall()
        return [dict(r) for r in rows]


class TMRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get(self, src_norm: str) -> TMEntry | None:
        row = self._conn.execute(_SQL_SELECT_TM, (src_norm, src_norm)).fetchone()
        if not row:
            return None
        return TMEntry(
            src_norm=row["src_norm"],
            target=row["target"],
            uses=row["uses"],
            origin=Origin(row["origin"]),
            updated_at=row["updated_at"],
        )

    def save(self, entry: TMEntry) -> None:
        self._conn.execute(
            _SQL_UPSERT_TM,
            (
                entry.src_norm,
                entry.target,
                entry.uses,
                entry.origin.value,
                entry.updated_at or utc_timestamp(),
            ),
        )

    def all_entries(self) -> list[TMEntry]:
        rows = self._conn.execute(_SQL_SELECT_ALL_TM).fetchall()
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


class BatchRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, batch: BatchLog) -> None:
        stamp = batch.created_at or utc_timestamp()
        self._conn.execute(
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


class GlossaryRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get_all(self) -> list[GlossaryEntry]:
        rows = self._conn.execute(_SQL_SELECT_GLOSSARY).fetchall()
        res: list[GlossaryEntry] = []
        for r in rows:
            aliases_raw = r["aliases"]
            aliases = tuple(json.loads(aliases_raw)) if aliases_raw else ()
            res.append(
                GlossaryEntry(
                    term=r["term"],
                    translation=r["translation"],
                    kind=r["kind"],
                    case_sensitive=bool(r["case_sensitive"]),
                    aliases=aliases,
                    notes=r["notes"],
                )
            )
        return res

    def save(self, entry: GlossaryEntry) -> None:
        self._conn.execute(
            _SQL_UPSERT_GLOSSARY,
            (
                entry.term,
                entry.translation,
                entry.kind,
                1 if entry.case_sensitive else 0,
                json.dumps(list(entry.aliases), ensure_ascii=False),
                entry.notes,
            ),
        )

    def save_bulk(self, entries: Sequence[GlossaryEntry]) -> None:
        if not entries:
            return
        payload = [
            (
                e.term,
                e.translation,
                e.kind,
                1 if e.case_sensitive else 0,
                json.dumps(list(e.aliases), ensure_ascii=False),
                e.notes,
            )
            for e in entries
        ]
        self._conn.executemany(_SQL_UPSERT_GLOSSARY, payload)
