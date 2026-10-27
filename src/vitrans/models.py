"""Domain types shared by the parser, the store and the pipeline stages."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class UnitKind(str, enum.Enum):
    DIALOGUE = "dialogue"
    STRING = "string"
    UI = "ui"


class Status(str, enum.Enum):
    PENDING = "pending"
    TRANSLATED = "translated"
    QA_FLAGGED = "qa_flagged"
    REVIEWED = "reviewed"
    OBSOLETE = "obsolete"
    FAILED = "failed"


class Origin(str, enum.Enum):
    LLM = "llm"
    TM = "tm"
    MANUAL = "manual"
    SEED = "seed"
    IMPORTED = "imported"


LOCKED_ORIGINS = frozenset({Origin.MANUAL, Origin.IMPORTED})


@dataclass(frozen=True, slots=True)
class Unit:
    """A translatable text, identified by uid and independent of where it is written."""

    uid: str
    kind: UnitKind
    seq: int
    source: str
    status: Status
    ref: str | None = None
    speaker: str | None = None
    label: str | None = None
    masked_source: str | None = None
    target: str | None = None
    origin: Origin | None = None
    qa_flags: tuple[str, ...] = ()
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class Slot:
    """A line in a translation file that receives the target text of a unit.

    ``anchor_line`` points to the line that identifies the slot (the ``translate`` header for
    dialogue, the ``old`` line for strings). ``fingerprint`` covers the anchor and the target
    line so that a file edited since extraction is detected before anything is overwritten.
    """

    uid: str
    file: str
    line: int
    anchor_line: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ApplyTarget:
    slot: Slot
    kind: UnitKind
    target: str


@dataclass(frozen=True, slots=True)
class TMEntry:
    """Translation memory entry for reuse and suggestions."""

    src_norm: str
    target: str
    uses: int = 1
    origin: Origin = Origin.LLM
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    """Fixed glossary term definition."""

    term: str
    translation: str
    kind: str = "general"
    case_sensitive: bool = False
    aliases: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CharacterProfile:
    """Character personality, speech style, and addressing rules."""

    tag: str
    display_name: str
    gender: str = "unknown"
    role: str = ""
    speech_style: str = ""
    pronoun_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SceneSummary:
    """Summary of a dialogue scene for macro-context."""

    scene_id: str
    label: str = ""
    first_seq: int = 0
    last_seq: int = 0
    summary: str = ""


@dataclass(frozen=True, slots=True)
class BatchLog:
    """Execution log for an LLM translation batch."""

    batch_id: str
    unit_uids: tuple[str, ...]
    tokens_in: int = 0
    tokens_out: int = 0
    attempts: int = 1
    error: str | None = None
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class QAIssue:
    """A quality issue detected for a translated unit."""

    uid: str
    code: str
    severity: str  # "error", "warning", "info"
    message: str
