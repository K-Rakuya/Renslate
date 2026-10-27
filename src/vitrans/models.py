"""Domain types shared by the parser, the store and the pipeline stages."""

from __future__ import annotations

import enum
from dataclasses import dataclass


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
