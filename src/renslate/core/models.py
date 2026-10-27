"""Domain types shared by the parser, the store and the pipeline stages."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


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


class RelationshipKind(str, enum.Enum):
    """Semantic relationship classification between characters."""

    FAMILY_ELDER = "FAMILY_ELDER"
    FAMILY_SIBLING = "FAMILY_SIBLING"
    ROMANCE = "ROMANCE"
    MENTOR_DISCIPLE = "MENTOR_DISCIPLE"
    SUPERIOR_SUBORDINATE = "SUPERIOR_SUBORDINATE"
    FRIENDSHIP_CLOSE = "FRIENDSHIP_CLOSE"
    FRIENDSHIP_CASUAL = "FRIENDSHIP_CASUAL"
    FORMAL_ACQUAINTANCE = "FORMAL_ACQUAINTANCE"
    RIVAL_HOSTILE = "RIVAL_HOSTILE"
    SERVANT_MASTER = "SERVANT_MASTER"

    @classmethod
    def from_str(cls, value: str) -> RelationshipKind:
        """Parse string to RelationshipKind case-insensitively and resiliently."""
        norm = value.strip().upper().replace(" ", "_").replace("-", "_")
        try:
            return cls(norm)
        except ValueError:
            if norm in cls.__members__:
                return cls[norm]
            msg = f"Unknown relationship kind: {value!r}"
            raise ValueError(msg) from None


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
class RelationshipEdge:
    """A directed relationship edge from source_tag to target_tag with addressing pronouns."""

    source_tag: str
    target_tag: str
    kind: RelationshipKind
    speaker_pronoun: str = ""
    listener_pronoun: str = ""
    confidence: float = 1.0
    source_type: str = "manual"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize relationship edge to dictionary."""
        return {
            "source_tag": self.source_tag,
            "target_tag": self.target_tag,
            "kind": self.kind.value,
            "speaker_pronoun": self.speaker_pronoun,
            "listener_pronoun": self.listener_pronoun,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RelationshipEdge:
        """Deserialize dictionary to RelationshipEdge."""
        if not isinstance(data, (dict, Mapping)):
            msg = f"Expected dict or Mapping for RelationshipEdge, got {type(data).__name__}"
            raise TypeError(msg)

        source_tag = str(data["source_tag"])
        target_tag = str(data["target_tag"])
        kind_raw = data["kind"]
        if isinstance(kind_raw, RelationshipKind):
            kind = kind_raw
        elif isinstance(kind_raw, str):
            kind = RelationshipKind.from_str(kind_raw)
        else:
            msg = f"Expected RelationshipKind or str for kind, got {type(kind_raw).__name__}"
            raise TypeError(msg)

        return cls(
            source_tag=source_tag,
            target_tag=target_tag,
            kind=kind,
            speaker_pronoun=str(data.get("speaker_pronoun", "")),
            listener_pronoun=str(data.get("listener_pronoun", "")),
            confidence=float(data.get("confidence", 1.0)),
            source_type=str(data.get("source_type", "manual")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class CharacterProfile:
    """Character personality, speech style, and addressing rules."""

    tag: str
    display_name: str
    gender: str = "unknown"
    role: str = ""
    speech_style: str = ""
    pronoun_map: dict[str, str] = field(default_factory=dict)
    archetype: str = ""
    personality_traits: tuple[str, ...] = ()
    tone_and_voice: str = ""
    socioeconomic_status: str = ""
    sample_dialogue: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.personality_traits, tuple):
            object.__setattr__(
                self,
                "personality_traits",
                tuple(self.personality_traits) if self.personality_traits else (),
            )
        if not isinstance(self.sample_dialogue, tuple):
            object.__setattr__(
                self,
                "sample_dialogue",
                tuple(self.sample_dialogue) if self.sample_dialogue else (),
            )

    def to_dict(self, include_tag: bool = True) -> dict[str, Any]:
        """Serialize character profile to dictionary for JSON persistence."""
        d: dict[str, Any] = {
            "display_name": self.display_name,
            "gender": self.gender,
            "role": self.role,
            "speech_style": self.speech_style,
            "pronoun_map": dict(self.pronoun_map),
            "archetype": self.archetype,
            "personality_traits": list(self.personality_traits),
            "tone_and_voice": self.tone_and_voice,
            "socioeconomic_status": self.socioeconomic_status,
            "sample_dialogue": list(self.sample_dialogue),
        }
        if include_tag:
            return {"tag": self.tag, **d}
        return d

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        tag: str | None = None,
    ) -> CharacterProfile:
        """Deserialize dictionary to CharacterProfile.

        If tag is provided explicitly, it takes precedence over data['tag'].
        """
        if not isinstance(data, (dict, Mapping)):
            msg = f"Expected dict or Mapping for CharacterProfile, got {type(data).__name__}"
            raise TypeError(msg)

        resolved_tag = str(
            tag if tag is not None and str(tag).strip() else data.get("tag", "")
        ).strip()
        if not resolved_tag:
            msg = "CharacterProfile requires a non-empty 'tag'"
            raise ValueError(msg)

        raw_traits = data.get("personality_traits")
        if isinstance(raw_traits, (list, tuple)):
            traits = tuple(str(x) for x in raw_traits)
        elif isinstance(raw_traits, str) and raw_traits.strip():
            traits = tuple(x.strip() for x in raw_traits.split(",") if x.strip())
        else:
            traits = ()

        raw_dialogue = data.get("sample_dialogue")
        if isinstance(raw_dialogue, (list, tuple)):
            dialogue = tuple(str(x) for x in raw_dialogue)
        elif isinstance(raw_dialogue, str) and raw_dialogue.strip():
            dialogue = (raw_dialogue.strip(),)
        else:
            dialogue = ()

        raw_pronouns = data.get("pronoun_map")
        if isinstance(raw_pronouns, dict):
            pronoun_map = {str(k): str(v) for k, v in raw_pronouns.items()}
        else:
            pronoun_map = {}

        raw_disp = data.get("display_name")
        disp = (
            str(raw_disp).strip()
            if raw_disp is not None and str(raw_disp).strip()
            else resolved_tag
        )

        return cls(
            tag=resolved_tag,
            display_name=disp,
            gender=str(data.get("gender") or "unknown"),
            role=str(data.get("role") or ""),
            speech_style=str(data.get("speech_style") or ""),
            pronoun_map=pronoun_map,
            archetype=str(data.get("archetype") or ""),
            personality_traits=traits,
            tone_and_voice=str(data.get("tone_and_voice") or ""),
            socioeconomic_status=str(data.get("socioeconomic_status") or ""),
            sample_dialogue=dialogue,
        )


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


__all__ = [
    "LOCKED_ORIGINS",
    "ApplyTarget",
    "BatchLog",
    "CharacterProfile",
    "GlossaryEntry",
    "Origin",
    "QAIssue",
    "RelationshipEdge",
    "RelationshipKind",
    "SceneSummary",
    "Slot",
    "Status",
    "TMEntry",
    "Unit",
    "UnitKind",
]

