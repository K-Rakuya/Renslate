"""Domain types shared by the parser, the store and the pipeline stages."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
class SceneMacroContext:
    """Hierarchical macro context of a dialogue scene for prompt enrichment.

    Encapsulates scene boundaries, physical setting, atmosphere, active participants,
    and narrative summary to guide LLM persona consistency across long script arcs.
    """

    scene_id: str
    label: str
    file: str = ""
    summary: str = ""
    location: str = ""
    atmosphere: str = ""
    active_characters: tuple[str, ...] = ()
    temporal_anchor: str = ""
    first_seq: int = 0
    last_seq: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.active_characters, tuple):
            object.__setattr__(
                self,
                "active_characters",
                tuple(self.active_characters) if self.active_characters else (),
            )

    def is_empty(self) -> bool:
        """Return True if no narrative context is populated."""
        return not (
            self.summary
            or self.location
            or self.atmosphere
            or self.temporal_anchor
            or self.active_characters
        )

    def format_prompt_header(self) -> str:
        """Format macro context into standardized Vietnamese prompt header block for LLM."""
        lines = ["BỐI CẢNH PHÂN CẢNH & TÓM TẮT CẢNH HIỆN TẠI (SCENE MACRO CONTEXT):"]
        if self.label:
            lines.append(f"- Phân cảnh (Scene Label): {self.label}")
        if self.location:
            lines.append(f"- Địa điểm (Location): {self.location}")
        if self.atmosphere:
            lines.append(f"- Không khí / Cảm xúc (Atmosphere): {self.atmosphere}")
        if self.temporal_anchor:
            lines.append(f"- Thời điểm (Time): {self.temporal_anchor}")
        if self.active_characters:
            lines.append(f"- Nhân vật hiện diện (Characters): {', '.join(self.active_characters)}")
        if self.summary:
            lines.append(f"- Tóm tắt tình huống (Summary): {self.summary}")
        return "\n".join(lines)

    def with_updates(self, **kwargs: Any) -> SceneMacroContext:
        """Return a copy of the scene macro context with updated fields."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scene macro context to dictionary for JSON persistence."""
        return {
            "scene_id": self.scene_id,
            "label": self.label,
            "file": self.file,
            "summary": self.summary,
            "location": self.location,
            "atmosphere": self.atmosphere,
            "active_characters": list(self.active_characters),
            "temporal_anchor": self.temporal_anchor,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SceneMacroContext:
        """Deserialize dictionary or Mapping to SceneMacroContext."""
        if not isinstance(data, (dict, Mapping)):
            msg = f"Expected dict or Mapping for SceneMacroContext, got {type(data).__name__}"
            raise TypeError(msg)

        scene_id = str(data.get("scene_id") or "").strip()
        if not scene_id:
            msg = "SceneMacroContext requires a non-empty 'scene_id'"
            raise ValueError(msg)

        label = str(data.get("label") or "").strip()
        file_path = str(data.get("file") or "").strip()
        summary = str(data.get("summary") or "").strip()
        location = str(data.get("location") or "").strip()
        atmosphere = str(data.get("atmosphere") or "").strip()
        temporal_anchor = str(data.get("temporal_anchor") or "").strip()

        raw_chars = data.get("active_characters")
        if isinstance(raw_chars, (list, tuple)):
            active_characters = tuple(str(c).strip() for c in raw_chars if str(c).strip())
        elif isinstance(raw_chars, str) and raw_chars.strip():
            active_characters = tuple(c.strip() for c in raw_chars.split(",") if c.strip())
        else:
            active_characters = ()

        raw_meta = data.get("metadata")
        metadata = dict(raw_meta) if isinstance(raw_meta, dict) else {}

        first_seq = int(data.get("first_seq") or 0)
        last_seq = int(data.get("last_seq") or 0)

        return cls(
            scene_id=scene_id,
            label=label,
            file=file_path,
            summary=summary,
            location=location,
            atmosphere=atmosphere,
            active_characters=active_characters,
            temporal_anchor=temporal_anchor,
            first_seq=first_seq,
            last_seq=last_seq,
            metadata=metadata,
        )


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
    "SceneMacroContext",
    "SceneSummary",
    "Slot",
    "Status",
    "TMEntry",
    "Unit",
    "UnitKind",
]

