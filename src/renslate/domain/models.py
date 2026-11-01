"""Domain models and value objects for the Renslate translation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from renslate.core.models import Status, Unit

__all__ = [
    "BatchExecutionResult",
    "DiscourseSnapshot",
    "QualityVerdict",
    "SpriteState",
    "StageSnapshot",
    "TranslationResult",
    "TranslationStats",
    "UnitFailure",
]


@dataclass(frozen=True, slots=True)
class SpriteState:
    """Represents the visual state of a character sprite on the stage."""

    tag: str
    expression: str = ""
    position: str = ""


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    """Snapshot of active sprites and scene backdrop at a specific narrative point."""

    scene_bg: str = ""
    active_sprites: tuple[SpriteState, ...] = ()

    @property
    def active_tags(self) -> set[str]:
        return {s.tag.lower() for s in self.active_sprites}


@dataclass(frozen=True, slots=True)
class DiscourseSnapshot:
    """Discourse framing for a dialogue turn."""

    speaker: str | None
    listener: str | None
    is_monologue: bool = False
    detected_emotion: str | None = None
    stage: StageSnapshot | None = None


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Domain outcome for an individual unit translation."""

    unit: Unit
    target_text: str
    qa_flags: tuple[str, ...] = ()
    status: Status = Status.TRANSLATED


@dataclass(frozen=True, slots=True)
class UnitFailure:
    """Detailed record of a translation failure for retry or diagnostic reporting."""

    unit: Unit
    failure_type: str
    message: str
    raw_output: str = ""


@dataclass(slots=True)
class TranslationStats:
    """Cumulative operational metrics for a translation run."""

    total_pending: int = 0
    seeded: int = 0
    tm_reused: int = 0
    tm_context_skipped: int = 0
    llm_translated: int = 0
    failed: int = 0
    batches_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BatchExecutionResult:
    """Outcome of processing a single translation batch."""

    success_results: tuple[TranslationResult, ...] = ()
    failures: tuple[UnitFailure, ...] = ()
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def success_count(self) -> int:
        return len(self.success_results)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


@dataclass(frozen=True, slots=True)
class QualityVerdict:
    """Verdict of a quality audit on translated text."""

    is_valid: bool
    qa_flags: tuple[str, ...] = ()
    error_code: str | None = None
    reason: str | None = None
