"""Outbound port interfaces (Protocols) for external dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from renslate.core.models import CharacterProfile, RelationshipEdge, Unit
from renslate.domain.models import TranslationResult, UnitFailure

__all__ = [
    "DiscourseRepositoryPort",
    "LLMClientPort",
    "TelemetryPort",
    "UnitRepositoryPort",
]


class UnitRepositoryPort(Protocol):
    """Abstraction for loading and persisting translation units and their audit states."""

    def get_pending_units(self, limit: int, offset_seq: int = 0) -> Sequence[Unit]:
        """Fetch a batch of pending units ordered by sequence."""
        ...

    def save_batch_results(self, results: Sequence[TranslationResult]) -> None:
        """Persist translated units atomically within a transaction."""
        ...

    def mark_failed_units(self, failures: Sequence[UnitFailure]) -> None:
        """Mark units as failed with failure codes and detailed messages."""
        ...

    def get_total_pending_count(self) -> int:
        """Return count of remaining pending units."""
        ...


class LLMClientPort(Protocol):
    """Abstraction for communicating with an inference backend (vLLM, llama.cpp, etc.)."""

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
        retries: int | None = None,
    ) -> tuple[dict[str, str], int, int]:
        """Generate a structured JSON completion; returns (data, prompt_tokens, completion_tokens)."""
        ...


class DiscourseRepositoryPort(Protocol):
    """Abstraction for querying character profiles and relational edges."""

    def get_characters(self) -> dict[str, CharacterProfile]:
        """Retrieve all registered character profiles."""
        ...

    def get_relationship_edge(self, speaker: str, listener: str) -> RelationshipEdge | None:
        """Retrieve the relational edge between speaker and listener if available."""
        ...


class TelemetryPort(Protocol):
    """Abstraction for progress reporting and observability sinks."""

    def emit_progress(
        self,
        translated_count: int,
        total_count: int,
        failed_count: int,
        batch_tokens: int,
    ) -> None:
        """Emit real-time translation metrics."""
        ...

    def emit_batch_failure(
        self,
        batch_id: str,
        failed_count: int,
        retry_level: str,
    ) -> None:
        """Emit notification when units in a batch fail."""
        ...
