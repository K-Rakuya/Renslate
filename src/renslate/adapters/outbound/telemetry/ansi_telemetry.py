"""Outbound adapter for telemetry and event dispatching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["RichAnsiTelemetryAdapter"]


@dataclass(slots=True)
class RichAnsiTelemetryAdapter:
    """Dispatches operational events to logging or TUI callbacks."""

    on_progress: Callable[[int, int, int, int], None] | None = None
    on_failure: Callable[[str, int, str], None] | None = None

    def emit_progress(
        self,
        translated_count: int,
        total_count: int,
        failed_count: int,
        batch_tokens: int,
    ) -> None:
        if self.on_progress is not None:
            self.on_progress(translated_count, total_count, failed_count, batch_tokens)

    def emit_batch_failure(
        self,
        batch_id: str,
        failed_count: int,
        retry_level: str,
    ) -> None:
        if self.on_failure is not None:
            self.on_failure(batch_id, failed_count, retry_level)
