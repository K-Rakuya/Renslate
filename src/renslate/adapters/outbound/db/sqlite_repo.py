"""Outbound adapter connecting the application ports to SQLite storage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from renslate.core.db import Database
from renslate.core.models import Status, Unit
from renslate.domain.models import TranslationResult, UnitFailure

__all__ = ["SQLiteUnitRepositoryAdapter"]


@dataclass(slots=True)
class SQLiteUnitRepositoryAdapter:
    """Implements UnitRepositoryPort via Renslate's Database (SQLite WAL mode)."""

    db: Database

    def get_pending_units(self, limit: int, offset_seq: int = 0) -> Sequence[Unit]:
        return self.db.pending_units(limit=limit)

    def save_batch_results(self, results: Sequence[TranslationResult]) -> None:
        if not results:
            return
        with self.db.transaction():
            for r in results:
                self.db.update_translation(
                    uid=r.unit.uid,
                    target=r.target_text,
                    qa_flags=r.qa_flags,
                    status=r.status,
                )

    def mark_failed_units(self, failures: Sequence[UnitFailure]) -> None:
        if not failures:
            return
        with self.db.transaction():
            for f in failures:
                self.db.mark_failed(
                    uid=f.unit.uid,
                    error_reason=f"error:{f.failure_type}: {f.message}",
                )

    def get_total_pending_count(self) -> int:
        return len(self.db.units_by_status(Status.PENDING))
