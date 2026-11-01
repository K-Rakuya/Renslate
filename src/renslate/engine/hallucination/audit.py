"""Audit persistence for Hallucination Guard scan results into SQLite qa_audit_log."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence

from renslate.core.db import Database
from renslate.engine.hallucination.models import HallucinationScanResult

_LOG = logging.getLogger("renslate.engine.hallucination.audit")

__all__ = [
    "persist_hallucination_audit",
]


def persist_hallucination_audit(
    db: Database,
    results: Sequence[HallucinationScanResult],
    stage: str = "llm_hallucination_guard",
) -> int:
    """Record hallucination scan results into the SQLite qa_audit_log table."""
    inserted = 0
    with db.transaction():
        for r in results:
            verdict = "FAIL" if r.has_hallucination else "PASS"
            if r.severity == "warning" and not r.has_hallucination:
                verdict = "WARNING"
            try:
                db.log_qa_audit(
                    unit_uid=r.unit_id,
                    stage=stage,
                    verdict=verdict,
                    issue_type=r.category or "none",
                    explanation=r.explanation,
                    suggested_correction=r.suggested_correction,
                    confidence=r.confidence,
                )
                inserted += 1
            except (sqlite3.IntegrityError, sqlite3.OperationalError) as err:
                _LOG.warning("Could not persist audit record for %s: %s", r.unit_id, err)
    return inserted
