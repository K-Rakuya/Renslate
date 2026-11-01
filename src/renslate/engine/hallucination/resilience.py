"""Fail-open resilience policies for LLM network timeouts and parse errors."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from http.client import RemoteDisconnected

from renslate.core.models import Unit
from renslate.engine.hallucination.models import (
    HallucinationBatchReport,
    HallucinationScanResult,
)

_LOG = logging.getLogger("renslate.engine.hallucination.resilience")

__all__ = [
    "build_fail_open_batch_report",
    "fail_open_results",
    "is_fail_open_exception",
]


def is_fail_open_exception(exc: BaseException) -> bool:
    """Determine if exception warrants fail-open fallback rather than hard error."""
    if isinstance(exc, (TimeoutError, json.JSONDecodeError, RemoteDisconnected, ConnectionError)):
        return True
    exc_name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in exc_name or "timeout" in msg:
        return True
    if "connect" in exc_name or "connection" in msg or "disconnected" in msg:
        return True
    if "json" in msg or "decode" in msg or "unparseable" in msg:
        return True
    return bool(
        "500" in msg
        or "502" in msg
        or "503" in msg
        or "504" in msg
        or "server error" in msg
    )


def fail_open_results(
    units: Sequence[Unit],
    reason: str,
    category: str = "SCAN_SKIPPED_TIMEOUT",
) -> list[HallucinationScanResult]:
    """Construct non-blocking pass verdicts under fail-open resilience policy."""
    _LOG.warning("Hallucination scan failed open for %d units: %s", len(units), reason)
    return [
        HallucinationScanResult(
            unit_id=u.uid,
            has_hallucination=False,
            category=category,
            confidence=0.0,
            explanation=f"Fail-open: {reason}",
            suggested_correction="",
            severity="warning",
        )
        for u in units
    ]


def build_fail_open_batch_report(
    batch_id: str,
    units: Sequence[Unit],
    reason: str,
) -> HallucinationBatchReport:
    """Construct a batch report for a failed-open scan pass."""
    results = fail_open_results(units, reason)
    return HallucinationBatchReport(
        batch_id=batch_id,
        results=tuple(results),
        has_hallucinations=False,
        scanned_count=len(units),
        flagged_count=0,
        scan_duration_ms=0.0,
        error=reason,
    )
