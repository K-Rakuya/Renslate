"""Bridge between Hallucination Guard detections and RetryLadder repair loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from renslate.core.models import Unit
from renslate.engine.hallucination.models import HallucinationScanResult
from renslate.engine.repair import UnitFailure, format_failure_feedback

__all__ = [
    "format_hallucination_feedback",
    "scan_results_to_unit_failures",
]


def scan_results_to_unit_failures(
    results: Sequence[HallucinationScanResult],
    units_by_id: Mapping[str, Unit],
    min_confidence: float = 0.70,
) -> list[UnitFailure]:
    """Convert scan detections meeting confidence threshold into RetryLadder UnitFailure objects."""
    failures: list[UnitFailure] = []
    for r in results:
        if not r.has_hallucination:
            continue
        if r.confidence < min_confidence:
            continue
        unit = units_by_id.get(r.unit_id)
        if unit is None:
            continue
        detail = r.explanation
        if r.suggested_correction:
            detail = f"{detail} (Gợi ý: {r.suggested_correction})"
        failures.append(
            UnitFailure(
                unit=unit,
                code=r.category or "hallucination",
                detail=detail,
                candidate=unit.target or "",
            )
        )
    return failures


def format_hallucination_feedback(failures: Sequence[UnitFailure]) -> str:
    """Format negative feedback instructions for the next translation retry prompt."""
    return format_failure_feedback(failures)
