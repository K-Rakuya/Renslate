"""Use case orchestrating QA validation rules and continuous HallucinationGuard scans."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from renslate.core.models import Status, UnitKind
from renslate.domain.models import TranslationResult, UnitFailure

_LOG = logging.getLogger("renslate.pipeline.audit")

__all__ = ["QualityAuditUseCase"]


@dataclass(slots=True)
class QualityAuditUseCase:
    """Orchestrates QA syntactic rule checks and hallucination audit verification."""

    guard: Any | None = None

    def audit_batch_results(
        self,
        candidate_results: Sequence[TranslationResult],
        context_prompt: str = "",
    ) -> tuple[list[TranslationResult], list[UnitFailure]]:
        """Run HallucinationGuard on dialogue units and filter out hallucinations.

        Returns (accepted_results, failed_units).
        """
        if self.guard is None or not candidate_results:
            return list(candidate_results), []

        # Only audit actual dialogue units containing alphanumeric characters
        units_to_audit = [
            replace(r.unit, target=r.target_text)
            for r in candidate_results
            if r.unit.kind == UnitKind.DIALOGUE
            and any(c.isalnum() for c in r.unit.source)
            and not (
                r.unit.source.strip().startswith("[")
                and r.unit.source.strip().endswith("]")
                and ":" in r.unit.source
            )
        ]

        if not units_to_audit:
            return list(candidate_results), []

        guard_results = self.guard.scan_batch(units_to_audit, context_window=context_prompt)
        hal_by_id = {res.unit_id: res for res in guard_results if res.has_hallucination}

        accepted: list[TranslationResult] = []
        failures: list[UnitFailure] = []

        for r in candidate_results:
            hal = hal_by_id.get(r.unit.uid)
            if hal is not None and hal.confidence >= 0.70:
                _LOG.info(
                    "Dòng %s bị HallucinationGuard gắn cờ %s (độ tin cậy %.2f): %s",
                    r.unit.uid,
                    hal.category,
                    hal.confidence,
                    hal.explanation,
                )
                detail = hal.explanation
                if hal.suggested_correction:
                    detail = f"{detail} (Gợi ý: {hal.suggested_correction})"
                failures.append(
                    UnitFailure(
                        r.unit,
                        hal.category or "hallucination",
                        detail,
                        r.target_text,
                    )
                )
            else:
                flags = r.qa_flags
                status = r.status
                if hal is not None:
                    flags = tuple(sorted(set(flags) | {f"qa_hallucination_{hal.category.lower()}"}))
                    status = Status.QA_FLAGGED
                accepted.append(
                    TranslationResult(
                        unit=r.unit,
                        target_text=r.target_text,
                        qa_flags=flags,
                        status=status,
                    )
                )

        return accepted, failures
