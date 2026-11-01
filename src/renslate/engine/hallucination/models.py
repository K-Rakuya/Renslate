"""Domain data models for the LLM Hallucination Guard pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "HallucinationBatchReport",
    "HallucinationScanResult",
]


@dataclass(frozen=True, slots=True)
class HallucinationScanResult:
    """Individual unit hallucination scan verdict adhering to PROJECT.md interface contract."""

    unit_id: str
    has_hallucination: bool
    category: str = ""
    confidence: float = 0.0
    explanation: str = ""
    suggested_correction: str = ""
    severity: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "has_hallucination": self.has_hallucination,
            "category": self.category,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "suggested_correction": self.suggested_correction,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HallucinationScanResult:
        raw_conf = data.get("confidence")
        try:
            conf = float(raw_conf) if raw_conf is not None else 0.0
            if math.isnan(conf) or math.isinf(conf):
                conf = 1.0 if (data.get("has_hallucination") or data.get("category")) else 0.0
        except (ValueError, TypeError):
            conf = 0.0
        return cls(
            unit_id=str(data.get("unit_id") or data.get("uid") or ""),
            has_hallucination=bool(data.get("has_hallucination", False)),
            category=str(data.get("category") or data.get("issue_type") or ""),
            confidence=conf,
            explanation=str(data.get("explanation", "")),
            suggested_correction=str(data.get("suggested_correction", "")),
            severity=str(data.get("severity", "")),
        )


@dataclass(frozen=True, slots=True)
class HallucinationBatchReport:
    """Comprehensive batch scanning audit report."""

    batch_id: str
    results: tuple[HallucinationScanResult, ...] = ()
    has_hallucinations: bool = False
    scanned_count: int = 0
    flagged_count: int = 0
    scan_duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "results": [r.to_dict() for r in self.results],
            "has_hallucinations": self.has_hallucinations,
            "scanned_count": self.scanned_count,
            "flagged_count": self.flagged_count,
            "scan_duration_ms": self.scan_duration_ms,
            "error": self.error,
        }
