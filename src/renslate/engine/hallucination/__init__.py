"""Continuous LLM Hallucination Guard pipeline package."""

from __future__ import annotations

from renslate.engine.hallucination.audit import persist_hallucination_audit
from renslate.engine.hallucination.feedback import (
    format_hallucination_feedback,
    scan_results_to_unit_failures,
)
from renslate.engine.hallucination.models import (
    HallucinationBatchReport,
    HallucinationScanResult,
)
from renslate.engine.hallucination.resilience import (
    build_fail_open_batch_report,
    fail_open_results,
    is_fail_open_exception,
)
from renslate.engine.hallucination.scanner import (
    HALLUCINATION_GUARD_JSON_SCHEMA,
    HALLUCINATION_GUARD_SYSTEM_PROMPT,
    HallucinationGuard,
)
from renslate.engine.hallucination.taxonomy import (
    ALL_CATEGORIES,
    CATEGORY_ALIASES,
    CONTEXT_LEAK,
    FOREIGN_ARTIFACT,
    PRONOUN_ERROR,
    SEMANTIC_INVERSION,
    SPEAKER_MISATTRIBUTION,
    TERMINOLOGY_DRIFT,
    TONE_REGISTER_VIOLATION,
    HallucinationCategory,
    is_valid_category,
    normalize_category,
)

__all__ = [
    "ALL_CATEGORIES",
    "CATEGORY_ALIASES",
    "CONTEXT_LEAK",
    "FOREIGN_ARTIFACT",
    "HALLUCINATION_GUARD_JSON_SCHEMA",
    "HALLUCINATION_GUARD_SYSTEM_PROMPT",
    "PRONOUN_ERROR",
    "SEMANTIC_INVERSION",
    "SPEAKER_MISATTRIBUTION",
    "TERMINOLOGY_DRIFT",
    "TONE_REGISTER_VIOLATION",
    "HallucinationBatchReport",
    "HallucinationCategory",
    "HallucinationGuard",
    "HallucinationScanResult",
    "build_fail_open_batch_report",
    "fail_open_results",
    "format_hallucination_feedback",
    "is_fail_open_exception",
    "is_valid_category",
    "normalize_category",
    "persist_hallucination_audit",
    "scan_results_to_unit_failures",
]
