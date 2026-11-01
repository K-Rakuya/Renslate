"""Taxonomy of translation communication breakdowns and LLM hallucinations."""

from __future__ import annotations

from enum import Enum

__all__ = [
    "ALL_CATEGORIES",
    "CATEGORY_ALIASES",
    "CONTEXT_LEAK",
    "FOREIGN_ARTIFACT",
    "PRONOUN_ERROR",
    "SEMANTIC_INVERSION",
    "SPEAKER_MISATTRIBUTION",
    "TERMINOLOGY_DRIFT",
    "TONE_REGISTER_VIOLATION",
    "HallucinationCategory",
    "is_valid_category",
    "normalize_category",
]


class HallucinationCategory(str, Enum):
    """The 7 canonical hallucination and translation defect categories."""

    SPEAKER_MISATTRIBUTION = "SPEAKER_MISATTRIBUTION"
    SEMANTIC_INVERSION = "SEMANTIC_INVERSION"
    PRONOUN_ERROR = "PRONOUN_ERROR"
    TONE_REGISTER_VIOLATION = "TONE_REGISTER_VIOLATION"
    TERMINOLOGY_DRIFT = "TERMINOLOGY_DRIFT"
    CONTEXT_LEAK = "CONTEXT_LEAK"
    FOREIGN_ARTIFACT = "FOREIGN_ARTIFACT"


SPEAKER_MISATTRIBUTION = HallucinationCategory.SPEAKER_MISATTRIBUTION.value
SEMANTIC_INVERSION = HallucinationCategory.SEMANTIC_INVERSION.value
PRONOUN_ERROR = HallucinationCategory.PRONOUN_ERROR.value
TONE_REGISTER_VIOLATION = HallucinationCategory.TONE_REGISTER_VIOLATION.value
TERMINOLOGY_DRIFT = HallucinationCategory.TERMINOLOGY_DRIFT.value
CONTEXT_LEAK = HallucinationCategory.CONTEXT_LEAK.value
FOREIGN_ARTIFACT = HallucinationCategory.FOREIGN_ARTIFACT.value

ALL_CATEGORIES: frozenset[str] = frozenset(c.value for c in HallucinationCategory)

CATEGORY_ALIASES: dict[str, str] = {
    # RFC-001 and E2E specification synonyms
    "factual_hallucination": SEMANTIC_INVERSION,
    "critical_omission": SEMANTIC_INVERSION,
    "pronoun_inversion": PRONOUN_ERROR,
    "register_breakdown": TONE_REGISTER_VIOLATION,
    "emotional_distortion": TONE_REGISTER_VIOLATION,
    "continuity_breakdown": SPEAKER_MISATTRIBUTION,
    "leaked_artifacts": FOREIGN_ARTIFACT,
    "leaked_artifact": CONTEXT_LEAK,
    "pronoun_drift": PRONOUN_ERROR,
    # Uppercase variations
    "FACTUAL_HALLUCINATION": SEMANTIC_INVERSION,
    "CRITICAL_OMISSION": SEMANTIC_INVERSION,
    "PRONOUN_INVERSION": PRONOUN_ERROR,
    "REGISTER_BREAKDOWN": TONE_REGISTER_VIOLATION,
    "EMOTIONAL_DISTORTION": TONE_REGISTER_VIOLATION,
    "CONTINUITY_BREAKDOWN": SPEAKER_MISATTRIBUTION,
    "LEAKED_ARTIFACTS": FOREIGN_ARTIFACT,
    "SPEAKER_MISATTRIBUTION": SPEAKER_MISATTRIBUTION,
    "SEMANTIC_INVERSION": SEMANTIC_INVERSION,
    "PRONOUN_ERROR": PRONOUN_ERROR,
    "TONE_REGISTER_VIOLATION": TONE_REGISTER_VIOLATION,
    "TERMINOLOGY_DRIFT": TERMINOLOGY_DRIFT,
    "CONTEXT_LEAK": CONTEXT_LEAK,
    "FOREIGN_ARTIFACT": FOREIGN_ARTIFACT,
}


def normalize_category(category: str) -> str:
    """Normalize a raw issue type string into its canonical HallucinationCategory."""
    c = category.strip()
    upper = c.upper()
    if upper in ALL_CATEGORIES:
        return upper
    lower = c.lower()
    if lower in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[lower]
    if upper in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[upper]
    return upper if upper else "UNKNOWN"


def is_valid_category(category: str) -> bool:
    """Check if category string is a recognized canonical category or approved alias."""
    c = category.strip()
    return (
        c.upper() in ALL_CATEGORIES
        or c.lower() in CATEGORY_ALIASES
        or c.upper() in CATEGORY_ALIASES
    )
