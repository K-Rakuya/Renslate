"""Translation pipeline, local LLM client, context budgeting, QA, TM, and analysis."""

from __future__ import annotations

from renslate.engine.analyze import (
    bootstrap_characters_with_llm,
    extract_character_definitions,
    infer_character_gender_nlp,
    infer_character_role_nlp,
    run_analysis,
    segment_scenes,
)
from renslate.engine.context import (
    QUOTED_RULES,
    ContextBuilder,
    LookbackItem,
    estimate_tokens,
)
from renslate.engine.glossary import (
    Glossary,
    clean_glossary,
    is_valid_glossary_term,
    mine_glossary_candidates,
    translate_glossary_with_llm,
)
from renslate.engine.graph import RelationshipGraph
from renslate.engine.llm import LLMClient, extract_json_object
from renslate.engine.qa import (
    QAChecker,
    clean_translation,
    repair_glossary_leakage,
)
from renslate.engine.tm import FuzzyMatch, TranslationMemory, normalize_source
from renslate.engine.translate import (
    ProgressCallback,
    TranslationPipeline,
    TranslationStats,
    partition_batches,
)
from renslate.engine.tsvio import export_tsv, import_tsv

__all__ = [
    "QUOTED_RULES",
    # Context
    "ContextBuilder",
    "FuzzyMatch",
    # Glossary
    "Glossary",
    # LLM
    "LLMClient",
    "LookbackItem",
    "ProgressCallback",
    # QA
    "QAChecker",
    "RelationshipGraph",
    # TM
    "TranslationMemory",
    # Translation pipeline
    "TranslationPipeline",
    "TranslationStats",
    "bootstrap_characters_with_llm",
    "clean_glossary",
    "clean_translation",
    "estimate_tokens",
    # TSV I/O
    "export_tsv",
    "extract_character_definitions",
    "extract_json_object",
    "import_tsv",
    "infer_character_gender_nlp",
    "infer_character_role_nlp",
    "is_valid_glossary_term",
    "mine_glossary_candidates",
    "normalize_source",
    "partition_batches",
    "repair_glossary_leakage",
    # Analysis
    "run_analysis",
    "segment_scenes",
    "translate_glossary_with_llm",
]
