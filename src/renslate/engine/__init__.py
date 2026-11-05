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
from renslate.engine.graph import RelationshipGraph
from renslate.engine.llm import LLMClient, extract_json_object
from renslate.engine.qa import (
    QAChecker,
    clean_translation,
)
from renslate.engine.relationships import bootstrap_relationships_with_llm
from renslate.engine.scenes import summarize_scenes_with_llm, summarize_single_scene_with_llm
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
    "bootstrap_relationships_with_llm",
    "clean_translation",
    "estimate_tokens",
    # TSV I/O
    "export_tsv",
    "extract_character_definitions",
    "extract_json_object",
    "import_tsv",
    "infer_character_gender_nlp",
    "infer_character_role_nlp",
    "normalize_source",
    "partition_batches",
    # Analysis
    "run_analysis",
    "segment_scenes",
    "summarize_scenes_with_llm",
    "summarize_single_scene_with_llm",
]
