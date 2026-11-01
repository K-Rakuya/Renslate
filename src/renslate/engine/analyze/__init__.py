"""Analysis subpackage: stylometrics, sampling, persona inference, tropes, and pipeline."""

from __future__ import annotations

from renslate.core.db import (
    load_characters_from_json,
    save_characters_to_json,
    sync_characters,
)
from renslate.engine.analyze.persona import (
    PERSONA_SYSTEM_PROMPT,
    bootstrap_characters_with_llm,
    build_persona_user_prompt,
    collect_character_narration_context,
    collect_incoming_vocatives,
    extract_interaction_network,
    extract_interactive_dialogue_exchanges,
    extract_salient_locations,
    find_protagonist_tag,
    infer_character_gender_nlp,
    infer_character_role_nlp,
    infer_persona_fallback,
    infer_semantic_relationship,
    is_collective_entity,
    is_non_human_or_meta,
    resolve_pronoun_pair_from_relationship,
)
from renslate.engine.analyze.pipeline import (
    extract_character_definitions,
    extract_scene_macro_contexts,
    run_analysis,
    segment_scenes,
)
from renslate.engine.analyze.sampling import stratified_speaker_samples
from renslate.engine.analyze.stylometrics import StylometricSummary, extract_stylometrics
from renslate.engine.analyze.tropes import (
    COLLECTIVE_TAGS,
    INNER_MONOLOGUE_TAGS,
    NARRATOR_TAGS,
    NON_HUMAN_TAGS,
    NON_INTERACTION_TAGS,
    PROTAGONIST_TAGS,
    ArchetypeThresholds,
    detect_archetype,
)

__all__ = [
    "COLLECTIVE_TAGS",
    "INNER_MONOLOGUE_TAGS",
    "NARRATOR_TAGS",
    "NON_HUMAN_TAGS",
    "NON_INTERACTION_TAGS",
    "PERSONA_SYSTEM_PROMPT",
    "PROTAGONIST_TAGS",
    "ArchetypeThresholds",
    "StylometricSummary",
    "bootstrap_characters_with_llm",
    "build_persona_user_prompt",
    "collect_character_narration_context",
    "collect_incoming_vocatives",
    "detect_archetype",
    "extract_character_definitions",
    "extract_interaction_network",
    "extract_interactive_dialogue_exchanges",
    "extract_salient_locations",
    "extract_scene_macro_contexts",
    "extract_stylometrics",
    "find_protagonist_tag",
    "infer_character_gender_nlp",
    "infer_character_role_nlp",
    "infer_persona_fallback",
    "infer_semantic_relationship",
    "is_collective_entity",
    "is_non_human_or_meta",
    "load_characters_from_json",
    "resolve_pronoun_pair_from_relationship",
    "run_analysis",
    "save_characters_to_json",
    "segment_scenes",
    "stratified_speaker_samples",
    "sync_characters",
]
