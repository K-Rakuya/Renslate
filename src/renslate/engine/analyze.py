"""Backward compatibility shim forwarding to renslate.engine.analyze package."""

from __future__ import annotations

from renslate.engine.analyze import (
    COLLECTIVE_TAGS,
    INNER_MONOLOGUE_TAGS,
    NARRATOR_TAGS,
    NON_HUMAN_TAGS,
    NON_INTERACTION_TAGS,
    PERSONA_SYSTEM_PROMPT,
    PROTAGONIST_TAGS,
    ArchetypeThresholds,
    StylometricSummary,
    bootstrap_characters_with_llm,
    build_persona_user_prompt,
    collect_character_narration_context,
    collect_incoming_vocatives,
    detect_archetype,
    extract_character_definitions,
    extract_interaction_network,
    extract_interactive_dialogue_exchanges,
    extract_salient_locations,
    extract_scene_macro_contexts,
    extract_stylometrics,
    find_protagonist_tag,
    infer_character_gender_nlp,
    infer_character_role_nlp,
    infer_persona_fallback,
    infer_semantic_relationship,
    is_collective_entity,
    is_non_human_or_meta,
    load_characters_from_json,
    resolve_pronoun_pair_from_relationship,
    run_analysis,
    save_characters_to_json,
    segment_scenes,
    stratified_speaker_samples,
    sync_characters,
)

# Legacy aliases for internal sets
_PROTAGONIST_TAGS = PROTAGONIST_TAGS
_NARRATOR_TAGS = NARRATOR_TAGS
_INNER_MONOLOGUE_TAGS = INNER_MONOLOGUE_TAGS
_COLLECTIVE_TAGS = COLLECTIVE_TAGS
_NON_HUMAN_TAGS = NON_HUMAN_TAGS
_NON_INTERACTION_TAGS = NON_INTERACTION_TAGS

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
