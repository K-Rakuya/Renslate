"""Modular context architecture subpackage for renslate."""

from __future__ import annotations

from renslate.engine.context.builder import (
    DEFAULT_SYSTEM_PROMPT,
    QUOTED_RULES,
    ContextBuilder,
    ContextBundle,
    LookbackItem,
    build_transgraph_frame,
)
from renslate.engine.context.macro import (
    assemble_scene_macro_context,
    infer_scene_cues,
    update_scene_macro_from_flow,
)
from renslate.engine.context.prompt_frame import (
    _format_pronoun_pair,
    _format_relational_rule,
    format_rich_dialogue_line,
)
from renslate.engine.context.resolver import (
    _KINSHIP_WORDS,
    _LEADING_VOCATIVE_RE,
    _MEDIAL_VOCATIVE_RE,
    _NON_DIALOGUE_TAGS,
    _PROTAGONIST_TAGS,
    _TRAILING_VOCATIVE_RE,
    InterlocutorResolver,
    ResolvedInterlocutor,
    _analyze_monologue_and_dialogue,
    _extract_unit_file,
    _is_scene_boundary,
    _strip_outer_quotes,
)
from renslate.engine.context.tokens import (
    _CHARS_PER_TOKEN_CJK,
    _CHARS_PER_TOKEN_EN,
    _CHARS_PER_TOKEN_VI,
    estimate_tokens,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "QUOTED_RULES",
    "_CHARS_PER_TOKEN_CJK",
    "_CHARS_PER_TOKEN_EN",
    "_CHARS_PER_TOKEN_VI",
    "_KINSHIP_WORDS",
    "_LEADING_VOCATIVE_RE",
    "_MEDIAL_VOCATIVE_RE",
    "_NON_DIALOGUE_TAGS",
    "_PROTAGONIST_TAGS",
    "_TRAILING_VOCATIVE_RE",
    "ContextBuilder",
    "ContextBundle",
    "InterlocutorResolver",
    "LookbackItem",
    "ResolvedInterlocutor",
    "_analyze_monologue_and_dialogue",
    "_extract_unit_file",
    "_format_pronoun_pair",
    "_format_relational_rule",
    "_is_scene_boundary",
    "_strip_outer_quotes",
    "assemble_scene_macro_context",
    "build_transgraph_frame",
    "estimate_tokens",
    "format_rich_dialogue_line",
    "infer_scene_cues",
    "update_scene_macro_from_flow",
]
