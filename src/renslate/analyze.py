"""Backward compatibility shim forwarding to renslate.engine.analyze."""

from __future__ import annotations

import sys

from renslate.engine import analyze as _target
from renslate.engine.analyze import *  # noqa: F403
from renslate.engine.analyze import (
    bootstrap_characters_with_llm,
    extract_character_definitions,
    infer_character_gender_nlp,
    infer_character_role_nlp,
    run_analysis,
    segment_scenes,
)

__all__ = [
    "bootstrap_characters_with_llm",
    "extract_character_definitions",
    "infer_character_gender_nlp",
    "infer_character_role_nlp",
    "run_analysis",
    "segment_scenes",
]

sys.modules[__name__] = _target
