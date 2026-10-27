"""Backward compatibility shim forwarding to renslate.engine.glossary."""

from __future__ import annotations

import sys

from renslate.engine import glossary as _target
from renslate.engine.glossary import *  # noqa: F403
from renslate.engine.glossary import (
    Glossary,
    clean_glossary,
    is_valid_glossary_term,
    mine_glossary_candidates,
    translate_glossary_with_llm,
)

__all__ = [
    "Glossary",
    "clean_glossary",
    "is_valid_glossary_term",
    "mine_glossary_candidates",
    "translate_glossary_with_llm",
]

sys.modules[__name__] = _target
