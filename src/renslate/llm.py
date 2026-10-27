"""Backward compatibility shim forwarding to renslate.engine.llm."""

from __future__ import annotations

import sys

from renslate.engine import llm as _target
from renslate.engine.llm import *  # noqa: F403
from renslate.engine.llm import (
    LLMClient,
    extract_json_object,
)

__all__ = [
    "LLMClient",
    "extract_json_object",
]

sys.modules[__name__] = _target
