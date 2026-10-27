"""Backward compatibility shim forwarding to renslate.core.errors."""

from __future__ import annotations

import sys

from renslate.core import errors as _target
from renslate.core.errors import *  # noqa: F403
from renslate.core.errors import (
    ApplyError,
    ConfigError,
    LLMError,
    MarkupError,
    ProjectError,
    QAError,
    RenslateError,
    SdkError,
    TranslationError,
)

__all__ = [
    "ApplyError",
    "ConfigError",
    "LLMError",
    "MarkupError",
    "ProjectError",
    "QAError",
    "RenslateError",
    "SdkError",
    "TranslationError",
]

sys.modules[__name__] = _target
