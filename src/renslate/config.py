"""Backward compatibility shim forwarding to renslate.core.config."""

from __future__ import annotations

import sys

from renslate.core import config as _target
from renslate.core.config import *  # noqa: F403
from renslate.core.config import (
    BatchConfig,
    Config,
    LLMConfig,
    QAConfig,
    StyleConfig,
    TMConfig,
)

__all__ = [
    "BatchConfig",
    "Config",
    "LLMConfig",
    "QAConfig",
    "StyleConfig",
    "TMConfig",
]

sys.modules[__name__] = _target
