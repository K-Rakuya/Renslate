"""Backward compatibility shim forwarding to renslate.engine.translate."""

from __future__ import annotations

import sys

from renslate.engine import translate as _target
from renslate.engine.translate import *  # noqa: F403
from renslate.engine.translate import (
    ProgressCallback,
    TranslationPipeline,
    TranslationStats,
    partition_batches,
)

__all__ = [
    "ProgressCallback",
    "TranslationPipeline",
    "TranslationStats",
    "partition_batches",
]

sys.modules[__name__] = _target
