"""Backward compatibility shim forwarding to renslate.core.models."""

from __future__ import annotations

import sys

from renslate.core import models as _target
from renslate.core.models import *  # noqa: F403
from renslate.core.models import (
    LOCKED_ORIGINS,
    ApplyTarget,
    BatchLog,
    CharacterProfile,
    KinshipKind,
    Origin,
    QAIssue,
    RelationalVector,
    RelationshipEdge,
    RelationshipKind,
    SceneMacroContext,
    SceneSummary,
    Slot,
    Status,
    TMEntry,
    Unit,
    UnitKind,
)
from renslate.domain.relational import project_relational_vector

__all__ = [
    "LOCKED_ORIGINS",
    "ApplyTarget",
    "BatchLog",
    "CharacterProfile",
    "KinshipKind",
    "Origin",
    "QAIssue",
    "RelationalVector",
    "RelationshipEdge",
    "RelationshipKind",
    "SceneMacroContext",
    "SceneSummary",
    "Slot",
    "Status",
    "TMEntry",
    "Unit",
    "UnitKind",
    "project_relational_vector",
]

sys.modules[__name__] = _target
