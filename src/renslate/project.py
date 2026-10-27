"""Backward compatibility shim forwarding to renslate.core.project."""

from __future__ import annotations

import sys

from renslate.core import project as _target
from renslate.core.project import *  # noqa: F403
from renslate.core.project import (
    WORK_DIR_NAME,
    Project,
)

__all__ = [
    "WORK_DIR_NAME",
    "Project",
]

sys.modules[__name__] = _target
