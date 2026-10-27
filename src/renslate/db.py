"""Backward compatibility shim forwarding to renslate.core.db."""

from __future__ import annotations

import sys

from renslate.core import db as _target
from renslate.core.db import *  # noqa: F403
from renslate.core.db import (
    MIGRATIONS,
    Database,
    utc_timestamp,
)

__all__ = [
    "MIGRATIONS",
    "Database",
    "utc_timestamp",
]

sys.modules[__name__] = _target
