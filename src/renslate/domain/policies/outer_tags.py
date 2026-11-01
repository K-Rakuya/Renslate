"""Pure domain policy for extracting and realigning outer formatting markup tags."""

from __future__ import annotations

import re

__all__ = ["OuterTagPolicy", "realign_outer_tags"]

_OUTER_TAGS_PREFIX_RE = re.compile(r"^(?:\{[^\}]+\})+")
_OUTER_TAGS_SUFFIX_RE = re.compile(r"(?:\{[^\}]+\})+$")


def realign_outer_tags(source: str, translation: str) -> str:
    """Extract outer tag prefix and suffix from source string and realign on translation."""
    pref_match = _OUTER_TAGS_PREFIX_RE.match(source)
    suff_match = _OUTER_TAGS_SUFFIX_RE.search(source)
    if not pref_match or not suff_match:
        return translation

    prefix = pref_match.group(0)
    suffix = suff_match.group(0)
    inner = _OUTER_TAGS_PREFIX_RE.sub("", translation)
    inner = _OUTER_TAGS_SUFFIX_RE.sub("", inner).strip()

    src_inner = source[len(prefix) : len(source) - len(suffix)]
    if src_inner.startswith(" ") and not inner.startswith(" "):
        inner = " " + inner
    if src_inner.endswith(" ") and not inner.endswith(" "):
        inner = inner + " "
    return f"{prefix}{inner}{suffix}"


class OuterTagPolicy:
    """Encapsulates outer tag realignment logic."""

    @staticmethod
    def realign(source: str, translation: str) -> str:
        return realign_outer_tags(source, translation)
