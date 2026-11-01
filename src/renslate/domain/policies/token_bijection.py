"""Pure domain policy enforcing 1-to-1 bijection of protected placeholder tokens."""

from __future__ import annotations

import re

__all__ = ["TokenBijectionPolicy"]

_PLACEHOLDER_RE = re.compile(r"⟦([A-Za-z0-9_/]+)⟧")


class TokenBijectionPolicy:
    """Verifies that protected AST placeholder tokens maintain a strict bijection."""

    @staticmethod
    def inspect_placeholders(
        expected_keys: set[str],
        translated_text: str,
    ) -> tuple[bool, list[str]]:
        """Verify presence and uniqueness of expected placeholder keys in translated text.

        Returns (is_valid, issues).
        """
        found_tokens = _PLACEHOLDER_RE.findall(translated_text)
        issues: list[str] = []

        seen: set[str] = set()
        duplicates: set[str] = set()
        for tok in found_tokens:
            if tok in seen:
                duplicates.add(tok)
            seen.add(tok)

        if duplicates:
            for d in sorted(duplicates):
                issues.append(f"duplicate_placeholder: ⟦{d}⟧")

        missing = expected_keys - seen
        if missing:
            for m in sorted(missing):
                issues.append(f"missing_placeholder: ⟦{m}⟧")

        unknown = seen - expected_keys
        if unknown:
            for u in sorted(unknown):
                issues.append(f"unknown_placeholder: ⟦{u}⟧")

        return len(issues) == 0, issues
