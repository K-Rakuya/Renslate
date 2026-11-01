"""Pure domain policy enforcing pronoun consistency and prohibiting semantic inversion."""

from __future__ import annotations

import re

__all__ = ["AntiInversionPolicy"]


class AntiInversionPolicy:
    """Enforces non-inversion invariants between first-person and second-person pronouns.

    In Visual Novel translation to Vietnamese:
    - First person (I, me, my) MUST map to self_pronoun.
    - Second person (you, your) MUST map to listener_pronoun.
    - Reversing them (e.g. asking a question to listener using self_pronoun) is an inversion flaw.
    """

    @staticmethod
    def inspect_pronoun_pair(
        source_text: str,
        target_text: str,
        self_pronoun: str | None,
        listener_pronoun: str | None,
    ) -> tuple[bool, str | None]:
        """Check if target text inverts the expected address forms.

        Returns (is_valid, reason_if_invalid).
        """
        if not self_pronoun or not listener_pronoun:
            return True, None

        self_p = self_pronoun.strip().lower()
        lis_p = listener_pronoun.strip().lower()

        if self_p == lis_p:
            return True, None

        # Detect classic inversion pattern: English question with 'you' translated using self_pronoun
        # e.g. "Do you want..." or "Why are you..." translated as "Em muốn..." when self is "em" and listener is "anh"
        src_lower = source_text.lower()
        tgt_lower = target_text.lower()

        has_english_you_question = (
            ("do you " in src_lower or "why are you " in src_lower or "are you " in src_lower)
            and "?" in src_lower
        )
        if has_english_you_question:
            # If the translated question begins with self_pronoun instead of listener_pronoun:
            self_pattern = rf"^\s*{re.escape(self_p)}\s+(?:có|muốn|đang|lại|nghĩ|thích)"
            if re.search(self_pattern, tgt_lower) and lis_p not in tgt_lower:
                return False, f"SEMANTIC_INVERSION: câu hỏi hướng tới đối phương nhưng lại dùng từ tự xưng '{self_p}'"

        return True, None
