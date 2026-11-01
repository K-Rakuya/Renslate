"""Discourse attention policy for group dialogue disambiguation (>2 speakers).

Enforces strict isolated vocative detection and 3rd-person reference filtering.
Prevents incorrectly treating characters mentioned in 3rd person (e.g. 'Did you see Keiko?')
as the addressed listener.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = ["DiscourseAttentionPolicy"]

# Prepositions that indicate candidate is an indirect object or topic, NOT addressee
_THIRD_PERSON_PREPS = (
    r"\b(?:about|with|for|to|at|from|after|before|like|unlike|of|against|"
    r"behind|beside|by|towards|between|into|regarding|concerning)\s+"
)

# Transitive verbs and speech verbs where candidate is the direct object
_TRANSITIVE_VERBS_BEFORE = (
    r"\b(?:see|saw|seen|talk|talked|ask|asked|tell|told|know|knew|help|helped|"
    r"meet|met|visit|visited|hear|heard|remember|remembered|find|found|"
    r"warn|warned|save|saved|stop|stopped|kill|killed|call|called|remind|reminded|"
    r"hate|hated|love|loved|like|liked|trust|trusted|doubt|doubted)\s+"
)

# Auxiliary/verbs following candidate where candidate is the grammatical subject
_VERBS_AFTER = (
    r"\s+(?:is|was|are|were|will|would|can|could|should|might|has|had|have|did|does|"
    r"said|told|went|saw|knows|knew|thinks|thought|believes|believed|loves|loved|"
    r"hates|hated|wants|wanted|needs|needed|cried|smiled|laughed|nodded|replied)\b"
)

# Clause / question introducers before candidate
_CLAUSE_INTRODUCERS = r"\b(?:that|this|if|whether|why|how|where|when|what|which)\s+"


class DiscourseAttentionPolicy:
    """Discourse attention and interlocutor disambiguation policy."""

    @staticmethod
    def is_third_person_reference(text: str, candidate_name: str) -> bool:
        """Return True if candidate_name is used as a 3rd-person reference, not an addressee."""
        name_clean = candidate_name.strip()
        if not name_clean:
            return False

        c_esc = re.escape(name_clean)

        # 1. Possessive form (e.g. Keiko's book, Keiko' bag)
        if re.search(rf"\b{c_esc}['\u2019]s?\b", text, re.IGNORECASE):
            return True

        # 2. Preceded by preposition (e.g. "talk to Keiko", "about Keiko")
        if re.search(rf"{_THIRD_PERSON_PREPS}{c_esc}\b", text, re.IGNORECASE):
            return True

        # 3. Preceded by transitive verb (e.g. "Did you see Keiko?", "I know Keiko")
        if re.search(rf"{_TRANSITIVE_VERBS_BEFORE}{c_esc}\b", text, re.IGNORECASE):
            return True

        # 4. Followed by indicative verb (e.g. "Keiko said no", "Keiko was late")
        if re.search(rf"\b{c_esc}{_VERBS_AFTER}", text, re.IGNORECASE):
            return True

        # 5. Preceded by clause introducers (e.g. "I think that Keiko...")
        return bool(re.search(rf"{_CLAUSE_INTRODUCERS}{c_esc}\b", text, re.IGNORECASE))

    @staticmethod
    def is_strict_isolated_vocative(text: str, candidate_name: str) -> bool:
        """Return True only if candidate_name is clearly isolated as a direct vocative addressee."""
        name_clean = candidate_name.strip()
        if not name_clean:
            return False

        # If it's a 3rd-person reference grammatically, reject immediately
        if DiscourseAttentionPolicy.is_third_person_reference(text, name_clean):
            return False

        c_esc = re.escape(name_clean)

        # 1. Isolated at start: "Name, ..." or "Hey Name! ..."
        lead_pattern = (
            rf"^[“\"'«„「『(\uff08]*\s*(?:(?:hey|hi|hello|chào|này|ơi|oi|dear)\s+)?"
            rf"{c_esc}\s*[,!?:;\-—\uff0c\uff01\uff1f\uff1a\uff1b](?:\s+|$)"
        )
        if re.search(lead_pattern, text, re.IGNORECASE):
            return True

        # 2. Isolated at end: "... , Name." or "... , hey Name!"
        trail_pattern = (
            rf"(?:[,;—\-\uff0c\uff1b—]|(?:\.{2,}|…|~|\u301c))\s*"
            rf"(?:(?:hey|hi|hello|chào|này|ơi|oi|dear)\s+)?{c_esc}\s*"
            rf"[\.!\?~…'\"”»\uff09\)」』]*$"
        )
        if re.search(trail_pattern, text, re.IGNORECASE):
            return True

        # 3. Isolated in middle between punctuation: "..., Name, ..."
        medial_pattern = (
            rf"(?:[,;—\-\uff0c\uff1b—]|(?:\.{2,}|…|~|\u301c))\s*{c_esc}\s*"
            rf"(?:[,;—\-\uff0c\uff1b—]|(?:\.{2,}|…|~|\u301c))"
        )
        return bool(re.search(medial_pattern, text, re.IGNORECASE))

    @classmethod
    def disambiguate_group_turn(
        cls,
        text: str,
        speaker: str | None,
        candidate_names: Sequence[str],
        stage_tags: set[str] | None = None,
    ) -> str | None:
        """Identify the intended addressee among multiple characters in a group dialogue."""
        spk_clean = speaker.strip().lower() if speaker else ""

        valid_candidates = [
            c for c in candidate_names
            if c.strip().lower() != spk_clean
            and (not stage_tags or c.strip().lower() in stage_tags)
        ]

        for cand in valid_candidates:
            if cls.is_strict_isolated_vocative(text, cand):
                return cand

        return None
