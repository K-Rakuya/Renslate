"""Reference-free syntactic calque density scoring for anti-translationese confidence gating."""

from __future__ import annotations

import re

from renslate.core.textio import normalize_nfc

__all__ = ["compute_calque_density"]

_STARTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^chà(?:\s*[,.!?…~]|\s+thì\b)", re.IGNORECASE),
    re.compile(
        r"^(?:bạn|cậu|anh|em|ngươi|mày)\s+thấy\s+đấy(?:\s*[,.!?…~]|\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^như\s+(?:bạn|cậu|anh|em|ngươi|mày|chúng\s+ta|ai\s+cũng)?\s*(?:đã\s+)?biết(?:\s*[,.!?…~]|\b)",
        re.IGNORECASE,
    ),
    re.compile(r"^nói\s+cách\s+khác(?:\s*[,.!?…~]|\b)", re.IGNORECASE),
    re.compile(
        r"^(?:thành\s+thật|thật\s+lòng)\s+mà\s+nói(?:\s*[,.!?…~]|\b)",
        re.IGNORECASE,
    ),
    re.compile(r"^sau\s+tất\s+cả(?:\s*[,.!?…~]|\b)", re.IGNORECASE),
)

_EN_STARTER_RE = re.compile(
    r"^\s*(?:well|you see|as you know|in other words|to be honest|after all)\b",
    re.IGNORECASE,
)

_AGENTIVE_PASSIVE_RE = re.compile(
    r"\b(bị|được)\s+(?:(?!\b(?:bị|được)\b)\S+\s+){0,5}bởi\s+(?!vì\b|lẽ\b|do\b)\S+",
    re.IGNORECASE,
)

_EN_PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|been|being|be)\s+(?:\w+\s+)?\w+(?:ed|en|d|t)\b"
    r"|\bby\s+(?:him|her|them|me|us|you|whom|[A-Z]\w+|the\s+\w+)\b",
    re.IGNORECASE,
)

_DA_PASSIVE_RE = re.compile(
    r"\bđã\s+(?:bị|được)\s+(?:(?!\b(?:bị|được)\b)\S+\s+){0,5}bởi\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"\b\w+\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
_LEADING_PUNCT_RE = re.compile(r"^[“\"'«„「『—\-\s]+")


def compute_calque_density(source: str, target: str) -> float:
    """Evaluate target text against source to compute a normalized syntactic calque score.

    Returns a scalar float in the range [0.0, 1.0]. Higher values indicate greater
    translationese and rigid English structural mimicry (formulaic starters,
    agentive passive 'bị/được ... bởi ...', and excessive possessive density of 'của').
    Does not modify or substitute the input strings.
    """
    tgt_clean = normalize_nfc(target).strip()
    if not tgt_clean:
        return 0.0

    src_clean = source.strip()
    total_score = 0.0

    raw_sentences = _SENTENCE_SPLIT_RE.split(tgt_clean)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return 0.0

    starter_flagged = False
    for sentence in sentences:
        core_sentence = _LEADING_PUNCT_RE.sub("", sentence)
        for pattern in _STARTER_PATTERNS:
            if pattern.search(core_sentence):
                starter_flagged = True
                break
        if starter_flagged:
            break

    if starter_flagged:
        total_score += 0.30 if _EN_STARTER_RE.search(src_clean) else 0.25

    has_agentive_passive = bool(_AGENTIVE_PASSIVE_RE.search(tgt_clean))
    if has_agentive_passive:
        has_en_passive = bool(_EN_PASSIVE_RE.search(src_clean))
        total_score += 0.40 if has_en_passive else 0.35
        if _DA_PASSIVE_RE.search(tgt_clean):
            total_score += 0.10

    possessive_score = 0.0
    for sentence in sentences:
        words = _WORD_RE.findall(sentence)
        if not words:
            continue
        cua_count = sum(1 for w in words if w.lower() == "của")
        if len(words) < 12 and cua_count > 2:
            possessive_score = max(possessive_score, 0.40)
        elif (len(words) < 12 and cua_count == 2) or (len(words) >= 12 and cua_count >= 3):
            possessive_score = max(possessive_score, 0.20)

    all_words = _WORD_RE.findall(tgt_clean)
    if all_words:
        total_cua = sum(1 for w in all_words if w.lower() == "của")
        if total_cua >= 2:
            cua_ratio = total_cua / len(all_words)
            if cua_ratio >= 0.20:
                possessive_score = max(possessive_score, 0.35)

    total_score += possessive_score

    for sentence in sentences:
        words = _WORD_RE.findall(sentence)
        if len(words) < 16:
            da_count = sum(1 for w in words if w.lower() == "đã")
            if da_count >= 2:
                total_score += 0.10
                break

    return round(min(1.0, max(0.0, total_score)), 4)
