"""Stylometrics extraction: sentence metrics, punctuation rates, type-token ratio."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "StylometricSummary",
    "extract_stylometrics",
]

_RE_TAG_STRIP = re.compile(r"\{[^}]*\}")
_RE_VAR_STRIP = re.compile(r"\[[^\]]*\]")
_RE_WORD_TOKENS = re.compile(r"\b[^\W\d_]+(?:-[^\W\d_]+)*(?:'[^\W\d_]+)?\b", re.UNICODE)
_RE_SENTENCE_SPLIT = re.compile(r"(?:[.!?…\u3002\uff01\uff1f]+|\n+)\s*")
_RE_STUTTER = re.compile(r"\b[a-zA-Z]-[a-zA-Z]+(?:'[a-zA-Z]+)?\b")
_RE_INTERJECTIONS = re.compile(
    r"\b(uh+|um+|ah+|eh+|oh+|er+|erm|hm+|gah|ack|tch|sigh|oof|urk|eep|geez|whoa|huh|phew|hmph|tsk|ara|kuh)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StylometricSummary:
    """Stylometric metrics extracted from a character's dialogue corpus."""

    total_lines: int
    total_words: int
    unique_words: int
    type_token_ratio: float
    total_sentences: int
    avg_sentence_length: float
    exclamation_count: int
    exclamation_rate: float
    question_count: int
    question_rate: float
    ellipsis_count: int
    ellipsis_rate: float
    hesitation_count: int
    hesitation_rate: float
    top_interjections: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "total_lines": self.total_lines,
            "total_words": self.total_words,
            "unique_words": self.unique_words,
            "type_token_ratio": self.type_token_ratio,
            "total_sentences": self.total_sentences,
            "avg_sentence_length": self.avg_sentence_length,
            "exclamation_count": self.exclamation_count,
            "exclamation_rate": self.exclamation_rate,
            "question_count": self.question_count,
            "question_rate": self.question_rate,
            "ellipsis_count": self.ellipsis_count,
            "ellipsis_rate": self.ellipsis_rate,
            "hesitation_count": self.hesitation_count,
            "hesitation_rate": self.hesitation_rate,
            "top_interjections": list(self.top_interjections),
        }

    def summary_text(self) -> str:
        """Formatted human-readable and LLM-readable summary string."""
        if self.total_lines == 0 or self.total_words == 0:
            return "Không có dữ liệu thoại thống kê"

        interj_str = ", ".join(self.top_interjections) if self.top_interjections else "không có"
        return (
            f"TTR: {self.type_token_ratio:.2f} ({self.unique_words}/{self.total_words} từ); "
            f"Độ dài câu TB: {self.avg_sentence_length:.1f} từ; "
            f"Cảm thán (!): {self.exclamation_rate:.2f}/câu; "
            f"Nghi vấn (?): {self.question_rate:.2f}/câu; "
            f"Chấm lửng (...): {self.ellipsis_rate:.2f}/câu; "
            f"Thán từ/ngập ngừng: {self.hesitation_rate:.2f}/câu ({interj_str})"
        )


def extract_stylometrics(dialogues: Sequence[str]) -> StylometricSummary:
    """Extract stylometric metrics from a character's dialogue lines using Python stdlib."""
    if not dialogues:
        return StylometricSummary(
            total_lines=0,
            total_words=0,
            unique_words=0,
            type_token_ratio=0.0,
            total_sentences=0,
            avg_sentence_length=0.0,
            exclamation_count=0,
            exclamation_rate=0.0,
            question_count=0,
            question_rate=0.0,
            ellipsis_count=0,
            ellipsis_rate=0.0,
            hesitation_count=0,
            hesitation_rate=0.0,
            top_interjections=(),
        )

    total_words = 0
    words_counter: Counter[str] = Counter()
    total_sentences = 0
    excl_count = 0
    ques_count = 0
    ellip_count = 0
    interjection_counter: Counter[str] = Counter()

    for raw in dialogues:
        cleaned = _RE_TAG_STRIP.sub("", raw)
        cleaned = _RE_VAR_STRIP.sub("", cleaned).strip()
        if not cleaned:
            continue

        excl_count += len(re.findall(r"!|\uff01", cleaned))
        ques_count += len(re.findall(r"\?|\uff1f", cleaned))
        ellip_count += len(re.findall(r"\.{2,}|…", cleaned))

        for m in _RE_STUTTER.finditer(cleaned):
            interjection_counter[m.group(0).lower()] += 1
        for m in _RE_INTERJECTIONS.finditer(cleaned):
            interjection_counter[m.group(0).lower()] += 1

        line_words = _RE_WORD_TOKENS.findall(cleaned)
        total_words += len(line_words)
        for w in line_words:
            words_counter[w.lower()] += 1

        sents = [s for s in _RE_SENTENCE_SPLIT.split(cleaned) if s.strip()]
        total_sentences += len(sents)

    unique_words = len(words_counter)
    ttr = (unique_words / total_words) if total_words > 0 else 0.0
    sents_denom = max(1, total_sentences)
    avg_len = (total_words / sents_denom) if total_sentences > 0 else 0.0
    hesitation_count = sum(interjection_counter.values())

    return StylometricSummary(
        total_lines=len(dialogues),
        total_words=total_words,
        unique_words=unique_words,
        type_token_ratio=round(ttr, 4),
        total_sentences=total_sentences,
        avg_sentence_length=round(avg_len, 2),
        exclamation_count=excl_count,
        exclamation_rate=round(excl_count / sents_denom, 2) if total_sentences else 0.0,
        question_count=ques_count,
        question_rate=round(ques_count / sents_denom, 2) if total_sentences else 0.0,
        ellipsis_count=ellip_count,
        ellipsis_rate=round(ellip_count / sents_denom, 2) if total_sentences else 0.0,
        hesitation_count=hesitation_count,
        hesitation_rate=round(hesitation_count / sents_denom, 2) if total_sentences else 0.0,
        top_interjections=tuple(k for k, _ in interjection_counter.most_common(5)),
    )
