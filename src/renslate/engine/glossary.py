"""Glossary management, boundary-aware matching, and term candidate extraction."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from renslate.core.db import Database
from renslate.core.models import GlossaryEntry
from renslate.core.textio import normalize_nfc

_SENTENCE_START = re.compile(r'(?:^|[.!?…]\s+)(?:["\']|\b)([A-Z][a-z]+)\b')
_CAPITALIZED_PHRASE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b")
_WORD_RUN = re.compile(r"\b[A-Za-z0-9_'\-]+\b")
_PLACEHOLDER = re.compile(r"⟦[^⟧]+⟧")


def _compile_pattern(term: str, case_sensitive: bool) -> re.Pattern[str]:
    """Compile regex with proper word/punctuation boundaries for term."""
    escaped = re.escape(term)
    prefix = r"(?<!\w)" if term[:1].isalnum() or term[:1] == "_" else ""
    suffix = r"(?!\w)" if term[-1:].isalnum() or term[-1:] == "_" else ""
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(f"{prefix}{escaped}{suffix}", flags)


class Glossary:
    """Manages glossary terms and matches them against text batches."""

    def __init__(self, entries: Sequence[GlossaryEntry] = ()) -> None:
        self._entries: list[GlossaryEntry] = list(entries)
        self._compiled: list[tuple[re.Pattern[str], GlossaryEntry]] = []
        if self._entries:
            self._entries.sort(key=lambda e: len(e.term), reverse=True)
            self._recompile()

    @classmethod
    def load_db(cls, db: Database) -> Glossary:
        """Load glossary from SQLite database."""
        return cls(db.get_glossary())

    @classmethod
    def load_file(cls, path: Path) -> Glossary:
        """Load glossary from JSON file."""
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return cls()
        entries: list[GlossaryEntry] = []
        for item in data:
            if not isinstance(item, dict) or "term" not in item or "translation" not in item:
                continue
            entries.append(
                GlossaryEntry(
                    term=str(item["term"]),
                    translation=str(item["translation"]),
                    kind=str(item.get("kind", "general")),
                    case_sensitive=bool(item.get("case_sensitive", False)),
                    aliases=tuple(str(a) for a in item.get("aliases", ())),
                    notes=str(item.get("notes", "")),
                )
            )
        return cls(entries)

    def save_file(self, path: Path) -> None:
        """Export glossary to JSON file."""
        payload = [
            {
                "term": e.term,
                "translation": e.translation,
                "kind": e.kind,
                "case_sensitive": e.case_sensitive,
                "aliases": list(e.aliases),
                "notes": e.notes,
            }
            for e in self._entries
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def save_db(self, db: Database) -> None:
        """Persist all entries to SQLite database, replacing existing entries."""
        with db.transaction():
            db.execute("DELETE FROM glossary")
            for e in self._entries:
                db.save_glossary_entry(e)

    def add(self, entry: GlossaryEntry) -> None:
        """Add an entry, ordering by term length descending (longest term first)."""
        self._entries.append(entry)
        self._entries.sort(key=lambda e: len(e.term), reverse=True)
        self._recompile()

    def _recompile(self) -> None:
        """Rebuild regex patterns for all terms and their aliases."""
        self._compiled.clear()
        for e in self._entries:
            # Main term
            pat = _compile_pattern(e.term, e.case_sensitive)
            self._compiled.append((pat, e))
            # Aliases
            for alias in e.aliases:
                if alias.strip():
                    pat_alias = _compile_pattern(alias.strip(), e.case_sensitive)
                    self._compiled.append((pat_alias, e))

    @property
    def entries(self) -> list[GlossaryEntry]:
        return list(self._entries)

    def match_text(self, text: str) -> list[GlossaryEntry]:
        """Find all glossary entries that appear in text."""
        # Strip synthetic placeholders to avoid false matches on placeholder tokens
        clean_text = _PLACEHOLDER.sub(" ", text)
        matched: list[GlossaryEntry] = []
        seen_terms: set[str] = set()
        for pattern, entry in self._compiled:
            if entry.term in seen_terms:
                continue
            if pattern.search(clean_text):
                matched.append(entry)
                seen_terms.add(entry.term)
        return matched

    def match_batch(self, texts: Iterable[str]) -> list[GlossaryEntry]:
        """Find all glossary entries present across a collection of texts."""
        combined = "\n".join(texts)
        return self.match_text(combined)


_NOISE_TERM = re.compile(
    r"^(?:[A-Z]{1,2}|(?:AH|GH|HM|OH|UG|HA|HE|MM|SH|PST|URG)[A-Z]*|"
    r"[A-Z]*(?:AH|GH|HM|OH|UG|HA|HE){2,}[A-Z]*)$",
    re.IGNORECASE,
)


def _make_plural_aliases(term: str) -> list[str]:
    """Generate standard English plural aliases for a term."""
    words = term.split()
    if not words:
        return []
    last = words[-1]
    if last.endswith(("s", "x", "z", "ch", "sh")):
        p_last = last + "es"
    elif last.endswith("y") and len(last) > 1 and last[-2].lower() not in "aeiou":
        p_last = last[:-1] + "ies"
    else:
        p_last = last + "s"
    plural_term = " ".join([*words[:-1], p_last])
    return [plural_term] if plural_term.lower() != term.lower() else []


def translate_glossary_with_llm(
    entries: Sequence[GlossaryEntry],
    llm: Any,
    *,
    batch_size: int = 30,
) -> list[GlossaryEntry]:
    """Translate glossary entries using the local LLM in batches with zero hardcoding."""
    untranslated = [e for e in entries if not e.translation.strip()]
    if not untranslated:
        return list(entries)

    translated_map: dict[str, str] = {}
    system_prompt = (
        "Bạn là chuyên gia dịch thuật và bản địa hóa game visual novel. "
        "Nhiệm vụ của bạn là dịch danh sách thuật ngữ, danh từ riêng, tổ chức, "
        "địa danh, vật phẩm... từ tiếng Anh sang tiếng Việt chuẩn xác theo phong cách game.\n"
        "Quy tắc:\n"
        "- Tên riêng người, địa danh đặc thù: giữ nguyên hoặc phiên âm tự nhiên nếu cần.\n"
        "- Thuật ngữ, vai vế, danh xưng, địa điểm, vật phẩm, chiêu thức: dịch chính xác, súc tích "
        "(ví dụ: Sect -> Tông môn, Elder -> Trưởng lão, Force Field -> Kết giới...).\n"
        '- Trả về DUY NHẤT một đối tượng JSON dạng: {"Term": "Bản dịch tiếng Việt"}.\n'
        "Tuyệt đối không giải thích, không markdown code fence ngoài JSON."
    )

    for i in range(0, len(untranslated), batch_size):
        chunk = untranslated[i : i + batch_size]
        terms = [e.term for e in chunk]
        user_prompt = (
            'Dịch các thuật ngữ sau sang tiếng Việt dưới dạng JSON object {"Term": "Dịch"}:\n'
            + json.dumps(terms, ensure_ascii=False)
        )
        try:
            result_dict, _, _ = llm.complete_json(system_prompt, user_prompt)
            for term in terms:
                trans = result_dict.get(term, "").strip()
                if not trans:
                    # Case-insensitive lookup fallback
                    for k, v in result_dict.items():
                        if k.strip().lower() == term.lower() and v.strip():
                            trans = v.strip()
                            break
                if trans:
                    translated_map[term] = trans
        except Exception:
            # Continue with other batches if one fails
            continue

    result_entries: list[GlossaryEntry] = []
    for e in entries:
        if e.term in translated_map and not e.translation.strip():
            auto_aliases = set(e.aliases)
            for p in _make_plural_aliases(e.term):
                auto_aliases.add(p)
            result_entries.append(
                GlossaryEntry(
                    term=e.term,
                    translation=translated_map[e.term],
                    kind="auto_llm",
                    # Enforce case-insensitive matching for robust text detection
                    case_sensitive=False,
                    aliases=tuple(sorted(auto_aliases)),
                    notes="Dịch tự động bởi LLM",
                )
            )
        else:
            result_entries.append(e)

    return result_entries


_ENGLISH_PRONOUNS = frozenset(
    {
        "i",
        "me",
        "my",
        "myself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
    }
)

_ENGLISH_STOP_WORDS = frozenset(
    {
        # Articles
        "a",
        "an",
        "the",
        # Common particles & conjunctions
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "this",
        "that",
        "these",
        "those",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "really",
        "well",
        "okay",
        "yeah",
        "yes",
        "still",
        "now",
        "today",
        "sometimes",
        "maybe",
        "luckily",
        "finally",
        "since",
        "even",
        # Common auxiliary / modal verbs & contractions
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "come",
        "go",
        "get",
        "let",
        "see",
        "look",
        "make",
        "take",
        "think",
        "know",
        "don",
        "doesn",
        "didn",
        "won",
        "wouldn",
        "couldn",
        "shouldn",
        "isn",
        "aren",
        "wasn",
        "weren",
        "hasn",
        "haven",
        "hadn",
        # Common numbers & quantifiers
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        # Fillers & exclamations
        "damn",
        "fuck",
        "shit",
        "sigh",
        "wait",
        "help",
        "stop",
        "please",
        "wow",
        "whoa",
        "hmm",
        "huh",
        "oh",
        "ah",
        "ugh",
        "grahhh",
        "ahhh",
        "aghh",
        "dumbass",
        "screw",
        "whatever",
        "everyone",
        "people",
        "everything",
        "first",
        "last",
        "right",
        "left",
        "great",
        "nice",
        "regular",
        "strange",
        # Common generic vocabulary that must not be proper nouns
        "thank",
        "thanks",
        "relax",
        "time",
        "times",
        "day",
        "days",
        "night",
        "nights",
        "way",
        "ways",
        "thing",
        "things",
        "place",
        "places",
        "human",
        "humans",
        "person",
        "persons",
        "man",
        "men",
        "woman",
        "women",
        "boy",
        "boys",
        "girl",
        "girls",
        "kid",
        "kids",
        "child",
        "children",
        "water",
        "room",
        "rooms",
        "bathroom",
        "bathrooms",
        "bath",
        "door",
        "doors",
        "house",
        "home",
        "head",
        "eye",
        "eyes",
        "hand",
        "hands",
        "face",
        "heart",
        "mind",
        "voice",
        "sound",
        "noise",
        "sleep",
        "dream",
        "thought",
        "thoughts",
        "sorry",
        "bye",
        "gulp",
        "eeh",
        "excuse",
        "hello",
        "hi",
        "welcome",
        "listen",
        "hey",
        "haa",
        "aah",
        "ghg",
        "haha",
        "hahaha",
        # Pronouns
        *_ENGLISH_PRONOUNS,
    }
)

_EXCLUDED_COMMON_PHRASES = frozenset(
    {
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "excuse me",
        "never mind",
        "no problem",
        "of course",
        "thank you",
        "thanks a lot",
        "wait a minute",
        "what the hell",
        "oh my god",
        "by the way",
        "come on",
        "hold on",
        "let's go",
        "see you",
        "take care",
        "right now",
        "every day",
        "every night",
        "long time",
        "a lot",
        "so much",
        "all right",
        "no way",
        "don't worry",
        "i'm sorry",
        "high school",
        "middle school",
        "living room",
        "dining room",
        "front door",
        "back door",
        "coffee shop",
        "cell phone",
        "ice cream",
        "first floor",
        "second floor",
        "last night",
        "next week",
        "yesterday morning",
        "chapter one",
        "day one",
        "few minutes",
        "long ago",
        "one day",
        "next day",
        "last time",
        "first time",
        "good luck",
        "have fun",
        "listen to",
        "look at",
        "think about",
        "believe me",
        "tell me",
        "ask me",
        "wake up",
        "shut up",
        "stand up",
        "sit down",
        "get out",
        "come in",
    }
)

_CONTRACTION_STARTS = frozenset(
    {
        "don't",
        "can't",
        "won't",
        "let's",
        "i'm",
        "you're",
        "it's",
        "that's",
        "what's",
        "there's",
        "how's",
        "why's",
        "didn't",
        "couldn't",
        "wouldn't",
        "shouldn't",
        "hasn't",
        "haven't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
    }
)

_NUMBER_OR_TIME_WORDS = frozenset(
    {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "last",
        "next",
        "minute",
        "minutes",
        "hour",
        "hours",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "time",
        "times",
    }
)

_PREFIX_CLEAN = re.compile(
    r"^(?:A|An|The|My|Your|His|Her|Our|Their|This|That|These|Those|Some|Like|But|And|If|So|Because|As|With)\s+",
    re.IGNORECASE,
)


def is_valid_glossary_term(term: str, exclude_terms: set[str] | None = None) -> bool:
    """Validate whether a term is a genuine domain noun/proper entity and not grammar noise."""
    t_clean = term.strip()
    if len(t_clean) < 3 or t_clean.isdigit():
        return False
    if _NOISE_TERM.match(t_clean):
        return False

    # Filter out hex colors (e.g. #A05A61, A05A61, DBB498)
    clean_hex = t_clean.lstrip("#")
    if (
        len(clean_hex) in (3, 6, 8)
        and all(c in "0123456789abcdefABCDEF" for c in clean_hex)
        and (len(clean_hex) >= 6 or any(c.isdigit() for c in clean_hex))
    ):
        return False

    t_lower = t_clean.lower()
    if t_lower in _EXCLUDED_COMMON_PHRASES:
        return False

    if exclude_terms and t_lower in exclude_terms:
        return False

    # Check font names, file extensions, or technical artifacts
    tech_artifacts = (".ttf", ".otf", ".png", ".jpg", ".rpy", ".rpyc", "zcool", "pysync", "py_sync")
    if any(tech in t_lower for tech in tech_artifacts):
        return False

    words = [w.lower() for w in t_clean.split()]
    if not words:
        return False

    # Reject conversational contractions at start (e.g. "Don't Worry", "Let's Go")
    if words[0] in _CONTRACTION_STARTS:
        return False

    # Reject single-word stop words
    if len(words) == 1 and words[0] in _ENGLISH_STOP_WORDS:
        return False

    # Reject multi-word phrases composed entirely of stop words
    if all(w in _ENGLISH_STOP_WORDS for w in words):
        return False

    # Reject multi-word phrases that contain English pronouns (e.g. "Because I", "If I")
    if any(w in _ENGLISH_PRONOUNS for w in words):
        return False

    # Reject phrases starting or ending with pure prepositions/conjunctions
    if words[0] in {
        "and",
        "but",
        "or",
        "if",
        "so",
        "because",
        "like",
        "with",
        "since",
        "as",
        "to",
        "for",
        "of",
        "in",
        "on",
        "at",
        "by",
        "from",
    }:
        return False
    if words[-1] in {
        "and",
        "but",
        "or",
        "if",
        "so",
        "with",
        "to",
        "for",
        "of",
        "in",
        "on",
        "at",
        "by",
        "from",
    }:
        return False

    # Reject phrases that consist purely of time/number words
    return not all(w in _NUMBER_OR_TIME_WORDS or w in _ENGLISH_STOP_WORDS for w in words)


def clean_glossary(
    entries: Sequence[GlossaryEntry],
    exclude_terms: set[str] | None = None,
) -> list[GlossaryEntry]:
    """Sanitize and deduplicate glossary entries, filtering out noise and character names."""
    seen_terms: set[str] = set()
    cleaned_entries: list[GlossaryEntry] = []

    # Sort entries by occurrences count (from notes) or translation presence
    def _priority_score(e: GlossaryEntry) -> tuple[int, int]:
        has_trans = 1 if e.translation.strip() else 0
        m = re.search(r"(\d+)\s*lần", e.notes)
        count = int(m.group(1)) if m else 1
        return (has_trans, count)

    sorted_input = sorted(entries, key=_priority_score, reverse=True)

    for e in sorted_input:
        clean_t = _PREFIX_CLEAN.sub("", e.term).strip()
        target_term = clean_t if len(clean_t) >= 3 else e.term
        if not is_valid_glossary_term(target_term, exclude_terms):
            continue
        if target_term.lower() in seen_terms:
            continue

        seen_terms.add(target_term.lower())
        auto_aliases = set(e.aliases)
        for p in _make_plural_aliases(target_term):
            auto_aliases.add(p)

        cleaned_entries.append(
            GlossaryEntry(
                term=target_term,
                translation=e.translation.strip(),
                kind=e.kind,
                case_sensitive=False,  # Enforce case_sensitive=False
                aliases=tuple(sorted(auto_aliases)),
                notes=e.notes,
            )
        )
    return cleaned_entries


def mine_glossary_candidates(
    sources: Sequence[str],
    min_count: int = 2,
    exclude_terms: set[str] | None = None,
    extra_terms: Sequence[str] | None = None,
) -> list[tuple[str, int]]:
    """Mine candidate proper nouns and recurring terminology from source texts.

    Filters out grammatical stop words, pronouns, clause starters (e.g. "If I"),
    font names, hex colors, and known character names.
    """
    counts: Counter[str] = Counter()
    effective_min = 1 if (min_count <= 2 and len(sources) < 200) else min_count

    for text in sources:
        # Normalize
        norm = normalize_nfc(text)

        # Exclude the first capitalized word in sentences
        sentence_starts = set(_SENTENCE_START.findall(norm))

        # Multi-word capitalized phrases (e.g., "Student Council", "Sacred Sword")
        for match in _CAPITALIZED_PHRASE.finditer(norm):
            phrase = match.group(0).strip()
            # Clean leading conversational words e.g. "My Inner Sanctum" -> "Inner Sanctum"
            clean_phrase = _PREFIX_CLEAN.sub("", phrase).strip()
            target_term = clean_phrase if len(clean_phrase) >= 3 else phrase

            # Validate term validity
            if not is_valid_glossary_term(target_term, exclude_terms):
                continue

            # Ignore single letters or purely sentence-starting single words
            words = target_term.split()
            if len(words) == 1 and words[0] in sentence_starts:
                continue

            counts[target_term] += 1

    if extra_terms:
        for t in extra_terms:
            t_clean = _PREFIX_CLEAN.sub("", t).strip()
            if is_valid_glossary_term(t_clean, exclude_terms):
                counts[t_clean] += max(2, effective_min)

    candidates = [
        (term, count)
        for term, count in counts.most_common()
        if count >= effective_min and not term.isdigit()
    ]
    return candidates
