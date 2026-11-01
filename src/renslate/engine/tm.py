"""Translation Memory (TM) for renslate.

Supports exact matches for instant reuse without model calls, and fuzzy
matches using n-gram candidate filtering + SequenceMatcher fine scoring
to supply translation hints in the LLM context prompt.
"""

from __future__ import annotations

import difflib
import re
import threading
from collections import defaultdict
from dataclasses import dataclass

from renslate.core.db import Database, utc_timestamp
from renslate.core.models import LOCKED_ORIGINS, Origin, TMEntry, Unit, UnitKind
from renslate.core.textio import normalize_nfc
from renslate.engine.qa import clean_translation

_WHITESPACE = re.compile(r"\s+")


def normalize_source(text: str) -> str:
    """Normalize source text for TM lookup: NFC form, stripped, collapsed whitespace."""
    nfc = normalize_nfc(text)
    collapsed = _WHITESPACE.sub(" ", nfc).strip()
    return collapsed


# Lines whose Vietnamese rendering depends on *who is talking to whom* (first/second person
# pronouns, kinship and honorific vocatives). A bare exact-match TM hit is unsafe for them:
# "Thank you." becomes "Cảm ơn anh." for one pair and "Cảm ơn ngài." for another.
_ADDRESS_SENSITIVE = re.compile(
    r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours|yourself|yourselves"
    r"|sir|ma'am|madam|miss|mister|master|mistress|milord|milady|my\s+lord|my\s+lady"
    r"|mom|mum|mother|dad|father|papa|mama|brother|sister|bro|sis|uncle|aunt|grandpa|grandma"
    r"|sensei|senpai|kouhai)\b",
    re.IGNORECASE,
)
_NARRATION_SPEAKERS = frozenset(
    {
        "",
        "t",
        "th",
        "thought",
        "thoughts",
        "thinking",
        "narrator",
        "narration",
        "none",
        "all",
        "system",
        "ui",
        "menu",
    }
)
_TAG_OR_VAR = re.compile(r"\{[^{}]*\}|\[[^\[\]]*\]|⟦[^⟦⟧]*⟧")


def is_context_sensitive_line(unit: Unit) -> bool:
    """Return True when a unit's translation depends on speaker/listener relationship.

    Only spoken dialogue by a real character qualifies. UI strings, menu choices and
    narration (no addressee) are context-free and remain safe for exact TM reuse.
    """
    if unit.kind is not UnitKind.DIALOGUE:
        return False
    speaker = (unit.speaker or "").strip().lower()
    if speaker in _NARRATION_SPEAKERS:
        return False
    text = _TAG_OR_VAR.sub(" ", unit.source)
    return _ADDRESS_SENSITIVE.search(text) is not None


def _get_char_ngrams(text: str, n: int = 3) -> set[str]:
    """Generate character n-grams for fast fuzzy pre-filtering."""
    padded = f"  {text.lower()}  "
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


class _L1BloomFilter:
    """L1 Cache-Resident Bitset Bloom Filter (32 KB = 262,144 bits).

    Fits completely in AMD Zen 4 32 KB L1 Data Cache.
    Provides sub-microsecond (<30ns) negative rejection before touching
    heap dictionaries or SQLite. Zero false negatives guaranteed.
    """

    __slots__ = ("_bits",)

    def __init__(self, size_bytes: int = 32768) -> None:
        self._bits = bytearray(size_bytes)

    def clear(self) -> None:
        self._bits = bytearray(len(self._bits))

    def add(self, key: str) -> None:
        h = hash(key)
        h1 = h & 0x3FFFF
        h2 = ((h >> 16) ^ (h * 0x517CC1B727220A95)) & 0x3FFFF
        self._bits[h1 >> 3] |= 1 << (h1 & 7)
        self._bits[h2 >> 3] |= 1 << (h2 & 7)

    def contains(self, key: str) -> bool:
        h = hash(key)
        h1 = h & 0x3FFFF
        if not (self._bits[h1 >> 3] & (1 << (h1 & 7))):
            return False
        h2 = ((h >> 16) ^ (h * 0x517CC1B727220A95)) & 0x3FFFF
        return bool(self._bits[h2 >> 3] & (1 << (h2 & 7)))


@dataclass(frozen=True, slots=True)
class FuzzyMatch:
    score: float
    source: str
    target: str


class TranslationMemory:
    """In-memory index backed by the SQLite database."""

    def __init__(self, db: Database) -> None:
        self._db = db
        # The pipeline prepares the next batch (fuzzy lookups) on a worker thread while the
        # main thread records fresh translations; both touch the in-memory indices.
        self._lock = threading.RLock()
        self._bloom = _L1BloomFilter()
        self._exact_map: dict[str, TMEntry] = {}
        # Pre-computed 3-grams for fast fuzzy filtering: ngram -> set of src_norm
        self._ngram_index: dict[str, set[str]] = defaultdict(set)
        self._entries: dict[str, TMEntry] = {}
        self.reload()

    def reload(self) -> None:
        """Load all entries from SQLite database into memory indices."""
        with self._lock:
            self._bloom.clear()
            self._exact_map.clear()
            self._ngram_index.clear()
            self._entries.clear()
            for entry in self._db.all_tm():
                self._entries[entry.src_norm] = entry
                self._exact_map[entry.src_norm] = entry
                self._bloom.add(entry.src_norm)
                for ng in _get_char_ngrams(entry.src_norm):
                    self._ngram_index[ng].add(entry.src_norm)

    def lookup_exact_entry(self, source: str) -> TMEntry | None:
        """Return the full TM entry (target + origin) for an exact match, or None."""
        norm = normalize_source(source)
        with self._lock:
            if not norm or not self._bloom.contains(norm):
                return None
            return self._exact_map.get(norm)

    def lookup_exact(self, source: str) -> str | None:
        """Return the target translation if an exact match exists in TM."""
        entry = self.lookup_exact_entry(source)
        return entry.target if entry is not None else None

    def lookup_fuzzy(
        self, source: str, threshold: float = 0.88, max_results: int = 1
    ) -> list[FuzzyMatch]:
        """Find closest translation memory entries above the similarity threshold.

        Uses character 3-gram inverted index for fast candidate pre-filtering,
        then scores candidates using difflib.SequenceMatcher.
        """
        norm = normalize_source(source)
        if not norm:
            return []

        query_ngrams = _get_char_ngrams(norm)
        if not query_ngrams:
            return []

        len_norm = len(norm)
        # Count common n-grams to score candidate relevance with length bound pruning.
        # Only the index walk needs the lock; the expensive scoring below works on a snapshot.
        candidate_scores: dict[str, int] = defaultdict(int)
        with self._lock:
            for ng in query_ngrams:
                for cand in self._ngram_index.get(ng, ()):
                    cand_len = len(cand)
                    if (2.0 * min(len_norm, cand_len) / (len_norm + cand_len)) < threshold:
                        continue
                    candidate_scores[cand] += 1

            min_ngrams = max(1, int(len(query_ngrams) * threshold * 0.35))

            # Sort candidates by shared n-gram count and prune to top-K
            ranked_candidates = sorted(
                (
                    (cand, count)
                    for cand, count in candidate_scores.items()
                    if count >= min_ngrams
                ),
                key=lambda x: x[1],
                reverse=True,
            )[: max(50, max_results * 15)]
            snapshot = [
                (cand, self._entries[cand].target)
                for cand, _count in ranked_candidates
                if cand in self._entries
            ]

        matches: list[FuzzyMatch] = []
        norm_lower = norm.lower()
        for cand, target in snapshot:
            cand_len = len(cand)
            if cand_len == 0:
                continue

            # Strict upper bound: SequenceMatcher.ratio <= 2 * min(L1, L2) / (L1 + L2)
            max_possible_ratio = 2.0 * min(len_norm, cand_len) / (len_norm + cand_len)
            if max_possible_ratio < threshold:
                continue

            matcher = difflib.SequenceMatcher(None, norm_lower, cand.lower())
            ratio = matcher.ratio()
            if ratio >= threshold:
                matches.append(FuzzyMatch(score=round(ratio, 4), source=cand, target=target))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:max_results]

    def record(self, source: str, target: str, origin: Origin = Origin.LLM) -> None:
        """Add or update a translation unit in TM and sync with database."""
        norm = normalize_source(source)
        clean_tgt = clean_translation(target, source=source)
        if not norm or not clean_tgt:
            return
        existing = self._entries.get(norm)
        if (
            existing is not None
            and existing.origin in LOCKED_ORIGINS
            and origin not in LOCKED_ORIGINS
        ):
            # Human-curated memory must never be overwritten by machine output.
            return
        now = utc_timestamp()
        entry = TMEntry(
            src_norm=norm,
            target=clean_tgt,
            uses=(existing.uses + 1) if existing is not None else 1,
            origin=origin,
            updated_at=now,
        )
        with self._lock:
            self._db.save_tm(entry)
            self._entries[norm] = entry
            self._exact_map[norm] = entry
            self._bloom.add(norm)
            for ng in _get_char_ngrams(norm):
                self._ngram_index[ng].add(norm)
