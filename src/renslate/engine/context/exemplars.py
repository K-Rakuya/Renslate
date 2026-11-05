"""TransGraph dyad-aware dynamic TM exemplar retrieval and micro-framing."""

from __future__ import annotations

import difflib
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from renslate.core.db import Database

__all__ = [
    "ExemplarItem",
    "TMDyadExemplarRetriever",
    "format_exemplars_section",
]


@dataclass(slots=True, frozen=True)
class ExemplarItem:
    """A verified translation exemplar retrieved from project history."""

    source: str
    target: str
    similarity: float = 1.0


def format_exemplars_section(exemplars: Sequence[ExemplarItem]) -> str:
    """Format verified exemplars into prompt section."""
    if not exemplars:
        return ""
    lines = ["### CÂU MẪU ĐÃ ĐƯỢC XÁC NHẬN TRONG DỰ ÁN (PROJECT-VERIFIED EXEMPLARS):"]
    for item in exemplars:
        lines.append(f"- {item.source} => {item.target}")
    return "\n".join(lines)


class TMDyadExemplarRetriever:
    """Retrieves verified translation exemplars grounded in character interaction context."""

    def __init__(self, db: Database | Any) -> None:
        self._db = db

    def retrieve_for_dyad(
        self,
        speaker: str | None,
        listener: str | None,
        sources: Sequence[str],
        max_results: int = 2,
        min_similarity: float = 0.6,
    ) -> list[ExemplarItem]:
        """Retrieve verified exemplars matching active dyad or source tokens."""
        if not sources or self._db is None or max_results <= 0:
            return []

        cleaned_sources = [s.strip().lower() for s in sources if s and s.strip()]
        if not cleaned_sources:
            return []

        candidates: list[tuple[str, str, int, bool]] = []
        seen_pairs: set[tuple[str, str]] = set()

        if speaker and hasattr(self._db, "execute"):
            try:
                cur = self._db.execute(
                    "SELECT source, target, origin, status FROM units "
                    "WHERE speaker = ? AND target IS NOT NULL AND target != '' "
                    "AND (origin IN ('manual', 'imported', 'seed', 'user', 'review') "
                    "OR status = 'reviewed')",
                    (speaker,),
                )
                for row in cur.fetchall():
                    src = str(row["source"]).strip()
                    tgt = str(row["target"]).strip()
                    if src and tgt and (src, tgt) not in seen_pairs:
                        seen_pairs.add((src, tgt))
                        candidates.append((src, tgt, 2, True))
            except (sqlite3.Error, KeyError, TypeError):
                pass

        if hasattr(self._db, "execute"):
            try:
                cur = self._db.execute(
                    "SELECT src_norm, target, uses, origin FROM tm "
                    "WHERE target IS NOT NULL AND target != '' "
                    "AND (origin IN ('manual', 'imported', 'seed', 'user', 'review', 'reviewed') "
                    "OR uses >= 2) ORDER BY uses DESC",
                )
                for row in cur.fetchall():
                    src = str(row["src_norm"]).strip()
                    tgt = str(row["target"]).strip()
                    uses = int(row["uses"]) if hasattr(row, "__getitem__") else 1
                    if src and tgt and (src, tgt) not in seen_pairs:
                        seen_pairs.add((src, tgt))
                        candidates.append((src, tgt, uses, False))
            except (sqlite3.Error, KeyError, TypeError):
                pass
        elif hasattr(self._db, "all_tm"):
            try:
                for entry in self._db.all_tm():
                    src = entry.src_norm.strip()
                    tgt = entry.target.strip()
                    origin_val = getattr(entry.origin, "value", str(entry.origin)).lower()
                    is_verified = (
                        origin_val in ("manual", "imported", "seed", "user", "review", "reviewed")
                        or entry.uses >= 2
                    )
                    if is_verified and src and tgt and (src, tgt) not in seen_pairs:
                        seen_pairs.add((src, tgt))
                        candidates.append((src, tgt, entry.uses, False))
            except Exception:
                pass

        if not candidates:
            return []

        scored: list[tuple[float, bool, int, str, str]] = []
        for cand_src, cand_tgt, uses, is_speaker_match in candidates:
            cand_norm = cand_src.lower()
            best_sim = 0.0
            for query in cleaned_sources:
                if query == cand_norm:
                    sim = 1.0
                else:
                    sim = difflib.SequenceMatcher(None, query, cand_norm).ratio()
                if sim > best_sim:
                    best_sim = sim

            if best_sim >= min_similarity:
                scored.append((best_sim, is_speaker_match, uses, cand_src, cand_tgt))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

        results: list[ExemplarItem] = []
        for sim, _spk, _uses, src, tgt in scored:
            results.append(ExemplarItem(source=src, target=tgt, similarity=round(sim, 3)))
            if len(results) >= max_results:
                break

        return results

    @staticmethod
    def format_exemplars_section(exemplars: Sequence[ExemplarItem]) -> str:
        """Format verified exemplars into prompt section."""
        return format_exemplars_section(exemplars)
