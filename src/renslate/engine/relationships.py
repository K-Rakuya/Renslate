"""LLM-assisted discovery of character relationships and Vietnamese address forms.

The heuristic relationship inference in ``analyze`` guesses from keywords in role/name
fields (and assumes "protagonist + female character" means romance). Vietnamese address
forms (tôi/anh/em/cậu/ngươi/...) depend on age, intimacy, setting and tone, none of which
keywords capture. This module reads *actual exchanges* between each frequently interacting
pair, asks the local LLM to judge the relationship and the two address forms, validates the
answer, and stores directed ``RelationshipEdge`` objects with ``source_type="llm"``.

Human-curated (``manual``) edges are never touched. Results are persisted per pair so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from renslate.core.db import Database, save_characters_to_json
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.analyze import is_collective_entity, is_non_human_or_meta
from renslate.engine.context import InterlocutorResolver
from renslate.engine.graph import RelationshipGraph

_LOG = logging.getLogger("renslate.relationships")

__all__ = [
    "RELATIONSHIP_SYSTEM_PROMPT",
    "PairEvidence",
    "bootstrap_relationships_with_llm",
    "build_relationship_prompt",
    "collect_pair_evidence",
    "parse_pair_result",
]

_META_DONE_PREFIX = "rel_llm_done:"
_META_BOOTSTRAPPED = "relationships_bootstrapped"
_MAX_PRONOUN_LEN = 24
_TAG_OR_VAR = re.compile(r"\{[^{}]*\}|\[[^\[\]]*\]|⟦[^⟦⟧]*⟧")
_WS = re.compile(r"\s+")

RELATIONSHIP_SYSTEM_PROMPT = (
    "Bạn là biên dịch viên visual novel giàu kinh nghiệm, chuyên chọn cách xưng hô tiếng Việt.\n"
    "Với mỗi cặp nhân vật, đọc các đoạn thoại thực tế rồi xác định:\n"
    "1. kind: một trong " + ", ".join(k.value for k in RelationshipKind) + ".\n"
    "2. a_to_b: cách nhân vật A nói với B, gồm 'self' (A tự xưng) và 'other' (A gọi B).\n"
    "3. b_to_a: tương tự theo chiều ngược lại.\n"
    "4. confidence: số từ 0 đến 1 thể hiện mức chắc chắn.\n"
    "5. evidence: một câu ngắn nêu căn cứ.\n\n"
    "Nguyên tắc:\n"
    "- Dựa vào bằng chứng trong đoạn thoại: tuổi tác/vai vế, độ thân thiết, bối cảnh "
    "(hiện đại, giả tưởng, cổ trang), giọng điệu (kính cẩn, thân mật, thù địch).\n"
    "- Cách xưng hô có thể BẤT ĐỐI XỨNG (ví dụ một bên gọi 'em', bên kia gọi 'cậu').\n"
    "- 'self' và 'other' phải là từ/cụm từ ngắn tiếng Việt, khác nhau, không phải tên riêng.\n"
    "- Không đủ bằng chứng về tình cảm thì KHÔNG đoán là ROMANCE; chọn kind trung tính "
    "(FORMAL_ACQUAINTANCE hoặc FRIENDSHIP_CASUAL) và cách xưng hô lịch sự.\n"
    "- Nếu quan hệ thay đổi theo cốt truyện, chọn cách xưng hô đại diện nhất và ghi chú trong "
    "evidence.\n\n"
    "Trả về DUY NHẤT một đối tượng JSON, khóa là chuỗi cặp đúng như được cung cấp:\n"
    '{"tagA|tagB": {"kind": "FRIENDSHIP_CLOSE", '
    '"a_to_b": {"self": "...", "other": "..."}, '
    '"b_to_a": {"self": "...", "other": "..."}, '
    '"confidence": 0.8, "evidence": "..."}}\n'
    "Không giải thích ngoài JSON."
)


@dataclass(slots=True)
class PairEvidence:
    """Dialogue evidence for one unordered character pair, in story order."""

    a: str
    b: str
    turns: int = 0
    lines: list[tuple[str, str]] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.a}|{self.b}"


def _clean_line(text: str, limit: int = 180) -> str:
    cleaned = _WS.sub(" ", _TAG_OR_VAR.sub(" ", text)).strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def collect_pair_evidence(
    units: Sequence[Unit],
    characters: Mapping[str, CharacterProfile],
) -> dict[tuple[str, str], PairEvidence]:
    """Count who talks to whom (via the interlocutor resolver) and keep the exchanges."""
    dialogue = sorted((u for u in units if u.kind is UnitKind.DIALOGUE), key=lambda u: u.seq)
    if not dialogue:
        return {}

    known_names = {tag: p.display_name for tag, p in characters.items() if p.display_name}
    participants: dict[str, set[str]] = defaultdict(set)
    for u in dialogue:
        if u.speaker and u.speaker.strip():
            participants[u.label or ""].add(u.speaker.strip())

    def _skip(tag: str) -> bool:
        disp = characters[tag].display_name if tag in characters else ""
        return is_non_human_or_meta(tag, disp) or is_collective_entity(tag, disp)

    evidence: dict[tuple[str, str], PairEvidence] = {}
    resolver = InterlocutorResolver()
    prev: Unit | None = None
    for u in dialogue:
        res = resolver.resolve(
            u,
            prev_unit=prev,
            active_speakers=participants[u.label or ""],
            known_names=known_names,
        )
        prev = u
        spk = (u.speaker or "").strip()
        lis = (res.listener or "").strip()
        if not spk or not lis or res.is_monologue or spk.lower() == lis.lower():
            continue
        if _skip(spk) or _skip(lis):
            continue
        a, b = sorted((spk, lis))
        ev = evidence.setdefault((a, b), PairEvidence(a, b))
        ev.turns += 1
        ev.lines.append((spk, _clean_line(u.source)))
    return evidence


def _stratified(lines: Sequence[tuple[str, str]], k: int) -> list[tuple[str, str]]:
    """Pick k lines evenly across the story so the model sees how the pair evolves."""
    usable = [ln for ln in lines if len(ln[1]) >= 8]
    if len(usable) <= k:
        return list(usable)
    step = (len(usable) - 1) / (k - 1) if k > 1 else 0
    idx = sorted({round(i * step) for i in range(k)})
    return [usable[i] for i in idx]


def _describe(tag: str, characters: Mapping[str, CharacterProfile]) -> str:
    p = characters.get(tag)
    if p is None:
        return f"- {tag}"
    parts = [f"- {tag}: {p.display_name or tag}"]
    if p.gender and p.gender != "unknown":
        parts.append(f"giới tính {p.gender}")
    if p.role:
        parts.append(f"vai trò {p.role}")
    if p.archetype:
        parts.append(f"hình mẫu {p.archetype}")
    return " | ".join(parts)


def build_relationship_prompt(
    pairs: Sequence[PairEvidence],
    characters: Mapping[str, CharacterProfile],
    *,
    lines_per_pair: int = 8,
) -> str:
    """Compose the user prompt for a group of pairs."""
    blocks: list[str] = []
    for ev in pairs:
        samples = _stratified(ev.lines, lines_per_pair)
        sample_text = "\n".join(f'  [{spk}] "{text}"' for spk, text in samples) or "  (không có)"
        blocks.append(
            f'Cặp "{ev.key}" ({ev.turns} lượt thoại):\n'
            f"{_describe(ev.a, characters)}\n{_describe(ev.b, characters)}\n"
            f"Các đoạn thoại theo thứ tự cốt truyện:\n{sample_text}"
        )
    return "Xác định quan hệ và cách xưng hô cho các cặp sau:\n\n" + "\n\n".join(blocks)


@dataclass(frozen=True, slots=True)
class _ParsedPair:
    kind: RelationshipKind
    a_to_b: tuple[str, str]
    b_to_a: tuple[str, str]
    confidence: float
    evidence: str


def _clean_pronoun(value: Any) -> str:
    text = _WS.sub(" ", str(value or "")).strip().strip("'\"“”")
    if not text or len(text) > _MAX_PRONOUN_LEN or "\n" in text:
        return ""
    return text


def _parse_direction(raw: Any) -> tuple[str, str] | None:
    """Accept {"self","other"} objects or "xưng - gọi" strings."""
    if isinstance(raw, Mapping):
        self_p = _clean_pronoun(raw.get("self") or raw.get("speaker"))
        other_p = _clean_pronoun(raw.get("other") or raw.get("listener"))
    elif isinstance(raw, str) and " - " in raw:
        left, _, right = raw.partition(" - ")
        self_p, other_p = _clean_pronoun(left), _clean_pronoun(right)
    else:
        return None
    if not self_p or not other_p or self_p.lower() == other_p.lower():
        return None
    return self_p, other_p


def parse_pair_result(data: Any) -> _ParsedPair | None:
    """Validate one pair's LLM answer; return None when unusable."""
    if not isinstance(data, Mapping):
        return None
    try:
        kind = RelationshipKind.from_str(str(data.get("kind", "")))
    except ValueError:
        return None
    a_to_b = _parse_direction(data.get("a_to_b"))
    b_to_a = _parse_direction(data.get("b_to_a"))
    if a_to_b is None and b_to_a is None:
        return None
    try:
        confidence = float(data.get("confidence", 0.8))
    except (TypeError, ValueError):
        confidence = 0.8
    confidence = min(0.9, max(0.3, confidence))
    evidence = _clean_line(str(data.get("evidence") or ""), limit=200)
    return _ParsedPair(
        kind=kind,
        a_to_b=a_to_b or ("", ""),
        b_to_a=b_to_a or ("", ""),
        confidence=confidence,
        evidence=evidence,
    )


def _apply_pair(
    graph: RelationshipGraph,
    profiles: dict[str, CharacterProfile],
    ev: PairEvidence,
    parsed: _ParsedPair,
) -> tuple[int, set[str]]:
    """Write the directed edges; return (edges written, tags whose profile changed)."""
    written = 0
    changed: set[str] = set()
    note = f"LLM: {parsed.evidence}" if parsed.evidence else "LLM"
    for src, tgt, forms in ((ev.a, ev.b, parsed.a_to_b), (ev.b, ev.a, parsed.b_to_a)):
        if not forms[0] or not forms[1]:
            continue
        existing = graph.get_edge(src, tgt)
        if existing is not None and existing.source_type == "manual":
            continue
        edge = RelationshipEdge(
            source_tag=src,
            target_tag=tgt,
            kind=parsed.kind,
            speaker_pronoun=forms[0],
            listener_pronoun=forms[1],
            confidence=parsed.confidence,
            source_type="llm",
            notes=note,
        )
        if not graph.add_edge(edge):
            continue
        written += 1
        # Keep the profile's pronoun_map consistent with the graph so the prompt never gives
        # two different rules for one pair. Entries typed by a human (no edge behind them)
        # are left alone.
        prof = profiles.get(src)
        if prof is not None:
            rule = f"{forms[0]} - {forms[1]}"
            current = prof.pronoun_map.get(tgt)
            if current is None or existing is not None:
                if current != rule:
                    profiles[src] = dataclasses.replace(
                        prof, pronoun_map={**prof.pronoun_map, tgt: rule}
                    )
                    changed.add(src)
    return written, changed


def bootstrap_relationships_with_llm(
    project: Project,
    db: Database,
    llm: Any,
    *,
    max_pairs: int = 30,
    min_turns: int = 4,
    pairs_per_call: int = 3,
    lines_per_pair: int = 8,
    force: bool = False,
    graph: RelationshipGraph | None = None,
) -> dict[str, int]:
    """Ask the LLM to judge the most active character pairs and store the result.

    Returns counters: ``pairs`` (considered), ``resolved`` (usable answers),
    ``edges`` (directed edges written).
    """
    stats = {"pairs": 0, "resolved": 0, "edges": 0}
    characters = db.get_characters()
    evidence = collect_pair_evidence(db.all_units(), characters)
    ranked = sorted(
        (ev for ev in evidence.values() if ev.turns >= min_turns),
        key=lambda ev: ev.turns,
        reverse=True,
    )[: max(0, max_pairs)]

    graph = graph if graph is not None else RelationshipGraph.sync(db, project.relationships_path)

    todo: list[PairEvidence] = []
    for ev in ranked:
        if not force and db.get_meta(_META_DONE_PREFIX + ev.key):
            continue
        e_ab, e_ba = graph.get_bilateral(ev.a, ev.b)
        if (
            e_ab is not None
            and e_ba is not None
            and e_ab.source_type == "manual"
            and e_ba.source_type == "manual"
        ):
            continue
        todo.append(ev)
    stats["pairs"] = len(todo)

    profiles = dict(characters)
    changed_tags: set[str] = set()
    step = max(1, pairs_per_call)
    for i in range(0, len(todo), step):
        group = todo[i : i + step]
        prompt = build_relationship_prompt(group, characters, lines_per_pair=lines_per_pair)
        try:
            result, _, _ = llm.complete_json(RELATIONSHIP_SYSTEM_PROMPT, prompt, max_tokens=1024)
        except Exception as exc:
            _LOG.warning("Bootstrapping quan hệ cho %s thất bại: %s", [g.key for g in group], exc)
            continue
        if not isinstance(result, Mapping):
            continue

        lowered = {str(k).strip().lower(): v for k, v in result.items()}
        with db.transaction():
            group_changed: set[str] = set()
            for ev in group:
                parsed = parse_pair_result(lowered.get(ev.key.lower()))
                if parsed is None:
                    continue
                written, changed = _apply_pair(graph, profiles, ev, parsed)
                stats["resolved"] += 1
                stats["edges"] += written
                group_changed |= changed
                db.set_meta(_META_DONE_PREFIX + ev.key, datetime.now(timezone.utc).isoformat())
            # Persist edges, profiles and the "done" flags together so a crash never leaves a
            # pair marked done without its edges.
            graph.save_to_db(db)
            for tag in group_changed:
                db.save_character(profiles[tag])
            changed_tags |= group_changed

    RelationshipGraph.sync(db, project.relationships_path, direction="db_to_json")

    if changed_tags:
        chars_file = project.characters_path
        save_characters_to_json(db.get_characters(), chars_file)
        db.set_meta("characters_json_hash", hashlib.sha256(chars_file.read_bytes()).hexdigest())
        db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())

    db.set_meta(_META_BOOTSTRAPPED, datetime.now(timezone.utc).isoformat())
    return stats
