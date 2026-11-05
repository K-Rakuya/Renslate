"""Pass 2 Structured LLM Distillation for 4D Relational Vectors and Personas.

Extracts multi-dimensional persona profiles and 4D Relational Vectors:
    R = <Kinship, DeltaHierarchy, Affinity, Register>
Applies algebraic projection Pi(R) into mathematically reciprocal RelationshipEdges
and persists vectors and character profiles to SQLite and JSON.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from renslate.core.db import Database, save_characters_to_json, utc_timestamp
from renslate.core.models import (
    CharacterProfile,
    KinshipKind,
    RelationalVector,
    RelationshipEdge,
    RelationshipKind,
)
from renslate.core.project import Project
from renslate.domain.relational import project_relational_vector
from renslate.engine.analyze.topology import SalienceWindow
from renslate.engine.graph import RelationshipGraph

_LOG = logging.getLogger("renslate.analyze.distill")

__all__ = [
    "DISTILLATION_SYSTEM_PROMPT",
    "DistillationResult",
    "DistilledCharacter",
    "DistilledRelationship",
    "apply_distillation_results",
    "build_distillation_prompt",
    "clean_json_markdown_fence",
    "parse_distillation_json",
    "run_pass2_distillation",
]

DISTILLATION_SYSTEM_PROMPT = """Bạn là chuyên gia ngôn ngữ học và bản địa hóa game visual novel cao cấp.
Nhiệm vụ của bạn là đọc các phân đoạn đối thoại nổi bật (High-Salience Windows) giữa các nhân vật và chưng cất thành:
1. Hồ sơ nhân vật (Personas): giới tính, vai trò, hình mẫu tâm lý (archetype), đặc trưng tính cách và phong thái.
2. Vector quan hệ 4 chiều (4D Relational Vector) cho từng cặp nhân vật:
   - kinship: một trong các giá trị sau:
     ["NONE", "SIBLING", "PARENT_CHILD", "AVUNCULAR", "GRAND_RELATION", "ROMANTIC_COUPLE", "MENTOR_DISCIPLE", "SOVEREIGN_SUBJECT", "SWORN_ALLIANCE"]
   - delta_hierarchy: độ chênh lệch thứ bậc từ -1.0 đến 1.0 (dương nếu source có vị thế cao hơn target, âm nếu thấp hơn).
   - affinity: độ thân mật / tình cảm từ -1.0 đến 1.0 (dương là thân thiết/yêu mến, âm là thù địch/ghét bỏ).
   - register: mức độ trang trọng từ 0.0 (rất suồng sã, thân mật) đến 1.0 (trang trọng, tôn nghiêm, cổ điển).

NGUYÊN TẮC:
- Bám sát bằng chứng câu thoại thực tế. Tuyệt đối không suy diễn quan hệ tình cảm nếu không có bằng chứng rõ ràng.
- Đảm bảo tính nhất quán logic trong quan hệ xã hội và thứ bậc gia đình / tổ chức.
- Trả về DUY NHẤT một chuỗi JSON hợp lệ theo định dạng cấu trúc dưới đây, không có văn bản giải thích nào khác ngoài JSON:

{
  "characters": [
    {
      "tag": "character_tag",
      "display_name_vi": "Tên tiếng Việt hoặc giữ nguyên",
      "gender": "male | female | unknown",
      "role": "Vai trò cốt truyện ngắn gọn",
      "archetype": "Hình mẫu tâm lý (tsundere, kuudere, loyal_knight, elder_mentor, protagonist, v.v.)",
      "personality_traits": ["nét 1", "nét 2"],
      "tone_and_voice": "Phong thái và ngữ điệu đối thoại",
      "socioeconomic_status": "Địa vị xã hội / giai cấp",
      "speech_style": "Phong cách ăn nói tiếng Việt"
    }
  ],
  "relationships": [
    {
      "source": "tag_a",
      "target": "tag_b",
      "kinship": "NONE",
      "delta_hierarchy": 0.0,
      "affinity": 0.5,
      "register": 0.4,
      "confidence": 0.9,
      "evidence": "Mô tả ngắn gọn căn cứ"
    }
  ]
}
"""

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DistilledCharacter:
    """Distilled character attributes from Pass 2 LLM analysis."""

    tag: str
    gender: str = "unknown"
    role: str = ""
    display_name_vi: str = ""
    archetype: str = ""
    personality_traits: tuple[str, ...] = ()
    tone_and_voice: str = ""
    socioeconomic_status: str = ""
    speech_style: str = ""
    sample_dialogue: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DistilledRelationship:
    """Distilled 4D relational vector between an interacting dyad."""

    source: str
    target: str
    vector: RelationalVector
    confidence: float = 0.8
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class DistillationResult:
    """Complete structured distillation output for all characters and pairs."""

    characters: tuple[DistilledCharacter, ...]
    relationships: tuple[DistilledRelationship, ...]


def clean_json_markdown_fence(raw_text: str) -> str:
    """Safely strip markdown code fences and conversational preamble/postamble from LLM response text."""
    cleaned = raw_text.strip()
    # 1. Regex match code block containing JSON anywhere in text
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # 2. Extract outermost JSON object { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1].strip()

    # 3. Fallback to stripping fences from start/end
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_distillation_json(raw: Any) -> DistillationResult:
    """Parse and validate LLM output into typed DistillationResult.

    Recovers safely from markdown code fences or wrapped strings.
    """
    data: dict[str, Any]
    if isinstance(raw, str):
        cleaned = clean_json_markdown_fence(raw)
        try:
            data = json.loads(cleaned)
        except Exception:
            _LOG.warning("Failed to decode distillation JSON: %s", raw[:200])
            return DistillationResult(characters=(), relationships=())
    elif isinstance(raw, Mapping):
        data = dict(raw)
    else:
        return DistillationResult(characters=(), relationships=())

    parsed_chars: list[DistilledCharacter] = []
    raw_chars = data.get("characters")
    if isinstance(raw_chars, Sequence):
        for item in raw_chars:
            if not isinstance(item, Mapping):
                continue
            tag = str(item.get("tag", "")).strip()
            if not tag:
                continue

            g = str(item.get("gender", "unknown")).lower().strip()
            if g not in ("male", "female", "unknown"):
                g = "unknown"

            raw_traits = item.get("personality_traits")
            if isinstance(raw_traits, (list, tuple)):
                traits = tuple(str(x).strip() for x in raw_traits if str(x).strip())
            elif isinstance(raw_traits, str) and raw_traits.strip():
                traits = tuple(x.strip() for x in raw_traits.split(",") if x.strip())
            else:
                traits = ()

            raw_samples = item.get("sample_dialogue")
            if isinstance(raw_samples, (list, tuple)):
                samples = tuple(str(x).strip() for x in raw_samples if str(x).strip())
            elif isinstance(raw_samples, str) and raw_samples.strip():
                samples = (raw_samples.strip(),)
            else:
                samples = ()

            parsed_chars.append(
                DistilledCharacter(
                    tag=tag,
                    gender=g,
                    role=str(item.get("role", "")).strip(),
                    display_name_vi=str(item.get("display_name_vi", "")).strip(),
                    archetype=str(item.get("archetype", "")).strip(),
                    personality_traits=traits,
                    tone_and_voice=str(item.get("tone_and_voice", "")).strip(),
                    socioeconomic_status=str(item.get("socioeconomic_status", "")).strip(),
                    speech_style=str(item.get("speech_style", "")).strip(),
                    sample_dialogue=samples,
                )
            )

    parsed_rels: list[DistilledRelationship] = []
    raw_rels = data.get("relationships")
    if isinstance(raw_rels, Sequence):
        for item in raw_rels:
            if not isinstance(item, Mapping):
                continue
            src = str(item.get("source", "")).strip()
            tgt = str(item.get("target", "")).strip()
            if not src or not tgt or src.lower() == tgt.lower():
                continue

            try:
                kinship = KinshipKind.from_str(str(item.get("kinship", "NONE")))
            except ValueError:
                kinship = KinshipKind.NONE

            try:
                dh = float(item.get("delta_hierarchy", 0.0))
            except (TypeError, ValueError):
                dh = 0.0
            dh = max(-1.0, min(1.0, dh))

            try:
                aff = float(item.get("affinity", 0.0))
            except (TypeError, ValueError):
                aff = 0.0
            aff = max(-1.0, min(1.0, aff))

            try:
                reg = float(item.get("register", 0.5))
            except (TypeError, ValueError):
                reg = 0.5
            reg = max(0.0, min(1.0, reg))

            try:
                conf = float(item.get("confidence", 0.8))
            except (TypeError, ValueError):
                conf = 0.8
            conf = max(0.0, min(1.0, conf))

            evidence = str(item.get("evidence", "")).strip()

            vec = RelationalVector(
                kinship=kinship,
                delta_hierarchy=dh,
                affinity=aff,
                register=reg,
            )

            parsed_rels.append(
                DistilledRelationship(
                    source=src,
                    target=tgt,
                    vector=vec,
                    confidence=conf,
                    evidence=evidence,
                )
            )

    return DistillationResult(
        characters=tuple(parsed_chars),
        relationships=tuple(parsed_rels),
    )


def build_distillation_prompt(
    salience_windows: Mapping[tuple[str, str], Sequence[SalienceWindow]] | Sequence[SalienceWindow],
    characters: Mapping[str, CharacterProfile],
    *,
    protagonist_tag: str | None = None,
) -> str:
    """Compose user prompt containing high-salience interaction windows for Pass 2."""
    windows_by_pair: dict[tuple[str, str], list[SalienceWindow]] = {}
    if isinstance(salience_windows, Mapping):
        for pair, wins in salience_windows.items():
            windows_by_pair[pair] = list(wins)
    else:
        for w in salience_windows:
            windows_by_pair.setdefault(w.pair, []).append(w)

    char_descriptions: list[str] = []
    for tag, p in sorted(characters.items()):
        is_mc = bool(protagonist_tag and tag.lower() == protagonist_tag.lower())
        mc_mark = " (PROTAGONIST / MC)" if is_mc else ""
        char_descriptions.append(
            f"- {p.tag}{mc_mark}: Tên='{p.display_name}', Giới tính='{p.gender}', Vai trò='{p.role}'"
        )

    pair_blocks: list[str] = []
    for (s1, s2), wins in sorted(windows_by_pair.items()):
        if not s1 or not s2:
            continue
        p1 = characters.get(s1) or characters.get(s1.lower())
        p2 = characters.get(s2) or characters.get(s2.lower())
        d1 = p1.display_name if p1 else s1
        d2 = p2.display_name if p2 else s2

        window_dialogues: list[str] = []
        for idx, w in enumerate(wins, 1):
            lines = []
            for u in w.units:
                spk = getattr(u, "speaker", "") or "Narrator"
                txt = getattr(u, "text", "") or getattr(u, "source", "") or ""
                lines.append(f'    [{spk}] "{txt}"')
            window_dialogues.append(f"  [Window {idx} (Score={w.score:.1f})]:\n" + "\n".join(lines))

        pair_blocks.append(
            f"=== Cặp tương tác: {s1} ({d1}) <-> {s2} ({d2}) ===\n"
            + "\n".join(window_dialogues)
        )

    prompt_parts = [
        "Dưới đây là danh sách nhân vật đã biết và các phân đoạn đối thoại nổi bật (High-Salience Windows):",
        "",
        "--- DANH SÁCH NHÂN VẬT ---",
        *(char_descriptions or ["(Chưa có hồ sơ trước)"]),
        "",
        "--- CÁC PHÂN ĐOẠN ĐỐI THOẠI NỔI BẬT ---",
        *(pair_blocks or ["(Không có đoạn đối thoại cụ thể)"]),
        "",
        "Hãy trích xuất hồ sơ nhân vật và 4D Relational Vector theo định dạng JSON đã hướng dẫn.",
    ]
    return "\n".join(prompt_parts)


def relationship_kind_from_vector(vec: RelationalVector) -> RelationshipKind:
    """Map 4D RelationalVector to closest semantic RelationshipKind."""
    if vec.kinship == KinshipKind.ROMANTIC_COUPLE:
        return RelationshipKind.ROMANCE
    if vec.kinship == KinshipKind.SIBLING:
        return RelationshipKind.FAMILY_SIBLING
    if vec.kinship in (KinshipKind.PARENT_CHILD, KinshipKind.AVUNCULAR, KinshipKind.GRAND_RELATION):
        return RelationshipKind.FAMILY_ELDER
    if vec.kinship == KinshipKind.MENTOR_DISCIPLE:
        return RelationshipKind.MENTOR_DISCIPLE
    if vec.kinship == KinshipKind.SOVEREIGN_SUBJECT:
        return RelationshipKind.SUPERIOR_SUBORDINATE
    if vec.affinity <= -0.3:
        return RelationshipKind.RIVAL_HOSTILE
    if vec.affinity >= 0.6:
        return RelationshipKind.FRIENDSHIP_CLOSE
    if vec.affinity >= 0.2:
        return RelationshipKind.FRIENDSHIP_CASUAL
    if abs(vec.delta_hierarchy) >= 0.3:
        return RelationshipKind.SUPERIOR_SUBORDINATE
    return RelationshipKind.FORMAL_ACQUAINTANCE


def apply_distillation_results(
    distilled: DistillationResult,
    db: Database,
    project: Project,
    *,
    graph: RelationshipGraph | None = None,
) -> tuple[int, int]:
    """Persist distilled personas and algebraically projected reciprocal relationship edges.

    Returns:
        (characters_updated_count, edges_written_count)
    """
    rel_graph = graph or RelationshipGraph.sync(db, project.relationships_path)
    existing_chars = db.get_characters()
    updated_profiles: dict[str, CharacterProfile] = dict(existing_chars)

    chars_updated = 0
    edges_written = 0

    # 1. Apply Distilled Character Personas
    with db.transaction():
        for dc in distilled.characters:
            tag = dc.tag
            cur = updated_profiles.get(tag) or next((p for p in updated_profiles.values() if p.tag.lower() == tag.lower()), None)
            canonical_tag = cur.tag if cur else tag

            disp = dc.display_name_vi or (cur.display_name if cur else canonical_tag.capitalize())
            gender = dc.gender if dc.gender != "unknown" else (cur.gender if cur else "unknown")
            role = dc.role or (cur.role if cur else "")
            archetype = dc.archetype or (cur.archetype if cur else "")
            traits = dc.personality_traits or (cur.personality_traits if cur else ())
            tone = dc.tone_and_voice or (cur.tone_and_voice if cur else "")
            status = dc.socioeconomic_status or (cur.socioeconomic_status if cur else "")
            style = dc.speech_style or (cur.speech_style if cur else "")
            samples = dc.sample_dialogue or (cur.sample_dialogue if cur else ())
            pmap = dict(cur.pronoun_map) if cur else {}

            prof = CharacterProfile(
                tag=canonical_tag,
                display_name=disp,
                gender=gender,
                role=role,
                speech_style=style,
                pronoun_map=pmap,
                archetype=archetype,
                personality_traits=traits,
                tone_and_voice=tone,
                socioeconomic_status=status,
                sample_dialogue=samples,
            )
            db.save_character(prof)
            updated_profiles[canonical_tag] = prof
            chars_updated += 1

    # 2. Algebraic Reciprocal Edge Projection
    for rel in distilled.relationships:
        src = rel.source
        tgt = rel.target
        prof_src = updated_profiles.get(src) or next((p for p in updated_profiles.values() if p.tag.lower() == src.lower()), None)
        prof_tgt = updated_profiles.get(tgt) or next((p for p in updated_profiles.values() if p.tag.lower() == tgt.lower()), None)

        canonical_src = prof_src.tag if prof_src else src
        canonical_tgt = prof_tgt.tag if prof_tgt else tgt

        g_src = prof_src.gender if prof_src else "unknown"
        g_tgt = prof_tgt.gender if prof_tgt else "unknown"

        vec_fwd = rel.vector
        vec_rev = vec_fwd.inverted()

        # Deterministic Algebraic Pronoun Projection enforcing reciprocal closure:
        # Pi(R_BA, g_B, g_A) == (p2, p1) when Pi(R_AB, g_A, g_B) == (p1, p2)
        p_fwd = project_relational_vector(vec_fwd, g_src, g_tgt)
        p_rev = project_relational_vector(vec_rev, g_tgt, g_src)

        kind = relationship_kind_from_vector(vec_fwd)
        kind_rev = relationship_kind_from_vector(vec_rev)

        note = f"Pass 2 Distillation: {rel.evidence}" if rel.evidence else "Pass 2 Distillation"

        edge_fwd = RelationshipEdge(
            source_tag=canonical_src,
            target_tag=canonical_tgt,
            kind=kind,
            speaker_pronoun=p_fwd[0],
            listener_pronoun=p_fwd[1],
            confidence=rel.confidence,
            source_type="llm",
            notes=note,
            vector=vec_fwd,
        )

        edge_rev = RelationshipEdge(
            source_tag=canonical_tgt,
            target_tag=canonical_src,
            kind=kind_rev,
            speaker_pronoun=p_rev[0],
            listener_pronoun=p_rev[1],
            confidence=rel.confidence,
            source_type="llm",
            notes=note,
            vector=vec_rev,
        )

        if rel_graph.add_edge(edge_fwd):
            edges_written += 1
        if rel_graph.add_edge(edge_rev):
            edges_written += 1

        # Update reciprocal pronoun maps
        if prof_src:
            rule_fwd = f"{p_fwd[0]} - {p_fwd[1]}"
            updated_profiles[canonical_src] = dataclasses.replace(
                prof_src, pronoun_map={**prof_src.pronoun_map, canonical_tgt: rule_fwd}
            )
        if prof_tgt:
            rule_rev = f"{p_rev[0]} - {p_rev[1]}"
            updated_profiles[canonical_tgt] = dataclasses.replace(
                prof_tgt, pronoun_map={**prof_tgt.pronoun_map, canonical_src: rule_rev}
            )

    # 3. Save graph and updated profiles
    rel_graph.save_to_db(db)
    rel_graph.save_to_json_file(project.relationships_path)

    with db.transaction():
        for prof in updated_profiles.values():
            db.save_character(prof)

    chars_file = project.characters_path
    save_characters_to_json(updated_profiles, chars_file)
    content = chars_file.read_bytes()
    db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
    db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())

    return chars_updated, edges_written


def run_pass2_distillation(
    project: Project,
    db: Database,
    llm: Any,
    salience_windows: Mapping[tuple[str, str], Sequence[SalienceWindow]] | Sequence[SalienceWindow],
    *,
    protagonist_tag: str | None = None,
) -> DistillationResult:
    """Execute Pass 2 Structured LLM Distillation."""
    characters = db.get_characters()
    user_prompt = build_distillation_prompt(
        salience_windows,
        characters,
        protagonist_tag=protagonist_tag,
    )

    try:
        raw_res, _, _ = llm.complete_json(
            DISTILLATION_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=2048,
        )
    except Exception as e:
        _LOG.warning("Pass 2 distillation LLM call failed: %s", e)
        return DistillationResult(characters=(), relationships=())

    distilled = parse_distillation_json(raw_res)
    apply_distillation_results(distilled, db, project)
    db.set_meta("pass2_distillation_completed", utc_timestamp())
    return distilled
