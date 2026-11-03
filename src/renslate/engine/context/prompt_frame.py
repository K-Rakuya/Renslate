"""Dynamic relational prompt frame formatting for dialogue units."""

from __future__ import annotations

from typing import TYPE_CHECKING

from renslate.core.models import CharacterProfile, Unit
from renslate.domain.policies.mob_roles import MobRoleClassifier
from renslate.engine.context.resolver import (
    _NON_DIALOGUE_TAGS,
    _PROTAGONIST_TAGS,
    ResolvedInterlocutor,
    invert_reciprocal_pronoun_rule,
)

if TYPE_CHECKING:
    from renslate.engine.graph import RelationshipGraph

__all__ = [
    "_format_pronoun_pair",
    "_format_relational_rule",
    "format_rich_dialogue_line",
]


def _format_pronoun_pair(rule_text: str) -> str:
    rule_clean = rule_text.strip()
    if " - " in rule_clean and not rule_clean.startswith("xưng"):
        parts = rule_clean.split(" - ", 1)
        return f"xưng '{parts[0].strip()}', gọi '{parts[1].strip()}'"
    return rule_clean


def _format_relational_rule(
    p_spk: CharacterProfile,
    p_tgt: CharacterProfile,
    rule_text: str,
) -> str:
    rule_clean = rule_text.strip()
    if " - " in rule_clean and not rule_clean.startswith("xưng"):
        parts = [p.strip() for p in rule_clean.split(" - ", 1)]
        self_p, target_p = parts[0], parts[1]
        return (
            f"  * Khi {p_spk.display_name} ({p_spk.tag}) nói với "
            f"{p_tgt.display_name} ({p_tgt.tag}): Tự xưng (I/me/my) là '{self_p}', "
            f"gọi đối phương (you/your) là '{target_p}'. "
            f"(QUY TẮC: Mọi câu có 'you' [Do you..., Why are you...] "
            f"BẮT BUỘC dịch là '{target_p}', TUYỆT ĐỐI CẤM dịch 'you' thành '{self_p}')."
        )
    return (
        f"  * {p_spk.display_name} ({p_spk.tag}) khi nói với "
        f"{p_tgt.display_name} ({p_tgt.tag}): {rule_clean}"
    )


_EMOTION_VIETNAMESE = {
    "blush": "e thẹn (blush)",
    "angry": "tức giận (angry)",
    "happy": "vui vẻ (happy)",
    "crying": "khóc/buồn bã (crying)",
    "sad": "buồn bã (sad)",
    "surprised": "bất ngờ (surprised)",
    "shocked": "bàng hoàng (shocked)",
    "smirk": "cười khẩy (smirk)",
    "smile": "mỉm cười (smile)",
    "worried": "lo lắng (worried)",
    "scared": "sợ hãi (scared)",
}


def _resolve_unit_emotion(unit: Unit, resolved_emotion: str | None, spk_lower: str) -> str:
    parts: list[str] = []
    if resolved_emotion:
        parts.append(f"Cảm xúc: {resolved_emotion}")

    if hasattr(unit, "stage") and unit.stage is not None:
        active_sprites = getattr(unit.stage, "active_sprites", ())
        for sp in active_sprites:
            if getattr(sp, "tag", "").lower() == spk_lower and getattr(sp, "expression", ""):
                raw_expr = sp.expression.strip().lower()
                vn_expr = _EMOTION_VIETNAMESE.get(raw_expr, raw_expr)
                parts.append(f"Biểu cảm: {vn_expr}")
                break

    return " | " + " | ".join(parts) if parts else ""


def format_rich_dialogue_line(
    unit: Unit,
    resolved: ResolvedInterlocutor,
    char_map: dict[str, CharacterProfile],
    graph: RelationshipGraph | None = None,
    concise: bool = False,
) -> str:
    """Format an individual dialogue line with rich relational and interlocutor framing."""
    text = unit.masked_source or unit.source
    if not unit.speaker:
        return f"{unit.uid} | {text}"

    spk_clean = unit.speaker.strip()
    spk_lower = spk_clean.lower()
    cp_spk = char_map.get(spk_clean) or char_map.get(spk_lower)

    # 1. Monologue / Soliloquy Case
    if resolved.is_monologue or spk_lower in _NON_DIALOGUE_TAGS:
        if spk_lower in _NON_DIALOGUE_TAGS:
            spk_label = unit.speaker
        elif cp_spk and cp_spk.display_name:
            spk_label = cp_spk.display_name
        else:
            spk_label = unit.speaker

        emotion_str = _resolve_unit_emotion(unit, resolved.detected_emotion, spk_lower)
        if concise and spk_lower not in _NON_DIALOGUE_TAGS:
            return f"{unit.uid} | [{spk_label}{emotion_str}] {text}"
        return f"{unit.uid} | [Độc thoại/Nội tâm: {spk_label}{emotion_str}] {text}"

    # 2. Interlocutor Case with Resolved Listener
    if resolved.listener:
        lis_clean = resolved.listener.strip()
        lis_lower = lis_clean.lower()
        cp_lis = char_map.get(lis_clean) or char_map.get(lis_lower)

        spk_name = cp_spk.display_name if cp_spk and cp_spk.display_name else spk_clean
        lis_name = cp_lis.display_name if cp_lis and cp_lis.display_name else lis_clean

        rel_part = ""
        edge = None
        if graph is not None:
            edge = graph.get_edge(spk_clean, lis_clean) or graph.get_edge(spk_lower, lis_lower)
            if edge is None and cp_spk and cp_lis:
                edge = graph.get_edge(cp_spk.tag, cp_lis.tag)

        if edge is not None:
            kind_str = edge.kind.value if hasattr(edge.kind, "value") else str(edge.kind)
            if edge.speaker_pronoun and edge.listener_pronoun:
                p_desc = (
                    f" ({spk_name} xưng '{edge.speaker_pronoun}', gọi '{edge.listener_pronoun}')"
                )
            elif edge.speaker_pronoun:
                p_desc = f" ({spk_name} xưng '{edge.speaker_pronoun}')"
            elif edge.listener_pronoun:
                p_desc = f" ({spk_name} gọi '{edge.listener_pronoun}')"
            else:
                p_desc = ""
            rel_part = f" | Quan hệ: {kind_str}{p_desc}"

        if not rel_part and cp_spk and cp_spk.pronoun_map:
            tgt_key = cp_lis.tag if cp_lis else lis_clean
            tgt_name = cp_lis.display_name if cp_lis else lis_clean
            p_rule = (
                cp_spk.pronoun_map.get(tgt_key)
                or cp_spk.pronoun_map.get(tgt_name)
                or cp_spk.pronoun_map.get(tgt_key.lower())
                or (cp_spk.pronoun_map.get(tgt_name.lower()) if tgt_name else None)
            )
            if not p_rule:
                is_mc = (
                    tgt_key.lower() in _PROTAGONIST_TAGS
                    or (cp_lis is not None and "protagonist" in (cp_lis.role or "").lower())
                )
                if is_mc:
                    p_rule = cp_spk.pronoun_map.get("mc") or cp_spk.pronoun_map.get("player")
            if p_rule and tgt_key.lower() != (cp_spk.tag or "").lower():
                rule_clean = p_rule.strip()
                if " - " in rule_clean and not rule_clean.startswith("xưng"):
                    parts = [p.strip() for p in rule_clean.split(" - ", 1)]
                    rel_part = f" | Quan hệ: {spk_name} xưng '{parts[0]}', gọi '{parts[1]}'"
                else:
                    rel_part = f" | Quan hệ: {rule_clean}"

        if not rel_part and cp_lis and cp_lis.pronoun_map:
            rev_key = cp_spk.tag if cp_spk else spk_clean
            rev_name = cp_spk.display_name if cp_spk else spk_clean
            rev_rule = (
                cp_lis.pronoun_map.get(rev_key)
                or cp_lis.pronoun_map.get(rev_name)
                or cp_lis.pronoun_map.get(rev_key.lower())
                or (cp_lis.pronoun_map.get(rev_name.lower()) if rev_name else None)
            )
            if not rev_rule:
                is_spk_mc = (
                    rev_key.lower() in _PROTAGONIST_TAGS
                    or (cp_spk is not None and "protagonist" in (cp_spk.role or "").lower())
                )
                if is_spk_mc:
                    rev_rule = cp_lis.pronoun_map.get("mc") or cp_lis.pronoun_map.get("player")
            if rev_rule and rev_key.lower() != (cp_lis.tag or "").lower():
                inv_p = invert_reciprocal_pronoun_rule(rev_rule)
                if inv_p:
                    parts = [p.strip() for p in inv_p.split(" - ", 1)]
                    rel_part = f" | Quan hệ: {spk_name} xưng '{parts[0]}', gọi '{parts[1]}'"
        if not rel_part and (
            (cp_spk is None and MobRoleClassifier.classify(spk_clean))
            or (cp_lis is None and MobRoleClassifier.classify(lis_clean))
            or (cp_spk and cp_spk.role in ("service", "authority"))
            or (cp_lis and cp_lis.role in ("service", "authority"))
        ):
            mob_pair = MobRoleClassifier.infer_pronoun_pair(spk_clean, lis_clean)
            if mob_pair:
                spk_pro, lis_pro = mob_pair
                rel_part = f" | Vai trò xã hội: {spk_name} xưng '{spk_pro}', gọi '{lis_pro}'"

        emotion_str = _resolve_unit_emotion(unit, resolved.detected_emotion, spk_lower)
        if concise:
            return f"{unit.uid} | [{spk_name} -> {lis_name}{emotion_str}] {text}"
        return (
            f"{unit.uid} | [Người nói: {spk_name} -> Người nghe: {lis_name}"
            f"{rel_part}{emotion_str}] {text}"
        )

    # 3. Dialogue without resolved listener
    if cp_spk and cp_spk.display_name and cp_spk.display_name.lower() != spk_lower:
        spk_label = f"{cp_spk.display_name} ({unit.speaker})"
    else:
        spk_label = unit.speaker

    emotion_str = _resolve_unit_emotion(unit, resolved.detected_emotion, spk_lower)
    if concise:
        return f"{unit.uid} | [{spk_label}{emotion_str}] {text}"
    return f"{unit.uid} | [Người nói: {spk_label}{emotion_str}] {text}"
