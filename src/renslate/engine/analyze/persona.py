"""Character persona inference, NLP heuristics, and LLM bootstrapping."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from renslate.core.db import (
    Database,
    save_characters_to_json,
)
from renslate.core.models import (
    CharacterProfile,
    RelationshipKind,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.analyze.sampling import stratified_speaker_samples
from renslate.engine.analyze.stylometrics import StylometricSummary, extract_stylometrics
from renslate.engine.analyze.tropes import (
    COLLECTIVE_TAGS,
    NON_HUMAN_TAGS,
    NON_INTERACTION_TAGS,
    PROTAGONIST_TAGS,
    detect_archetype,
)
from renslate.engine.context import infer_scene_cues
from renslate.engine.knowledge import collect_gender_evidence

_LOG = logging.getLogger("renslate.analyze.persona")

_FEMININE_CUES = re.compile(
    r"\b(?:she|her|hers|herself|girl|woman|lady|sister|daughter|mother|mom|queen|empress|female|maiden|princess|miss|mrs|ms|madam|heroine|sorceress|witch|milf)\b",
    re.IGNORECASE,
)
_MASCULINE_CUES = re.compile(
    r"\b(?:he|him|his|himself|boy|man|guy|brother|son|father|dad|king|emperor|male|prince|lord|sir|mr|hero|dude)\b",
    re.IGNORECASE,
)
_FEMININE_NAMES = re.compile(
    r"^(?:charlotte|alice|eileen|diana|evelyn|luna|iris|emily|sarah|mary|anna|clara|lucy|sophia|"
    r"olivia|emma|chloe|mia|lily|grace|zoe|eva|elena|stella|amber|ruby|rose|victoria|valerie|karyn|zuri|"
    r"kaede|tomoe|yui|aoi|akane|"
    r"[a-z]*(?:ko|ka|na|ri|mi|ne|tsu|ki|yu))$",
    re.IGNORECASE,
)
_MASCULINE_NAMES = re.compile(
    r"^(?:jin|orson|kevin|john|david|michael|alex|mike|bob|tom|jack|george|arthur|"
    r"[a-z]*(?:taro|ro|shi|to|ya|suke))$",
    re.IGNORECASE,
)

PERSONA_SYSTEM_PROMPT = (
    "Bạn là chuyên gia biên kịch và bản địa hóa game visual novel cao cấp.\n"
    "Nhiệm vụ của bạn là phân tích sâu hồ sơ tính cách nhân vật dựa trên:\n"
    "1. Danh xưng/hô ngữ người khác dùng khi gọi nhân vật (phản ánh địa vị, giai cấp, tôn ti "
    "trật tự).\n"
    "2. Mạng lưới bạn thoại và tương tác đối thoại 2 chiều thực tế (đặc biệt là với MC, bộc lộ "
    "thái độ, cảm xúc và quan hệ thân sơ).\n"
    "3. Không gian/phân cảnh xuất hiện tiêu biểu.\n"
    "4. Ngữ cảnh dẫn chuyện kịch bản (Narration), chỉ số hành vi ngôn ngữ (stylometrics), và các "
    "câu thoại mẫu thực tế để phục vụ dịch thuật sang tiếng Việt chuẩn xác và tự nhiên.\n\n"
    "NGUYÊN TẮC BẮT BUỘC:\n"
    "- Bám sát câu thoại, tương tác 2 chiều và ngữ cảnh kịch bản đã cung cấp. Tuyệt đối KHÔNG "
    "suy diễn hoặc gán ép các khuôn mẫu xa lạ nếu câu thoại không thể hiện.\n"
    "- Nếu kịch bản có ghi nhận chứng cứ giới tính (Nam/Nữ), bắt buộc tuân theo. Nếu không đủ căn "
    "cứ, hãy trả về 'unknown' thay vì đoán mò.\n"
    "- Với pronoun_map: Bắt buộc dùng đúng tag của đối tượng tương tác (đặc biệt là tag của Nhân vật chính "
    "MC được chỉ định) làm key xưng hô. Dựa vào các cặp thoại 2 chiều thực tế để xác định cặp "
    "xưng hô đối ứng tiếng Việt chính xác tuyệt đối (ví dụ: 'mc': 'anh - em', 'tôi - cậu', "
    "'cháu - chú', 'ta - ngươi'). Tuyệt đối không dùng các key trừu tượng không có trong kịch bản.\n\n"
    "Với mỗi nhân vật, hãy xác định đầy đủ các trường sau:\n"
    "- display_name_vi: Tên tiếng Việt tự nhiên (danh từ chung chỉ nghề nghiệp/chức danh như 'Guard', 'Mercenary' "
    "dịch nghĩa tự nhiên; tên riêng nước ngoài giữ nguyên).\n"
    "- gender: 'female', 'male', hoặc 'unknown'.\n"
    "- role: Vai trò cốt truyện ngắn gọn (ví dụ: 'Hiệp sĩ', 'Nhân vật chính', 'Thương nhân').\n"
    "- archetype: Hình mẫu tâm lý nhân vật chuẩn xác (ví dụ: 'tsundere', 'kuudere', 'dandere', 'elder_mentor', "
    "'loyal_knight', 'rebel_warrior', 'inquisitive_apprentice', 'childhood_friend', 'noble_lady', "
    "'cunning_villain', 'protagonist', 'narrator', 'supporting_cast' hoặc 'unknown').\n"
    "- personality_traits: Mảng 2-4 nét tính cách chủ đạo dựa trên bằng chứng kịch bản.\n"
    "- tone_and_voice: Mô tả giọng điệu, âm vực, và phong thái đối thoại thể hiện trong lời thoại.\n"
    "- socioeconomic_status: Địa vị hoặc tầng lớp xã hội thể hiện trong bối cảnh.\n"
    "- speech_style: Phong cách ăn nói tổng quát bằng tiếng Việt.\n"
    "- sample_dialogue: Mảng gồm 2-3 câu thoại tiêu biểu nhất được trích dẫn NGUYÊN VĂN từ các câu thoại mẫu "
    "đã cung cấp trong kịch bản. Tuyệt đối không tự bịa đặt câu thoại.\n"
    "- pronoun_map: Quy tắc xưng hô đối ứng tiếng Việt với các bạn thoại thực tế (key là tag bạn thoại, "
    "value có dạng 'xưng - gọi').\n\n"
    "Trả về DUY NHẤT một đối tượng JSON có khóa là các tag nhân vật được yêu cầu:\n"
    "{\n"
    '  "<character_tag>": {\n'
    '    "display_name_vi": "<tên hiển thị tiếng Việt>",\n'
    '    "gender": "female | male | unknown",\n'
    '    "role": "<vai trò cốt truyện>",\n'
    '    "archetype": "<hình mẫu tâm lý>",\n'
    '    "personality_traits": ["<nét tính cách 1>", "<nét tính cách 2>"],\n'
    '    "tone_and_voice": "<mô tả phong thái, ngữ điệu đối thoại>",\n'
    '    "socioeconomic_status": "<địa vị / tầng lớp xã hội>",\n'
    '    "speech_style": "<phong cách đối thoại tiếng Việt>",\n'
    '    "sample_dialogue": [\n'
    '      "<câu thoại trích dẫn nguyên văn 1 từ dữ liệu mẫu đã cung cấp>",\n'
    '      "<câu thoại trích dẫn nguyên văn 2 từ dữ liệu mẫu đã cung cấp>"\n'
    "    ],\n"
    '    "pronoun_map": {\n'
    '      "<partner_tag>": "<xưng> - <gọi>"\n'
    "    }\n"
    "  }\n"
    "}\n"
    "Tuyệt đối không giải thích ngoài JSON."
)


def infer_character_role_nlp(tag: str, line_count: int, total_lines: int) -> str:
    """Infer character role heuristic based on dialogue volume and protagonist tags."""
    tag_lower = tag.lower()
    if tag_lower in ("t", "th", "thought", "thoughts", "thinking"):
        return "Độc thoại nội tâm (Inner Monologue)"
    if tag_lower in ("narrator", "narration", "none"):
        return "Người dẫn chuyện (Narrator)"
    if tag_lower in PROTAGONIST_TAGS:
        return "Nhân vật chính (Protagonist)"
    if total_lines <= 0 or line_count == 0:
        return ""
    ratio = line_count / total_lines
    if ratio >= 0.04:
        return "Nhân vật chính tuyến (Main Cast)"
    if ratio >= 0.008:
        return "Nhân vật phụ (Supporting Cast)"
    return "Nhân vật quần chúng (NPC)"


def infer_character_gender_nlp(name: str, sources: Sequence[str]) -> str:
    """Infer grammatical character gender from universal names and pronoun co-occurrence."""
    name_clean = re.sub(r"\b([A-Za-z])\s+([a-z]{2,})\b", r"\1\2", name).strip()
    name_lower = name_clean.lower()
    is_fem_name = bool(_FEMININE_CUES.search(name_lower) or _FEMININE_NAMES.match(name_lower))
    is_masc_name = bool(_MASCULINE_CUES.search(name_lower) or _MASCULINE_NAMES.match(name_lower))

    if is_fem_name and not is_masc_name:
        return "female"
    if is_masc_name and not is_fem_name:
        return "male"

    try:
        name_pat = re.compile(rf"\b{re.escape(name_clean)}\b", re.IGNORECASE)
    except re.error:
        return "unknown"

    fem_matches = 0
    masc_matches = 0
    for text in sources:
        if name_pat.search(text):
            fem_matches += len(_FEMININE_CUES.findall(text))
            masc_matches += len(_MASCULINE_CUES.findall(text))

    if fem_matches >= 1 and (masc_matches == 0 or fem_matches >= masc_matches * 2):
        return "female"
    if masc_matches >= 1 and (fem_matches == 0 or masc_matches >= fem_matches * 2):
        return "male"

    return "unknown"


def is_collective_entity(tag: str, display_name: str = "") -> bool:
    """Check if tag or display_name refers to a collective group of people."""
    t = tag.lower().strip()
    if t in COLLECTIVE_TAGS:
        return True
    dn = display_name.lower().strip()
    return any(
        c in dn
        for c in (
            "everyone",
            "crowd",
            "mob",
            "audience",
            "people",
            "bystanders",
            "mọi người",
            "đám đông",
            "quần chúng",
        )
    )


def is_non_human_or_meta(tag: str, display_name: str = "") -> bool:
    """Check if tag or display_name refers to a non-human entity, system, or narrator."""
    t = tag.lower().strip()
    if t in NON_HUMAN_TAGS:
        return True
    dn = display_name.lower().strip()
    return any(
        nh in dn
        for nh in (
            "system",
            "announcer",
            "hệ thống",
            "người dẫn",
            "narrator",
            "nội tâm",
            "độc thoại",
            "voice",
        )
    )


def resolve_pronoun_pair_from_relationship(
    kind: RelationshipKind,
    speaker_gender: str,
    listener_gender: str,
    *,
    is_speaker_senior: bool = True,
) -> tuple[str, str]:
    """Map RelationshipKind and genders to authentic Vietnamese addressing pronouns."""
    spk_g = (speaker_gender or "unknown").lower()
    lis_g = (listener_gender or "unknown").lower()

    if kind == RelationshipKind.FAMILY_ELDER:
        if is_speaker_senior:
            spk = "mẹ" if spk_g == "female" else ("cha" if spk_g == "male" else "ta")
            return (spk, "con")
        lis = "mẹ" if lis_g == "female" else ("cha" if lis_g == "male" else "người")
        return ("con", lis)

    if kind == RelationshipKind.FAMILY_SIBLING:
        if is_speaker_senior:
            spk = "chị" if spk_g == "female" else "anh"
            return (spk, "em")
        lis = "chị" if lis_g == "female" else "anh"
        return ("em", lis)

    if kind == RelationshipKind.SUPERIOR_SUBORDINATE:
        if is_speaker_senior:
            return ("trẫm" if spk_g in ("male", "female") else "ta", "khanh")
        return ("thần", "bệ hạ")

    if kind == RelationshipKind.MENTOR_DISCIPLE:
        if is_speaker_senior:
            return ("ta", "con")
        return ("con", "thầy" if lis_g == "male" else "sư phụ")

    if kind == RelationshipKind.SERVANT_MASTER:
        if is_speaker_senior:
            lis = "cô" if lis_g == "female" else ("cậu" if lis_g == "male" else "ngươi")
            return ("ta", lis)
        lis = "tiểu thư" if lis_g == "female" else ("thiếu gia" if lis_g == "male" else "chủ nhân")
        return ("em" if spk_g == "female" else "tôi", lis)

    if kind == RelationshipKind.RIVAL_HOSTILE:
        return ("ta", "ngươi")

    if kind == RelationshipKind.FRIENDSHIP_CLOSE:
        return ("mình", "cậu")

    if kind == RelationshipKind.FRIENDSHIP_CASUAL:
        return ("tôi", "cậu")

    if kind == RelationshipKind.ROMANCE:
        if spk_g == "male" and lis_g == "female":
            return ("anh", "em")
        if spk_g == "female" and lis_g == "male":
            return ("em", "anh")
        if spk_g == "female" and lis_g == "female":
            return ("mình", "cậu")
        if spk_g == "male" and lis_g == "male":
            return ("tôi", "cậu")
        return ("anh", "em") if is_speaker_senior else ("em", "anh")

    # Default: FORMAL_ACQUAINTANCE
    lis = "cô" if lis_g == "female" else ("anh" if lis_g == "male" else "bạn")
    return ("tôi", lis)


_ALLOWED_ROLE_MODIFIERS = frozenset(
    {
        "the", "a", "an", "and", "of", "to", "elder", "older", "younger", "little",
        "big", "first", "second", "loving", "kind", "beloved", "gentle", "strict",
        "wise", "honored", "honourable", "head", "chief", "royal", "personal",
        "loyal", "trusted", "devoted", "step", "foster", "adoptive", "biological",
        "sworn", "sole", "only", "favorite", "favourite", "childhood", "s",
        "người", "đứa", "già", "nhỏ", "ruột", "kế", "nuôi", "đẻ", "hiền", "cả",
        "hai", "ba", "tư", "chính", "thân", "thân sinh", "yêu", "quý", "yêu quý",
        "hiền từ", "nghiêm nghị", "tổng", "trưởng", "hoàng gia", "trung thành",
        "tận tuỵ", "tận tụy", "kết nghĩa", "duy nhất", "đích tôn", "chưa cưới", "sắp cưới",
        "dinh", "thự", "thời", "thơ", "ấu", "không", "đội", "trời", "chung",
    }
)


def infer_semantic_relationship(
    tag: str,
    tc: dict[str, Any],
    partner_tag: str,
    partner_info: dict[str, Any],
    *,
    is_mc: bool = False,
    partner_is_mc: bool = False,
) -> tuple[RelationshipKind, bool] | None:
    """Infer semantic relationship kind and seniority between two interacting characters.

    RFC 001 Section 3.1: Speculative romance heuristic on opposite genders has been
    eliminated. Romance relationship is strictly assigned when supported by explicit
    semantic cues (e.g. lover, wife, husband, heroine). Ambiguous interactions default
    to formal acquaintance or friendship.
    """
    if is_non_human_or_meta(tag, str(tc.get("display_name", ""))) or is_non_human_or_meta(
        partner_tag, str(partner_info.get("display_name", ""))
    ):
        return None

    if is_collective_entity(
        partner_tag, str(partner_info.get("display_name", ""))
    ) or is_collective_entity(tag, str(tc.get("display_name", ""))):
        return (RelationshipKind.FORMAL_ACQUAINTANCE, True)

    def _combine_text(t: str, info: dict[str, Any]) -> str:
        parts = [
            t,
            str(info.get("display_name", "")),
            str(info.get("role", "")),
            str(info.get("speech_style", "")),
            str(info.get("archetype", "")),
            " ".join(str(x) for x in info.get("personality_traits", ())),
        ]
        return " ".join(parts).lower()

    text_self = _combine_text(tag, tc)
    text_partner = _combine_text(partner_tag, partner_info)

    def _is_mentioned_in(t: str, disp: str, text: str) -> bool:
        if t and len(t) >= 2 and re.search(rf"\b{re.escape(t.lower())}\b", text):
            return True
        return bool(disp and len(disp) >= 2 and re.search(rf"\b{re.escape(disp.lower())}\b", text))

    def _is_external_or_narrative(text: str, mc_cues: set[str]) -> bool:
        all_rel_cues = (
            r"father|mother|parent|dad|mom|daughter|son|child|cha|mẹ|bố|con|"
            r"sister|brother|chị|anh|em|wife|husband|vợ|chồng|lover|người yêu|"
            r"boyfriend|girlfriend|fiancee|fiance|master|mentor|teacher|sensei|sư phụ|thầy|"
            r"servant|maid|butler|apprentice|disciple|student|học trò|đồ nhi|đệ tử|"
            r"friend|bạn thân|thanh mai trúc mã|rival|enemy|đối thủ|kẻ thù"
        )
        m = re.search(
            rf"\b(?:{all_rel_cues})(?:\s+\w+){{0,2}}\s+(?:of|to|của)\s+([^,.;\n]+)",
            text,
            re.IGNORECASE,
        )
        if m:
            target_phrase = m.group(1).strip().lower()
            if not any(re.search(rf"\b{re.escape(k)}\b", target_phrase) for k in mc_cues):
                return True

        m_poss = re.search(
            rf"\b([A-Za-zÀ-ỹ0-9_]+)(?:'s|\u2019s)\s+(?:\w+\s+)*(?:{all_rel_cues})\b",
            text,
            re.IGNORECASE,
        )
        if m_poss:
            possessor = m_poss.group(1).strip().lower()
            if not any(re.search(rf"\b{re.escape(k)}\b", possessor) for k in mc_cues):
                return True

        if re.search(
            rf"\b(?:his|her|their)\s+(?:\w+\s+)*(?:{all_rel_cues})\b",
            text,
            re.IGNORECASE,
        ):
            return True

        return bool(
            re.search(
                r"\b(?:grieving|mourning|late|missing|lost|searching\s+for|devoted\s+(?:husband|wife|father|mother)|(?:husband|wife|father|mother|brother|sister)\s+and)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _is_bound_relation(
        source_tag_str: str,
        source_info: dict[str, Any],
        target_tag_str: str,
        target_info: dict[str, Any],
        cues: Sequence[str],
        *,
        is_target_protagonist: bool,
    ) -> bool:
        text = _combine_text(source_tag_str, source_info)
        has_any_cue = any(re.search(rf"\b{c}\b", text, re.I) for c in cues)
        if not has_any_cue:
            return False

        target_disp = str(target_info.get("display_name", ""))
        if _is_mentioned_in(target_tag_str, target_disp, text):
            return True

        if not is_target_protagonist:
            return False

        target_names = {target_tag_str.lower(), target_disp.lower()} - {"", "unknown"}
        mc_cues = {"nhân vật chính", "protagonist", "hero", "player", "mc", "main character"} | target_names
        cues_re = "|".join(re.escape(c) for c in cues)
        mc_re = "|".join(re.escape(m) for m in mc_cues)

        if re.search(rf"\b(?:{cues_re})(?:\s+\w+){{0,2}}\s+(?:of|to|của)\s+(?:the\s+)?(?:{mc_re})\b", text, re.I):
            return True
        if re.search(rf"\b(?:{mc_re})(?:'s|\u2019s| của)?\s+(?:\w+\s+){{0,2}}(?:{cues_re})\b", text, re.I):
            return True

        if _is_external_or_narrative(text, mc_cues):
            return False

        source_name_tokens = {source_tag_str.lower()} | set(
            re.findall(r"[\wÀ-ỹ]+", str(source_info.get("display_name", "")).lower())
        )
        allowed_words = _ALLOWED_ROLE_MODIFIERS | mc_cues | source_name_tokens
        role_self = str(source_info.get("role", "")).strip().lower()
        disp_self = str(source_info.get("display_name", "")).strip().lower()
        for candidate_str in (source_tag_str.lower(), role_self, disp_self):
            clean = re.sub(r"[()]", " / ", candidate_str).strip()
            parts = [p.strip() for p in re.split(r"[/,\u2013\u2014|]", clean) if p.strip()]
            for p in parts:
                if any(re.search(rf"\b{c}\b", p, re.I) for c in cues):
                    p_clean = p
                    for c in sorted(cues, key=len, reverse=True):
                        p_clean = re.sub(rf"\b{re.escape(c)}\b", " ", p_clean, flags=re.I)
                    remaining = {w.strip().lower() for w in re.findall(r"[\wÀ-ỹ]+", p_clean) if w.strip()}
                    if remaining.issubset(allowed_words):
                        return True

        return False

    # 1. Family Elder & Junior
    elder_cues = (
        "mother", "mom", "mama", "father", "dad", "papa", "parent", "uncle", "aunt",
        "grandpa", "grandma", "grandfather", "grandmother", "mẹ", "cha", "bố", "bác",
        "chú", "cô", "dì", "ông", "bà",
    )
    child_cues = (
        "daughter", "son", "child", "children", "con gái", "con trai", "đứa con", "cháu gái", "cháu trai",
    )
    partner_is_elder = _is_bound_relation(partner_tag, partner_info, tag, tc, elder_cues, is_target_protagonist=is_mc)
    self_is_elder = _is_bound_relation(tag, tc, partner_tag, partner_info, elder_cues, is_target_protagonist=partner_is_mc)
    partner_is_child = _is_bound_relation(partner_tag, partner_info, tag, tc, child_cues, is_target_protagonist=is_mc)
    self_is_child = _is_bound_relation(tag, tc, partner_tag, partner_info, child_cues, is_target_protagonist=partner_is_mc)

    if (partner_is_elder or self_is_child) and (is_mc or (not self_is_elder and not partner_is_child)):
        return (RelationshipKind.FAMILY_ELDER, False)

    if (self_is_elder or partner_is_child) and (partner_is_mc or (not partner_is_elder and not self_is_child)):
        return (RelationshipKind.FAMILY_ELDER, True)

    # 2. Sibling
    sibling_cues = ("sister", "brother", "sibling", "chị", "em gái", "em trai", "anh trai")
    partner_is_sib = _is_bound_relation(partner_tag, partner_info, tag, tc, sibling_cues, is_target_protagonist=is_mc)
    self_is_sib = _is_bound_relation(tag, tc, partner_tag, partner_info, sibling_cues, is_target_protagonist=partner_is_mc)

    if partner_is_sib or self_is_sib:
        self_has_junior = bool(re.search(r"\b(little|younger|junior|em gái|em trai|muội|đệ)\b", text_self))
        partner_has_junior = bool(re.search(r"\b(little|younger|junior|em gái|em trai|muội|đệ)\b", text_partner))
        self_has_senior = bool(re.search(r"\b(elder|older|senior|chị|anh trai|huynh|tỷ)\b", text_self))
        partner_has_senior = bool(re.search(r"\b(elder|older|senior|chị|anh trai|huynh|tỷ)\b", text_partner))

        if self_has_junior or partner_has_senior:
            is_senior = False
        elif self_has_senior or partner_has_junior or is_mc:
            is_senior = True
        elif partner_is_mc:
            is_senior = False
        else:
            is_senior = tag < partner_tag
        return (RelationshipKind.FAMILY_SIBLING, is_senior)

    # 3. Explicit Romance Cues (Prioritized before Royalty for intimate relationships)
    romance_cues = (
        "lover", "wife", "husband", "girlfriend", "boyfriend", "fiancee", "fiance",
        "bride", "vợ", "chồng", "người yêu", "bạn gái",
    )
    partner_is_romance = _is_bound_relation(partner_tag, partner_info, tag, tc, romance_cues, is_target_protagonist=is_mc)
    self_is_romance = _is_bound_relation(tag, tc, partner_tag, partner_info, romance_cues, is_target_protagonist=partner_is_mc)

    if partner_is_romance or self_is_romance:
        return (RelationshipKind.ROMANCE, is_mc)

    # Heroine role: only romantic between MC and designated heroine partner
    partner_is_heroine = bool(re.search(r"\bheroine\b", text_partner))
    self_is_heroine = bool(re.search(r"\bheroine\b", text_self))
    self_not_female = str(tc.get("gender", "")).lower() != "female"
    partner_not_female = str(partner_info.get("gender", "")).lower() != "female"
    if (is_mc and partner_is_heroine and self_not_female) or (
        partner_is_mc and self_is_heroine and partner_not_female
    ):
        return (RelationshipKind.ROMANCE, is_mc)

    # 4. Superior / Royalty
    supreme_royal_cues = (
        "queen", "king", "emperor", "empress", "bệ hạ", "nữ hoàng", "hoàng đế",
    )
    subordinate_royal_cues = (
        "princess", "prince", "lord", "commander", "captain", "công chúa", "hoàng tử", "chỉ huy",
    )

    self_has_supreme = any(re.search(rf"\b{c}\b", text_self) for c in supreme_royal_cues)
    partner_has_supreme = any(re.search(rf"\b{c}\b", text_partner) for c in supreme_royal_cues)
    self_has_sub = any(re.search(rf"\b{c}\b", text_self) for c in subordinate_royal_cues)
    partner_has_sub = any(re.search(rf"\b{c}\b", text_partner) for c in subordinate_royal_cues)

    if self_has_supreme and not partner_has_supreme:
        return (RelationshipKind.SUPERIOR_SUBORDINATE, True)
    if partner_has_supreme and not self_has_supreme:
        return (RelationshipKind.SUPERIOR_SUBORDINATE, False)
    if self_has_sub and not partner_has_sub and not partner_has_supreme:
        return (RelationshipKind.SUPERIOR_SUBORDINATE, True)
    if partner_has_sub and not self_has_sub and not self_has_supreme:
        return (RelationshipKind.SUPERIOR_SUBORDINATE, False)

    # 5. Mentor / Disciple
    mentor_cues = (
        "master", "mentor", "teacher", "sensei", "professor", "sư phụ", "thầy", "tông chủ",
    )
    disciple_cues = (
        "apprentice", "disciple", "student", "học trò", "đồ nhi", "đệ tử",
    )
    partner_is_mentor = _is_bound_relation(partner_tag, partner_info, tag, tc, mentor_cues, is_target_protagonist=is_mc)
    self_is_mentor = _is_bound_relation(tag, tc, partner_tag, partner_info, mentor_cues, is_target_protagonist=partner_is_mc)
    partner_is_disciple = _is_bound_relation(partner_tag, partner_info, tag, tc, disciple_cues, is_target_protagonist=is_mc)
    self_is_disciple = _is_bound_relation(tag, tc, partner_tag, partner_info, disciple_cues, is_target_protagonist=partner_is_mc)

    if (partner_is_mentor or self_is_disciple) and (is_mc or (not self_is_mentor and not partner_is_disciple)):
        return (RelationshipKind.MENTOR_DISCIPLE, False)
    if (self_is_mentor or partner_is_disciple) and (partner_is_mc or (not partner_is_mentor and not self_is_disciple)):
        return (RelationshipKind.MENTOR_DISCIPLE, True)

    # 6. Servant / Master
    servant_cues = ("maid", "butler", "servant", "attendant", "hầu gái", "người hầu", "quản gia")
    partner_is_servant = _is_bound_relation(partner_tag, partner_info, tag, tc, servant_cues, is_target_protagonist=is_mc)
    self_is_servant = _is_bound_relation(tag, tc, partner_tag, partner_info, servant_cues, is_target_protagonist=partner_is_mc)
    if partner_is_servant and not self_is_servant:
        return (RelationshipKind.SERVANT_MASTER, True)
    if self_is_servant and not partner_is_servant:
        return (RelationshipKind.SERVANT_MASTER, False)

    # 7. Rival / Hostile
    hostile_cues = (
        "rival", "enemy", "villain", "nemesis", "antagonist", "hostile", "traitor",
        "kẻ thù", "đối thủ", "phản bội", "kẻ thù không đội trời chung",
    )
    partner_is_hostile = _is_bound_relation(partner_tag, partner_info, tag, tc, hostile_cues, is_target_protagonist=is_mc)
    self_is_hostile = _is_bound_relation(tag, tc, partner_tag, partner_info, hostile_cues, is_target_protagonist=partner_is_mc)
    if partner_is_hostile or self_is_hostile:
        return (RelationshipKind.RIVAL_HOSTILE, True)

    # 8. Close Friend
    friend_cues = (
        "childhood friend", "best friend", "close friend", "thanh mai trúc mã", "bạn thân", "bạn thời thơ ấu",
    )
    partner_is_friend = _is_bound_relation(partner_tag, partner_info, tag, tc, friend_cues, is_target_protagonist=is_mc)
    self_is_friend = _is_bound_relation(tag, tc, partner_tag, partner_info, friend_cues, is_target_protagonist=partner_is_mc)
    if partner_is_friend or self_is_friend:
        return (RelationshipKind.FRIENDSHIP_CLOSE, True)

    # 9. Default: Formal Acquaintance
    return (RelationshipKind.FORMAL_ACQUAINTANCE, True)


def infer_persona_fallback(
    tag: str,
    dialogues: Sequence[str],
    stylometrics: StylometricSummary,
    role: str = "",
    gender: str = "",
    display_name: str = "",
) -> dict[str, Any]:
    """Infer persona attributes using archetype detector and dialogue samples."""
    archetype, traits, tone, status = detect_archetype(
        tag=tag,
        role=role,
        stylometrics=stylometrics,
        display_name=display_name,
    )

    clean_samples = [s.strip() for s in dialogues if s.strip() and len(s.strip()) > 5]
    if len(clean_samples) <= 3:
        sample_dlg = tuple(clean_samples)
    else:
        selected: list[str] = [clean_samples[0]]
        mid_idx = len(clean_samples) // 2
        if clean_samples[mid_idx] not in selected:
            selected.append(clean_samples[mid_idx])
        if clean_samples[-1] not in selected and len(selected) < 3:
            selected.append(clean_samples[-1])
        sample_dlg = tuple(selected[:3])

    return {
        "archetype": archetype,
        "personality_traits": traits,
        "tone_and_voice": tone,
        "socioeconomic_status": status,
        "sample_dialogue": sample_dlg,
    }


def find_protagonist_tag(
    characters: Mapping[str, CharacterProfile],
    units: Sequence[Unit],
) -> str | None:
    """Identify the protagonist / main character tag from project profiles and script."""
    for tag, prof in characters.items():
        t_low = tag.lower()
        if t_low in PROTAGONIST_TAGS:
            return tag
        if prof.role and "protagonist" in prof.role.lower():
            return tag
        if (
            prof.display_name
            and prof.display_name.lower() in ("mc", "player", "hero", "protagonist")
        ):
            return tag

    dialogue_units = [u for u in units if u.kind is UnitKind.DIALOGUE and u.speaker]
    if dialogue_units:
        counts = Counter(u.speaker for u in dialogue_units if u.speaker)
        for tag, _ in counts.most_common(5):
            if tag.lower() in PROTAGONIST_TAGS:
                return tag

    return None


def collect_character_narration_context(
    units: Sequence[Unit],
    tags: Sequence[str],
    characters: Mapping[str, CharacterProfile],
    max_lines: int = 3,
) -> dict[str, list[str]]:
    """Extract descriptive narration lines mentioning characters to ground persona analysis."""
    wanted_names: dict[str, list[str]] = {}
    for tag in tags:
        names = [tag]
        prof = characters.get(tag)
        if prof and prof.display_name and prof.display_name.lower() != tag.lower():
            names.append(prof.display_name)
        wanted_names[tag] = names

    patterns: dict[str, re.Pattern[str]] = {}
    for tag, names in wanted_names.items():
        sub_pats = [r"\b" + re.escape(n) + r"\b" for n in names if len(n) >= 2]
        if sub_pats:
            patterns[tag] = re.compile("|".join(sub_pats), re.IGNORECASE)

    collected: dict[str, list[str]] = {tag: [] for tag in tags}
    for u in units:
        if u.kind is not UnitKind.DIALOGUE or u.speaker:
            continue
        text = u.source.strip()
        if len(text) < 15 or len(text) > 220:
            continue
        for tag, pat in patterns.items():
            if len(collected[tag]) < max_lines and pat.search(text):
                clean_text = " ".join(text.split())
                if clean_text not in collected[tag]:
                    collected[tag].append(clean_text)

    return collected


def extract_interactive_dialogue_exchanges(
    units: Sequence[Unit],
    tags: Sequence[str],
    characters: Mapping[str, CharacterProfile],
    mc_tag: str | None = None,
    max_exchanges: int = 2,
) -> dict[str, list[str]]:
    """Extract consecutive two-way dialogue exchanges (e.g. MC <-> Character)."""
    dialogues = [
        u
        for u in units
        if u.kind is UnitKind.DIALOGUE
        and u.speaker
        and u.speaker.strip()
        and u.speaker.strip().lower() not in NON_INTERACTION_TAGS
    ]
    if not dialogues:
        return {t: [] for t in tags}

    display_names: dict[str, str] = {}
    for t in tags:
        p = characters.get(t)
        display_names[t] = p.display_name if (p and p.display_name) else t.capitalize()
    if mc_tag and mc_tag not in display_names:
        p = characters.get(mc_tag)
        display_names[mc_tag] = p.display_name if (p and p.display_name) else mc_tag.capitalize()

    tag_lookup = {t.lower(): t for t in tags}
    if mc_tag:
        tag_lookup[mc_tag.lower()] = mc_tag

    partner_counts: dict[str, Counter[str]] = {t: Counter() for t in tags}
    for i in range(len(dialogues) - 1):
        u1, u2 = dialogues[i], dialogues[i + 1]
        s1, s2 = (u1.speaker or "").strip(), (u2.speaker or "").strip()
        if u1.label and u2.label and u1.label != u2.label:
            continue
        if s1 and s2 and s1.lower() != s2.lower() and abs(u2.seq - u1.seq) <= 4:
            s1_key = tag_lookup.get(s1.lower()) or s1
            s2_key = tag_lookup.get(s2.lower()) or s2
            if s1_key in partner_counts:
                partner_counts[s1_key][s2_key] += 1
            if s2_key in partner_counts:
                partner_counts[s2_key][s1_key] += 1

    exchanges_by_tag: dict[str, list[str]] = {t: [] for t in tags}

    for tag in tags:
        target_partner: str | None = None
        if mc_tag and tag.lower() == mc_tag.lower():
            if partner_counts[tag]:
                target_partner = partner_counts[tag].most_common(1)[0][0]
        elif mc_tag and partner_counts[tag].get(mc_tag, 0) > 0:
            target_partner = mc_tag
        elif partner_counts[tag]:
            target_partner = partner_counts[tag].most_common(1)[0][0]

        if not target_partner:
            continue

        p_prof = characters.get(target_partner)
        p_name = display_names.get(
            target_partner,
            (
                p_prof.display_name
                if (p_prof and p_prof.display_name)
                else target_partner.capitalize()
            ),
        )
        tag_name = display_names.get(tag, tag.capitalize())

        candidates: list[str] = []
        for i in range(len(dialogues) - 1):
            u1, u2 = dialogues[i], dialogues[i + 1]
            s1, s2 = (u1.speaker or "").strip(), (u2.speaker or "").strip()
            if u1.label and u2.label and u1.label != u2.label:
                continue
            if abs(u2.seq - u1.seq) > 4:
                continue
            s1_k = tag_lookup.get(s1.lower()) or s1
            s2_k = tag_lookup.get(s2.lower()) or s2
            if (s1_k == tag and s2_k == target_partner) or (s1_k == target_partner and s2_k == tag):
                t1 = " ".join(u1.source.strip().split())
                t2 = " ".join(u2.source.strip().split())
                if 5 <= len(t1) <= 180 and 5 <= len(t2) <= 180:
                    n1 = tag_name if s1 == tag else p_name
                    n2 = tag_name if s2 == tag else p_name
                    cand = f'{n1}: "{t1}" ➔ {n2}: "{t2}"'
                    if cand not in candidates:
                        candidates.append(cand)

        if not candidates:
            continue

        if len(candidates) <= max_exchanges:
            exchanges_by_tag[tag] = candidates
        else:
            step = len(candidates) / max_exchanges
            picked = [candidates[int(j * step)] for j in range(max_exchanges)]
            exchanges_by_tag[tag] = picked

    return exchanges_by_tag


def collect_incoming_vocatives(
    units: Sequence[Unit],
    tags: Sequence[str],
    characters: Mapping[str, CharacterProfile],
    max_vocatives: int = 3,
) -> dict[str, list[str]]:
    """Extract vocatives, honorific titles, and direct address forms used by others to
    call characters."""
    dialogues = [u for u in units if u.kind is UnitKind.DIALOGUE]

    honorific_prefix = (
        r"\b(Lady|Lord|Master|Doctor|Dr\.|Professor|Prof\.|Captain|Commander|Princess|Prince|"
        r"King|Queen|Sister|Miss|Mr\.|Mrs\.|Ms\.|Sir|Madam|Mistress|Boss|Officer)\s+"
    )
    honorific_suffix = (
        r"(?:-(?:sama|san|kun|chan|senpai|sensei|dono|tan|shishou|neesan|niisan))\b"
    )

    name_patterns: dict[str, list[tuple[re.Pattern[str], str]]] = {}
    for tag in tags:
        names = [tag]
        prof = characters.get(tag)
        if prof and prof.display_name and prof.display_name.lower() != tag.lower():
            names.append(prof.display_name)

        pats: list[tuple[re.Pattern[str], str]] = []
        for n in names:
            if len(n) < 2:
                continue
            escaped = re.escape(n)
            pats.append(
                (
                    re.compile(
                        honorific_prefix + escaped + r"\b",
                        re.IGNORECASE,
                    ),
                    "prefix",
                )
            )
            pats.append(
                (
                    re.compile(
                        escaped + honorific_suffix,
                        re.IGNORECASE,
                    ),
                    "suffix",
                )
            )
            pats.append(
                (
                    re.compile(
                        r"(?:^|[.!?]\s*)" + escaped + r"[,!]|,\s*" + escaped + r"(?:[.!?]|$)",
                        re.IGNORECASE,
                    ),
                    "direct",
                )
            )
        name_patterns[tag] = pats

    vocatives_by_tag: dict[str, list[str]] = {t: [] for t in tags}
    vocative_counters: dict[str, Counter[str]] = {t: Counter() for t in tags}

    for u in dialogues:
        text = u.source.strip()
        if len(text) < 3 or len(text) > 300:
            continue
        spk = (u.speaker or "").strip()

        for tag in tags:
            if spk and spk.lower() == tag.lower():
                continue

            for pat, ptype in name_patterns[tag]:
                m = pat.search(text)
                if m:
                    raw_match = m.group(0).strip(" ,.!?\"'")
                    if ptype in ("prefix", "suffix"):
                        clean_form = " ".join(raw_match.split())
                        vocative_counters[tag][clean_form] += 3
                    elif ptype == "direct":
                        prof = characters.get(tag)
                        d_name = (
                            prof.display_name
                            if prof and prof.display_name
                            else tag.capitalize()
                        )
                        vocative_counters[tag][f"Gọi trực tiếp: '{d_name}'"] += 1

    for tag in tags:
        counts = vocative_counters[tag]
        if counts:
            top_items = [item for item, _ in counts.most_common(max_vocatives)]
            vocatives_by_tag[tag] = top_items

    return vocatives_by_tag


def extract_interaction_network(
    units: Sequence[Unit],
    tags: Sequence[str],
    characters: Mapping[str, CharacterProfile],
    max_partners: int = 3,
) -> dict[str, list[str]]:
    """Calculate the primary conversational co-occurrence network for characters."""
    dialogues = [
        u
        for u in units
        if u.kind is UnitKind.DIALOGUE
        and u.speaker
        and u.speaker.strip()
        and u.speaker.strip().lower() not in NON_INTERACTION_TAGS
    ]

    tag_lookup = {t.lower(): t for t in tags}
    partner_counts: dict[str, Counter[str]] = {t: Counter() for t in tags}
    for i in range(len(dialogues) - 1):
        u1, u2 = dialogues[i], dialogues[i + 1]
        s1, s2 = (u1.speaker or "").strip(), (u2.speaker or "").strip()
        if u1.label and u2.label and u1.label != u2.label:
            continue
        if not s1 or not s2 or s1.lower() == s2.lower() or abs(u2.seq - u1.seq) > 4:
            continue
        s1_key = tag_lookup.get(s1.lower())
        s2_key = tag_lookup.get(s2.lower())
        if s1_key:
            partner_counts[s1_key][s2_key or s2] += 1
        if s2_key:
            partner_counts[s2_key][s1_key or s1] += 1

    network_by_tag: dict[str, list[str]] = {t: [] for t in tags}
    for tag in tags:
        counts = partner_counts[tag]
        total = sum(counts.values())
        if total == 0:
            continue
        partners_formatted: list[str] = []
        for p_tag, c in counts.most_common(max_partners):
            prof = characters.get(p_tag)
            p_name = prof.display_name if prof and prof.display_name else p_tag.capitalize()
            pct = round((c / total) * 100)
            partners_formatted.append(f"{p_name} ({pct}%)")
        network_by_tag[tag] = partners_formatted

    return network_by_tag


def extract_salient_locations(
    units: Sequence[Unit],
    tags: Sequence[str],
    max_locations: int = 2,
) -> dict[str, list[str]]:
    """Determine the most frequent locations and scene contexts where characters speak."""
    tag_lookup = {t.lower(): t for t in tags}
    location_counts: dict[str, Counter[str]] = {t: Counter() for t in tags}
    dialogues = [u for u in units if u.kind is UnitKind.DIALOGUE and u.speaker]

    for u in dialogues:
        spk = (u.speaker or "").strip()
        spk_tag = tag_lookup.get(spk.lower())
        if spk_tag:
            lbl = u.label or ""
            loc, _ = infer_scene_cues(lbl)
            if loc:
                location_counts[spk_tag][loc] += 1
            elif lbl:
                lbl_lower = lbl.lower()
                if lbl_lower not in ("start", "prologue", "splashscreen", "main_menu", "common"):
                    clean_lbl = lbl.replace("_", " ").strip().title()
                    location_counts[spk_tag][f"Cảnh {clean_lbl}"] += 1

    out: dict[str, list[str]] = {}
    for tag in tags:
        counts = location_counts[tag]
        if counts:
            out[tag] = [loc for loc, _ in counts.most_common(max_locations)]
        else:
            out[tag] = []
    return out


def build_persona_user_prompt(
    chunk_tags: Sequence[str],
    existing_chars: Mapping[str, CharacterProfile],
    samples_by_speaker: Mapping[str, Sequence[str]],
    stylometrics_by_speaker: Mapping[str, StylometricSummary],
    narration_by_speaker: Mapping[str, Sequence[str]] | None = None,
    gender_evidence_by_speaker: Mapping[str, Any] | None = None,
    mc_tag: str | None = None,
    max_samples: int = 4,
    interactive_exchanges_by_speaker: Mapping[str, Sequence[str]] | None = None,
    vocatives_by_speaker: Mapping[str, Sequence[str]] | None = None,
    interaction_partners_by_speaker: Mapping[str, Sequence[str]] | None = None,
    salient_locations_by_speaker: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Build user prompt with stylometrics, script context, and formatted dialogue samples."""
    char_blocks: list[str] = []
    for tag in chunk_tags:
        prof = existing_chars.get(tag)
        disp = prof.display_name if prof else tag.capitalize()
        samples = samples_by_speaker.get(tag, [])
        sty = stylometrics_by_speaker.get(tag)
        sty_desc = sty.summary_text() if sty else "Không có dữ liệu thống kê"

        gender_hint = ""
        if gender_evidence_by_speaker and tag in gender_evidence_by_speaker:
            ev = gender_evidence_by_speaker[tag]
            m_count = getattr(ev, "male", 0)
            f_count = getattr(ev, "female", 0)
            if m_count >= 3 and m_count > f_count * 2:
                gender_hint = f" (Chứng cứ kịch bản: Nam, {m_count} đại từ he/him)"
            elif f_count >= 3 and f_count > m_count * 2:
                gender_hint = f" (Chứng cứ kịch bản: Nữ, {f_count} đại từ she/her)"

        loc_lines = (
            salient_locations_by_speaker.get(tag, []) if salient_locations_by_speaker else []
        )
        loc_formatted = ""
        if loc_lines:
            loc_formatted = f"\n  Không gian xuất hiện tiêu biểu: {', '.join(loc_lines)}"

        net_lines = (
            interaction_partners_by_speaker.get(tag, []) if interaction_partners_by_speaker else []
        )
        net_formatted = ""
        if net_lines:
            net_formatted = f"\n  Mạng lưới bạn thoại chính: {', '.join(net_lines)}"

        voc_lines = vocatives_by_speaker.get(tag, []) if vocatives_by_speaker else []
        voc_formatted = ""
        if voc_lines:
            voc_formatted = f"\n  Danh xưng/Hô ngữ người khác dùng khi gọi: {', '.join(voc_lines)}"

        narr_lines = narration_by_speaker.get(tag, []) if narration_by_speaker else []
        narr_formatted = ""
        if narr_lines:
            narr_formatted = "\n  Ngữ cảnh xuất hiện (Narration kịch bản):\n    " + "\n    ".join(
                f'- "{nl}"' for nl in narr_lines
            )

        exchanges = (
            interactive_exchanges_by_speaker.get(tag, [])
            if interactive_exchanges_by_speaker
            else []
        )
        exchanges_formatted = ""
        if exchanges:
            ex_header = "\n  Tương tác đối thoại 2 chiều thực tế (với MC / bạn thoại):\n    "
            exchanges_formatted = ex_header + "\n    ".join(
                f"- {ex}" for ex in exchanges
            )

        if samples:
            samples_formatted = "\n    ".join(
                f'{i+1}. "{s}"' for i, s in enumerate(samples[:max_samples])
            )
        else:
            samples_formatted = "    (Không có thoại)"

        block = (
            f"- Tag: {tag}\n"
            f"  Tên: {disp}{gender_hint}"
            f"{loc_formatted}"
            f"{net_formatted}"
            f"{voc_formatted}\n"
            f"  Chỉ số hành vi thoại (Stylometrics): {sty_desc}"
            f"{narr_formatted}"
            f"{exchanges_formatted}\n"
            f"  Thoại mẫu thực tế:\n{samples_formatted}"
        )
        char_blocks.append(block)

    prefix_parts = ["Phân tích hồ sơ tính cách và phong thái đối thoại cho các nhân vật sau:"]
    if mc_tag:
        mc_prof = existing_chars.get(mc_tag)
        mc_disp = mc_prof.display_name if mc_prof else mc_tag
        if any(t.lower() == mc_tag.lower() for t in chunk_tags):
            prefix_parts.append(
                f"\n[LƯU Ý NHÂN VẬT CHÍNH - MC]: Nhân vật '{mc_tag}' (tên: '{mc_disp}') là Nhân vật chính (MC). "
                f"Khi thiết lập pronoun_map cho MC, hãy dùng tag của các nhân vật khác làm key đối ứng "
                f"(TUYỆT ĐỐI KHÔNG dùng '{mc_tag}' làm key cho chính mình)."
            )
        else:
            prefix_parts.append(
                f"\n[LƯU Ý NHÂN VẬT CHÍNH - MC]: Nhân vật chính của game có tag là '{mc_tag}' "
                f"(tên: '{mc_disp}'). Khi thiết lập pronoun_map, BẮT BUỘC dùng tag '{mc_tag}' "
                f"làm key xưng hô đối ứng (định dạng: '{mc_tag}': '<xưng> - <gọi>')."
            )
    prefix_parts.append("\n\n")
    return "".join(prefix_parts) + "\n\n".join(char_blocks)


def bootstrap_characters_with_llm(
    project: Project,
    db: Database,
    llm: Any,
    *,
    max_chars: int = 25,
    chunk_size: int = 4,
) -> dict[str, CharacterProfile]:
    """Bootstrap character personalities, roles, and Vietnamese pronouns using the LLM."""
    all_units = db.all_units()
    speaker_counts: Counter[str] = Counter(u.speaker for u in all_units if u.speaker)
    top_speakers = [tag for tag, _ in speaker_counts.most_common(max_chars)]
    existing_chars = db.get_characters()

    if len(top_speakers) < max_chars:
        defined_candidates = [
            t
            for t in existing_chars
            if t not in top_speakers and t.lower() not in NON_INTERACTION_TAGS
        ]
        top_speakers.extend(defined_candidates[: max_chars - len(top_speakers)])

    if not top_speakers:
        return existing_chars

    samples_by_speaker = stratified_speaker_samples(all_units, top_speakers, per_speaker=6)
    narration_by_speaker = collect_character_narration_context(
        all_units, top_speakers, existing_chars
    )
    gender_evidence_by_speaker = collect_gender_evidence(all_units, existing_chars)
    mc_tag = find_protagonist_tag(existing_chars, all_units)

    stylometrics_by_speaker = {
        tag: extract_stylometrics(samples_by_speaker.get(tag, [])) for tag in top_speakers
    }

    interactive_exchanges_by_speaker = extract_interactive_dialogue_exchanges(
        all_units, top_speakers, existing_chars, mc_tag=mc_tag
    )
    vocatives_by_speaker = collect_incoming_vocatives(
        all_units, top_speakers, existing_chars
    )
    interaction_partners_by_speaker = extract_interaction_network(
        all_units, top_speakers, existing_chars
    )
    salient_locations_by_speaker = extract_salient_locations(
        all_units, top_speakers
    )

    tag_lookup: dict[str, str] = {}
    for t, p in existing_chars.items():
        tag_lookup[t.lower()] = t
        if p.display_name:
            tag_lookup[p.display_name.lower()] = t

    for i in range(0, len(top_speakers), chunk_size):
        chunk_tags = top_speakers[i : i + chunk_size]
        user_prompt = build_persona_user_prompt(
            chunk_tags,
            existing_chars,
            samples_by_speaker,
            stylometrics_by_speaker,
            narration_by_speaker=narration_by_speaker,
            gender_evidence_by_speaker=gender_evidence_by_speaker,
            mc_tag=mc_tag,
            max_samples=6,
            interactive_exchanges_by_speaker=interactive_exchanges_by_speaker,
            vocatives_by_speaker=vocatives_by_speaker,
            interaction_partners_by_speaker=interaction_partners_by_speaker,
            salient_locations_by_speaker=salient_locations_by_speaker,
        )

        try:
            result_dict, _, _ = llm.complete_json(
                PERSONA_SYSTEM_PROMPT, user_prompt, max_tokens=2048
            )
        except Exception as e:
            _LOG.warning("Bootstrapping nhóm nhân vật %s qua LLM thất bại: %s", chunk_tags, e)
            continue

        with db.transaction():
            for raw_key, data in result_dict.items():
                tag = tag_lookup.get(raw_key.lower(), raw_key)
                if tag not in existing_chars and raw_key in existing_chars:
                    tag = raw_key

                if isinstance(data, str):
                    s = data.strip()
                    if s.startswith("{") and s.endswith("}"):
                        try:
                            import ast

                            data = ast.literal_eval(s)
                        except Exception:
                            try:
                                data = json.loads(s)
                            except Exception:
                                continue
                if not isinstance(data, dict):
                    continue

                cur = existing_chars.get(tag)
                disp = cur.display_name if cur else tag.capitalize()
                vi_name = str(data.get("display_name_vi") or "").strip()
                if (
                    vi_name
                    and vi_name.lower() != disp.lower()
                    and not any(
                        disp.lower() == n.lower() for n in ("mc", "player", "hero", "protagonist")
                    )
                ):
                    disp = vi_name

                gender = data.get("gender") or (cur.gender if cur else "unknown")
                if gender in (None, "", "unknown") and tag in gender_evidence_by_speaker:
                    gev = gender_evidence_by_speaker[tag]
                    if gev.male >= 3 and gev.male > gev.female * 2:
                        gender = "male"
                    elif gev.female >= 3 and gev.female > gev.male * 2:
                        gender = "female"

                role = data.get("role") or (cur.role if cur else "")
                speech_style = data.get("speech_style") or (cur.speech_style if cur else "")
                pronoun_map = data.get("pronoun_map") or (cur.pronoun_map if cur else {})

                if isinstance(pronoun_map, str):
                    s_p = pronoun_map.strip()
                    if s_p.startswith("{") and s_p.endswith("}"):
                        try:
                            import ast

                            pronoun_map = ast.literal_eval(s_p)
                        except Exception:
                            try:
                                pronoun_map = json.loads(s_p)
                            except Exception:
                                pronoun_map = {}
                if not isinstance(pronoun_map, dict):
                    pronoun_map = {}

                if mc_tag and mc_tag.lower() != "mc":
                    if "mc" in pronoun_map and mc_tag not in pronoun_map:
                        pronoun_map[mc_tag] = pronoun_map["mc"]
                    elif mc_tag in pronoun_map and "mc" not in pronoun_map:
                        pronoun_map["mc"] = pronoun_map[mc_tag]

                for self_k in (tag, tag.lower()):
                    pronoun_map.pop(self_k, None)
                if mc_tag and tag.lower() == mc_tag.lower():
                    pronoun_map.pop("mc", None)
                    pronoun_map.pop(mc_tag, None)

                archetype = str(data.get("archetype") or "").strip()
                tone_and_voice = str(data.get("tone_and_voice") or "").strip()
                socioeconomic_status = str(data.get("socioeconomic_status") or "").strip()

                raw_traits = data.get("personality_traits")
                if isinstance(raw_traits, (list, tuple)):
                    traits = tuple(
                        str(x).strip() for x in raw_traits if str(x).strip()
                    )
                elif isinstance(raw_traits, str) and raw_traits.strip():
                    traits = tuple(x.strip() for x in raw_traits.split(",") if x.strip())
                else:
                    traits = ()

                raw_samples = data.get("sample_dialogue")
                if isinstance(raw_samples, (list, tuple)):
                    samples_tuple = tuple(
                        str(x).strip() for x in raw_samples if str(x).strip()
                    )
                elif isinstance(raw_samples, str) and raw_samples.strip():
                    samples_tuple = (raw_samples.strip(),)
                else:
                    samples_tuple = ()

                sty = stylometrics_by_speaker.get(tag) or extract_stylometrics([])
                fb = infer_persona_fallback(
                    tag,
                    samples_by_speaker.get(tag, []),
                    sty,
                    role=str(role),
                    gender=str(gender),
                    display_name=disp,
                )
                if not archetype:
                    archetype = cur.archetype if cur and cur.archetype else fb["archetype"]
                if not traits:
                    traits = (
                        cur.personality_traits
                        if cur and cur.personality_traits
                        else fb["personality_traits"]
                    )
                if not tone_and_voice:
                    tone_and_voice = (
                        cur.tone_and_voice
                        if cur and cur.tone_and_voice
                        else fb["tone_and_voice"]
                    )
                if not socioeconomic_status:
                    socioeconomic_status = (
                        cur.socioeconomic_status
                        if cur and cur.socioeconomic_status
                        else fb["socioeconomic_status"]
                    )
                if not samples_tuple:
                    samples_tuple = (
                        cur.sample_dialogue
                        if cur and cur.sample_dialogue
                        else fb["sample_dialogue"]
                    )

                profile = CharacterProfile(
                    tag=tag,
                    display_name=disp,
                    gender=str(gender),
                    role=str(role),
                    speech_style=str(speech_style),
                    pronoun_map={str(k): str(v) for k, v in pronoun_map.items()},
                    archetype=archetype,
                    personality_traits=traits,
                    tone_and_voice=tone_and_voice,
                    socioeconomic_status=socioeconomic_status,
                    sample_dialogue=samples_tuple,
                )
                db.save_character(profile)
                existing_chars[tag] = profile

    chars_file = project.characters_path
    save_characters_to_json(existing_chars, chars_file)
    content = chars_file.read_bytes()
    db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
    db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())
    return existing_chars
