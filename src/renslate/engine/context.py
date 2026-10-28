"""Context assembly and prompt engineering with token budgeting for renslate."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from renslate.core.config import Config
from renslate.core.models import (
    CharacterProfile,
    GlossaryEntry,
    SceneMacroContext,
    SceneSummary,
    Unit,
)
from renslate.engine.tm import FuzzyMatch

if TYPE_CHECKING:
    from renslate.engine.graph import RelationshipGraph

_CHARS_PER_TOKEN_EN = 3.6
_CHARS_PER_TOKEN_VI = 2.8

_NON_DIALOGUE_TAGS = frozenset(
    {
        "t",
        "th",
        "thought",
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

_LEADING_VOCATIVE_RE = re.compile(
    r"^[“\"'«„「『(\uff08]*\s*(?:(?:hey|hi|hello|chào|này|ơi|oi|dear)\s+)?"
    r"([A-Za-zÀ-ỹ0-9_'\- ]+?)\s*"
    r"(?:\.{2,}|…|~|\u301c|[,!?:;\-—\uff0c\uff01\uff1f\uff1a\uff1b])(?:\s+|$)",
    re.IGNORECASE,
)

_TRAILING_VOCATIVE_RE = re.compile(
    r"(?:[,;—\-\uff0c\uff1b—]|(?:\.{2,}|…|~|\u301c))\s*(?:(?:hey|hi|hello|chào|này|ơi|oi|dear)\s+)?"
    r"([A-Za-zÀ-ỹ0-9_'\- ]+?)\s*[\.!\?~…'\"”»\uff09\)」』]*$",
    re.IGNORECASE,
)

_MEDIAL_VOCATIVE_RE = re.compile(
    r"(?:[,;—\-\uff0c\uff1b—]|(?:\.{2,}|…|~|\u301c))\s*([A-Za-zÀ-ỹ0-9_'\-]+)\s*"
    r"(?:[,;—\-\uff0c\uff1b—]|(?:\.{2,}|…|~|\u301c))",
    re.IGNORECASE,
)

_KINSHIP_WORDS = frozenset(
    {
        "father",
        "dad",
        "papa",
        "mother",
        "mom",
        "mama",
        "brother",
        "sister",
        "uncle",
        "aunt",
        "cha",
        "mẹ",
        "bố",
        "chị",
        "em",
        "lord",
        "master",
        "sensei",
        "bệ hạ",
        "sư phụ",
        "tiên sinh",
        "king",
        "sir",
        "duke",
    }
)


def _strip_outer_quotes(text: str) -> str:
    """Strip matching pairs of outer quotation marks and Japanese brackets."""
    t = text.strip()
    quote_pairs = (
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("\u2018", "\u2019"),
        ("«", "»"),
        ("„", "”"),
        ("„", '"'),
        ("「", "」"),
    )
    changed = True
    while changed and len(t) >= 2:
        changed = False
        for op, cl in quote_pairs:
            if t.startswith(op) and t.endswith(cl):
                t = t[len(op) : -len(cl)].strip()
                changed = True
                break
    return t


def _analyze_monologue_and_dialogue(text: str) -> tuple[bool, str]:
    """Classify monologue state and extract stripped spoken dialogue in O(1) time."""
    t = text.strip()
    if t.startswith("『") and t.endswith("』"):
        return True, ""

    unquoted = _strip_outer_quotes(t)
    if (unquoted.startswith("(") and unquoted.endswith(")")) or (
        unquoted.startswith("\uff08") and unquoted.endswith("\uff09")
    ):
        return True, ""

    # Check for parenthesized stage direction: e.g. "(Laughs) Hey Liana, look at this!"
    stage_match = re.match(r"^[(\uff08][^)\uff09\n]+[)\uff09]\s*", unquoted)
    if stage_match:
        remainder = unquoted[stage_match.end() :].strip()
        if re.search(r"\w", remainder):
            return False, remainder
        return True, ""

    if unquoted.startswith("(") or unquoted.startswith("\uff08"):
        return True, ""

    return False, unquoted


@dataclass(frozen=True, slots=True)
class ResolvedInterlocutor:
    """Resolved dialogue participants and communicative modality."""

    speaker: str | None
    listener: str | None = None
    is_monologue: bool = False
    detected_emotion: str | None = None
    resolution_method: str = "default"


class InterlocutorResolver:
    """Real-time O(1) zero-token resolver for interlocutor turn-taking and vocatives."""

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}

    def reset(self) -> None:
        """Clear dynamic alias associations for a new scene."""
        self._aliases.clear()

    @staticmethod
    def _detect_emotion(text: str) -> str | None:
        """Lightweight O(1) punctuation and marker-based emotion detection."""
        t = text.strip()
        if "!?" in t or "?!" in t:
            return "surprised"
        if "..." in t or "…" in t:
            return "hesitant"
        if "~" in t or "\u301c" in t:
            return "affectionate"
        if t.endswith("!"):
            if t.isupper() and len(t) > 3:
                return "shouting"
            return "emphatic"
        if t.endswith("?"):
            return "inquiring"
        return None

    def _build_name_map(
        self,
        active_speakers: set[str] | Sequence[str] | None,
        known_names: dict[str, str] | None,
    ) -> dict[str, str]:
        """Construct normalized tag and display name index for O(1) vocative lookup."""
        name_map: dict[str, str] = {}
        if known_names:
            for tag, disp in known_names.items():
                t = tag.strip().lower()
                name_map[t] = tag
                d = disp.strip().lower()
                name_map[d] = tag
                words = [w.strip() for w in d.split() if len(w.strip()) > 2]
                for w in words:
                    if w not in _KINSHIP_WORDS and w not in name_map:
                        name_map[w] = tag
        if active_speakers:
            for tag in active_speakers:
                t = tag.strip().lower()
                if t not in name_map:
                    name_map[t] = tag
        return name_map

    @staticmethod
    def _clean_cand(raw: str) -> str:
        c = raw.strip().lower()
        for prefix in ("chào ", "hey ", "hi ", "hello ", "này ", "dear "):
            if c.startswith(prefix):
                c = c[len(prefix) :].strip()
        for suffix in (" ơi", " này", " ạ", " oi"):
            if c.endswith(suffix):
                c = c[: -len(suffix)].strip()
        return c

    def _extract_vocative(
        self,
        text: str,
        name_map: dict[str, str],
        speaker: str | None,
    ) -> tuple[str | None, str | None]:
        """Extract vocatively addressed character from text, returning (target_tag, method)."""
        clean = text.strip()
        speaker_norm = speaker.strip().lower() if speaker else None

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
        if not sentences:
            sentences = [clean]

        # 1. Leading vocative in first sentence (highest priority)
        m = _LEADING_VOCATIVE_RE.match(sentences[0])
        if m:
            cand = self._clean_cand(m.group(1))
            tag = name_map.get(cand) or self._aliases.get(cand)
            if tag and tag.lower() != speaker_norm:
                return tag, "vocative"

        # 2. Trailing vocative in last sentence or any sentence (in reverse)
        for s in reversed(sentences):
            m = _TRAILING_VOCATIVE_RE.search(s)
            if m:
                cand = self._clean_cand(m.group(1))
                tag = name_map.get(cand) or self._aliases.get(cand)
                if tag and tag.lower() != speaker_norm:
                    return tag, "vocative"

        # 3. Leading vocative in subsequent sentences
        for s in sentences[1:]:
            m = _LEADING_VOCATIVE_RE.match(s)
            if m:
                cand = self._clean_cand(m.group(1))
                tag = name_map.get(cand) or self._aliases.get(cand)
                if tag and tag.lower() != speaker_norm:
                    return tag, "vocative"

        # 4. Medial vocative
        for s in sentences:
            for m in _MEDIAL_VOCATIVE_RE.finditer(s):
                cand = self._clean_cand(m.group(1))
                tag = name_map.get(cand) or self._aliases.get(cand)
                if tag and tag.lower() != speaker_norm:
                    return tag, "vocative"

        return None, None

    def _resolve_kinship_target(
        self,
        cand: str,
        spk: str | None,
        active_speakers: set[str] | Sequence[str] | None,
        known_names: dict[str, str] | None,
        prev_unit: Unit | None,
        src: str,
    ) -> str | None:
        """Resolve kinship title addressee using candidate scoring."""
        if cand in self._aliases:
            target = self._aliases[cand]
            if not spk or target.lower() != spk.lower():
                return target

        if not active_speakers:
            return None

        spk_lower = spk.lower() if spk else ""
        candidates = [a for a in active_speakers if a.lower() != spk_lower]
        if not candidates:
            return None
        if len(candidates) == 1:
            target = candidates[0]
            self._aliases[cand] = target
            return target

        prev_spk = prev_unit.speaker.lower() if prev_unit and prev_unit.speaker else None

        def _score(c: str) -> float:
            score = 0.0
            disp = (known_names.get(c) if known_names else None) or c
            disp_lower = disp.lower()
            c_lower = c.lower()

            disp_words = set(re.findall(r"[\wÀ-ỹ]+", disp_lower))
            if cand in disp_words or cand in disp_lower:
                score += 50.0

            for kw in _KINSHIP_WORDS:
                if kw in disp_words:
                    score += 20.0
                    break

            name_tokens = {c_lower} | {
                w for w in disp_words if len(w) > 2 and w not in _KINSHIP_WORDS
            }
            for tok in name_tokens:
                if re.search(r"\b" + re.escape(tok) + r"\b", src, re.I):
                    score -= 100.0
                    break

            if prev_spk and c_lower == prev_spk:
                score += 10.0
                if re.search(r"\b(?:him|he|his|her|she|hers)\b", src, re.I):
                    score += 15.0

            return score

        scored = [(c, _score(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored[0][1] > scored[1][1]:
            target = scored[0][0]
            self._aliases[cand] = target
            return target

        if prev_spk:
            for c, _ in scored:
                if c.lower() == prev_spk:
                    self._aliases[cand] = c
                    return c

        target = scored[0][0]
        self._aliases[cand] = target
        return target

    def resolve(
        self,
        unit: Unit,
        prev_unit: Unit | None = None,
        active_speakers: set[str] | Sequence[str] | None = None,
        known_names: dict[str, str] | None = None,
    ) -> ResolvedInterlocutor:
        """Resolve speaker, listener, and monologue state in O(1) time."""
        spk = unit.speaker.strip() if unit.speaker else None
        spk_lower = spk.lower() if spk else ""
        src = unit.source.strip()
        emotion = self._detect_emotion(src)

        # 1. Monologue / Soliloquy classification
        is_mono, clean_dial = _analyze_monologue_and_dialogue(src)
        if not spk or spk_lower in _NON_DIALOGUE_TAGS or is_mono:
            return ResolvedInterlocutor(
                speaker=spk,
                listener=None,
                is_monologue=True,
                detected_emotion=emotion,
                resolution_method="monologue",
            )

        name_map = self._build_name_map(active_speakers, known_names)

        # 2. Vocative Detection on clean dialogue
        vocative_target, _ = self._extract_vocative(clean_dial, name_map, spk)
        if vocative_target:
            return ResolvedInterlocutor(
                speaker=spk,
                listener=vocative_target,
                is_monologue=False,
                detected_emotion=emotion,
                resolution_method="vocative",
            )

        # 3. Kinship / Title Address Fallback (leading and trailing)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_dial) if s.strip()]
        if not sentences:
            sentences = [clean_dial]

        for s in sentences:
            m_lead = _LEADING_VOCATIVE_RE.match(s)
            if m_lead:
                cand = self._clean_cand(m_lead.group(1))
                if cand in _KINSHIP_WORDS:
                    target = self._resolve_kinship_target(
                        cand=cand,
                        spk=spk,
                        active_speakers=active_speakers,
                        known_names=known_names,
                        prev_unit=prev_unit,
                        src=clean_dial,
                    )
                    if target:
                        return ResolvedInterlocutor(
                            speaker=spk,
                            listener=target,
                            is_monologue=False,
                            detected_emotion=emotion,
                            resolution_method="vocative",
                        )

            m_trail = _TRAILING_VOCATIVE_RE.search(s)
            if m_trail:
                cand = self._clean_cand(m_trail.group(1))
                if cand in _KINSHIP_WORDS:
                    target = self._resolve_kinship_target(
                        cand=cand,
                        spk=spk,
                        active_speakers=active_speakers,
                        known_names=known_names,
                        prev_unit=prev_unit,
                        src=clean_dial,
                    )
                    if target:
                        return ResolvedInterlocutor(
                            speaker=spk,
                            listener=target,
                            is_monologue=False,
                            detected_emotion=emotion,
                            resolution_method="vocative",
                        )

        # 4. Turn-Taking Adjacency
        if prev_unit and prev_unit.speaker:
            prev_spk = prev_unit.speaker.strip()
            prev_src = prev_unit.source.strip()
            prev_is_mono, _ = _analyze_monologue_and_dialogue(prev_src)
            prev_is_non_dialogue = prev_spk.lower() in _NON_DIALOGUE_TAGS or prev_is_mono
            if prev_spk.lower() != spk_lower and not prev_is_non_dialogue:
                # Dynamically register any vocative titles used in current unit to prev_spk
                for s in sentences:
                    m_t = _TRAILING_VOCATIVE_RE.search(s)
                    if m_t:
                        c = self._clean_cand(m_t.group(1))
                        if c in _KINSHIP_WORDS and c not in self._aliases:
                            self._aliases[c] = prev_spk
                return ResolvedInterlocutor(
                    speaker=spk,
                    listener=prev_spk,
                    is_monologue=False,
                    detected_emotion=emotion,
                    resolution_method="turn_taking",
                )

        return ResolvedInterlocutor(
            speaker=spk,
            listener=None,
            is_monologue=False,
            detected_emotion=emotion,
            resolution_method="default",
        )


def infer_scene_cues(label: str) -> tuple[str, str]:
    """Extract location and atmospheric cues from Ren'Py scene label."""
    norm = label.lower().replace("-", "_").strip()
    location_parts: list[str] = []
    atmosphere_parts: list[str] = []

    loc_keywords = {
        "forest": "Rừng rậm",
        "woods": "Rừng",
        "classroom": "Lớp học",
        "school": "Trường học",
        "castle": "Lâu đài",
        "palace": "Cung điện",
        "room": "Phòng riêng",
        "bedroom": "Phòng ngủ",
        "dungeon": "Ngục tối",
        "garden": "Vườn hoa",
        "street": "Khu phố",
        "market": "Khu chợ",
        "alley": "Con hẻm",
        "hall": "Đại sảnh",
        "beach": "Bãi biển",
        "river": "Bờ sông",
        "lake": "Bờ hồ",
        "mountain": "Đỉnh núi",
    }
    atmos_keywords = {
        "night": "Đêm khuya / Tĩnh mịch",
        "midnight": "Nửa đêm / Bí ẩn",
        "sunset": "Hoàng hôn / Trầm buồn",
        "dusk": "Chập tối",
        "morning": "Buổi sáng / Tươi vui",
        "dawn": "Bình minh",
        "rain": "Trời mưa / Trầm lắng",
        "storm": "Bão tố / Căng thẳng",
        "dark": "U ám / Nguy hiểm",
        "tense": "Căng thẳng",
        "battle": "Chiến đấu quyết liệt",
        "romance": "Lãng mạn / Thân mật",
        "comedy": "Vui vẻ / Hài hước",
        "climax": "Cao trào kịch tính",
    }

    for k, v in loc_keywords.items():
        if k in norm:
            location_parts.append(v)
    for k, v in atmos_keywords.items():
        if k in norm:
            atmosphere_parts.append(v)

    loc_str = ", ".join(location_parts) if location_parts else ""
    atmos_str = ", ".join(atmosphere_parts) if atmosphere_parts else ""
    return loc_str, atmos_str


def format_rich_dialogue_line(
    unit: Unit,
    resolved: ResolvedInterlocutor,
    char_map: dict[str, CharacterProfile],
    graph: RelationshipGraph | None = None,
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

        emotion_str = (
            f" | Cảm xúc: {resolved.detected_emotion}" if resolved.detected_emotion else ""
        )
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
        elif cp_spk and cp_spk.pronoun_map:
            tgt_key = cp_lis.tag if cp_lis else lis_clean
            tgt_name = cp_lis.display_name if cp_lis else lis_clean
            p_rule = (
                cp_spk.pronoun_map.get(tgt_key)
                or cp_spk.pronoun_map.get(tgt_name)
                or cp_spk.pronoun_map.get(tgt_key.lower())
            )
            if p_rule:
                rule_clean = p_rule.strip()
                if " - " in rule_clean and not rule_clean.startswith("xưng"):
                    parts = [p.strip() for p in rule_clean.split(" - ", 1)]
                    rel_part = f" | Quan hệ: {spk_name} xưng '{parts[0]}', gọi '{parts[1]}'"
                else:
                    rel_part = f" | Quan hệ: {rule_clean}"

        emotion_str = (
            f" | Cảm xúc: {resolved.detected_emotion}" if resolved.detected_emotion else ""
        )
        return (
            f"{unit.uid} | [Người nói: {spk_name} -> Người nghe: {lis_name}"
            f"{rel_part}{emotion_str}] {text}"
        )

    # 3. Dialogue without resolved listener
    if cp_spk and cp_spk.display_name and cp_spk.display_name.lower() != spk_lower:
        spk_label = f"{cp_spk.display_name} ({unit.speaker})"
    else:
        spk_label = unit.speaker

    emotion_str = f" | Cảm xúc: {resolved.detected_emotion}" if resolved.detected_emotion else ""
    return f"{unit.uid} | [Người nói: {spk_label}{emotion_str}] {text}"


def estimate_tokens(text: str, is_vietnamese: bool = False) -> int:
    """Heuristic token estimation based on character count and language density."""
    ratio = _CHARS_PER_TOKEN_VI if is_vietnamese else _CHARS_PER_TOKEN_EN
    return max(1, math.ceil(len(text) / ratio))


QUOTED_RULES = (
    "QUY TẮC BẮT BUỘC:\n"
    "1. Giữ NGUYÊN VẸN 100% mọi placeholder đặc biệt có dạng ⟦V1⟧, ⟦T1⟧, ⟦/T1⟧, ⟦E1⟧, ⟦L1⟧...\n"
    "   Tuyệt đối không dịch, không xoá, không đổi hoa/thường, không sửa nội dung bên trong.\n"
    "2. BẢNG THUẬT NGỮ (GLOSSARY) LÀ TỐI THƯỢNG:\n"
    "   - Bắt buộc dịch đúng 100% theo Glossary đã cung cấp (VD: Sect -> Tông môn...).\n"
    "   - TUYỆT ĐỐI KHÔNG để sót từ tiếng Anh trong câu dịch khi từ đó đã có trong Glossary.\n"
    "   - Giữ nhất quán tuyệt đối trong cùng một câu và giữa các câu liên tiếp.\n"
    "3. QUY TẮC XƯNG HÔ ĐỐI ỨNG (PRONOUNS) - TUYỆT ĐỐI CẤM ĐẢO NGƯỢC:\n"
    "   Khi nhân vật nói chuyện theo quy tắc [xưng X, gọi Y]:\n"
    "   - ĐẠI TỪ NGÔI THỨ NHẤT (I, me, my, we): BẮT BUỘC dịch là X (tự xưng). "
    "TUYỆT ĐỐI CẤM dịch thành Y.\n"
    "   - ĐẠI TỪ NGÔI THỨ HAI (you, your, yourself): BẮT BUỘC dịch là Y (gọi đối phương). "
    "TUYỆT ĐỐI CẤM dịch thành X.\n"
    "   - CẢNH BÁO ĐẢO NGÔI (LỖI PHỔ BIẾN BẮT BUỘC TRÁNH):\n"
    "     * Khi nhân vật xưng 'em', gọi 'anh': Mọi câu hỏi hướng tới đối phương có 'you' "
    "(VD: 'Do you want to die?', 'Why are you sleeping here?') BẮT BUỘC dịch là 'Anh...' "
    "(VD: 'Anh muốn chết sao?', 'Sao anh lại ngủ ở đây?'). TUYỆT ĐỐI CẤM dùng từ tự xưng "
    "'Em' làm chủ ngữ (như 'Em muốn chết sao?') vì sẽ biến câu hỏi thành tự hỏi bản thân!\n"
    "     * Khi nhân vật xưng 'anh', gọi 'em': Khi nói 'Did I catch you...?', "
    "BẮT BUỘC dịch là 'Anh bắt gặp em...?' (TUYỆT ĐỐI CẤM dịch 'I' thành 'em'!).\n"
    "     * Khi nhân vật xưng 'anh', gọi 'em': Khi nói 'I think...', "
    "BẮT BUỘC dịch là 'Anh nghĩ...' (TUYỆT ĐỐI CẤM xưng 'tôi' hoặc 'em' thất thường!).\n"
    "   - TUYỆT ĐỐI KHÔNG đảo ngược vai vế (ví dụ: Đệ tử xưng 'ta' gọi bề trên 'ngươi' là SAI;\n"
    "     phải xưng 'con/đệ tử' gọi 'sư phụ/cha/mẹ').\n"
    "   - TUYỆT ĐỐI KHÔNG đổi ngôi xưng hô thất thường trong cùng một câu thoại.\n"
    "4. Lời thoại súc tích, tự nhiên, văn phong visual novel, tránh dịch word-by-word.\n"
    "   Giữ độ dài vừa phải để không làm tràn textbox.\n"
    "5. QUY TẮC NỘI DUNG LỜI THOẠI:\n"
    "   - Nhãn [Người nói: ...] ở dòng nguồn chỉ dùng làm ngữ cảnh xưng hô.\n"
    "   - TUYỆT ĐỐI KHÔNG chép lại nhãn [Người nói: ...] hay tên nhân vật vào chuỗi kết quả.\n"
    '   - TUYỆT ĐỐI KHÔNG bọc bản dịch trong dấu ngoặc kép ngoài cùng (không dùng " hay \\").\n'
    "   - TUYỆT ĐỐI CHỈ dùng chữ quốc ngữ tiếng Việt. TUYỆT ĐỐI CẤM sử dụng bất kỳ "
    "chữ Hán / tiếng Trung (Hanzi/Kanji) hay lặp từ vô tận.\n"
    '6. ĐỊNH DẠNG ĐẦU RA: Trả về DUY NHẤT một đối tượng JSON: {"id_1": "Bản dịch 1", ...}\n'
    "   - Bắt buộc giữ NGUYÊN VẸN 100% ID của từng dòng làm key JSON "
    '(bao gồm tiền tố s: hay d:, VD: "s:abc" -> key là "s:abc", không được lược bỏ "s:").\n'
    "   - Đóng ngoặc nhọn } và DỪNG NGAY LẬP TỨC. Tuyệt đối không thêm lời giải thích, "
    "chú thích hay đối tượng JSON thứ hai nào ngoài JSON."
)

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là chuyên gia bản địa hóa game visual novel Ren'Py "
    "sang tiếng Việt tự nhiên và giàu cảm xúc.\n\n" + QUOTED_RULES
)


@dataclass(frozen=True, slots=True)
class LookbackItem:
    speaker: str | None
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """Encapsulates all contextual domain assets assembled for a translation prompt.

    Provides a unified container for glossary matches, character profiles, scene context,
    lookback conversation history, fuzzy matches, relationship graph, and interlocutor resolver.
    """

    glossary_matches: Sequence[GlossaryEntry] = ()
    characters: dict[str, CharacterProfile] = field(default_factory=dict)
    scene_summary: SceneSummary | None = None
    scene_macro: SceneMacroContext | None = None
    lookback: Sequence[LookbackItem] = ()
    fuzzy_matches: dict[str, list[FuzzyMatch]] = field(default_factory=dict)
    graph: RelationshipGraph | None = None
    resolver: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.glossary_matches, tuple):
            object.__setattr__(self, "glossary_matches", tuple(self.glossary_matches))
        if not isinstance(self.lookback, tuple):
            object.__setattr__(self, "lookback", tuple(self.lookback))


class ContextBuilder:
    """Builds optimized prompts respecting token budgets and context priorities."""

    def __init__(self, config: Config, custom_style_guide: str = "") -> None:
        self.config = config
        self.custom_style_guide = custom_style_guide

    def build_system_prompt(self) -> str:
        """Compose the cached system prompt with style guidelines."""
        prompt = DEFAULT_SYSTEM_PROMPT
        if self.config.style.default_register:
            prompt += f"\nGiọng điệu mặc định: {self.config.style.default_register}."
        if self.config.style.notes:
            prompt += f"\nLưu ý phong cách: {self.config.style.notes}."
        if self.custom_style_guide.strip():
            prompt += f"\n\nHướng dẫn phong cách riêng của game:\n{self.custom_style_guide.strip()}"
        return prompt

    @staticmethod
    def calculate_output_budget(batch_size: int, source_tokens: int = 0, n_ctx: int = 6144) -> int:
        """Calculate reserved completion token budget for a batch size.

        Dynamically sizes completion token reservation based on source token volume
        while respecting bounds [350, min(3000, n_ctx // 2)].
        """
        if source_tokens > 0:
            target = int(source_tokens * 1.35) + batch_size * 25 + 64
        else:
            target = batch_size * 80 + 150
        max_bound = min(3000, n_ctx // 2) if n_ctx >= 6000 else (n_ctx // 2)
        return max(350, min(max_bound, target))

    def _compact_lookback(
        self,
        lookback: Sequence[LookbackItem],
        available_budget: int,
    ) -> tuple[str | None, int]:
        """Sliding suffix compaction: fits maximal recent conversation turns into budget."""
        if not lookback or available_budget < 60:
            return None, available_budget

        selected_lines: list[str] = []
        accumulated_tokens = 0
        header = "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC (để nối mạch):"
        header_tokens = estimate_tokens(header)

        if header_tokens > available_budget:
            return None, available_budget

        budget_for_lines = available_budget - header_tokens

        recent_items = list(lookback)[-self.config.batch.lookback :]
        for item in reversed(recent_items):
            spk = f"[{item.speaker}] " if item.speaker else ""
            tgt = item.target if len(item.target) <= 150 else item.target[:140] + "..."
            src = item.source if len(item.source) <= 150 else item.source[:140] + "..."
            line = f'- {spk}"{src}" => "{tgt}"'
            line_tokens = estimate_tokens(line)

            if accumulated_tokens + line_tokens <= budget_for_lines:
                selected_lines.append(line)
                accumulated_tokens += line_tokens
            else:
                break

        if not selected_lines:
            return None, available_budget

        selected_lines.reverse()
        block = header + "\n" + "\n".join(selected_lines)
        total_tokens = header_tokens + accumulated_tokens
        return block, available_budget - total_tokens

    def build_user_prompt(
        self,
        batch_units: Sequence[Unit],
        glossary_matches: Sequence[GlossaryEntry] = (),
        characters: dict[str, CharacterProfile] | None = None,
        scene_summary: SceneSummary | None = None,
        lookback: Sequence[LookbackItem] = (),
        fuzzy_matches: dict[str, list[FuzzyMatch]] | None = None,
        graph: RelationshipGraph | None = None,
        resolver: Any | None = None,
        scene_macro: SceneMacroContext | None = None,
        bundle: ContextBundle | None = None,
    ) -> str:
        """Assemble the user prompt prioritizing critical context within token limits."""
        if bundle is not None:
            if not glossary_matches and bundle.glossary_matches:
                glossary_matches = bundle.glossary_matches
            if characters is None and bundle.characters:
                characters = bundle.characters
            scene_summary = scene_summary or bundle.scene_summary
            scene_macro = scene_macro or bundle.scene_macro
            if not lookback and bundle.lookback:
                lookback = bundle.lookback
            fuzzy_matches = fuzzy_matches if fuzzy_matches is not None else bundle.fuzzy_matches
            graph = graph or bundle.graph
            resolver = resolver or bundle.resolver

        characters = characters or {}
        fuzzy_matches = fuzzy_matches or {}

        # Character lookup index mapping both tag and display_name (case-insensitive)
        char_map: dict[str, CharacterProfile] = {}
        for p in characters.values():
            char_map[p.tag] = p
            char_map[p.tag.lower()] = p
            if p.display_name:
                char_map[p.display_name] = p
                char_map[p.display_name.lower()] = p

        lines_section = ["CÁC DÒNG CẦN DỊCH (bắt buộc dịch đầy đủ đúng id):"]

        active_resolver = resolver
        if active_resolver is None and graph is not None:
            active_resolver = InterlocutorResolver()

        if active_resolver is not None:
            active_speakers = {
                u.speaker.strip()
                for u in batch_units
                if u.speaker and u.speaker.strip().lower() not in _NON_DIALOGUE_TAGS
            }
            known_names = {
                p.tag: p.display_name for p in characters.values() if p.display_name
            }
            for p in characters.values():
                if p.display_name:
                    known_names[p.display_name] = p.display_name

            prev_unit: Unit | None = None
            for u in batch_units:
                if u.speaker:
                    resolved = active_resolver.resolve(
                        unit=u,
                        prev_unit=prev_unit,
                        active_speakers=active_speakers,
                        known_names=known_names,
                    )
                else:
                    resolved = ResolvedInterlocutor(
                        speaker=None,
                        listener=None,
                        is_monologue=True,
                    )
                lines_section.append(
                    format_rich_dialogue_line(u, resolved, char_map, graph=graph)
                )
                prev_unit = u
        else:
            for u in batch_units:
                text = u.masked_source or u.source
                if u.speaker:
                    spk_clean = u.speaker.strip()
                    spk_lower = spk_clean.lower()
                    cp = char_map.get(spk_clean) or char_map.get(spk_lower)
                    if spk_lower in _NON_DIALOGUE_TAGS:
                        speaker_tag = f"[Độc thoại/Nội tâm: {u.speaker}] "
                    elif cp and cp.display_name and cp.display_name.lower() != spk_lower:
                        speaker_tag = f"[Người nói: {cp.display_name} ({u.speaker})] "
                    else:
                        speaker_tag = f"[Người nói: {u.speaker}] "
                else:
                    speaker_tag = ""
                lines_section.append(f"{u.uid} | {speaker_tag}{text}")
        batch_text = "\n".join(lines_section)

        system_text = self.build_system_prompt()
        sys_tokens = estimate_tokens(system_text)
        batch_tokens = estimate_tokens(batch_text)
        batch_source_tokens = sum(estimate_tokens(u.masked_source or u.source) for u in batch_units)
        reserve_output = self.calculate_output_budget(
            len(batch_units),
            source_tokens=batch_source_tokens,
            n_ctx=self.config.llm.n_ctx,
        )
        max_context = self.config.llm.n_ctx
        available_budget = max_context - sys_tokens - batch_tokens - reserve_output

        sections: list[str] = []

        if glossary_matches and available_budget > 100:
            gl_lines = ["THUẬT NGỮ BẮT BUỘC (GLOSSARY):"]
            candidate_terms: list[str] = []
            for g in glossary_matches:
                if g.translation.strip():
                    note_str = f" ({g.notes})" if g.notes else ""
                    gl_lines.append(f"- {g.term} -> {g.translation}{note_str}")
                elif g.kind in ("name", "location", "proper", "candidate") or not g.kind:
                    candidate_terms.append(g.term)

            if len(gl_lines) > 1:
                gl_block = "\n".join(gl_lines)
                tokens = estimate_tokens(gl_block)
                if tokens <= available_budget:
                    sections.append(gl_block)
                    available_budget -= tokens

            known_char_names = {
                cp.display_name.lower() for cp in characters.values() if cp.display_name
            } | set(characters.keys())

            filtered_candidates = [
                ct for ct in candidate_terms if ct.lower() not in known_char_names
            ]

            if filtered_candidates and available_budget > 60:
                cand_block = (
                    "DANH TỪ RIÊNG & THUẬT NGỮ CẦN ĐỒNG NHẤT:\n- "
                    + ", ".join(filtered_candidates[:8])
                    + "\n(dịch nghĩa chính xác hoặc phiên âm tự nhiên"
                    + " phù hợp ngữ cảnh, không bỏ sót)"
                )
                tokens = estimate_tokens(cand_block)
                if tokens <= available_budget:
                    sections.append(cand_block)
                    available_budget -= tokens

        speakers_in_batch = [u.speaker for u in batch_units if u.speaker]
        distinct_speakers = list(dict.fromkeys(speakers_in_batch))

        dialogue_speakers = [s for s in distinct_speakers if s.lower() not in _NON_DIALOGUE_TAGS]
        active_dialogue_speakers = list(dict.fromkeys(dialogue_speakers))

        # If only 1 dialogue speaker in batch, inspect recent lookback for the partner
        if len(active_dialogue_speakers) == 1 and lookback:
            single_lower = active_dialogue_speakers[0].lower()
            for item in reversed(lookback):
                if (
                    item.speaker
                    and item.speaker.lower() not in _NON_DIALOGUE_TAGS
                    and item.speaker.lower() != single_lower
                ):
                    active_dialogue_speakers.append(item.speaker)
                    break

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

        # Relational pronoun extraction between dialogue participants
        relational_lines: list[str] = []
        pairs_seen: set[tuple[str, str]] = set()
        for i in range(len(active_dialogue_speakers)):
            for j in range(i + 1, len(active_dialogue_speakers)):
                s_a, s_b = active_dialogue_speakers[i], active_dialogue_speakers[j]
                p_a = char_map.get(s_a) or char_map.get(s_a.lower())
                p_b = char_map.get(s_b) or char_map.get(s_b.lower())
                if p_a and p_b and p_a.tag != p_b.tag:
                    pair_key = (p_a.tag, p_b.tag) if p_a.tag < p_b.tag else (p_b.tag, p_a.tag)
                    if pair_key in pairs_seen:
                        continue
                    pairs_seen.add(pair_key)
                    rule_a_to_b = p_a.pronoun_map.get(p_b.tag) or p_a.pronoun_map.get(
                        p_b.display_name
                    )
                    rule_b_to_a = p_b.pronoun_map.get(p_a.tag) or p_b.pronoun_map.get(
                        p_a.display_name
                    )
                    if graph is not None:
                        edge_ab = graph.get_edge(p_a.tag, p_b.tag) or graph.get_edge(s_a, s_b)
                        if edge_ab and edge_ab.speaker_pronoun and edge_ab.listener_pronoun:
                            rule_a_to_b = f"{edge_ab.speaker_pronoun} - {edge_ab.listener_pronoun}"
                        edge_ba = graph.get_edge(p_b.tag, p_a.tag) or graph.get_edge(s_b, s_a)
                        if edge_ba and edge_ba.speaker_pronoun and edge_ba.listener_pronoun:
                            rule_b_to_a = f"{edge_ba.speaker_pronoun} - {edge_ba.listener_pronoun}"
                    if rule_a_to_b or rule_b_to_a:
                        relational_lines.append(
                            f"- {p_a.display_name} và {p_b.display_name} đang đối thoại trực tiếp "
                            f"(cặp {p_a.tag} - {p_b.tag}):"
                        )
                        if rule_a_to_b:
                            relational_lines.append(_format_relational_rule(p_a, p_b, rule_a_to_b))
                        if rule_b_to_a:
                            relational_lines.append(_format_relational_rule(p_b, p_a, rule_b_to_a))

        if relational_lines and available_budget > 80:
            rel_block = "QUAN HỆ XƯNG HÔ ĐỐI THOẠI TRONG PHÂN ĐOẠN:\n" + "\n".join(relational_lines)
            tokens = estimate_tokens(rel_block)
            if tokens <= available_budget:
                sections.append(rel_block)
                available_budget -= tokens

        char_lines: list[str] = []
        for spk in sorted(set(distinct_speakers)):
            char_prof = char_map.get(spk) or char_map.get(spk.lower())
            if char_prof is not None:
                g_raw = (char_prof.gender or "unknown").strip().lower()
                gender_str = (
                    "Nữ" if g_raw == "female" else ("Nam" if g_raw == "male" else "Chưa rõ")
                )
                char_info = f"- {char_prof.display_name} (tag: {char_prof.tag}): {gender_str}"
                if char_prof.role:
                    char_info += f" | Vai trò: {char_prof.role}"
                if char_prof.speech_style:
                    char_info += f" | Giọng: {char_prof.speech_style}"
                if char_prof.pronoun_map:
                    pronoun_parts = []
                    for k, v in char_prof.pronoun_map.items():
                        if k in ("to_elder", "to_peer", "to_junior", "to_enemy"):
                            label = {
                                "to_elder": "với bề trên",
                                "to_peer": "với bạn bè/đồng môn",
                                "to_junior": "với bề dưới",
                                "to_enemy": "với kẻ thù",
                            }.get(k, k)
                            pronoun_parts.append(f"{label} ({v})")
                        else:
                            tgt_char = char_map.get(k) or char_map.get(k.lower())
                            tgt_name = (
                                f"{tgt_char.display_name} ({k})"
                                if tgt_char and tgt_char.display_name.lower() != k.lower()
                                else k
                            )
                            pronoun_parts.append(f"với {tgt_name} ({_format_pronoun_pair(v)})")
                    if pronoun_parts:
                        char_info += f" | Xưng hô: {'; '.join(pronoun_parts)}"
                char_lines.append(char_info)
        if char_lines and available_budget > 80:
            ch_block = "HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:\n" + "\n".join(char_lines)
            tokens = estimate_tokens(ch_block)
            if tokens <= available_budget:
                sections.append(ch_block)
                available_budget -= tokens

        resolved_macro: SceneMacroContext | None = scene_macro
        if resolved_macro is None and scene_summary is not None:
            if isinstance(scene_summary, SceneMacroContext):
                resolved_macro = scene_summary
            else:
                loc = getattr(scene_summary, "location", "")
                atmos = getattr(scene_summary, "atmosphere", "")
                resolved_macro = SceneMacroContext(
                    scene_id=scene_summary.scene_id,
                    label=scene_summary.label,
                    first_seq=scene_summary.first_seq,
                    last_seq=scene_summary.last_seq,
                    summary=scene_summary.summary,
                    location=loc,
                    atmosphere=atmos,
                )

        if (
            resolved_macro is not None
            and (resolved_macro.summary or resolved_macro.label or not resolved_macro.is_empty())
            and available_budget > 80
        ):
            loc = resolved_macro.location
            atmos = resolved_macro.atmosphere
            if (not loc or not atmos) and resolved_macro.label:
                inferred_loc, inferred_atmos = infer_scene_cues(resolved_macro.label)
                loc = loc or inferred_loc
                atmos = atmos or inferred_atmos
                resolved_macro = resolved_macro.with_updates(location=loc, atmosphere=atmos)

            sc_block = resolved_macro.format_prompt_header()
            tokens = estimate_tokens(sc_block)
            if tokens <= available_budget:
                sections.append(sc_block)
                available_budget -= tokens

        if self.config.batch.lookback > 0 and lookback and available_budget > 60:
            lb_block, available_budget = self._compact_lookback(lookback, available_budget)
            if lb_block is not None:
                sections.append(lb_block)

        if fuzzy_matches and available_budget > 100:
            fz_lines: list[str] = []
            for uid, matches in fuzzy_matches.items():
                for m in matches:
                    pct = int(m.score * 100)
                    fz_lines.append(
                        f'- Gợi ý cho {uid} (khớp {pct}%): "{m.source}" => "{m.target}"'
                    )
            if fz_lines:
                fz_block = "GỢI Ý TỪ BỘ NHỚ DỊCH:\n" + "\n".join(fz_lines)
                tokens = estimate_tokens(fz_block)
                if tokens <= available_budget:
                    sections.append(fz_block)
                    available_budget -= tokens

        sections.append(batch_text)
        return "\n\n".join(sections)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "QUOTED_RULES",
    "ContextBuilder",
    "ContextBundle",
    "InterlocutorResolver",
    "LookbackItem",
    "ResolvedInterlocutor",
    "estimate_tokens",
    "format_rich_dialogue_line",
    "infer_scene_cues",
]

