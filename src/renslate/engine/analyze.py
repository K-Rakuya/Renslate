"""Analysis stage: character discovery, scene segmentation, and glossary mining."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from renslate.core.db import (
    Database,
    load_characters_from_json,
    save_characters_to_json,
    sync_characters,
    utc_timestamp,
)
from renslate.core.models import (
    CharacterProfile,
    GlossaryEntry,
    RelationshipEdge,
    RelationshipKind,
    SceneSummary,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.glossary import (
    Glossary,
    _make_plural_aliases,
    clean_glossary,
    mine_glossary_candidates,
    translate_glossary_with_llm,
)
from renslate.engine.graph import RelationshipGraph

_LOG = logging.getLogger("renslate.analyze")

__all__ = [
    "StylometricSummary",
    "bootstrap_characters_with_llm",
    "extract_character_definitions",
    "extract_stylometrics",
    "infer_character_gender_nlp",
    "infer_character_role_nlp",
    "infer_persona_fallback",
    "infer_semantic_relationship",
    "is_collective_entity",
    "is_non_human_or_meta",
    "load_characters_from_json",
    "resolve_pronoun_pair_from_relationship",
    "run_analysis",
    "save_characters_to_json",
    "segment_scenes",
    "sync_characters",
]

_RE_CHARACTER_DEF = re.compile(
    r'(?:define|default)\s+(?P<tag>\w+)\s*=\s*(?:Character|DynamicCharacter)\s*\(\s*(?P<name>_[^(]*\([^)]*\)|"(?:[^"\\]|\\.)*"|\'[^\']*\'|[A-Za-z_]\w*)'
)
_RE_SCENE_STMT = re.compile(r"^\s*scene\s+\w+")

_PROTAGONIST_TAGS = frozenset({"mc", "player", "hero", "protag", "protagonist", "main", "you"})
_NARRATOR_TAGS = frozenset({"narrator", "narration", "none"})
_INNER_MONOLOGUE_TAGS = frozenset({"t", "th", "thought", "thoughts", "thinking"})
_COLLECTIVE_TAGS = frozenset(
    {
        "all",
        "everyone",
        "everybody",
        "both",
        "crowd",
        "mob",
        "audience",
        "group",
        "people",
        "students",
        "guards",
        "soldiers",
        "villagers",
        "townsfolk",
        "citizens",
        "bystanders",
        "patrons",
        "girls",
        "boys",
        "children",
        "workers",
    }
)
_NON_HUMAN_TAGS = frozenset(
    {
        "system",
        "announcer",
        "voice",
        "unknown_voice",
        "mysterious_voice",
        "phone",
        "tv",
        "radio",
        "sign",
        "letter",
        "note",
        "message",
        "intercom",
        "broadcast",
        "robot",
        "ai",
        "dog",
        "cat",
        "monster",
        "dragon",
        "wolf",
        "beast",
        "creature",
        "spirit",
        "ghost",
        "narrator",
        "narration",
        "narr",
        "none",
        "n",
        "t",
        "th",
        "thought",
        "thoughts",
        "thinking",
    }
)
_NON_INTERACTION_TAGS = frozenset(_NON_HUMAN_TAGS | {"none"})

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
    type_token_ratio: float  # unique_words / total_words (0.0 if empty)
    total_sentences: int
    avg_sentence_length: float  # words per sentence (0.0 if empty)
    exclamation_count: int
    exclamation_rate: float  # exclamations per sentence
    question_count: int
    question_rate: float  # questions per sentence
    ellipsis_count: int
    ellipsis_rate: float  # ellipses per sentence
    hesitation_count: int
    hesitation_rate: float  # hesitation markers per sentence
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


def infer_character_role_nlp(tag: str, line_count: int, total_lines: int) -> str:
    """Infer character role heuristic based on dialogue volume and protagonist tags."""
    tag_lower = tag.lower()
    if tag_lower in ("t", "th", "thought", "thoughts", "thinking"):
        return "Độc thoại nội tâm (Inner Monologue)"
    if tag_lower in ("narrator", "narration", "none"):
        return "Người dẫn chuyện (Narrator)"
    if tag_lower in _PROTAGONIST_TAGS:
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

    # Search for English pronoun co-occurrence in sentences mentioning the character name
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


def extract_character_definitions(project_dir: Path) -> dict[str, str]:
    """Scan game script files for Ren'Py Character definitions."""
    characters: dict[str, str] = {}
    game_dir = project_dir / "game"
    if game_dir.is_dir():
        for rpy in sorted(game_dir.rglob("*.rpy")):
            # Skip translation directories
            if "tl" in rpy.parts:
                continue
            try:
                content = rpy.read_text(encoding="utf-8-sig")
            except (UnicodeDecodeError, OSError):
                continue

            for m in _RE_CHARACTER_DEF.finditer(content):
                tag = m.group("tag")
                raw_name = m.group("name").strip()
                if (raw_name.startswith('_("') and raw_name.endswith('")')) or (
                    raw_name.startswith("_('") and raw_name.endswith("')")
                ):
                    name = raw_name[3:-2]
                elif (raw_name.startswith('"') and raw_name.endswith('"')) or (
                    raw_name.startswith("'") and raw_name.endswith("'")
                ):
                    name = raw_name[1:-1]
                elif raw_name in ("None", ""):
                    name = tag.capitalize()
                elif raw_name.endswith("_name"):
                    base = raw_name[:-5].strip()
                    if base.lower() in _PROTAGONIST_TAGS or tag.lower() in _PROTAGONIST_TAGS:
                        name = "Protagonist"
                    else:
                        name = base.capitalize()
                elif tag.lower() in _PROTAGONIST_TAGS:
                    name = "Protagonist"
                else:
                    name = raw_name
                characters[tag] = re.sub(r"\b([A-Za-z])\s+([a-z]{2,})\b", r"\1\2", name).strip()

    # Fallback to direct .rpyc bytecode parsing if no .rpy scripts contained definitions
    if not characters:
        with contextlib.suppress(Exception):
            from renslate.renpy.rpyc import extract_character_definitions_from_rpyc

            characters = extract_character_definitions_from_rpyc(project_dir)

    return characters


def segment_scenes(db: Database) -> list[SceneSummary]:
    """Group dialogue units into coherent scenes based on labels and sequence continuity."""
    units = db.all_units()
    dialogues = [u for u in units if u.kind is UnitKind.DIALOGUE]
    if not dialogues:
        return []

    scenes: list[SceneSummary] = []
    current_label = dialogues[0].label or "prologue"
    first_seq = dialogues[0].seq
    last_seq = dialogues[0].seq
    scene_idx = 1

    for i in range(1, len(dialogues)):
        u = dialogues[i]
        label = u.label or current_label
        # New scene if label changes or sequence jump > 50
        if label != current_label or (u.seq - last_seq > 50):
            scene_id = f"scene_{scene_idx}_{current_label}"
            scenes.append(
                SceneSummary(
                    scene_id=scene_id,
                    label=current_label,
                    first_seq=first_seq,
                    last_seq=last_seq,
                    summary=f"Cảnh '{current_label}' (từ dòng {first_seq} đến {last_seq})",
                )
            )
            scene_idx += 1
            current_label = label
            first_seq = u.seq
        last_seq = u.seq

    # Final scene
    scene_id = f"scene_{scene_idx}_{current_label}"
    scenes.append(
        SceneSummary(
            scene_id=scene_id,
            label=current_label,
            first_seq=first_seq,
            last_seq=last_seq,
            summary=f"Cảnh '{current_label}' (từ dòng {first_seq} đến {last_seq})",
        )
    )

    return scenes


def is_collective_entity(tag: str, display_name: str = "") -> bool:
    """Check if tag or display_name refers to a collective group of people."""
    t = tag.lower().strip()
    if t in _COLLECTIVE_TAGS:
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
    if t in _NON_HUMAN_TAGS:
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


def infer_semantic_relationship(
    tag: str,
    tc: dict[str, Any],
    partner_tag: str,
    partner_info: dict[str, Any],
    *,
    is_mc: bool = False,
    partner_is_mc: bool = False,
) -> tuple[RelationshipKind, bool] | None:
    """Infer semantic relationship kind and seniority between two interacting characters."""
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

    # 1. Family Elder
    elder_cues = (
        "mother", "mom", "mama", "father", "dad", "papa", "parent", "uncle", "aunt",
        "grandpa", "grandma", "grandfather", "grandmother", "mẹ", "cha", "bố", "bác",
        "chú", "cô", "dì", "ông", "bà",
    )
    if any(re.search(rf"\b{c}\b", text_partner) for c in elder_cues) and (
        is_mc or not any(re.search(rf"\b{c}\b", text_self) for c in elder_cues)
    ):
        return (RelationshipKind.FAMILY_ELDER, False)
    if any(re.search(rf"\b{c}\b", text_self) for c in elder_cues) and (
        partner_is_mc or not any(re.search(rf"\b{c}\b", text_partner) for c in elder_cues)
    ):
        return (RelationshipKind.FAMILY_ELDER, True)

    # 2. Sibling
    sibling_cues = ("sister", "brother", "sibling", "chị", "em gái", "em trai", "anh trai")
    if any(re.search(rf"\b{c}\b", text_partner) for c in sibling_cues) or any(
        re.search(rf"\b{c}\b", text_self) for c in sibling_cues
    ):
        is_senior = bool(
            re.search(r"\b(elder|older|chị|anh trai)\b", text_self)
            or not re.search(r"\b(elder|older|chị|anh trai)\b", text_partner)
        )
        return (RelationshipKind.FAMILY_SIBLING, is_senior)

    # 3. Superior / Royalty
    royal_cues = (
        "queen", "king", "emperor", "empress", "princess", "prince", "lord",
        "commander", "captain", "bệ hạ", "nữ hoàng", "hoàng đế", "công chúa", "chỉ huy",
    )
    if any(re.search(rf"\b{c}\b", text_partner) for c in royal_cues):
        return (RelationshipKind.SUPERIOR_SUBORDINATE, False)
    if any(re.search(rf"\b{c}\b", text_self) for c in royal_cues):
        return (RelationshipKind.SUPERIOR_SUBORDINATE, True)

    # 4. Mentor / Disciple
    mentor_cues = (
        "master", "mentor", "teacher", "sensei", "professor", "sư phụ", "thầy", "tông chủ",
    )
    if any(re.search(rf"\b{c}\b", text_partner) for c in mentor_cues):
        return (RelationshipKind.MENTOR_DISCIPLE, False)
    if any(re.search(rf"\b{c}\b", text_self) for c in mentor_cues):
        return (RelationshipKind.MENTOR_DISCIPLE, True)

    # 5. Servant / Master
    servant_cues = ("maid", "butler", "servant", "attendant", "hầu gái", "người hầu", "quản gia")
    if any(re.search(rf"\b{c}\b", text_partner) for c in servant_cues):
        return (RelationshipKind.SERVANT_MASTER, True)
    if any(re.search(rf"\b{c}\b", text_self) for c in servant_cues):
        return (RelationshipKind.SERVANT_MASTER, False)

    # 6. Rival / Hostile
    hostile_cues = (
        "rival", "enemy", "villain", "nemesis", "antagonist", "hostile", "traitor",
        "kẻ thù", "đối thủ", "phản bội",
    )
    if any(re.search(rf"\b{c}\b", text_partner) for c in hostile_cues) or any(
        re.search(rf"\b{c}\b", text_self) for c in hostile_cues
    ):
        return (RelationshipKind.RIVAL_HOSTILE, True)

    # 7. Explicit Romance Cues
    romance_cues = (
        "lover", "wife", "husband", "girlfriend", "boyfriend", "fiancee", "fiance",
        "heroine", "bride", "vợ", "chồng", "người yêu", "bạn gái",
    )
    if any(re.search(rf"\b{c}\b", text_partner) for c in romance_cues) or any(
        re.search(rf"\b{c}\b", text_self) for c in romance_cues
    ):
        return (RelationshipKind.ROMANCE, is_mc)

    # 8. Visual Novel Protagonist + Heroine / Main Cast heuristic
    partner_gender = str(partner_info.get("gender", "unknown"))
    partner_role = str(partner_info.get("role", ""))
    is_generic_npc = (
        "quần chúng" in partner_role
        or "NPC" in partner_role
        or bool(re.search(r"\b(npc|villager|guard|merchant|soldier|passerby)\b", text_partner))
    )
    if (
        (is_mc and partner_gender == "female")
        or (partner_is_mc and tc.get("gender") == "female")
    ) and not is_generic_npc:
        return (RelationshipKind.ROMANCE, is_mc)

    # 9. Close Friend
    friend_cues = (
        "childhood friend", "best friend", "close friend", "thanh mai trúc mã", "bạn thân",
    )
    if any(re.search(rf"\b{c}\b", text_partner) for c in friend_cues) or any(
        re.search(rf"\b{c}\b", text_self) for c in friend_cues
    ):
        return (RelationshipKind.FRIENDSHIP_CLOSE, True)

    # 10. Default: Formal Acquaintance
    return (RelationshipKind.FORMAL_ACQUAINTANCE, True)


def infer_persona_fallback(
    tag: str,
    dialogues: Sequence[str],
    stylometrics: StylometricSummary,
    role: str = "",
    gender: str = "",
    display_name: str = "",
) -> dict[str, Any]:
    """Infer persona attributes purely from stylometric metrics and heuristic signals."""
    tag_lower = tag.lower().strip()
    disp_lower = display_name.lower().strip()
    role_lower = role.lower().strip()
    traits: tuple[str, ...]

    # 1. Exact Meta Roles Check
    if tag_lower in _PROTAGONIST_TAGS or "protagonist" in role_lower:
        archetype = "protagonist"
        traits = ("thích nghi", "quyết đoán", "trung thực")
        tone = "Điềm đạm, tự nhiên, mang góc nhìn nhân vật chính."
        status = "Nhân vật chính / Phiêu lưu giả (Protagonist)"
    elif tag_lower in _NARRATOR_TAGS:
        archetype = "narrator"
        traits = ("quan sát", "khách quan", "điềm tĩnh")
        tone = "Trung tính, truyền cảm, nhịp điệu đều đặn."
        status = "Người dẫn chuyện (Narrator)"
    elif tag_lower in _INNER_MONOLOGUE_TAGS:
        archetype = "inner_monologue"
        traits = ("nội tâm", "suy tư", "nhạy bén")
        tone = "Thầm thì, tự vấn, nhiều khoảng lặng suy tưởng."
        status = "Độc thoại nội tâm"
    else:
        # 2. Stylometric Archetype Classifier
        excl_rate = stylometrics.exclamation_rate
        ellip_rate = stylometrics.ellipsis_rate
        ques_rate = stylometrics.question_rate
        hesit_rate = stylometrics.hesitation_rate
        avg_len = stylometrics.avg_sentence_length
        ttr = stylometrics.type_token_ratio
        top_interj = stylometrics.top_interjections

        has_stutter = any("-" in w for w in top_interj) or hesit_rate >= 0.25

        if (excl_rate >= 0.7 and (has_stutter or ellip_rate >= 0.2)) or any(
            w in top_interj for w in ("hmph", "b-baka", "tch")
        ):
            archetype = "tsundere"
            traits = ("kiêu hãnh", "khẩu xà tâm phật", "dễ bối rối")
            tone = "Sắc sảo, gay gắt bề ngoài nhưng dễ ngập ngừng, bối rối khi xúc động."
        elif ellip_rate >= 0.5 and excl_rate < 0.3:
            if has_stutter or any(w in top_interj for w in ("eep", "uh", "um")):
                archetype = "dandere"
                traits = ("nhút nhát", "e thẹn", "nhạy cảm")
                tone = "Nhỏ nhẹ, hay ngập ngừng, rụt rè và e ấp."
            else:
                archetype = "kuudere"
                traits = ("điềm tĩnh", "lãnh đạm", "ít nói")
                tone = "Trầm lắng, ngắn gọn, lạnh lùng và tiết chế cảm xúc."
        elif excl_rate >= 1.0 and ellip_rate < 0.25:
            archetype = "rebel_warrior"
            traits = ("nhiệt huyết", "bộc trực", "can trường")
            tone = "Mạnh mẽ, sang sảng, quyết liệt và tràn đầy năng lượng."
        elif avg_len >= 12.0 and ttr >= 0.70 and excl_rate < 0.4:
            archetype = "elder_mentor"
            traits = ("thông thái", "điềm đạm", "sâu sắc")
            tone = "Trang trọng, từ tốn, câu từ trau chuốt và uy nghiêm."
        elif ques_rate >= 0.50:
            archetype = "inquisitive_apprentice"
            traits = ("tò mò", "nhiệt tình", "khiêm tốn")
            tone = "Háo hức, hay thắc mắc, nhịp điệu nhanh và cởi mở."
        else:
            archetype = "supporting_cast"
            traits = ("trung tính", "hòa nhã")
            tone = "Tự nhiên, phù hợp với vai trò hỗ trợ cốt truyện."

        # 3. Socioeconomic Status Cues
        combined_text = f"{tag_lower} {disp_lower} {role_lower}"
        if any(
            w in combined_text
            for w in (
                "queen", "king", "prince", "princess", "lady", "lord",
                "noble", "empress", "duke", "baron", "nữ hoàng", "hoàng đế", "công chúa",
            )
        ):
            status = "Quý tộc / Hoàng gia (Nobility)"
        elif any(
            w in combined_text
            for w in (
                "knight", "commander", "captain", "general", "guard",
                "soldier", "warrior", "hiệp sĩ", "lính gác", "chỉ huy",
            )
        ):
            status = "Quân đội / Hiệp sĩ (Military)"
        elif any(
            w in combined_text
            for w in (
                "mage", "wizard", "witch", "priest", "scholar",
                "archmage", "sorceress", "deity", "goddess", "pháp sư", "thầy",
            )
        ):
            status = "Học giả ma thuật / Giáo sĩ (Religious / Scholarly)"
        elif any(
            w in combined_text
            for w in ("merchant", "trader", "innkeeper", "barman", "bartender", "thương nhân")
        ):
            status = "Thương nhân / Tiểu thương (Merchant)"
        else:
            status = "Thường dân (Commoner)"

    # 4. Canonical Sample Dialogue Selection (2-3 representative lines)
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


PERSONA_SYSTEM_PROMPT = (
    "Bạn là chuyên gia biên kịch và bản địa hóa game visual novel cao cấp.\n"
    "Nhiệm vụ của bạn là phân tích sâu hồ sơ tính cách nhân vật dựa trên tên, "
    "chỉ số hành vi ngôn ngữ (stylometrics), và các câu thoại mẫu thực tế để phục vụ dịch thuật "
    "sang tiếng Việt chuẩn xác và tự nhiên.\n\n"
    "Với mỗi nhân vật, hãy xác định đầy đủ các trường sau:\n"
    "- display_name_vi: Tên tiếng Việt tự nhiên (danh từ chung như 'Guard', 'Mercenary' "
    "dịch là 'Lính gác', 'Lính đánh thuê'; tên riêng người như 'Charlotte', 'Alice' giữ nguyên).\n"
    "- gender: 'female', 'male', hoặc 'unknown'.\n"
    "- role: Vai trò cốt truyện ngắn gọn (VD: 'Nữ hiệp sĩ kiêu hãnh', 'Nhân vật chính').\n"
    "- archetype: Hình mẫu tâm lý chuẩn visual novel: "
    "'tsundere', 'kuudere', 'dandere', 'elder_mentor', 'loyal_knight', 'rebel_warrior', "
    "'inquisitive_apprentice', 'childhood_friend', 'noble_lady', 'genki_girl', "
    "'cunning_villain', 'protagonist', 'narrator', 'supporting_cast'...\n"
    "- personality_traits: Mảng 2-4 nét tính cách chủ đạo "
    "(VD: ['kiêu hãnh', 'khẩu xà tâm phật', 'dễ bối rối']).\n"
    "- tone_and_voice: Mô tả giọng điệu, âm vực, và phong thái đối thoại "
    "(VD: 'Sắc sảo, đanh đá nhưng lắp bắp khi bối rối; âm vực cao').\n"
    "- socioeconomic_status: Địa vị / giai cấp xã hội (VD: 'Tiểu thư quý tộc', "
    "'Đại pháp sư hoàng gia', 'Thường dân', 'Hiệp sĩ lang bạt').\n"
    "- speech_style: Phong cách ăn nói tổng quát bằng tiếng Việt "
    "(VD: 'Kiêu kỳ, đanh thép, ngập ngừng khi xúc động').\n"
    "- sample_dialogue: Mảng gồm 2-3 câu thoại tiêu biểu nhất của nhân vật được chọn từ "
    "các câu thoại mẫu đã cung cấp.\n"
    "- pronoun_map: Quy tắc xưng hô đối ứng tiếng Việt (VD: 'mc': 'tôi - cậu', "
    "'to_elder': 'xưng con, gọi ngài/bác', 'to_peer': 'xưng tôi, gọi cậu', "
    "'to_junior': 'xưng ta, gọi ngươi', 'to_enemy': 'xưng ta, gọi ngươi').\n\n"
    "Trả về DUY NHẤT một đối tượng JSON có khóa là các tag nhân vật:\n"
    "{\n"
    '  "tag_name": {\n'
    '    "display_name_vi": "Charlotte",\n'
    '    "gender": "female",\n'
    '    "role": "Tiểu thư quý tộc / Nữ chính",\n'
    '    "archetype": "tsundere",\n'
    '    "personality_traits": ["kiêu hãnh", "khẩu xà tâm phật", "dễ bối rối"],\n'
    '    "tone_and_voice": "Sắc sảo, gay gắt bề ngoài nhưng dễ ngập ngừng, bối rối khi xúc động",\n'
    '    "socioeconomic_status": "Tiểu thư quý tộc cấp cao",\n'
    '    "speech_style": "Kiêu kỳ, hay mỉa mai, ngập ngừng khi xúc động",\n'
    '    "sample_dialogue": [\n'
    '      "B-Baka! It\'s not like I waited for you or anything...!",\n'
    '      "W-What are you staring at?! Turn around, idiot!"\n'
    "    ],\n"
    '    "pronoun_map": {\n'
    '      "mc": "tôi - cậu",\n'
    '      "to_elder": "xưng cháu/con, gọi bác/ngài",\n'
    '      "to_peer": "xưng tôi, gọi cậu",\n'
    '      "to_junior": "xưng ta, gọi ngươi",\n'
    '      "to_enemy": "xưng ta, gọi ngươi"\n'
    "    }\n"
    "  }\n"
    "}\n"
    "Tuyệt đối không giải thích ngoài JSON."
)


def build_persona_user_prompt(
    chunk_tags: Sequence[str],
    existing_chars: Mapping[str, CharacterProfile],
    samples_by_speaker: Mapping[str, Sequence[str]],
    stylometrics_by_speaker: Mapping[str, StylometricSummary],
) -> str:
    """Build user prompt with stylometrics and formatted dialogue samples."""
    char_blocks: list[str] = []
    for tag in chunk_tags:
        prof = existing_chars.get(tag)
        disp = prof.display_name if prof else tag.capitalize()
        samples = samples_by_speaker.get(tag, [])
        sty = stylometrics_by_speaker.get(tag)
        sty_desc = sty.summary_text() if sty else "Không có dữ liệu thống kê"

        if samples:
            samples_formatted = "\n    ".join(f'{i+1}. "{s}"' for i, s in enumerate(samples[:4]))
        else:
            samples_formatted = "    (Không có thoại)"

        block = (
            f"- Tag: {tag}\n"
            f"  Tên: {disp}\n"
            f"  Chỉ số hành vi thoại (Stylometrics): {sty_desc}\n"
            f"  Thoại mẫu thực tế:\n{samples_formatted}"
        )
        char_blocks.append(block)

    prefix = "Phân tích hồ sơ tính cách và phong thái đối thoại cho các nhân vật sau:\n\n"
    return prefix + "\n\n".join(char_blocks)


def bootstrap_characters_with_llm(
    project: Project,
    db: Database,
    llm: Any,
    *,
    max_chars: int = 25,
    chunk_size: int = 6,
) -> dict[str, CharacterProfile]:
    """Bootstrap character personalities, roles, and Vietnamese pronouns using the local LLM.

    Processes prominent speakers in focused chunks for maximum depth, accuracy,
    and resilience against output truncation.
    """
    all_units = db.all_units()
    speaker_counts: Counter[str] = Counter(u.speaker for u in all_units if u.speaker)
    top_speakers = [tag for tag, _ in speaker_counts.most_common(max_chars)]
    existing_chars = db.get_characters()

    # Fill remaining capacity with defined characters from game definitions
    if len(top_speakers) < max_chars:
        defined_candidates = [
            t
            for t in existing_chars
            if t not in top_speakers and t.lower() not in _NON_INTERACTION_TAGS
        ]
        top_speakers.extend(defined_candidates[: max_chars - len(top_speakers)])

    if not top_speakers:
        return existing_chars

    # Collect 3-4 diverse sample lines for each speaker
    samples_by_speaker: dict[str, list[str]] = {}
    for u in all_units:
        if u.speaker in top_speakers and u.source.strip():
            lst = samples_by_speaker.setdefault(u.speaker, [])
            if len(lst) < 4 and len(u.source) > 8:
                lst.append(u.source.strip())

    stylometrics_by_speaker = {
        tag: extract_stylometrics(samples_by_speaker.get(tag, [])) for tag in top_speakers
    }

    # Build bidirectional lookup table (by tag, lower tag, display name, lower display name)
    tag_lookup: dict[str, str] = {}
    for t, p in existing_chars.items():
        tag_lookup[t.lower()] = t
        if p.display_name:
            tag_lookup[p.display_name.lower()] = t

    # Process speakers in focused chunks
    for i in range(0, len(top_speakers), chunk_size):
        chunk_tags = top_speakers[i : i + chunk_size]
        user_prompt = build_persona_user_prompt(
            chunk_tags, existing_chars, samples_by_speaker, stylometrics_by_speaker
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
                # Resolve tag using lookup
                tag = tag_lookup.get(raw_key.lower(), raw_key)
                if tag not in existing_chars and raw_key in existing_chars:
                    tag = raw_key

                # Defensive unpacking for stringified dicts (ast/json)
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
                role = data.get("role") or (cur.role if cur else "")
                speech_style = data.get("speech_style") or (cur.speech_style if cur else "")
                pronoun_map = data.get("pronoun_map") or (cur.pronoun_map if cur else {})

                # Defensive unpacking for pronoun_map if stringified
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

    # Update characters.json file
    chars_file = project.characters_path
    save_characters_to_json(existing_chars, chars_file)
    content = chars_file.read_bytes()
    db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
    db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())
    return existing_chars


def run_analysis(
    project: Project,
    db: Database,
    *,
    mine_glossary: bool = True,
    enrich_llm: bool = False,
    llm: Any = None,
) -> dict[str, int]:
    """Run full analysis pipeline: discover characters, segment scenes, and glossary."""
    stats = {"characters": 0, "scenes": 0, "glossary_candidates": 0, "glossary_translated": 0}

    # Discover characters from scripts and dialogue speaker tags
    defined_chars = extract_character_definitions(project.root)
    existing_chars = db.get_characters()

    # Also collect all speakers appearing in dialogue units
    all_units = db.all_units()
    dialogue_speakers = {u.speaker for u in all_units if u.speaker}
    speaker_counts: Counter[str] = Counter(u.speaker for u in all_units if u.speaker)
    sources = [u.source for u in all_units]
    total_units_count = len(all_units)

    # Collect dialogue lines grouped by speaker for stylometric analysis
    dialogue_by_speaker: dict[str, list[str]] = {}
    for u in all_units:
        if u.kind is UnitKind.DIALOGUE and u.speaker and u.source.strip():
            dialogue_by_speaker.setdefault(u.speaker, []).append(u.source)

    # Merge defined + discovered
    all_tags = set(defined_chars.keys()) | dialogue_speakers
    rel_graph = RelationshipGraph.sync(db, project.relationships_path)

    # Collect speaker adjacency / interaction network
    interactions: dict[str, Counter[str]] = {}
    prev_speaker: str | None = None
    for u in all_units:
        if u.kind is UnitKind.DIALOGUE and u.speaker:
            spk = u.speaker
            if spk.lower() not in _NON_INTERACTION_TAGS:
                if prev_speaker and prev_speaker != spk:
                    interactions.setdefault(prev_speaker, Counter())[spk] += 1
                    interactions.setdefault(spk, Counter())[prev_speaker] += 1
                    rel_graph.record_interaction(prev_speaker, spk)
                prev_speaker = spk

    chars_file = project.characters_path
    file_chars: dict[str, CharacterProfile] = {}
    if chars_file.is_file():
        with contextlib.suppress(Exception):
            file_chars = load_characters_from_json(chars_file)

    with db.transaction():
        temp_chars: dict[str, dict[str, Any]] = {}
        for tag in sorted(all_tags):
            display_name = defined_chars.get(tag, tag.capitalize())
            fc = file_chars.get(tag)
            ec = existing_chars.get(tag)

            # Prioritize: 1) manual file, 2) existing DB, 3) statistical NLP inference
            gender = (
                (fc.gender if fc and fc.gender != "unknown" else "")
                or (ec.gender if ec and ec.gender != "unknown" else "")
            )
            if not gender or gender == "unknown":
                gender = infer_character_gender_nlp(display_name, sources)

            role = (fc.role if fc and fc.role else "") or (ec.role if ec and ec.role else "")
            if not role:
                role = infer_character_role_nlp(tag, speaker_counts[tag], total_units_count)

            speech_style = (fc.speech_style if fc else "") or (ec.speech_style if ec else "")
            pronoun_map = (
                (dict(fc.pronoun_map) if fc and fc.pronoun_map else {})
                or (dict(ec.pronoun_map) if ec and ec.pronoun_map else {})
            )

            speaker_lines = dialogue_by_speaker.get(tag, [])
            sty = extract_stylometrics(speaker_lines)
            fb = infer_persona_fallback(
                tag, speaker_lines, sty, role=role, gender=gender, display_name=display_name
            )

            archetype = (
                (fc.archetype if fc and fc.archetype else "")
                or (ec.archetype if ec and ec.archetype else "")
                or fb["archetype"]
            )
            personality_traits = (
                (fc.personality_traits if fc and fc.personality_traits else ())
                or (ec.personality_traits if ec and ec.personality_traits else ())
                or fb["personality_traits"]
            )
            tone_and_voice = (
                (fc.tone_and_voice if fc and fc.tone_and_voice else "")
                or (ec.tone_and_voice if ec and ec.tone_and_voice else "")
                or fb["tone_and_voice"]
            )
            socioeconomic_status = (
                (fc.socioeconomic_status if fc and fc.socioeconomic_status else "")
                or (ec.socioeconomic_status if ec and ec.socioeconomic_status else "")
                or fb["socioeconomic_status"]
            )
            sample_dialogue = (
                (fc.sample_dialogue if fc and fc.sample_dialogue else ())
                or (ec.sample_dialogue if ec and ec.sample_dialogue else ())
                or fb["sample_dialogue"]
            )

            temp_chars[tag] = {
                "display_name": (fc.display_name if fc and fc.display_name else display_name),
                "gender": gender,
                "role": role,
                "speech_style": speech_style,
                "pronoun_map": pronoun_map,
                "archetype": archetype,
                "personality_traits": personality_traits,
                "tone_and_voice": tone_and_voice,
                "socioeconomic_status": socioeconomic_status,
                "sample_dialogue": sample_dialogue,
            }

        # Pass 2: Graph-guided and semantic reciprocal pairwise addressing
        has_explicit_mc = any(t.lower() in _PROTAGONIST_TAGS for t in temp_chars)
        for tag, tc in temp_chars.items():
            pronoun_map = dict(tc["pronoun_map"])
            if has_explicit_mc:
                is_mc = tag.lower() in _PROTAGONIST_TAGS
            else:
                is_mc = tc["role"] == "Nhân vật chính (Protagonist)"

            # Exclude non-human or meta entities from interpersonal pronoun mapping
            if is_non_human_or_meta(tag, tc["display_name"]):
                profile = CharacterProfile(
                    tag=tag,
                    display_name=tc["display_name"],
                    gender=tc["gender"],
                    role=tc["role"],
                    speech_style=tc["speech_style"],
                    pronoun_map=pronoun_map,
                    archetype=tc.get("archetype", ""),
                    personality_traits=tc.get("personality_traits", ()),
                    tone_and_voice=tc.get("tone_and_voice", ""),
                    socioeconomic_status=tc.get("socioeconomic_status", ""),
                    sample_dialogue=tc.get("sample_dialogue", ()),
                )
                db.save_character(profile)
                stats["characters"] += 1
                continue

            for partner_tag, _ in interactions.get(tag, Counter()).most_common(5):
                if partner_tag in pronoun_map:
                    continue

                partner_info = temp_chars.get(partner_tag, {})
                if is_non_human_or_meta(partner_tag, partner_info.get("display_name", "")):
                    continue

                if is_collective_entity(partner_tag, partner_info.get("display_name", "")):
                    pronoun_map[partner_tag] = "tôi - mọi người"
                    continue

                if is_collective_entity(tag, tc["display_name"]):
                    pronoun_map[partner_tag] = "mọi người - bạn"
                    continue

                partner_is_mc = (
                    partner_tag.lower() in _PROTAGONIST_TAGS
                    if has_explicit_mc
                    else partner_info.get("role") == "Nhân vật chính (Protagonist)"
                )

                # 1. First check RelationshipGraph for bilateral edge
                edge_out, edge_in = rel_graph.get_bilateral(tag, partner_tag)
                spk_pro = ""
                lis_pro = ""

                if edge_out and edge_out.speaker_pronoun and edge_out.listener_pronoun:
                    spk_pro = edge_out.speaker_pronoun
                    lis_pro = edge_out.listener_pronoun
                elif edge_in and edge_in.kind:
                    if edge_in.kind in (
                        RelationshipKind.RIVAL_HOSTILE,
                        RelationshipKind.FRIENDSHIP_CLOSE,
                        RelationshipKind.FRIENDSHIP_CASUAL,
                        RelationshipKind.FORMAL_ACQUAINTANCE,
                    ):
                        spk_pro, lis_pro = resolve_pronoun_pair_from_relationship(
                            edge_in.kind,
                            tc["gender"],
                            partner_info.get("gender", "unknown"),
                        )
                    else:
                        spk_pro, lis_pro = resolve_pronoun_pair_from_relationship(
                            edge_in.kind,
                            tc["gender"],
                            partner_info.get("gender", "unknown"),
                            is_speaker_senior=False,
                        )
                elif edge_out and edge_out.kind:
                    spk_pro, lis_pro = resolve_pronoun_pair_from_relationship(
                        edge_out.kind,
                        tc["gender"],
                        partner_info.get("gender", "unknown"),
                        is_speaker_senior=True,
                    )

                # 2. Dynamic semantic relationship inferencing if no graph edge
                if not spk_pro or not lis_pro:
                    inferred = infer_semantic_relationship(
                        tag, tc, partner_tag, partner_info, is_mc=is_mc, partner_is_mc=partner_is_mc
                    )
                    if inferred is not None:
                        kind, is_senior = inferred
                        spk_pro, lis_pro = resolve_pronoun_pair_from_relationship(
                            kind,
                            tc["gender"],
                            partner_info.get("gender", "unknown"),
                            is_speaker_senior=is_senior,
                        )
                        rel_graph.add_edge(
                            RelationshipEdge(
                                source_tag=tag,
                                target_tag=partner_tag,
                                kind=kind,
                                speaker_pronoun=spk_pro,
                                listener_pronoun=lis_pro,
                                confidence=0.7,
                                source_type="heuristic",
                            )
                        )
                        if rel_graph.get_edge(partner_tag, tag) is None:
                            if kind in (
                                RelationshipKind.RIVAL_HOSTILE,
                                RelationshipKind.FRIENDSHIP_CLOSE,
                                RelationshipKind.FRIENDSHIP_CASUAL,
                                RelationshipKind.FORMAL_ACQUAINTANCE,
                            ):
                                recip_spk, recip_lis = resolve_pronoun_pair_from_relationship(
                                    kind,
                                    partner_info.get("gender", "unknown"),
                                    tc["gender"],
                                )
                            else:
                                recip_spk, recip_lis = resolve_pronoun_pair_from_relationship(
                                    kind,
                                    partner_info.get("gender", "unknown"),
                                    tc["gender"],
                                    is_speaker_senior=not is_senior,
                                )
                            rel_graph.add_edge(
                                RelationshipEdge(
                                    source_tag=partner_tag,
                                    target_tag=tag,
                                    kind=kind,
                                    speaker_pronoun=recip_spk,
                                    listener_pronoun=recip_lis,
                                    confidence=0.7,
                                    source_type="heuristic",
                                )
                            )

                if spk_pro and lis_pro:
                    pronoun_map[partner_tag] = f"{spk_pro} - {lis_pro}"

            profile = CharacterProfile(
                tag=tag,
                display_name=tc["display_name"],
                gender=tc["gender"],
                role=tc["role"],
                speech_style=tc["speech_style"],
                pronoun_map=pronoun_map,
                archetype=tc.get("archetype", ""),
                personality_traits=tc.get("personality_traits", ()),
                tone_and_voice=tc.get("tone_and_voice", ""),
                socioeconomic_status=tc.get("socioeconomic_status", ""),
                sample_dialogue=tc.get("sample_dialogue", ()),
            )
            db.save_character(profile)
            stats["characters"] += 1

        # Persist relationship graph to SQLite and relationships.json
        rel_graph.save_to_db(db)
        rel_graph.save_to_json_file(project.relationships_path)

    # Save to characters.json for human review/editing
    all_profiles = db.get_characters()
    save_characters_to_json(all_profiles, chars_file)
    content = chars_file.read_bytes()
    db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
    db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())

    # Segment scenes
    scenes = segment_scenes(db)
    with db.transaction():
        for sc in scenes:
            db.save_scene(sc)
            stats["scenes"] += 1

    # Mine glossary candidates
    glossary_file = project.work_dir / "glossary.json"
    exclude_character_names = {
        p.display_name.lower() for p in all_profiles.values() if p.display_name
    } | {tag.lower() for tag in all_tags}

    if mine_glossary:
        extra_terms: list[str] = []
        game_dir = project.root / "game"
        if game_dir.is_dir():
            for f in sorted(game_dir.iterdir()):
                if f.suffix in (".rpy", ".rpyc") and not f.name.startswith(
                    ("00", "gui", "options", "screens", "updater")
                ):
                    clean_stem = (
                        f.stem.replace("_events", "")
                        .replace("_event", "")
                        .replace("_store", " Store")
                        .replace("_clan", " Clan")
                        .replace("_world", " World")
                    )
                    words = clean_stem.replace("_", " ").title().split()
                    c_cand = " ".join(words)
                    if len(c_cand) >= 3 and not c_cand.isdigit():
                        extra_terms.append(c_cand)

        candidates = mine_glossary_candidates(
            sources,
            min_count=2,
            exclude_terms=exclude_character_names,
            extra_terms=extra_terms,
        )
        existing_glossary = Glossary.load_file(glossary_file)
        known_terms = {e.term.lower() for e in existing_glossary.entries}

        new_entries: list[GlossaryEntry] = list(existing_glossary.entries)
        for term, _count in candidates:
            if term.lower() not in known_terms:
                entry = GlossaryEntry(
                    term=term,
                    translation="",  # Left for human or initial translation
                    kind="candidate",
                    case_sensitive=False,
                    aliases=tuple(_make_plural_aliases(term)),
                    notes=f"Khai thác tự động (xuất hiện {_count} lần)",
                )
                new_entries.append(entry)
                known_terms.add(term.lower())
                stats["glossary_candidates"] += 1

        cleaned_entries = clean_glossary(new_entries, exclude_terms=exclude_character_names)
        updated_glossary = Glossary(cleaned_entries)
        updated_glossary.save_file(glossary_file)
        updated_glossary.save_db(db)

    # Optional autonomous LLM bootstrapping
    if enrich_llm and llm is not None:
        _LOG.info("Bắt đầu LLM bootstrapping cho hồ sơ nhân vật và thuật ngữ...")
        bootstrap_characters_with_llm(project, db, llm, max_chars=15)
        db.set_meta("characters_bootstrapped", utc_timestamp())
        if mine_glossary:
            cur_glossary = Glossary.load_db(db)
            cleaned = clean_glossary(cur_glossary.entries, exclude_terms=exclude_character_names)
            untranslated = [e for e in cleaned if not e.translation.strip()]
            if untranslated:

                def _freq(e: GlossaryEntry) -> int:
                    m = re.search(r"(\d+)\s*lần", e.notes)
                    return int(m.group(1)) if m else 1

                untranslated.sort(key=_freq, reverse=True)
                to_trans = untranslated[:150]
                enriched = translate_glossary_with_llm(to_trans, llm)
                trans_map = {e.term: e for e in enriched if e.translation.strip()}
                final_entries = [trans_map.get(e.term, e) for e in cleaned]
                Glossary(final_entries).save_file(glossary_file)
                Glossary(final_entries).save_db(db)
                stats["glossary_translated"] = sum(
                    1 for e in final_entries if e.translation.strip()
                )
            db.set_meta("glossary_bootstrapped", utc_timestamp())

    db.set_meta("analysis_completed", utc_timestamp())
    return stats
