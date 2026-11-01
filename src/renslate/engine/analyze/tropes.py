"""Archetype, trope, and socioeconomic status detection without arbitrary magic constants."""

from __future__ import annotations

from dataclasses import dataclass

from renslate.engine.analyze.stylometrics import StylometricSummary

__all__ = [
    "COLLECTIVE_TAGS",
    "INNER_MONOLOGUE_TAGS",
    "NARRATOR_TAGS",
    "NON_HUMAN_TAGS",
    "NON_INTERACTION_TAGS",
    "PROTAGONIST_TAGS",
    "ArchetypeThresholds",
    "detect_archetype",
]

PROTAGONIST_TAGS = frozenset({"mc", "player", "hero", "protag", "protagonist", "main", "you"})
NARRATOR_TAGS = frozenset({"narrator", "narration", "none"})
INNER_MONOLOGUE_TAGS = frozenset({"t", "th", "thought", "thoughts", "thinking"})
COLLECTIVE_TAGS = frozenset(
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
NON_HUMAN_TAGS = frozenset(
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
NON_INTERACTION_TAGS = frozenset(NON_HUMAN_TAGS | {"none"})

NOBILITY_KEYWORDS = frozenset(
    {
        "queen",
        "king",
        "prince",
        "princess",
        "lady",
        "lord",
        "noble",
        "empress",
        "duke",
        "baron",
        "nữ hoàng",
        "hoàng đế",
        "công chúa",
    }
)

MILITARY_KEYWORDS = frozenset(
    {
        "knight",
        "commander",
        "captain",
        "general",
        "guard",
        "soldier",
        "warrior",
        "hiệp sĩ",
        "lính gác",
        "chỉ huy",
    }
)

SCHOLARLY_KEYWORDS = frozenset(
    {
        "mage",
        "wizard",
        "witch",
        "priest",
        "scholar",
        "archmage",
        "sorceress",
        "deity",
        "goddess",
        "pháp sư",
        "thầy",
    }
)

MERCHANT_KEYWORDS = frozenset(
    {"merchant", "trader", "innkeeper", "barman", "bartender", "thương nhân"}
)


@dataclass(frozen=True, slots=True)
class ArchetypeThresholds:
    """Configurable thresholds for stylometric archetype classification."""

    tsundere_min_exclamation_rate: float = 0.70
    tsundere_min_ellipsis_rate: float = 0.20
    tsundere_min_hesitation_rate: float = 0.25
    dandere_min_ellipsis_rate: float = 0.50
    dandere_max_exclamation_rate: float = 0.30
    rebel_min_exclamation_rate: float = 1.00
    rebel_max_ellipsis_rate: float = 0.25
    elder_min_avg_length: float = 12.0
    elder_min_ttr: float = 0.70
    elder_max_exclamation_rate: float = 0.40
    apprentice_min_question_rate: float = 0.50


_DEFAULT_THRESHOLDS = ArchetypeThresholds()


def detect_archetype(
    tag: str,
    role: str,
    stylometrics: StylometricSummary,
    display_name: str = "",
    thresholds: ArchetypeThresholds = _DEFAULT_THRESHOLDS,
) -> tuple[str, tuple[str, ...], str, str]:
    """Detect visual novel archetype, personality traits, tone, and socioeconomic status.

    Returns:
        (archetype, traits, tone, status)
    """
    tag_lower = tag.lower().strip()
    disp_lower = display_name.lower().strip()
    role_lower = role.lower().strip()

    # 1. Exact Meta Roles Check
    if tag_lower in PROTAGONIST_TAGS or "protagonist" in role_lower:
        return (
            "protagonist",
            ("thích nghi", "quyết đoán", "trung thực"),
            "Điềm đạm, tự nhiên, mang góc nhìn nhân vật chính.",
            "Nhân vật chính / Phiêu lưu giả (Protagonist)",
        )
    if tag_lower in NARRATOR_TAGS:
        return (
            "narrator",
            ("quan sát", "khách quan", "điềm tĩnh"),
            "Trung tính, truyền cảm, nhịp điệu đều đặn.",
            "Người dẫn chuyện (Narrator)",
        )
    if tag_lower in INNER_MONOLOGUE_TAGS:
        return (
            "inner_monologue",
            ("nội tâm", "suy tư", "nhạy bén"),
            "Thầm thì, tự vấn, nhiều khoảng lặng suy tưởng.",
            "Độc thoại nội tâm",
        )

    # 2. Stylometric Archetype Classifier
    excl_rate = stylometrics.exclamation_rate
    ellip_rate = stylometrics.ellipsis_rate
    ques_rate = stylometrics.question_rate
    hesit_rate = stylometrics.hesitation_rate
    avg_len = stylometrics.avg_sentence_length
    ttr = stylometrics.type_token_ratio
    top_interj = stylometrics.top_interjections

    has_stutter = (
        any("-" in w for w in top_interj)
        or hesit_rate >= thresholds.tsundere_min_hesitation_rate
    )

    traits: tuple[str, ...]
    if (
        excl_rate >= thresholds.tsundere_min_exclamation_rate
        and (has_stutter or ellip_rate >= thresholds.tsundere_min_ellipsis_rate)
    ) or any(w in top_interj for w in ("hmph", "b-baka", "tch")):
        archetype = "tsundere"
        traits = ("kiêu hãnh", "khẩu xà tâm phật", "dễ bối rối")
        tone = "Sắc sảo, gay gắt bề ngoài nhưng dễ ngập ngừng, bối rối khi xúc động."
    elif (
        ellip_rate >= thresholds.dandere_min_ellipsis_rate
        and excl_rate < thresholds.dandere_max_exclamation_rate
    ):
        if has_stutter or any(w in top_interj for w in ("eep", "uh", "um")):
            archetype = "dandere"
            traits = ("nhút nhát", "e thẹn", "nhạy cảm")
            tone = "Nhỏ nhẹ, hay ngập ngừng, rụt rè và e ấp."
        else:
            archetype = "kuudere"
            traits = ("điềm tĩnh", "lãnh đạm", "ít nói")
            tone = "Trầm lắng, ngắn gọn, lạnh lùng và tiết chế cảm xúc."
    elif (
        excl_rate >= thresholds.rebel_min_exclamation_rate
        and ellip_rate < thresholds.rebel_max_ellipsis_rate
    ):
        archetype = "rebel_warrior"
        traits = ("nhiệt huyết", "bộc trực", "can trường")
        tone = "Mạnh mẽ, sang sảng, quyết liệt và tràn đầy năng lượng."
    elif (
        avg_len >= thresholds.elder_min_avg_length
        and ttr >= thresholds.elder_min_ttr
        and excl_rate < thresholds.elder_max_exclamation_rate
    ):
        archetype = "elder_mentor"
        traits = ("thông thái", "điềm đạm", "sâu sắc")
        tone = "Trang trọng, từ tốn, câu từ trau chuốt và uy nghiêm."
    elif ques_rate >= thresholds.apprentice_min_question_rate:
        archetype = "inquisitive_apprentice"
        traits = ("tò mò", "nhiệt tình", "khiêm tốn")
        tone = "Háo hức, hay thắc mắc, nhịp điệu nhanh và cởi mở."
    else:
        archetype = "supporting_cast"
        traits = ("trung tính", "hòa nhã")
        tone = "Tự nhiên, phù hợp với vai trò hỗ trợ cốt truyện."

    # 3. Socioeconomic Status Cues
    combined_text = f"{tag_lower} {disp_lower} {role_lower}"
    if any(w in combined_text for w in NOBILITY_KEYWORDS):
        status = "Quý tộc / Hoàng gia (Nobility)"
    elif any(w in combined_text for w in MILITARY_KEYWORDS):
        status = "Quân đội / Hiệp sĩ (Military)"
    elif any(w in combined_text for w in SCHOLARLY_KEYWORDS):
        status = "Học giả ma thuật / Giáo sĩ (Religious / Scholarly)"
    elif any(w in combined_text for w in MERCHANT_KEYWORDS):
        status = "Thương nhân / Tiểu thương (Merchant)"
    else:
        status = "Thường dân (Commoner)"

    return archetype, traits, tone, status
