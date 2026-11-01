"""Mob / Anonymous NPC Social Role Taxonomy and Relational Inferencing.

Provides classification and Vietnamese honorific/pronoun pairing for non-protagonist,
anonymous, or background NPCs (waiter, police, doctor, student, crowd, etc.).
"""

from __future__ import annotations

import enum
from typing import Final

from renslate.core.models import CharacterProfile

__all__ = ["MobRoleCategory", "MobRoleClassifier"]


class MobRoleCategory(enum.Enum):
    """Broad social role category of an anonymous NPC."""

    SERVICE = "service"
    AUTHORITY = "authority"
    PEER = "peer"


_SERVICE_KEYWORDS: Final[frozenset[str]] = frozenset({
    "waiter", "waitress", "maid", "bartender", "cashier", "clerk", "nurse",
    "receptionist", "vendor", "shopkeeper", "barista", "driver", "butler",
    "servant", "bồi_bàn", "phục_vụ", "hầu_gái", "thu_ngân", "tiếp_tân",
    "người_bán_hàng", "tài_xế",
})

_AUTHORITY_KEYWORDS: Final[frozenset[str]] = frozenset({
    "doctor", "officer", "police", "cop", "guard", "security", "judge",
    "boss", "teacher", "professor", "soldier", "captain", "commander",
    "detective", "inspector", "king", "queen", "lord", "master",
    "bác_sĩ", "cảnh_sát", "bảo_vệ", "quan_tòa", "giáo_viên", "thầy_giáo",
    "cô_giáo", "giáo_sư", "thuyền_trưởng", "chỉ_huy", "thanh_tra", "bệ_hạ",
})

_PEER_KEYWORDS: Final[frozenset[str]] = frozenset({
    "student", "classmate", "child", "kid", "boy", "girl", "stranger",
    "pedestrian", "passerby", "crowd", "mob", "citizen", "villager",
    "customer", "patron", "guest", "neighbor", "friend", "học_sinh",
    "bạn_học", "người_lạ", "người_qua_đường", "dân_làng", "khách_hàng",
    "khách", "hàng_xóm", "bạn_bè",
})


class MobRoleClassifier:
    """Classifies anonymous NPCs into social roles and infers Vietnamese address dynamics."""

    @classmethod
    def classify(cls, tag_or_name: str) -> MobRoleCategory | None:
        """Classify a character tag or name into a MobRoleCategory."""
        normalized = tag_or_name.strip().lower().replace("-", "_").replace(" ", "_")
        tokens = set(normalized.split("_"))

        if tokens.intersection(_SERVICE_KEYWORDS) or any(k in normalized for k in _SERVICE_KEYWORDS):
            return MobRoleCategory.SERVICE
        if tokens.intersection(_AUTHORITY_KEYWORDS) or any(k in normalized for k in _AUTHORITY_KEYWORDS):
            return MobRoleCategory.AUTHORITY
        if tokens.intersection(_PEER_KEYWORDS) or any(k in normalized for k in _PEER_KEYWORDS):
            return MobRoleCategory.PEER

        return None

    @classmethod
    def infer_pronoun_pair(
        cls,
        speaker_tag: str,
        listener_tag: str | None,
    ) -> tuple[str, str] | None:
        """Infer (speaker_pronoun, listener_pronoun) in Vietnamese based on social hierarchy."""
        spk_cat = cls.classify(speaker_tag)
        lis_cat = cls.classify(listener_tag) if listener_tag else None

        if spk_cat is None and lis_cat is None:
            return None

        # 1. Speaker is SERVICE
        if spk_cat is MobRoleCategory.SERVICE:
            if lis_cat is MobRoleCategory.AUTHORITY:
                return ("tôi", "ngài")
            return ("tôi", "quý khách")

        # 2. Listener is SERVICE (Speaker is customer or peer)
        if lis_cat is MobRoleCategory.SERVICE:
            return ("tôi", "cậu")

        # 3. Speaker is AUTHORITY
        if spk_cat is MobRoleCategory.AUTHORITY:
            return ("tôi", "anh/chị")

        # 4. Listener is AUTHORITY
        if lis_cat is MobRoleCategory.AUTHORITY:
            return ("tôi", "ngài")

        # 5. Both are PEER
        if spk_cat is MobRoleCategory.PEER or lis_cat is MobRoleCategory.PEER:
            return ("tôi", "bạn")

        return None

    @classmethod
    def create_mob_profile(cls, tag: str, display_name: str | None = None) -> CharacterProfile:
        """Create a synthetic CharacterProfile for an anonymous NPC."""
        cat = cls.classify(tag)
        role = cat.value if cat else "npc"
        disp = display_name or tag.replace("_", " ").title()

        return CharacterProfile(
            tag=tag,
            display_name=disp,
            role=role,
            archetype=f"mob_{role}",
        )
