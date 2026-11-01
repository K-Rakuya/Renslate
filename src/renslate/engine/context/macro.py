"""Scene macro context assembly and AST scene flow integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from renslate.core.models import SceneMacroContext

__all__ = [
    "assemble_scene_macro_context",
    "infer_scene_cues",
    "update_scene_macro_from_flow",
]


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


def update_scene_macro_from_flow(
    macro: SceneMacroContext,
    flow_items: Sequence[Any],
) -> SceneMacroContext:
    """Refine SceneMacroContext using AST scene flow items (Scene, Show, Hide, active sprites)."""
    if not flow_items:
        return macro

    active_chars = set(macro.active_characters)
    location = macro.location
    atmosphere = macro.atmosphere
    sprites: set[str] = set()

    for item in flow_items:
        # 1. Register active sprites if present
        item_sprites = getattr(item, "active_sprites", ())
        if item_sprites:
            sprites.update(item_sprites)
            active_chars.update(item_sprites)

        kind = getattr(item, "kind", "")
        name = getattr(item, "name", "")

        # 2. Scene background statement often conveys location
        if kind == "scene" and name:
            if not location:
                loc_cues, atm_cues = infer_scene_cues(name)
                location = loc_cues or name
                if atm_cues and not atmosphere:
                    atmosphere = atm_cues

        # 3. Show statement registers characters in the scene
        elif kind == "show" and name:
            active_chars.add(name)

    metadata = dict(macro.metadata) if macro.metadata else {}
    if sprites:
        metadata["active_sprites"] = tuple(sorted(sprites))

    return macro.with_updates(
        location=location or macro.location,
        atmosphere=atmosphere or macro.atmosphere,
        active_characters=tuple(sorted(active_chars)),
        metadata=metadata,
    )


def assemble_scene_macro_context(
    scene_id: str,
    label: str = "",
    active_characters: Sequence[str] = (),
    location: str = "",
    atmosphere: str = "",
    summary: str = "",
    previous_summary: str = "",
    temporal_anchor: str = "",
    first_seq: int = 0,
    last_seq: int = 0,
    scene_flow_items: Sequence[Any] | None = None,
) -> SceneMacroContext:
    """Assemble a standardized SceneMacroContext from label, sequence bounds, and flow items."""
    inferred_loc, inferred_atm = infer_scene_cues(label) if label else ("", "")
    effective_loc = location or inferred_loc
    effective_atm = atmosphere or inferred_atm

    metadata: dict[str, Any] = {}
    if previous_summary:
        metadata["previous_summary"] = previous_summary

    macro = SceneMacroContext(
        scene_id=scene_id,
        label=label,
        first_seq=first_seq,
        last_seq=last_seq,
        summary=summary,
        location=effective_loc,
        atmosphere=effective_atm,
        temporal_anchor=temporal_anchor,
        active_characters=tuple(sorted(set(active_characters))),
        metadata=metadata,
    )

    if scene_flow_items:
        macro = update_scene_macro_from_flow(macro, scene_flow_items)

    return macro
