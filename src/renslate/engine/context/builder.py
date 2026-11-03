"""Hierarchical multi-tier context builder with Knapsack token budgeting."""

from __future__ import annotations

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
from renslate.engine.context.macro import infer_scene_cues
from renslate.engine.context.prompt_frame import (
    _format_pronoun_pair,
    _format_relational_rule,
    format_rich_dialogue_line,
)
from renslate.engine.context.resolver import (
    _NON_DIALOGUE_TAGS,
    _PROTAGONIST_TAGS,
    InterlocutorResolver,
    ResolvedInterlocutor,
    _is_scene_boundary,
    invert_reciprocal_pronoun_rule,
)
from renslate.engine.context.tokens import estimate_tokens
from renslate.engine.tm import FuzzyMatch

if TYPE_CHECKING:
    from renslate.engine.graph import RelationshipGraph

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "QUOTED_RULES",
    "ContextBuilder",
    "ContextBundle",
    "LookbackItem",
]

_invert_reciprocal_pronoun_rule = invert_reciprocal_pronoun_rule


QUOTED_RULES = (
    "QUY TẮC BẮT BUỘC:\n"
    "1. Giữ NGUYÊN VẸN 100% mọi placeholder đặc biệt có dạng ⟦V1⟧, ⟦T1⟧, ⟦/T1⟧, ⟦E1⟧, ⟦L1⟧...\n"
    "   Tuyệt đối không dịch, không xoá, không đổi hoa/thường, không sửa nội dung bên trong.\n"
    "2. QUY TẮC XƯNG HÔ ĐỐI ỨNG (PRONOUNS) - TUYỆT ĐỐI CẤM ĐẢO NGƯỢC:\n"
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
    "3. Lời thoại súc tích, tự nhiên, văn phong visual novel, tránh dịch word-by-word.\n"
    "   Giữ độ dài vừa phải để không làm tràn textbox.\n"
    "4. QUY TẮC NỘI DUNG LỜI THOẠI:\n"
    "   - Nhãn [Người nói: ...] ở dòng nguồn chỉ dùng làm ngữ cảnh xưng hô.\n"
    "   - TUYỆT ĐỐI KHÔNG chép lại nhãn [Người nói: ...] hay tên nhân vật vào chuỗi kết quả.\n"
    '   - TUYỆT ĐỐI KHÔNG bọc bản dịch trong dấu ngoặc kép ngoài cùng (không dùng " hay \\").\n'
    "   - TUYỆT ĐỐI CHỈ dùng chữ quốc ngữ tiếng Việt. TUYỆT ĐỐI CẤM sử dụng bất kỳ "
    "chữ Hán / tiếng Trung (Hanzi/Kanji) hay lặp từ vô tận.\n"
    '5. ĐỊNH DẠNG ĐẦU RA: Trả về DUY NHẤT một đối tượng JSON: {"id_1": "Bản dịch 1", ...}\n'
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
        while respecting bounds [350, min(max_bound, target)].
        """
        if source_tokens > 0:
            target = int(source_tokens * 1.6) + batch_size * 40 + 128
        else:
            target = batch_size * 80 + 150
        max_bound = min(4096, n_ctx // 2) if n_ctx >= 8000 else min(3000, n_ctx // 2)
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
        retry_feedback: str = "",
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

        # Strictly order dialogue units by sequence (seq) to ensure unbroken chronological context
        batch_units = sorted(batch_units, key=lambda u: (u.seq, u.uid))

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

        present_characters: set[str] = set()
        if scene_macro and scene_macro.active_characters:
            present_characters.update(scene_macro.active_characters)
        elif scene_summary:
            sc_chars = getattr(scene_summary, "active_characters", None)
            if sc_chars:
                present_characters.update(sc_chars)

        resolved_listeners_in_batch: set[str] = set()
        resolved_addressed_pairs: list[tuple[str, str]] = []

        if active_resolver is not None:
            active_speakers = {
                u.speaker.strip()
                for u in batch_units
                if u.speaker and u.speaker.strip().lower() not in _NON_DIALOGUE_TAGS
            }
            known_names = {p.tag: p.display_name for p in characters.values() if p.display_name}
            for p in characters.values():
                if p.display_name:
                    known_names[p.display_name] = p.display_name

            prev_unit: Unit | None = None
            for u in batch_units:
                if prev_unit is not None and _is_scene_boundary(u, prev_unit):
                    active_resolver.reset()
                    prev_unit = None

                if u.speaker:
                    resolved = active_resolver.resolve(
                        unit=u,
                        prev_unit=prev_unit,
                        active_speakers=active_speakers,
                        known_names=known_names,
                        present_characters=present_characters if present_characters else None,
                    )
                    if resolved and resolved.listener:
                        resolved_listeners_in_batch.add(resolved.listener.lower())
                        if not resolved.is_monologue and u.speaker.strip().lower() != resolved.listener.strip().lower():
                            resolved_addressed_pairs.append((u.speaker.strip(), resolved.listener.strip()))
                else:
                    resolved = ResolvedInterlocutor(
                        speaker=None,
                        listener=None,
                        is_monologue=True,
                    )
                is_concise = getattr(self.config.batch, "concise_tags", False)
                lines_section.append(
                    format_rich_dialogue_line(
                        u, resolved, char_map, graph=graph, concise=is_concise
                    )
                )
                prev_unit = u
        else:
            is_concise = getattr(self.config.batch, "concise_tags", False)
            for u in batch_units:
                text = u.masked_source or u.source
                if u.speaker:
                    spk_clean = u.speaker.strip()
                    spk_lower = spk_clean.lower()
                    cp = char_map.get(spk_clean) or char_map.get(spk_lower)
                    if is_concise:
                        if spk_lower in _NON_DIALOGUE_TAGS:
                            speaker_tag = f"[{u.speaker}] "
                        elif cp and cp.display_name:
                            speaker_tag = f"[{cp.display_name}] "
                        else:
                            speaker_tag = f"[{u.speaker}] "
                    else:
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
                    if (
                        present_characters
                        and item.speaker.lower() not in {p.lower() for p in present_characters}
                    ):
                        continue
                    active_dialogue_speakers.append(item.speaker)
                    break

        # =========================================================================
        # LAYER 2: STABLE GAME-LEVEL CHARACTER PROFILES (Deterministic Stable Prefix)
        # Positioned at token 0 of user_prompt immediately following system prompt.
        # Deterministically sorted across all known characters in the visual novel
        # so that KV cache prefix remains 100% identical across all batches in the game.
        # =========================================================================
        char_lines: list[str] = []
        unique_profiles: dict[str, CharacterProfile] = {}
        for p in characters.values():
            tag_key = (p.tag or "").strip().lower()
            if tag_key and tag_key not in unique_profiles:
                unique_profiles[tag_key] = p

        for spk in distinct_speakers:
            spk_clean = spk.strip().lower()
            if spk_clean not in unique_profiles and spk_clean not in _NON_DIALOGUE_TAGS:
                cp = char_map.get(spk) or char_map.get(spk_clean)
                if cp is not None:
                    tag_k = (cp.tag or "").strip().lower()
                    if tag_k and tag_k not in unique_profiles:
                        unique_profiles[tag_k] = cp

        sorted_profiles = sorted(
            unique_profiles.values(),
            key=lambda p: (p.tag or p.display_name or "").lower(),
        )

        for char_prof in sorted_profiles:
            g_raw = (char_prof.gender or "unknown").strip().lower()
            gender_str = "Nữ" if g_raw == "female" else ("Nam" if g_raw == "male" else "Chưa rõ")
            char_info = f"- {char_prof.display_name} (tag: {char_prof.tag}): {gender_str}"
            if char_prof.role:
                char_info += f" | Vai trò: {char_prof.role}"
            if char_prof.speech_style:
                char_info += f" | Giọng: {char_prof.speech_style}"
            if char_prof.pronoun_map:
                pronoun_parts = []
                for k, v in sorted(char_prof.pronoun_map.items()):
                    if k.lower() == (char_prof.tag or "").lower():
                        continue
                    if char_prof.display_name and k.lower() == char_prof.display_name.lower():
                        continue
                    is_self_mc = (
                        (char_prof.tag or "").lower() in _PROTAGONIST_TAGS
                        or "protagonist" in (char_prof.role or "").lower()
                    )
                    if is_self_mc and k.lower() in ("mc", "player"):
                        continue
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
                        if not tgt_char and k.lower() in ("mc", "player"):
                            for p_prof in char_map.values():
                                if p_prof.tag.lower() in _PROTAGONIST_TAGS or "protagonist" in (p_prof.role or "").lower():
                                    tgt_char = p_prof
                                    break
                        if tgt_char and (tgt_char.tag == char_prof.tag or tgt_char is char_prof):
                            continue
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
            else:
                # Retain both active speakers AND resolved listeners in the batch (resolves FLAW-07)
                batch_spk_set = {s.lower() for s in distinct_speakers} | resolved_listeners_in_batch
                fallback_lines = [
                    line
                    for line, prof in zip(char_lines, sorted_profiles, strict=False)
                    if (prof.tag and prof.tag.lower() in batch_spk_set)
                    or (prof.display_name and prof.display_name.lower() in batch_spk_set)
                ]
                if fallback_lines:
                    fb_block = "HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:\n" + "\n".join(fallback_lines)
                    fb_tokens = estimate_tokens(fb_block)
                    if fb_tokens <= available_budget:
                        sections.append(fb_block)
                        available_budget -= fb_tokens

        # =========================================================================
        # LAYER 3: DYNAMIC BATCH CONTEXT (Ordered from Slowest to Fastest Changing)
        # =========================================================================

        # 3.1 Scene Macro Context (Semi-stable across scene)
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

        # 3.2 Dynamic Interlocutor Dialogue Talking Pairs
        relational_lines: list[str] = []
        pairs_seen: set[tuple[str, str]] = set()

        talking_pairs: list[tuple[str, str]] = []
        if resolved_addressed_pairs:
            talking_pairs = list(resolved_addressed_pairs)
        else:
            if len(active_dialogue_speakers) == 2:
                talking_pairs.append((active_dialogue_speakers[0], active_dialogue_speakers[1]))
            elif len(active_dialogue_speakers) > 2:
                prev_spk_turn: str | None = None
                for u in batch_units:
                    if u.speaker and u.speaker.strip().lower() not in _NON_DIALOGUE_TAGS:
                        curr_spk = u.speaker.strip()
                        if prev_spk_turn and prev_spk_turn.lower() != curr_spk.lower():
                            talking_pairs.append((prev_spk_turn, curr_spk))
                        prev_spk_turn = curr_spk
            elif len(active_dialogue_speakers) == 1 and lookback:
                single_lower = active_dialogue_speakers[0].lower()
                for item in reversed(lookback):
                    if (
                        item.speaker
                        and item.speaker.lower() not in _NON_DIALOGUE_TAGS
                        and item.speaker.lower() != single_lower
                    ):
                        if (
                            present_characters
                            and item.speaker.lower() not in {p.lower() for p in present_characters}
                        ):
                            continue
                        talking_pairs.append((active_dialogue_speakers[0], item.speaker))
                        break

        for s_a, s_b in talking_pairs:
            p_a = char_map.get(s_a) or char_map.get(s_a.lower())
            p_b = char_map.get(s_b) or char_map.get(s_b.lower())
            if p_a and p_b and p_a.tag != p_b.tag:
                pair_key = (p_a.tag, p_b.tag) if p_a.tag < p_b.tag else (p_b.tag, p_a.tag)
                if pair_key in pairs_seen:
                    continue
                pairs_seen.add(pair_key)
                rule_a_to_b = (
                    p_a.pronoun_map.get(p_b.tag)
                    or p_a.pronoun_map.get(p_b.tag.lower())
                    or (p_a.pronoun_map.get(p_b.display_name) if p_b.display_name else None)
                    or (p_a.pronoun_map.get(p_b.display_name.lower()) if p_b.display_name else None)
                )
                if not rule_a_to_b:
                    is_b_mc = (
                        p_b.tag.lower() in _PROTAGONIST_TAGS
                        or "protagonist" in (p_b.role or "").lower()
                    )
                    if is_b_mc:
                        rule_a_to_b = p_a.pronoun_map.get("mc") or p_a.pronoun_map.get("player")

                rule_b_to_a = (
                    p_b.pronoun_map.get(p_a.tag)
                    or p_b.pronoun_map.get(p_a.tag.lower())
                    or (p_b.pronoun_map.get(p_a.display_name) if p_a.display_name else None)
                    or (p_b.pronoun_map.get(p_a.display_name.lower()) if p_a.display_name else None)
                )
                if not rule_b_to_a:
                    is_a_mc = (
                        p_a.tag.lower() in _PROTAGONIST_TAGS
                        or "protagonist" in (p_a.role or "").lower()
                    )
                    if is_a_mc:
                        rule_b_to_a = p_b.pronoun_map.get("mc") or p_b.pronoun_map.get("player")
                if graph is not None:
                    edge_ab = graph.get_edge(p_a.tag, p_b.tag) or graph.get_edge(s_a, s_b)
                    if edge_ab and edge_ab.speaker_pronoun and edge_ab.listener_pronoun:
                        rule_a_to_b = f"{edge_ab.speaker_pronoun} - {edge_ab.listener_pronoun}"
                    edge_ba = graph.get_edge(p_b.tag, p_a.tag) or graph.get_edge(s_b, s_a)
                    if edge_ba and edge_ba.speaker_pronoun and edge_ba.listener_pronoun:
                        rule_b_to_a = f"{edge_ba.speaker_pronoun} - {edge_ba.listener_pronoun}"
                if rule_a_to_b and not rule_b_to_a:
                    rule_b_to_a = _invert_reciprocal_pronoun_rule(rule_a_to_b)
                elif rule_b_to_a and not rule_a_to_b:
                    rule_a_to_b = _invert_reciprocal_pronoun_rule(rule_b_to_a)
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

        # 3.4 Lookback History
        if self.config.batch.lookback > 0 and lookback and available_budget > 60:
            lb_block, available_budget = self._compact_lookback(lookback, available_budget)
            if lb_block is not None:
                sections.append(lb_block)

        # 3.5 Fuzzy Matches / Translation Memory
        if fuzzy_matches and available_budget > 100:
            fz_lines: list[str] = []
            for uid, matches in fuzzy_matches.items():
                for m in matches:
                    pct = int(m.score * 100)
                    fz_lines.append(
                        f'- Gợi ý cho {uid} (khớp {pct}%): "{m.source}" => "{m.target}"'
                    )
            if fz_lines:
                fz_block = (
                    "GỢI Ý TỪ BỘ NHỚ DỊCH:\n"
                    "(Chỉ để tham khảo thuật ngữ và giọng văn. Đại từ xưng hô PHẢI theo quy tắc "
                    "của cặp nhân vật đang đối thoại ở trên, không sao chép máy móc.)\n"
                    + "\n".join(fz_lines)
                )
                tokens = estimate_tokens(fz_block)
                if tokens <= available_budget:
                    sections.append(fz_block)
                    available_budget -= tokens

        # 3.6 Batch Dialogue Lines
        if retry_feedback:
            sections.append(retry_feedback)
        sections.append(batch_text)
        return "\n\n".join(sections)
