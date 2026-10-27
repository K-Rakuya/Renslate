"""Context assembly and prompt engineering with token budgeting for vitrans."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from vitrans.config import Config
from vitrans.models import CharacterProfile, GlossaryEntry, SceneSummary, Unit
from vitrans.tm import FuzzyMatch

_CHARS_PER_TOKEN_EN = 3.6
_CHARS_PER_TOKEN_VI = 2.8


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
    "3. XƯNG HÔ ĐỐI ỨNG (PRONOUNS) NHẤT QUÁN & ĐÚNG VAI VẾ:\n"
    "   - Tuân thủ nghiêm ngặt quan hệ vai vế trong Hồ sơ nhân vật và ngữ cảnh đối thoại.\n"
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
    def calculate_output_budget(batch_size: int) -> int:
        """Calculate reserved completion token budget for a batch size."""
        return max(500, min(3000, batch_size * 80 + 150))

    def build_user_prompt(
        self,
        batch_units: Sequence[Unit],
        glossary_matches: Sequence[GlossaryEntry] = (),
        characters: dict[str, CharacterProfile] | None = None,
        scene_summary: SceneSummary | None = None,
        lookback: Sequence[LookbackItem] = (),
        fuzzy_matches: dict[str, list[FuzzyMatch]] | None = None,
    ) -> str:
        """Assemble the user prompt prioritizing critical context within token limits."""
        characters = characters or {}
        fuzzy_matches = fuzzy_matches or {}

        # 1. Mandatory batch lines
        lines_section = ["CÁC DÒNG CẦN DỊCH (bắt buộc dịch đầy đủ đúng id):"]
        for u in batch_units:
            text = u.masked_source or u.source
            speaker_tag = f"[Người nói: {u.speaker}] " if u.speaker else ""
            lines_section.append(f"{u.uid} | {speaker_tag}{text}")
        batch_text = "\n".join(lines_section)

        # Token Budgeting
        system_text = self.build_system_prompt()
        sys_tokens = estimate_tokens(system_text)
        batch_tokens = estimate_tokens(batch_text)
        # Reserve token budget for response (proportional to batch size)
        reserve_output = self.calculate_output_budget(len(batch_units))
        max_context = self.config.llm.n_ctx
        available_budget = max_context - sys_tokens - batch_tokens - reserve_output

        sections: list[str] = []

        # Priority 2: Glossary terms present in the batch
        if glossary_matches and available_budget > 100:
            gl_lines = ["THUẬT NGỮ BẮT BUỘC (GLOSSARY):"]
            candidate_terms: list[str] = []
            for g in glossary_matches:
                if g.translation.strip():
                    note_str = f" ({g.notes})" if g.notes else ""
                    gl_lines.append(f"- {g.term} -> {g.translation}{note_str}")
                else:
                    candidate_terms.append(g.term)

            if len(gl_lines) > 1:
                gl_block = "\n".join(gl_lines)
                tokens = estimate_tokens(gl_block)
                if tokens <= available_budget:
                    sections.append(gl_block)
                    available_budget -= tokens

            if candidate_terms and available_budget > 60:
                cand_block = (
                    "DANH TỪ RIÊNG & THUẬT NGỮ CẦN ĐỒNG NHẤT:\n- "
                    + ", ".join(candidate_terms[:12])
                    + " (chú ý dịch đồng nhất hoặc giữ nguyên nếu là tên người/địa danh)"
                )
                tokens = estimate_tokens(cand_block)
                if tokens <= available_budget:
                    sections.append(cand_block)
                    available_budget -= tokens

        # Priority 3: Character profiles for speakers in current batch
        speakers_in_batch = {u.speaker for u in batch_units if u.speaker}
        char_lines: list[str] = []
        for spk in sorted(speakers_in_batch):
            if spk in characters:
                p = characters[spk]
                g_raw = (p.gender or "unknown").strip().lower()
                gender_str = (
                    "Nữ" if g_raw == "female" else ("Nam" if g_raw == "male" else "Chưa rõ")
                )
                char_info = f"- {p.display_name} (tag: {p.tag}): {gender_str}"
                if p.role:
                    char_info += f" | Vai trò: {p.role}"
                if p.speech_style:
                    char_info += f" | Giọng: {p.speech_style}"
                if p.pronoun_map:
                    pronoun_parts = []
                    for k, v in p.pronoun_map.items():
                        if k == "to_elder":
                            pronoun_parts.append(f"với bề trên ({v})")
                        elif k == "to_peer":
                            pronoun_parts.append(f"với bạn bè/đồng môn ({v})")
                        elif k == "to_junior":
                            pronoun_parts.append(f"với bề dưới ({v})")
                        elif k == "to_enemy":
                            pronoun_parts.append(f"với kẻ thù ({v})")
                        else:
                            pronoun_parts.append(f"{k}: {v}")
                    char_info += f" | Xưng hô: {'; '.join(pronoun_parts)}"
                char_lines.append(char_info)
        if char_lines and available_budget > 80:
            ch_block = "HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:\n" + "\n".join(char_lines)
            tokens = estimate_tokens(ch_block)
            if tokens <= available_budget:
                sections.append(ch_block)
                available_budget -= tokens

        # Priority 4: Scene summary
        if scene_summary and scene_summary.summary and available_budget > 80:
            sc_block = f"TÓM TẮT CẢNH HIỆN TẠI:\n{scene_summary.summary}"
            tokens = estimate_tokens(sc_block)
            if tokens <= available_budget:
                sections.append(sc_block)
                available_budget -= tokens

        # Priority 5: Lookback (preceding 3-6 translated lines)
        if self.config.batch.lookback > 0 and lookback and available_budget > 100:
            lb_lines = ["NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC (để nối mạch):"]
            for item in list(lookback)[-self.config.batch.lookback :]:
                spk = f"[{item.speaker}] " if item.speaker else ""
                lb_lines.append(f'- {spk}"{item.source}" => "{item.target}"')
            lb_block = "\n".join(lb_lines)
            tokens = estimate_tokens(lb_block)
            if tokens <= available_budget:
                sections.append(lb_block)
                available_budget -= tokens

        # Priority 6: TM fuzzy suggestions
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

        # Assemble prompt with context first, then batch to translate at the end
        sections.append(batch_text)
        return "\n\n".join(sections)
