"""Interactive TUI Control Board for visual feature toggling and configuration."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from renslate.ui.theme import (
    BOX_LIGHT_HORIZ,
    BOX_LIGHT_VERT,
    BOX_ROUND_BL,
    BOX_ROUND_BR,
    BOX_ROUND_TL,
    BOX_ROUND_TR,
    C_BOLD,
    C_CYAN,
    C_DIM,
    C_EMERALD,
    C_RESET,
    C_SKY,
    C_SLATE,
    C_YELLOW,
    IS_TTY,
    visible_len,
)

__all__ = [
    "ControlBoardState",
    "ControlField",
    "OptionType",
    "create_default_control_fields",
    "render_control_board",
    "run_control_board",
]


class OptionType(enum.Enum):
    """Supported field input control styles."""

    BOOLEAN = "boolean"
    NUMBER_RANGE = "number_range"
    CHOICE = "choice"


@dataclass
class ControlField:
    """A configurable control switch or parameter with micro-interactions."""

    key: str
    label: str
    field_type: OptionType
    value: Any
    options: list[Any] = field(default_factory=list)
    min_val: float | int = 0
    max_val: float | int = 100
    step: float | int = 1
    description: str = ""
    recommendation: str = ""
    impact: str = ""

    def toggle(self) -> None:
        """Toggle boolean state."""
        if self.field_type == OptionType.BOOLEAN:
            self.value = not self.value

    def step_next(self) -> None:
        """Step value forward."""
        if self.field_type == OptionType.NUMBER_RANGE:
            new_val = self.value + self.step
            if isinstance(self.step, float):
                new_val = round(new_val, 4)
            self.value = min(self.max_val, new_val)
        elif self.field_type == OptionType.CHOICE and self.options:
            try:
                curr_idx = self.options.index(self.value)
                next_idx = (curr_idx + 1) % len(self.options)
                self.value = self.options[next_idx]
            except ValueError:
                self.value = self.options[0]

    def step_prev(self) -> None:
        """Step value backward."""
        if self.field_type == OptionType.NUMBER_RANGE:
            new_val = self.value - self.step
            if isinstance(self.step, float):
                new_val = round(new_val, 4)
            self.value = max(self.min_val, new_val)
        elif self.field_type == OptionType.CHOICE and self.options:
            try:
                curr_idx = self.options.index(self.value)
                prev_idx = (curr_idx - 1 + len(self.options)) % len(self.options)
                self.value = self.options[prev_idx]
            except ValueError:
                self.value = self.options[-1]

    def format_display_value(self, plain: bool = False) -> str:
        """Format field value into concise visual switch or slider."""
        use_color = (not plain) and IS_TTY
        c_emerald = C_EMERALD if use_color else ""
        c_slate = C_SLATE if use_color else ""
        c_dim = C_DIM if use_color else ""
        c_reset = C_RESET if use_color else ""

        if self.field_type == OptionType.BOOLEAN:
            if self.value:
                return f"{c_emerald}[● BẬT]{c_reset}"
            return f"{c_slate}[○ TẮT]{c_reset}"

        if self.field_type == OptionType.NUMBER_RANGE:
            val_str = f"{self.value:.2f}" if isinstance(self.value, float) else f"{self.value}"
            return f"{c_dim}◄{c_reset} {val_str} {c_dim}►{c_reset}"

        if self.field_type == OptionType.CHOICE:
            return f"{c_dim}◄{c_reset} {self.value} {c_dim}►{c_reset}"

        return str(self.value)


@dataclass
class ControlBoardState:
    """State machine governing active cursor, fields list, and config serialization."""

    fields: list[ControlField]
    active_idx: int = 0

    @property
    def current_field(self) -> ControlField:
        idx = max(0, min(self.active_idx, len(self.fields) - 1))
        return self.fields[idx]

    def move_up(self) -> None:
        """Move cursor up with wrap-around."""
        if not self.fields:
            return
        self.active_idx = (self.active_idx - 1 + len(self.fields)) % len(self.fields)

    def move_down(self) -> None:
        """Move cursor down with wrap-around."""
        if not self.fields:
            return
        self.active_idx = (self.active_idx + 1) % len(self.fields)

    def to_config_dict(self) -> dict[str, Any]:
        """Convert state into flat configuration dictionary for engine pipeline."""
        return {f.key: f.value for f in self.fields}


def create_default_control_fields() -> list[ControlField]:
    """Instantiate standard 2026 literary localization control fields."""
    return [
        ControlField(
            key="deliberate",
            label="2-Pass Deliberative Reflector",
            field_type=OptionType.BOOLEAN,
            value=True,
            description=(
                "Cơ chế tư duy chậm (System-2) tự động nhận diện câu dịch thô cứng, "
                "bị động gượng gạo và dùng LLM viết lại thành tiếng Việt văn học tự nhiên."
            ),
            recommendation="Khuyên dùng: BẬT cho mọi Visual Novel văn học.",
            impact="Khống chế trần ngân sách <= 25% số câu để giữ ~90% tốc độ 1-Pass.",
        ),
        ControlField(
            key="calque_threshold",
            label="Ngưỡng Calque kích hoạt",
            field_type=OptionType.NUMBER_RANGE,
            value=0.35,
            min_val=0.15,
            max_val=0.60,
            step=0.05,
            description=(
                "Mức độ dịch thô cứng tối thiểu để kích hoạt Deliberative Reflection. "
                "Càng thấp thì LLM càng trau chuốt nhiều câu thoại hơn."
            ),
            recommendation="Mặc định 0.35 là điểm cân bằng vàng giữa tốc độ và độ mượt.",
            impact="Giá trị thấp hơn (0.25) tăng độ mượt nhưng tăng số lượt gọi LLM.",
        ),
        ControlField(
            key="exemplars",
            label="Dyad TM Exemplars (câu mẫu)",
            field_type=OptionType.NUMBER_RANGE,
            value=2,
            min_val=0,
            max_val=4,
            step=1,
            description=(
                "Số lượng câu dịch mẫu đã được xác nhận của đúng cặp nhân vật đang đối thoại "
                "được nạp động từ database TM vào prompt."
            ),
            recommendation="2 câu mẫu giúp model nắm bắt chuẩn xác văn phong nhân vật.",
            impact="Chiếm thêm ~40 token prompt, hiệu quả cao.",
        ),
        ControlField(
            key="narrative_memory",
            label="Ring Buffer Lịch sử Cảnh",
            field_type=OptionType.NUMBER_RANGE,
            value=6,
            min_val=0,
            max_val=16,
            step=2,
            description=(
                "Số lượt đối thoại gần nhất có gắn thẻ sân khấu và người nghe được ghi nhớ. "
                "Tự động làm sạch khi chuyển phân cảnh Ren'Py."
            ),
            recommendation="6 lượt thoại là kích thước tối ưu cho ngữ cảnh đối thoại liên tiếp.",
            impact="Ngăn ngừa đứt gãy mạch xưng hô và rò rỉ bối cảnh.",
        ),
        ControlField(
            key="transgraph",
            label="TransGraph Micro-Framing 4D",
            field_type=OptionType.BOOLEAN,
            value=True,
            description=(
                "Khung vi mô quan hệ 4 chiều: Thân sơ, Quyền lực, Xúc cảm, Ngữ cảnh "
                "cùng toán tử chiếu đại số Pi(R) định hướng xưng hô chuẩn xác."
            ),
            recommendation="Khuyên dùng: BẬT để đảm bảo cặp xưng hô đối xứng (anh-em, ta-con).",
            impact="Tiêu tốn ~70 token prompt, triệt tiêu xưng hô sai ngôi.",
        ),
        ControlField(
            key="guard",
            label="Continuous Hallucination Guard",
            field_type=OptionType.BOOLEAN,
            value=True,
            description=(
                "Hệ thống thẩm định ảo giác liên tục quét phát hiện dịch lặp từ, "
                "tự bịa đặt tình tiết hoặc rò rỉ tiếng Trung/Nhật/Anh."
            ),
            recommendation="Khuyên dùng: BẬT để bảo đảm an toàn tuyệt đối cho bản dịch.",
            impact="Tự động sửa lỗi hoặc gắn cờ QA cảnh báo.",
        ),
        ControlField(
            key="batch_size",
            label="Kích thước lô (Batch lines)",
            field_type=OptionType.NUMBER_RANGE,
            value=8,
            min_val=1,
            max_val=25,
            step=1,
            description="Số lượng dòng thoại được gom dịch cùng lúc trong một lượt gọi LLM.",
            recommendation="8-12 dòng là kích thước lý tưởng cho LLM 14B-35B.",
            impact="Lô lớn hơn giúp dịch nhanh hơn nhưng tốn nhiều bộ nhớ context hơn.",
        ),
        ControlField(
            key="temperature",
            label="Temperature sinh mẫu LLM",
            field_type=OptionType.NUMBER_RANGE,
            value=0.30,
            min_val=0.0,
            max_val=1.0,
            step=0.05,
            description="Độ ngẫu nhiên và sáng tạo của model trong lời thoại.",
            recommendation="0.25 - 0.35 tạo lời thoại tiếng Việt tự nhiên và bám sát ngữ nghĩa.",
            impact="Càng cao thì văn phong càng phong phú nhưng dễ tăng rủi ro lệch nghĩa.",
        ),
        ControlField(
            key="concise_tags",
            label="Rút gọn nhãn thoại (--concise)",
            field_type=OptionType.BOOLEAN,
            value=False,
            description="Bỏ lặp lại quan hệ từng dòng khi trong phân cảnh đã có khung TransGraph tổng.",
            recommendation="Bật khi muốn tiết kiệm 25% token context cho game có nhiều nhân vật phụ.",
            impact="Tiết kiệm token prompt, tăng tốc độ sinh.",
        ),
        ControlField(
            key="auto_apply",
            label="Tự động ghi vào game sau dịch",
            field_type=OptionType.BOOLEAN,
            value=False,
            description="Sau khi dịch xong toàn bộ các dòng, tự động apply bản dịch vào tệp kịch bản game.",
            recommendation="Tắt nếu muốn review bản dịch trước qua TSV hoặc Dashboard.",
            impact="Tiện lợi cho quy trình dịch hoàn toàn tự động.",
        ),
        ControlField(
            key="setup_font",
            label="Thiết lập phông chữ tiếng Việt",
            field_type=OptionType.BOOLEAN,
            value=True,
            description="Tự động cấu hình phông chữ tiếng Việt (Be Vietnam Pro / Nunito) trong Ren'Py.",
            recommendation="BẬT để không bao giờ bị lỗi hiển thị ô vuông hoặc mất dấu trong game.",
            impact="Sinh tệp cấu hình renslate_setup.rpy vào thư mục game/.",
        ),
    ]


def render_control_board(
    state: ControlBoardState,
    project_name: str = "",
    progress_info: str = "",
    width: int = 80,
    plain: bool = False,
) -> str:
    """Render interactive dual-panel visual control board."""
    from renslate.ui.studio import wrap_text_to_lines

    w = max(60, width)
    use_color = (not plain) and IS_TTY

    c_cyan = C_CYAN if use_color else ""
    c_bold = C_BOLD if use_color else ""
    c_dim = C_DIM if use_color else ""
    c_sky = C_SKY if use_color else ""
    c_emerald = C_EMERALD if use_color else ""
    c_yellow = C_YELLOW if use_color else ""
    c_slate = C_SLATE if use_color else ""
    c_reset = C_RESET if use_color else ""

    box_tl = BOX_ROUND_TL
    box_tr = BOX_ROUND_TR
    box_bl = BOX_ROUND_BL
    box_br = BOX_ROUND_BR
    box_vert = BOX_LIGHT_VERT
    box_horiz = BOX_LIGHT_HORIZ

    lines: list[str] = []

    # 1. Header Box
    title_str = "RENSLATE VISUAL NOVEL LOCALIZATION ENGINE  ──  BẢNG ĐIỀU KHIỂN CẤU HÌNH"
    lines.append(f"{c_slate}{box_tl}{box_horiz * max(2, w - 2)}{box_tr}{c_reset}")
    t_pad = max(0, w - 4 - visible_len(title_str))
    lines.append(f"{c_slate}{box_vert}{c_reset} {c_bold}{c_sky}{title_str}{c_reset}{' ' * t_pad} {c_slate}{box_vert}{c_reset}")

    p_info = f"Dự án: [{project_name}]  │  Tiến độ: {progress_info}" if project_name else progress_info
    if p_info:
        p_pad = max(0, w - 4 - visible_len(p_info))
        lines.append(f"{c_slate}{box_vert}{c_reset} {c_dim}{p_info}{c_reset}{' ' * p_pad} {c_slate}{box_vert}{c_reset}")
    lines.append(f"{c_slate}{box_bl}{box_horiz * max(2, w - 2)}{box_br}{c_reset}")
    lines.append("")

    # 2. Dual-Panel Body
    # Left: options (width ~46), Right: info card (width ~34)
    left_w = max(38, int(w * 0.54))
    right_w = max(24, w - left_w - 3)

    header_left = f"{c_bold}CÁC TÙY CHỌN & CỜ ĐIỀU KHIỂN:{c_reset}"
    header_right = f"{c_bold}THÔNG TIN & KHUYẾN NGHỊ:{c_reset}"

    hl_pad = max(0, left_w - visible_len("CÁC TÙY CHỌN & CỜ ĐIỀU KHIỂN:"))
    hr_pad = max(0, right_w - visible_len("THÔNG TIN & KHUYẾN NGHỊ:"))
    lines.append(f" {header_left}{' ' * hl_pad}   {header_right}{' ' * hr_pad}")
    lines.append(f" {c_slate}{box_horiz * left_w}{c_reset}   {c_slate}{box_horiz * right_w}{c_reset}")

    # Build Left Lines
    left_lines: list[str] = []
    for idx, f in enumerate(state.fields):
        is_active = idx == state.active_idx
        cursor = f"{c_cyan}►{c_reset}" if (is_active and use_color) else ("►" if is_active else " ")
        val_str = f.format_display_value(plain)

        # Label truncation if needed
        avail_lbl = max(10, left_w - visible_len(val_str) - 5)
        lbl = f.label
        if visible_len(lbl) > avail_lbl:
            lbl = lbl[:avail_lbl - 3] + "..."

        sp_pad = max(1, left_w - 2 - visible_len(lbl) - visible_len(val_str))
        line_item = f"{cursor} {c_bold if is_active else ''}{lbl}{c_reset}{' ' * sp_pad}{val_str}"
        left_lines.append(line_item)

    # Build Right Lines (Context Info Card)
    curr = state.current_field
    right_lines: list[str] = [
        f"{c_bold}{c_cyan}[{curr.label.upper()}]{c_reset}",
    ]
    if curr.description:
        desc_wrapped = wrap_text_to_lines(curr.description, max_width=right_w - 2)
        for dw in desc_wrapped:
            right_lines.append(f"{c_slate}- {dw}{c_reset}")

    if curr.recommendation:
        right_lines.append(f"{c_slate}{box_horiz * right_w}{c_reset}")
        rec_wrapped = wrap_text_to_lines(curr.recommendation, max_width=right_w - 2)
        for rw in rec_wrapped:
            right_lines.append(f"{c_emerald}{rw}{c_reset}")

    if curr.impact:
        imp_wrapped = wrap_text_to_lines(f"Tác động: {curr.impact}", max_width=right_w - 2)
        for iw in imp_wrapped:
            right_lines.append(f"{c_yellow}{iw}{c_reset}")

    # Align left and right panel rows
    total_rows = max(len(left_lines), len(right_lines))
    for r in range(total_rows):
        l_text = left_lines[r] if r < len(left_lines) else ""
        r_text = right_lines[r] if r < len(right_lines) else ""
        l_pad = max(0, left_w - visible_len(l_text))
        r_pad = max(0, right_w - visible_len(r_text))
        lines.append(f" {l_text}{' ' * l_pad}   {r_text}{' ' * r_pad}")

    # 3. Footer Action Bar
    lines.append(f" {c_slate}{box_horiz * (w - 2)}{c_reset}")
    action_str = "[ENTER] Bắt đầu Dịch     [S] Lưu Cấu hình     [R] Khôi phục Mặc định     [Q] Thoát"
    lines.append(f" {c_bold}{action_str}{c_reset}")
    lines.append(f" {c_dim}(Dùng phím ↑/↓ để chọn dòng, Space để bật/tắt, ←/→ để tăng giảm giá trị){c_reset}")

    return "\n".join(lines)


def run_control_board(
    project: Any,
    initial_config: dict[str, Any] | None = None,
    key_reader: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    """Run interactive control board loop."""
    import sys

    from renslate.ui.wizard.prompts import read_single_key

    reader = key_reader or read_single_key
    fields = create_default_control_fields()
    if initial_config:
        for f in fields:
            if f.key in initial_config:
                f.value = initial_config[f.key]

    state = ControlBoardState(fields=fields)
    proj_name = getattr(project, "root", None)
    name_str = proj_name.name if proj_name else "Renpy Project"

    if not sys.stdin.isatty():
        return state.to_config_dict(), "START"

    while True:
        rendered = render_control_board(state, project_name=name_str)
        # Clear screen and draw
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(rendered + "\n")
        sys.stdout.flush()

        key = reader()
        if key is None or key in ("ESCAPE", "q", "Q"):
            return None, "QUIT"
        if key == "UP":
            state.move_up()
        elif key == "DOWN":
            state.move_down()
        elif key in ("LEFT", "h"):
            state.current_field.step_prev()
        elif key in ("RIGHT", "l"):
            state.current_field.step_next()
        elif key in ("SPACE", " "):
            state.current_field.toggle()
        elif key == "ENTER":
            return state.to_config_dict(), "START"
        elif key in ("s", "S"):
            # Save configuration to project config
            cfg_dict = state.to_config_dict()
            return cfg_dict, "SAVE"
        elif key == "TAB":
            return state.to_config_dict(), "SWITCH_STUDIO"
