"""Real-time full translation studio and in-place bilingual dialogue inspector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from renslate.core.models import LiveDialogueItem
from renslate.ui.theme import (
    BOX_LIGHT_BL,
    BOX_LIGHT_BR,
    BOX_LIGHT_HORIZ,
    BOX_LIGHT_TL,
    BOX_LIGHT_TR,
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
    char_width,
    visible_len,
)

__all__ = [
    "LiveDialogueItem",
    "StudioState",
    "commit_in_place_edit",
    "render_dialogue_card",
    "render_edit_modal",
    "render_studio_screen",
    "run_studio_session",
    "wrap_text_to_lines",
]


def wrap_text_to_lines(text: str, max_width: int) -> list[str]:
    """Word-wrap text to terminal width using exact Unicode cell widths without truncation."""
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = clean_text.split("\n")
    result_lines: list[str] = []

    limit = max(10, max_width)

    for p in paragraphs:
        if not p:
            result_lines.append("")
            continue

        words = p.split(" ")
        curr_line: list[str] = []
        curr_w = 0

        for w in words:
            word_w = sum(char_width(ch) for ch in w)
            space_w = 1 if curr_line else 0

            if curr_line and (curr_w + space_w + word_w > limit):
                result_lines.append(" ".join(curr_line))
                curr_line = [w]
                curr_w = word_w
            else:
                curr_line.append(w)
                curr_w += space_w + word_w

        if curr_line:
            result_lines.append(" ".join(curr_line))

    return result_lines if result_lines else [""]


@dataclass
class StudioState:
    """State machine governing live stream dialogue list, selection cursor, and editing."""

    units: list[LiveDialogueItem] = field(default_factory=list)
    selected_idx: int = 0
    scroll_offset: int = 0
    is_paused: bool = False
    filter_mode: str = "all"  # "all", "flagged", "reflected"
    is_editing: bool = False
    edit_buffer: str = ""
    status_message: str = ""

    def add_or_update_unit(self, item: LiveDialogueItem) -> None:
        """Add new unit or update existing one in-place."""
        for idx, u in enumerate(self.units):
            if u.uid == item.uid:
                self.units[idx] = item
                return
        self.units.append(item)

    def filtered_units(self) -> list[LiveDialogueItem]:
        """Return units matching active filter mode."""
        if self.filter_mode == "flagged":
            return [u for u in self.units if u.status == "qa_flagged" or u.qa_issues]
        if self.filter_mode == "reflected":
            return [u for u in self.units if u.origin == "reflected" or u.calque_score >= 0.35]
        return list(self.units)

    def selected_unit(self) -> LiveDialogueItem | None:
        """Return the currently focused dialogue unit."""
        flt = self.filtered_units()
        if not flt:
            return None
        idx = max(0, min(self.selected_idx, len(flt) - 1))
        return flt[idx]

    def move_up(self) -> None:
        """Move cursor up."""
        flt = self.filtered_units()
        if not flt:
            return
        self.selected_idx = max(0, self.selected_idx - 1)

    def move_down(self) -> None:
        """Move cursor down."""
        flt = self.filtered_units()
        if not flt:
            return
        self.selected_idx = min(len(flt) - 1, self.selected_idx + 1)

    def toggle_filter(self) -> str:
        """Cycle through filter modes: all -> flagged -> reflected -> all."""
        modes = ["all", "flagged", "reflected"]
        try:
            curr_pos = modes.index(self.filter_mode)
            self.filter_mode = modes[(curr_pos + 1) % len(modes)]
        except ValueError:
            self.filter_mode = "all"
        self.selected_idx = 0
        return self.filter_mode


def render_dialogue_card(
    item: LiveDialogueItem,
    width: int = 80,
    is_selected: bool = False,
    plain: bool = False,
) -> list[str]:
    """Render a full bilingual dialogue card with zero content truncation."""
    w = max(50, width)
    use_color = (not plain) and IS_TTY

    c_cyan = C_CYAN if use_color else ""
    c_bold = C_BOLD if use_color else ""
    c_dim = C_DIM if use_color else ""
    c_emerald = C_EMERALD if use_color else ""
    c_sky = C_SKY if use_color else ""
    c_yellow = C_YELLOW if use_color else ""
    c_reset = C_RESET if use_color else ""
    c_slate = C_SLATE if use_color else ""

    c_border = c_cyan if is_selected else c_slate
    cursor_mark = "►" if is_selected else " "

    # 1. Header Information
    spk_part = f"{item.speaker}" if item.speaker else "narrator"
    dyad_part = f"{spk_part} ──► {item.listener}" if item.listener else spk_part
    origin_part = "REFLECTED • System-2" if item.origin == "reflected" else item.origin.upper()
    calque_part = f"D_calque: {item.calque_score:.2f}"
    header_content = (
        f"[Seq: {item.seq}] {item.uid}  ──  {dyad_part}  [{origin_part} • {calque_part}]"
    )

    box_tl = BOX_ROUND_TL if is_selected else BOX_LIGHT_TL
    box_tr = BOX_ROUND_TR if is_selected else BOX_LIGHT_TR
    box_bl = BOX_ROUND_BL if is_selected else BOX_LIGHT_BL
    box_br = BOX_ROUND_BR if is_selected else BOX_LIGHT_BR
    box_vert = BOX_LIGHT_VERT
    box_horiz = BOX_LIGHT_HORIZ

    lines: list[str] = []

    # Card Top Border
    avail_top = w - 6 - visible_len(header_content)
    if avail_top > 2:
        top_line = (
            f"{cursor_mark}{c_border}{box_tl}{box_horiz} "
            f"{c_bold}{c_sky}{header_content}{c_reset} "
            f"{c_border}{box_horiz * max(1, avail_top)}{box_tr}{c_reset}"
        )
    else:
        top_line = f"{cursor_mark}{c_border}{box_tl}{box_horiz * max(2, w - 3)}{box_tr}{c_reset}"
    lines.append(top_line)

    interior_w = max(10, w - 6)

    # 2. ENG Source Section
    src_lines = wrap_text_to_lines(item.source, max_width=interior_w - 7)
    for idx, s_line in enumerate(src_lines):
        prefix = f"{c_dim}ENG:{c_reset} " if idx == 0 else "     "
        pad = max(0, interior_w - visible_len(prefix) - visible_len(s_line))
        line_str = (
            f" {c_border}{box_vert}{c_reset} {prefix}{s_line}{' ' * pad} {c_border}{box_vert}{c_reset}"
        )
        lines.append(line_str)

    # Divider between ENG and VIE
    lines.append(f" {c_border}{box_vert}{box_horiz * interior_w}{box_vert}{c_reset}")

    # 3. VIE Target Section
    tgt_lines = wrap_text_to_lines(item.target, max_width=interior_w - 7)
    for idx, t_line in enumerate(tgt_lines):
        prefix = f"{c_emerald}VIE:{c_reset} " if idx == 0 else "     "
        pad = max(0, interior_w - visible_len(prefix) - visible_len(t_line))
        line_str = (
            f" {c_border}{box_vert}{c_reset} {prefix}{c_bold}{t_line}{c_reset}{' ' * pad} "
            f"{c_border}{box_vert}{c_reset}"
        )
        lines.append(line_str)

    # 4. QA Issues (if any)
    if item.qa_issues:
        qa_str = f"{c_yellow}Cờ QA: {', '.join(item.qa_issues)}{c_reset}"
        pad = max(0, interior_w - visible_len(qa_str))
        lines.append(
            f" {c_border}{box_vert}{c_reset} {qa_str}{' ' * pad} {c_border}{box_vert}{c_reset}"
        )

    # Card Bottom Border
    bottom_line = f" {c_border}{box_bl}{box_horiz * interior_w}{box_br}{c_reset}"
    lines.append(bottom_line)

    return lines


def commit_in_place_edit(
    db: Any,
    unit_uid: str,
    new_target: str,
    qa_checker: Any = None,
) -> tuple[bool, str, list[Any]]:
    """Commit an edited translation directly to database and translation memory (TM)."""
    from renslate.core.models import Origin, Status
    from renslate.renpy.mask import reconcile_translation

    unit = db.get_unit(unit_uid)
    if unit is None:
        return False, f"Không tìm thấy dòng thoại có UID {unit_uid}", []

    cleaned, issues = reconcile_translation(unit.source, new_target)
    if issues:
        issue_names = [getattr(i, "code", str(i)) for i in issues]
        return False, f"Lỗi cú pháp placeholder/thẻ: {', '.join(issue_names)}", list(issues)

    with db.transaction():
        db.update_translation(
            uid=unit_uid,
            target=cleaned,
            origin=Origin.USER,
            status=Status.REVIEWED,
            qa_flags=(),
        )
        db.add_tm(src_norm=unit.source, target=cleaned, origin=Origin.USER)

    return True, "Đã cập nhật bản dịch vào cơ sở dữ liệu và TM thành công!", []


def render_edit_modal(
    item: LiveDialogueItem,
    current_input: str,
    width: int = 80,
    plain: bool = False,
) -> list[str]:
    """Render an interactive in-place editing modal."""
    w = max(50, width)
    use_color = (not plain) and IS_TTY

    c_cyan = C_CYAN if use_color else ""
    c_bold = C_BOLD if use_color else ""
    c_dim = C_DIM if use_color else ""
    c_emerald = C_EMERALD if use_color else ""
    c_slate = C_SLATE if use_color else ""
    c_reset = C_RESET if use_color else ""

    box_tl = BOX_ROUND_TL
    box_tr = BOX_ROUND_TR
    box_bl = BOX_ROUND_BL
    box_br = BOX_ROUND_BR
    box_vert = BOX_LIGHT_VERT
    box_horiz = BOX_LIGHT_HORIZ

    interior_w = max(10, w - 4)

    title = f"CHỈNH SỬA BẢN DỊCH TẠI CHỖ: {item.uid}"
    avail_title = interior_w - visible_len(title) - 2

    lines: list[str] = [
        f"{c_cyan}{box_tl}{box_horiz} {c_bold}{title}{c_reset}{c_cyan} {box_horiz * max(1, avail_title)}{box_tr}{c_reset}"
    ]

    # Source line
    src_lines = wrap_text_to_lines(f"Gốc: \"{item.source}\"", max_width=interior_w - 2)
    for s_line in src_lines:
        pad = max(0, interior_w - visible_len(s_line))
        lines.append(f"{c_cyan}{box_vert}{c_reset} {c_dim}{s_line}{c_reset}{' ' * pad} {c_cyan}{box_vert}{c_reset}")

    # Current line
    curr_lines = wrap_text_to_lines(f"Hiện tại: \"{item.target}\"", max_width=interior_w - 2)
    for c_line in curr_lines:
        pad = max(0, interior_w - visible_len(c_line))
        lines.append(f"{c_cyan}{box_vert}{c_reset} {c_slate}{c_line}{c_reset}{' ' * pad} {c_cyan}{box_vert}{c_reset}")

    # Divider
    lines.append(f"{c_cyan}{box_vert}{box_horiz * interior_w}{box_vert}{c_reset}")

    # Input line
    prompt_str = f"► {current_input}█"
    pad_in = max(0, interior_w - visible_len(prompt_str))
    lines.append(f"{c_cyan}{box_vert}{c_reset} {c_bold}{c_emerald}{prompt_str}{c_reset}{' ' * pad_in} {c_cyan}{box_vert}{c_reset}")

    # Bottom footer
    hint_str = "[ENTER] Lưu vào Database + TM    [ESC] Hủy bỏ"
    pad_h = max(0, interior_w - visible_len(hint_str))
    lines.append(f"{c_cyan}{box_vert}{c_reset} {c_dim}{hint_str}{c_reset}{' ' * pad_h} {c_cyan}{box_vert}{c_reset}")

    lines.append(f"{c_cyan}{box_bl}{box_horiz * interior_w}{box_br}{c_reset}")
    return lines


def render_studio_screen(
    state: StudioState,
    header_info: str = "",
    width: int = 80,
    height: int = 24,
    plain: bool = False,
) -> str:
    """Render full real-time bilingual studio interface."""
    w = max(60, width)
    use_color = (not plain) and IS_TTY

    c_bold = C_BOLD if use_color else ""
    c_dim = C_DIM if use_color else ""
    c_emerald = C_EMERALD if use_color else ""
    c_sky = C_SKY if use_color else ""
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

    # 1. Top Header
    title_str = "RENSLATE LIVE TRANSLATION STUDIO  ──  THEO DÕI & SỬA ĐỔI THỜI GIAN THỰC"
    lines.append(f"{c_slate}{box_tl}{box_horiz * max(2, w - 2)}{box_tr}{c_reset}")
    t_pad = max(0, w - 4 - visible_len(title_str))
    lines.append(
        f"{c_slate}{box_vert}{c_reset} {c_bold}{c_sky}{title_str}{c_reset}{' ' * t_pad} {c_slate}{box_vert}{c_reset}"
    )
    if header_info:
        h_pad = max(0, w - 4 - visible_len(header_info))
        lines.append(
            f"{c_slate}{box_vert}{c_reset} {c_emerald}{header_info}{c_reset}{' ' * h_pad} {c_slate}{box_vert}{c_reset}"
        )
    lines.append(f"{c_slate}{box_bl}{box_horiz * max(2, w - 2)}{box_br}{c_reset}")

    # 2. Sub-bar: Navigation and filter status
    flt_badge = f"{c_yellow}{state.filter_mode.upper()}{c_reset}"
    nav_str = (
        f"[TAB] Bảng Cờ Cấu hình    [F] Lọc: {flt_badge}    "
        f"[↑/↓ / PgUp/PgDn] Cuộn Dòng    [E] Sửa Dòng"
    )
    lines.append(f" {c_dim}{nav_str}{c_reset}")
    lines.append(f" {c_slate}{box_horiz * (w - 2)}{c_reset}")

    # 3. Content: Dialogue Cards
    flt_units = state.filtered_units()
    if not flt_units:
        empty_msg = "(Chưa có dòng thoại nào được dịch hoặc không khớp bộ lọc)"
        lines.append(f"\n   {c_dim}{empty_msg}{c_reset}\n")
    else:
        # Determine display window based on selected_idx and height
        sel_idx = max(0, min(state.selected_idx, len(flt_units) - 1))
        # Render visible cards
        # Show around 3-4 cards around sel_idx
        start_idx = max(0, sel_idx - 1)
        end_idx = min(len(flt_units), start_idx + 3)
        if end_idx - start_idx < 3 and start_idx > 0:
            start_idx = max(0, end_idx - 3)

        for i in range(start_idx, end_idx):
            unit_item = flt_units[i]
            is_active = (i == sel_idx)
            card_lines = render_dialogue_card(unit_item, width=w - 2, is_selected=is_active, plain=plain)
            lines.extend(card_lines)

    # 4. In-Place Edit Modal (if active)
    if state.is_editing:
        active_u = state.selected_unit()
        if active_u:
            lines.append("")
            modal_lines = render_edit_modal(active_u, current_input=state.edit_buffer, width=w - 4, plain=plain)
            lines.extend(modal_lines)

    # 5. Status / Feedback Message
    if state.status_message:
        lines.append(f" {c_emerald}✓ {state.status_message}{c_reset}")

    # 6. Bottom Action Bar
    lines.append(f" {c_slate}{box_horiz * (w - 2)}{c_reset}")
    pause_label = "Tiếp tục" if state.is_paused else "Tạm dừng"
    actions = f"[E] Sửa bản dịch    [R] Phản tư viết lại    [SPACE] {pause_label}    [Q] Thoát"
    lines.append(f" {c_bold}{actions}{c_reset}")

    return "\n".join(lines)


def run_studio_session(
    project: Any,
    state: StudioState | None = None,
    key_reader: Any = None,
    db: Any = None,
) -> tuple[StudioState, str]:
    """Run interactive studio viewer & in-place editor loop."""
    import sys

    from renslate.ui.wizard.prompts import read_single_key

    reader = key_reader or read_single_key
    st = state or StudioState()
    active_db = db or getattr(project, "open_db", None)

    if not sys.stdin.isatty():
        return st, "QUIT"

    while True:
        hdr = f"Đã nhận {len(st.units)} dòng thoại song ngữ"
        rendered = render_studio_screen(st, header_info=hdr)
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(rendered + "\n")
        sys.stdout.flush()

        key = reader()
        if key is None:
            return st, "QUIT"

        if st.is_editing:
            if key == "ENTER":
                # Commit edit
                cur_u = st.selected_unit()
                if cur_u and active_db:
                    success, msg, _ = commit_in_place_edit(active_db, cur_u.uid, st.edit_buffer)
                    if success:
                        cur_u.target = st.edit_buffer
                        cur_u.origin = "user"
                        st.status_message = "Đã lưu bản dịch mới vào SQLite & TM!"
                    else:
                        st.status_message = f"Lỗi: {msg}"
                st.is_editing = False
                st.edit_buffer = ""
            elif key in ("ESCAPE", "q"):
                st.is_editing = False
                st.edit_buffer = ""
                st.status_message = "Đã hủy chỉnh sửa."
            elif key in ("BACKSPACE", "\x7f", "\x08"):
                st.edit_buffer = st.edit_buffer[:-1]
            elif len(key) == 1 and key.isprintable():
                st.edit_buffer += key
            continue

        if key in ("q", "Q", "ESCAPE"):
            return st, "QUIT"
        elif key == "TAB":
            return st, "SWITCH_CONTROL_BOARD"
        elif key in ("UP", "k"):
            st.move_up()
        elif key in ("DOWN", "j"):
            st.move_down()
        elif key in ("f", "F"):
            mode = st.toggle_filter()
            st.status_message = f"Chế độ lọc: {mode.upper()}"
        elif key in ("SPACE", " "):
            st.is_paused = not st.is_paused
            st.status_message = "Đã tạm dừng nhận dòng mới." if st.is_paused else "Tiếp tục stream."
        elif key in ("e", "E"):
            cur_u = st.selected_unit()
            if cur_u:
                st.is_editing = True
                st.edit_buffer = cur_u.target
                st.status_message = f"Đang sửa dòng {cur_u.uid}..."
