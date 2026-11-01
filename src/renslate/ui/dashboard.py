"""Real-time ANSI terminal dashboard for translation pipeline monitoring.

Provides interactive live-refreshing monitoring of local LLM inference, SQLite progress,
corpus breakdown, and recent bilingual dialogue activity stream.
"""

from __future__ import annotations

import contextlib
import datetime
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from renslate.core.config import AppConfig
from renslate.ui.progress import format_duration, format_speed, render_progress_bar
from renslate.ui.theme import (
    BOX_LIGHT_HORIZ,
    BOX_LIGHT_T_LEFT,
    BOX_LIGHT_T_RIGHT,
    BOX_LIGHT_VERT,
    BOX_ROUND_BL,
    BOX_ROUND_BR,
    BOX_ROUND_TL,
    BOX_ROUND_TR,
    C_AMBER,
    C_BOLD,
    C_CYAN,
    C_DIM,
    C_EMERALD,
    C_GREEN,
    C_INDIGO,
    C_RED,
    C_RESET,
    C_ROSE,
    C_SKY,
    C_SLATE,
    C_YELLOW,
    IS_TTY,
    char_width,
    visible_len,
)

if TYPE_CHECKING:
    from renslate.core.project import Project


def _truncate_visible(text: str, max_cells: int) -> str:
    """Safely truncate text to fit max_cells terminal width."""
    if visible_len(text) <= max_cells:
        return text
    limit = max(0, max_cells - 3)
    curr: list[str] = []
    width = 0
    for ch in text:
        w = char_width(ch)
        if width + w > limit:
            break
        curr.append(ch)
        width += w
    return "".join(curr) + "..."


def _box_line(
    content: str,
    width: int,
    border_left: str = BOX_LIGHT_VERT,
    border_right: str = BOX_LIGHT_VERT,
    c_border: str = "",
    c_reset: str = "",
) -> str:
    """Render a single interior box line with precise Unicode padding."""
    vis = visible_len(content)
    pad = max(0, width - 4 - vis)
    return (
        f"{c_border}{border_left}{c_reset} {content}{' ' * pad} {c_border}{border_right}{c_reset}"
    )


def _box_divider(
    width: int,
    char_left: str = BOX_LIGHT_T_RIGHT,
    char_mid: str = BOX_LIGHT_HORIZ,
    char_right: str = BOX_LIGHT_T_LEFT,
    c_border: str = "",
    c_reset: str = "",
) -> str:
    """Render a horizontal box partition line."""
    mid = char_mid * max(0, width - 2)
    return f"{c_border}{char_left}{mid}{char_right}{c_reset}"


@dataclass
class DashboardState:
    """State object for runner pipeline live updates."""

    model_name: str = "Unknown"
    server_status: str = "Online"
    server_url: str = field(default_factory=lambda: AppConfig.server_base_url)
    total_units: int = 0
    completed_units: int = 0
    pending_units: int = 0
    failed_units: int = 0
    speed_u_s: float = 0.0
    elapsed_s: float = 0.0
    eta_s: float = 0.0
    current_batch: int = 0
    total_batches: int = 0
    batch_latency_s: float = 0.0
    lookback_size: int = 0
    guard_status: str = "Bật"
    guard_flagged: int = 0
    recent_errors: list[str] = field(default_factory=list)
    character_count: int = 0
    relation_count: int = 0
    scene_count: int = 0
    recent_speaker: str = ""
    recent_target: str = ""


class LiveDashboard:
    """Manages multi-panel terminal display during autonomous translation."""

    def __init__(self, state: DashboardState | None = None) -> None:
        self.state = state or DashboardState()
        self._last_rendered_lines: int = 0
        self._started: bool = False

    def start(self) -> None:
        """Initialize live display and hide terminal cursor."""
        if IS_TTY and not self._started:
            try:
                sys.stdout.write("\033[?25l")
                sys.stdout.flush()
            except Exception:
                pass
            self._started = True

    def stop(self) -> None:
        """Clean up live display and restore terminal cursor."""
        if IS_TTY and self._started:
            try:
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
            except Exception:
                pass
        self._started = False
        self._last_rendered_lines = 0

    def update(self, width: int = 78, plain: bool = False) -> None:
        """Update live dashboard display in-place line-by-line without scrolling."""
        if plain or not IS_TTY:
            sys.stdout.write(self.render(width=width, plain=True) + "\n")
            sys.stdout.flush()
            return

        if not self._started:
            self.start()

        content = self.render(width=width, plain=False)
        lines = content.splitlines()

        if self._last_rendered_lines > 0:
            # Move cursor up to first line of previous frame and clear downward
            sys.stdout.write(f"\033[{self._last_rendered_lines}F\033[J")

        sys.stdout.write(content + "\n")
        sys.stdout.flush()
        self._last_rendered_lines = len(lines)

    def render(self, width: int = 76, plain: bool = False) -> str:
        """Render multi-panel dashboard snapshot with modern visual hierarchy."""
        w = max(50, width)
        st = self.state
        pct = (st.completed_units / st.total_units * 100.0) if st.total_units > 0 else 0.0

        use_color = (not plain) and IS_TTY
        c_cyan = C_CYAN if use_color else ""
        c_bold = C_BOLD if use_color else ""
        c_dim = C_DIM if use_color else ""
        c_green = C_GREEN if use_color else ""
        c_yellow = C_YELLOW if use_color else ""
        c_red = C_RED if use_color else ""
        c_reset = C_RESET if use_color else ""
        c_sky = C_SKY if use_color else ""
        c_emerald = C_EMERALD if use_color else ""
        c_indigo = C_INDIGO if use_color else ""
        c_slate = C_SLATE if use_color else ""

        c_border = c_slate
        lines: list[str] = []

        # 1. Top border (rounded)
        lines.append(
            _box_divider(w, BOX_ROUND_TL, BOX_LIGHT_HORIZ, BOX_ROUND_TR, c_border, c_reset)
        )

        # 2. Header
        guard_badge = (
            f"{c_emerald}● Bật{c_reset}"
            if st.guard_status != "Tắt"
            else f"{c_dim}○ Tắt{c_reset}"
        )
        if st.guard_flagged > 0 and st.guard_status != "Tắt":
            guard_badge += f" ({c_yellow}{st.guard_flagged} cờ{c_reset})"

        server_badge = (
            f"{c_emerald}● Online{c_reset}"
            if st.server_status == "Online"
            else f"{c_red}○ Offline{c_reset}"
        )

        hdr_left = f"{c_bold}{c_sky}Renslate Autonomous Engine{c_reset}"
        hdr_right = (
            f"Model: {c_cyan}{st.model_name}{c_reset}  •  "
            f"Server: {server_badge}  •  "
            f"Guard: {guard_badge}"
        )

        if visible_len(hdr_left) + visible_len(hdr_right) + 6 <= w:
            sp_pad = w - 4 - visible_len(hdr_left) - visible_len(hdr_right)
            header_line = f"{hdr_left}{' ' * max(1, sp_pad)}{hdr_right}"
            lines.append(_box_line(header_line, w, c_border=c_border, c_reset=c_reset))
        else:
            lines.append(_box_line(hdr_left, w, c_border=c_border, c_reset=c_reset))
            lines.append(_box_line(hdr_right, w, c_border=c_border, c_reset=c_reset))

        # 3. Divider
        lines.append(
            _box_divider(
                w, BOX_LIGHT_T_RIGHT, BOX_LIGHT_HORIZ, BOX_LIGHT_T_LEFT, c_border, c_reset
            )
        )

        # 4. Progress bar line
        bar_len = max(10, min(32, w - 28))
        bar = render_progress_bar(pct, width=bar_len)
        prog_text = f"Progress: {c_emerald}{bar}{c_reset}"
        lines.append(_box_line(prog_text, w, c_border=c_border, c_reset=c_reset))

        # 5. Units line: completed/total, pending, failed
        fail_str = (
            f"{c_red}{st.failed_units}{c_reset}"
            if st.failed_units > 0 and use_color
            else str(st.failed_units)
        )
        units_text = (
            f"Units: {c_green}{st.completed_units}{c_reset}/{st.total_units}  •  "
            f"Pending: {c_yellow}{st.pending_units}{c_reset}  •  "
            f"Failed: {fail_str}"
        )
        lines.append(_box_line(units_text, w, c_border=c_border, c_reset=c_reset))

        # 6. Performance line: Speed, Batch, Latency, Lookback
        latency_str = f"{st.batch_latency_s:.2f}s" if st.batch_latency_s > 0 else "--"
        perf_text = (
            f"Speed: {format_speed(st.speed_u_s)}  •  "
            f"Batch: #{st.current_batch} ({latency_str})  •  "
            f"Lookback: {st.lookback_size}"
        )
        lines.append(_box_line(perf_text, w, c_border=c_border, c_reset=c_reset))

        # 7. Time line: Elapsed, ETA
        time_text = (
            f"Elapsed: {format_duration(st.elapsed_s)}  •  "
            f"ETA: {format_duration(st.eta_s)}"
        )
        lines.append(_box_line(time_text, w, c_border=c_border, c_reset=c_reset))

        # 8. Script Context section (if populated)
        has_context = (
            st.character_count > 0 or st.relation_count > 0 or st.scene_count > 0
        )
        if has_context:
            lines.append(
                _box_divider(
                    w, BOX_LIGHT_T_RIGHT, BOX_LIGHT_HORIZ, BOX_LIGHT_T_LEFT, c_border, c_reset
                )
            )
            ctx_parts: list[str] = []
            if st.character_count > 0:
                ctx_parts.append(f"{c_indigo}{st.character_count:,} nhân vật{c_reset}")
            if st.relation_count > 0:
                ctx_parts.append(f"{c_indigo}{st.relation_count:,} quan hệ{c_reset}")
            if st.scene_count > 0:
                ctx_parts.append(f"{c_cyan}{st.scene_count:,} phân cảnh{c_reset}")
            ctx_line = "Ngữ cảnh: " + "  •  ".join(ctx_parts)
            lines.append(_box_line(ctx_line, w, c_border=c_border, c_reset=c_reset))

        # 9. Recent dialogue preview (if populated)
        if st.recent_speaker or st.recent_target:
            lines.append(
                _box_divider(
                    w, BOX_LIGHT_T_RIGHT, BOX_LIGHT_HORIZ, BOX_LIGHT_T_LEFT, c_border, c_reset
                )
            )
            spk = f"[{st.recent_speaker}]" if st.recent_speaker else "[Dẫn truyện]"
            avail_w = max(10, w - visible_len(spk) - 18)
            tgt_clean = _truncate_visible(st.recent_target.replace("\n", " "), avail_w)
            recent_line = f"Mới nhất: {c_indigo}{spk}{c_reset} {c_bold}\"{tgt_clean}\"{c_reset}"
            lines.append(_box_line(recent_line, w, c_border=c_border, c_reset=c_reset))

        # 10. Bottom border (rounded)
        lines.append(
            _box_divider(w, BOX_ROUND_BL, BOX_LIGHT_HORIZ, BOX_ROUND_BR, c_border, c_reset)
        )

        # 11. Status / exit hint
        if use_color:
            lines.append(f"  {c_dim}[Ctrl+C] Dừng an toàn (ACID WAL){c_reset}")

        return "\n".join(lines)

    def __enter__(self) -> LiveDashboard:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.stop()


TerminalDashboard = LiveDashboard


def render_banner(
    title: str,
    width: int = 80,
    items: Sequence[tuple[str, str]] | None = None,
    plain: bool = False,
) -> str:
    """Render standardized startup banner."""
    use_color = (not plain) and IS_TTY
    c_cyan = C_CYAN if use_color else ""
    c_bold = C_BOLD if use_color else ""
    c_reset = C_RESET if use_color else ""

    lines: list[str] = []
    divider = "=" * width
    lines.append(f"{c_bold}{c_cyan}{divider}{c_reset}")
    lines.append(f"{c_bold}{title.center(width)}{c_reset}")
    lines.append(f"{c_bold}{c_cyan}{divider}{c_reset}")
    if items:
        for label, val in items:
            lines.append(f"  • {label:<18}: {val}")
        lines.append(f"{c_bold}{c_cyan}{divider}{c_reset}")
    return "\n".join(lines)


def render_summary_card(
    state: DashboardState,
    width: int = 80,
    plain: bool = False,
) -> str:
    """Render final session summary card."""
    use_color = (not plain) and IS_TTY
    c_cyan = C_CYAN if use_color else ""
    c_bold = C_BOLD if use_color else ""
    c_green = C_GREEN if use_color else ""
    c_red = C_RED if use_color else ""
    c_reset = C_RESET if use_color else ""

    pct = (state.completed_units / state.total_units * 100.0) if state.total_units > 0 else 100.0
    bar_str = render_progress_bar(pct, width=20)
    speed_per_min = state.speed_u_s * 60.0

    lines: list[str] = []
    divider = "=" * width
    lines.append(f"{c_bold}{c_cyan}{divider}{c_reset}")
    lines.append(f"{c_bold}{'TỔNG KẾT PHIÊN DỊCH RENSLATE'.center(width)}{c_reset}")
    lines.append(f"{c_bold}{c_cyan}{divider}{c_reset}")
    lines.append(f"  • Trạng thái       : {c_green}Hoàn tất phiên dịch{c_reset}")
    lines.append(f"  • Thời gian chạy   : {format_duration(state.elapsed_s)}")
    lines.append(
        f"  • Dịch phiên này   : {c_green}{state.completed_units:,}{c_reset} dòng thành công"
    )
    fail_c = c_red if state.failed_units > 0 else c_reset
    lines.append(f"  • Thất bại         : {fail_c}{state.failed_units:,}{c_reset} dòng")
    lines.append(
        f"  • Tốc độ trung bình: {state.speed_u_s:.2f} dòng/s (~{speed_per_min:.0f} dòng/phút)"
    )
    lines.append(
        f"  • Tiến độ toàn game: {c_cyan}{bar_str}{c_reset} "
        f"({state.completed_units:,} / {state.total_units:,} dòng)"
    )
    lines.append(f"  • Dòng còn lại     : {state.pending_units:,} dòng đang chờ")
    guard_txt = (
        f"{c_green}{state.guard_status}{c_reset}"
        if state.guard_status != "Tắt"
        else f"{c_reset}Tắt{c_reset}"
    )
    if state.guard_flagged > 0:
        guard_txt += f" ({c_red}{state.guard_flagged:,} lỗi được ngăn chặn & sửa tự động{c_reset})"
    lines.append(f"  • Kiểm soát ảo giác: {guard_txt}")
    db_msg = "Đã lưu an toàn (ACID SQLite) — Không mất mát dữ liệu."
    lines.append(f"  • Cơ sở dữ liệu    : {c_green}{db_msg}{c_reset}")
    lines.append(f"{c_bold}{c_cyan}{divider}{c_reset}")
    return "\n".join(lines)


def render_batch_card(
    batch_num: int,
    units_count: int,
    duration_s: float,
    speed: float,
    failed: int = 0,
    plain: bool = False,
) -> str:
    """Render single batch completion summary card."""
    use_color = (not plain) and IS_TTY
    c_red = C_RED if use_color else ""
    c_reset = C_RESET if use_color else ""

    fail_badge = f" | {c_red}{failed} lỗi{c_reset}" if failed > 0 else ""
    return (
        f"Lô #{batch_num} hoàn tất (+{units_count} dòng trong "
        f"{duration_s:.1f}s • {speed:.2f} dòng/s{fail_badge})"
    )


# ---------------------------------------------------------------------------
# Real-Time Interactive CLI Dashboard Subsystem
# ---------------------------------------------------------------------------


@dataclass
class DashboardSnapshot:
    """Comprehensive real-time snapshot of the localization project."""

    project_name: str
    target_lang: str
    model_name: str
    server_status: str
    server_url: str
    server_ping_ms: float
    total_units: int
    translated_units: int
    qa_flagged_units: int
    pending_units: int
    failed_units: int
    percent_complete: float
    total_slots: int
    kind_counts: dict[str, int]
    character_count: int
    glossary_count: int
    tm_count: int
    top_speakers: list[tuple[str, int]]
    recent_units: list[dict[str, Any]]
    updated_at: str
    scene_count: int = 0
    rel_count: int = 0
    audit_issues_count: int = 0


def _ping_server(
    base_url: str, default_model: str, timeout_s: float = 0.4
) -> tuple[bool, float, str]:
    """Test server availability and compute round-trip latency."""
    from renslate.ui.runner import check_server_health, detect_server_model

    t0 = time.perf_counter()
    online = check_server_health(base_url, timeout_s=timeout_s)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    model = (
        detect_server_model(base_url, default_model, timeout_s=timeout_s)
        if online
        else default_model
    )
    return online, latency_ms, model


def query_dashboard_snapshot(
    project: Project, filter_mode: str = "all", limit: int = 5
) -> DashboardSnapshot:
    """Fetch live metrics from SQLite database and local inference endpoint."""
    base_url = project.config.llm.base_url
    online, latency_ms, model_name = _ping_server(base_url, project.config.llm.model, timeout_s=0.4)
    server_status = "Online" if online else "Offline"

    with project.open_db() as db:
        cur = db.execute("SELECT status, COUNT(*) FROM units GROUP BY status")
        counts = {str(r[0]): int(r[1]) for r in cur.fetchall()}
        total_units = sum(counts.values())
        translated = counts.get("translated", 0) + counts.get("approved", 0)
        qa_flagged = counts.get("qa_flagged", 0)
        pending = counts.get("pending", 0)
        failed = counts.get("failed", 0)
        pct = (translated + qa_flagged) / total_units * 100.0 if total_units > 0 else 100.0

        kind_cur = db.execute("SELECT kind, COUNT(*) FROM units GROUP BY kind")
        kind_counts = {str(r[0]): int(r[1]) for r in kind_cur.fetchall()}

        slots = db.count_slots()
        char_row = db.execute("SELECT COUNT(*) FROM characters").fetchone()
        char_count = char_row[0] if char_row else 0
        gloss_row = db.execute("SELECT COUNT(*) FROM glossary").fetchone()
        glossary_count = gloss_row[0] if gloss_row else 0
        tm_row = db.execute("SELECT COUNT(*) FROM tm").fetchone()
        tm_count = tm_row[0] if tm_row else 0
        scene_row = db.execute("SELECT COUNT(*) FROM scenes").fetchone()
        scene_count = scene_row[0] if scene_row else 0
        try:
            audit_row = db.execute(
                "SELECT COUNT(*) FROM qa_audit_log WHERE verdict = 'FAIL'"
            ).fetchone()
            audit_issues_count = audit_row[0] if audit_row else 0
        except Exception:
            audit_issues_count = 0
        try:
            rel_row = db.execute("SELECT COUNT(*) FROM relationships").fetchone()
            rel_count = rel_row[0] if rel_row else 0
        except Exception:
            rel_count = 0

        top_chars_cur = db.execute(
            "SELECT speaker, COUNT(*) as cnt FROM units "
            "WHERE speaker IS NOT NULL AND speaker != '' "
            "GROUP BY speaker ORDER BY cnt DESC LIMIT 6"
        )
        top_speakers = [(str(r[0]), int(r[1])) for r in top_chars_cur.fetchall()]

        where_parts: list[str] = ["target IS NOT NULL AND target != ''"]
        if filter_mode == "dialogue":
            where_parts.append("kind = 'dialogue'")
        elif filter_mode == "flagged":
            where_parts.append("status = 'qa_flagged'")

        where_sql = "WHERE " + " AND ".join(where_parts)
        units_cur = db.execute(
            f"SELECT uid, seq, speaker, source, target, status FROM units "
            f"{where_sql} ORDER BY seq DESC LIMIT ?",
            (limit,),
        )
        recent_units = [
            {
                "uid": str(r[0]),
                "seq": int(r[1]),
                "speaker": str(r[2] or ""),
                "source": str(r[3]),
                "target": str(r[4] or ""),
                "status": str(r[5]),
            }
            for r in units_cur.fetchall()
        ]

    now_str = datetime.datetime.now().strftime("%H:%M:%S")

    return DashboardSnapshot(
        project_name=project.root.name,
        target_lang=project.config.target_lang,
        model_name=model_name,
        server_status=server_status,
        server_url=base_url,
        server_ping_ms=latency_ms,
        total_units=total_units,
        translated_units=translated,
        qa_flagged_units=qa_flagged,
        pending_units=pending,
        failed_units=failed,
        percent_complete=round(pct, 1),
        total_slots=slots,
        kind_counts=kind_counts,
        character_count=char_count,
        glossary_count=glossary_count,
        tm_count=tm_count,
        top_speakers=top_speakers,
        recent_units=recent_units,
        updated_at=now_str,
        scene_count=scene_count,
        rel_count=rel_count,
        audit_issues_count=audit_issues_count,
    )


def render_dashboard_screen(
    snapshot: DashboardSnapshot,
    width: int = 80,
    plain: bool = False,
    filter_mode: str = "all",
    interval: float = 2.0,
) -> str:
    """Render full-frame ANSI dashboard with rounded boxes and bilingual dialogue stream."""
    w = max(76, width)
    use_color = (not plain) and IS_TTY

    c_border = C_SLATE if use_color else ""
    c_bold = C_BOLD if use_color else ""
    c_dim = C_DIM if use_color else ""
    c_reset = C_RESET if use_color else ""
    c_sky = C_SKY if use_color else ""
    c_cyan = C_CYAN if use_color else ""
    c_emerald = C_EMERALD if use_color else ""
    c_amber = C_AMBER if use_color else ""
    c_rose = C_ROSE if use_color else ""
    c_indigo = C_INDIGO if use_color else ""

    lines: list[str] = []

    # 1. Top border
    lines.append(_box_divider(w, BOX_ROUND_TL, BOX_LIGHT_HORIZ, BOX_ROUND_TR, c_border, c_reset))

    # 2. Header
    title = (
        f"{c_bold}{c_sky}RENSLATE LOCALIZATION STUDIO{c_reset}  •  "
        f"{c_dim}REALTIME LIVE MONITOR{c_reset}"
    )
    lines.append(_box_line(title, w, c_border=c_border, c_reset=c_reset))

    sub1 = (
        f"Dự án: {c_bold}{snapshot.project_name}{c_reset}  •  "
        f"Đích: {c_emerald}{snapshot.target_lang.upper()}{c_reset}"
    )
    lines.append(_box_line(sub1, w, c_border=c_border, c_reset=c_reset))

    status_color = c_emerald if snapshot.server_status == "Online" else c_rose
    status_icon = "●" if snapshot.server_status == "Online" else "○"
    latency_info = (
        f" ({snapshot.server_ping_ms:.0f}ms)" if snapshot.server_status == "Online" else ""
    )
    sub2 = (
        f"Model : {c_cyan}{_truncate_visible(snapshot.model_name, 38)}{c_reset}  •  "
        f"LLM: {status_color}{status_icon} {snapshot.server_status}{latency_info}{c_reset}"
    )
    lines.append(_box_line(sub2, w, c_border=c_border, c_reset=c_reset))

    # 3. Progress Panel
    lines.append(
        _box_divider(w, BOX_LIGHT_T_RIGHT, BOX_LIGHT_HORIZ, BOX_LIGHT_T_LEFT, c_border, c_reset)
    )
    lines.append(
        _box_line(f"{c_bold}TIẾN ĐỘ TOÀN DỰ ÁN{c_reset}", w, c_border=c_border, c_reset=c_reset)
    )

    bar_len = max(16, w - 34)
    bar = render_progress_bar(snapshot.percent_complete, width=bar_len)
    total_done = snapshot.translated_units + snapshot.qa_flagged_units
    prog_line = f"{c_emerald}{bar}{c_reset}  ({total_done:,} / {snapshot.total_units:,} dòng)"
    lines.append(_box_line(prog_line, w, c_border=c_border, c_reset=c_reset))

    counts_line = (
        f"Đã dịch: {c_emerald}{snapshot.translated_units:,}{c_reset}  •  "
        f"Flagged: {c_amber}{snapshot.qa_flagged_units:,}{c_reset}  •  "
        f"Chờ: {c_dim}{snapshot.pending_units:,}{c_reset}  •  "
        f"Lỗi: {c_rose}{snapshot.failed_units:,}{c_reset}  •  "
        f"Slots: {snapshot.total_slots:,}"
    )
    lines.append(_box_line(counts_line, w, c_border=c_border, c_reset=c_reset))

    # 4. Corpus & Lore Breakdown
    lines.append(
        _box_divider(w, BOX_LIGHT_T_RIGHT, BOX_LIGHT_HORIZ, BOX_LIGHT_T_LEFT, c_border, c_reset)
    )
    lines.append(
        _box_line(
            f"{c_bold}PHÂN BỔ KỊCH BẢN & CƠ SỞ TRI THỨC{c_reset}",
            w,
            c_border=c_border,
            c_reset=c_reset,
        )
    )

    dlg_count = snapshot.kind_counts.get("dialogue", 0)
    str_count = snapshot.kind_counts.get("string", 0)
    ui_count = snapshot.kind_counts.get("ui", 0)
    corpus_line = (
        f"Hội thoại: {c_cyan}{dlg_count:,}{c_reset} dòng  •  "
        f"Chuỗi: {str_count:,} chuỗi  •  "
        f"Giao diện: {ui_count:,} mục"
    )
    lines.append(_box_line(corpus_line, w, c_border=c_border, c_reset=c_reset))

    speakers_text = ", ".join(f"{name} ({cnt:,})" for name, cnt in snapshot.top_speakers[:4])
    if not speakers_text:
        speakers_text = "(Chưa phân tích)"
    lore_line = f"Nhân vật chính : {c_indigo}{_truncate_visible(speakers_text, w - 24)}{c_reset}"
    lines.append(_box_line(lore_line, w, c_border=c_border, c_reset=c_reset))

    audit_part = (
        f"  •  Cờ ảo giác: {c_amber}{snapshot.audit_issues_count:,}{c_reset}"
        if snapshot.audit_issues_count > 0
        else ""
    )
    kb_line = (
        f"Phân cảnh: {c_cyan}{snapshot.scene_count:,}{c_reset} cảnh  •  "
        f"Quan hệ: {c_indigo}{snapshot.rel_count:,}{c_reset} cặp  •  "
        f"Bộ nhớ TM: {snapshot.tm_count:,} mẫu{audit_part}"
    )
    lines.append(_box_line(kb_line, w, c_border=c_border, c_reset=c_reset))

    # 5. Recent Dialogue Stream
    lines.append(
        _box_divider(w, BOX_LIGHT_T_RIGHT, BOX_LIGHT_HORIZ, BOX_LIGHT_T_LEFT, c_border, c_reset)
    )
    filter_label = filter_mode.upper()
    lines.append(
        _box_line(
            f"{c_bold}DÒNG THOẠI GẦN ĐÂY{c_reset}  [{c_sky}BỘ LỌC: {filter_label}{c_reset}]",
            w,
            c_border=c_border,
            c_reset=c_reset,
        )
    )

    if not snapshot.recent_units:
        lines.append(
            _box_line(
                f"{c_dim}(Không có dòng thoại phù hợp với bộ lọc hiện tại){c_reset}",
                w,
                c_border=c_border,
                c_reset=c_reset,
            )
        )
    else:
        max_text_w = w - 10
        for u in snapshot.recent_units:
            status_tag = u["status"]
            if status_tag in ("translated", "approved"):
                badge = f"{c_emerald}[{status_tag.upper()}]{c_reset}"
            elif status_tag == "qa_flagged":
                badge = f"{c_amber}[QA_FLAG]{c_reset}"
            else:
                badge = f"{c_dim}[{status_tag.upper()}]{c_reset}"

            speaker_tag = f"[{u['speaker']}]" if u["speaker"] else "[Dẫn truyện]"
            head = (
                f"{c_bold}#{u['seq']}{c_reset} • {u['uid']}  "
                f"{c_indigo}{speaker_tag}{c_reset}  {badge}"
            )
            lines.append(_box_line(head, w, c_border=c_border, c_reset=c_reset))

            en_txt = _truncate_visible(u["source"].replace("\n", " "), max_text_w)
            vi_txt = _truncate_visible(u["target"].replace("\n", " "), max_text_w)
            lines.append(
                _box_line(f"  {c_dim}EN:{c_reset} {en_txt}", w, c_border=c_border, c_reset=c_reset)
            )
            lines.append(
                _box_line(
                    f"  {c_emerald}VI:{c_reset} {c_bold}{vi_txt}{c_reset}",
                    w,
                    c_border=c_border,
                    c_reset=c_reset,
                )
            )

    # 6. Bottom Border
    lines.append(_box_divider(w, BOX_ROUND_BL, BOX_LIGHT_HORIZ, BOX_ROUND_BR, c_border, c_reset))

    # 7. Interactive Controls & Status
    controls = (
        f"  {c_bold}[q]{c_reset} Thoát   "
        f"{c_bold}[r]{c_reset} Tải lại   "
        f"{c_bold}[1]{c_reset} Tất cả   "
        f"{c_bold}[2]{c_reset} Kịch bản   "
        f"{c_bold}[3]{c_reset} QA Flagged"
    )
    lines.append(controls)
    status_footer = (
        f"  {c_dim}Cập nhật lúc: {snapshot.updated_at} "
        f"(Tự động làm mới sau {interval:.0f}s){c_reset}"
    )
    lines.append(status_footer)

    return "\n".join(lines)


def _read_key_timeout(timeout_s: float) -> str | None:
    """Non-blocking POSIX stdin read with timeout."""
    if not sys.stdin.isatty():
        time.sleep(timeout_s)
        return None
    try:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout_s)
            if rlist:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        time.sleep(timeout_s)
        return None


def run_live_dashboard(
    project: Project,
    interval: float = 2.0,
    once: bool = False,
    plain: bool = False,
    limit: int = 5,
    initial_filter: str = "all",
) -> int:
    """Launch interactive real-time terminal dashboard with auto-refresh and hotkeys."""
    cols = shutil.get_terminal_size((80, 24)).columns
    target_width = min(100, max(76, cols))

    if once or (not sys.stdout.isatty()):
        snapshot = query_dashboard_snapshot(project, filter_mode=initial_filter, limit=limit)
        screen = render_dashboard_screen(
            snapshot,
            width=target_width,
            plain=plain,
            filter_mode=initial_filter,
            interval=interval,
        )
        sys.stdout.write(screen + "\n")
        sys.stdout.flush()
        return 0

    filter_mode = initial_filter
    use_plain = plain or (not IS_TTY)

    # Enter alternate screen buffer & hide cursor
    sys.stdout.write("\033[?1049h\033[?25l")
    sys.stdout.flush()

    try:
        while True:
            current_cols = shutil.get_terminal_size((80, 24)).columns
            target_width = min(100, max(76, current_cols))

            snapshot = query_dashboard_snapshot(project, filter_mode=filter_mode, limit=limit)
            screen = render_dashboard_screen(
                snapshot,
                width=target_width,
                plain=use_plain,
                filter_mode=filter_mode,
                interval=interval,
            )

            # Draw from home position without screen flashing
            sys.stdout.write("\033[H" + screen + "\033[J")
            sys.stdout.flush()

            key = _read_key_timeout(interval)
            if key is None:
                continue

            key_lower = key.lower()
            if key_lower in ("q", "\x03", "\x1b"):
                break
            elif key_lower == "r":
                continue
            elif key == "1":
                filter_mode = "all"
            elif key == "2":
                filter_mode = "dialogue"
            elif key == "3":
                filter_mode = "flagged"

    except KeyboardInterrupt:
        pass
    finally:
        # Restore cursor and standard screen buffer
        sys.stdout.write("\033[?25h\033[?1049l\n")
        sys.stdout.flush()

    return 0
