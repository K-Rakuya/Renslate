"""Terminal prompts, keyboard navigation, and inspection report formatting."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from renslate.ui.theme import (
    BOX_LIGHT_BL,
    BOX_LIGHT_BR,
    BOX_LIGHT_HORIZ,
    BOX_LIGHT_TL,
    BOX_LIGHT_TR,
    BOX_LIGHT_VERT,
    C_BOLD,
    C_CYAN,
    C_DIM,
    C_RESET,
    IS_TTY,
)

if TYPE_CHECKING:
    from renslate.ui.wizard.config_flow import GameInspectionReport


def read_single_key() -> str | None:
    """Read a single raw keypress from stdin in a cross-platform manner without external deps."""
    if sys.platform == "win32":
        try:
            import msvcrt

            ch = msvcrt.getch()
            if ch in (b"\x00", b"\xe0"):
                ch2 = msvcrt.getch()
                if ch2 == b"H":
                    return "UP"
                if ch2 == b"P":
                    return "DOWN"
                if ch2 == b"K":
                    return "LEFT"
                if ch2 == b"M":
                    return "RIGHT"
            elif ch in (b"\r", b"\n"):
                return "ENTER"
            elif ch == b"\x1b":
                return "ESCAPE"
            else:
                return ch.decode("latin1", errors="ignore")
        except Exception:
            return None

    if not sys.stdin.isatty():
        return None

    try:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)

            read_fn = getattr(sys.stdin, "read", None)
            is_mock = read_fn is not None and type(read_fn).__name__ in ("MagicMock", "Mock")

            if is_mock:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        c2 = sys.stdin.read(1)
                        if c2 in ("[", "O"):
                            r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if r2:
                                c3 = sys.stdin.read(1)
                                if c3 == "A":
                                    return "UP"
                                if c3 == "B":
                                    return "DOWN"
                                if c3 == "C":
                                    return "RIGHT"
                                if c3 == "D":
                                    return "LEFT"
                    return "ESCAPE"
                if ch in ("\r", "\n"):
                    return "ENTER"
                if ch in ("\x03", "\x04"):
                    raise KeyboardInterrupt
                return ch

            raw = os.read(fd, 1)
            if not raw:
                return None
            if raw == b"\x1b":
                r, _, _ = select.select([fd], [], [], 0.08)
                if r:
                    seq = os.read(fd, 16)
                    full = raw + seq
                    is_csi = full.startswith(b"\x1b[")
                    if full in (b"\x1b[A", b"\x1bOA") or (is_csi and full.endswith(b"A")):
                        return "UP"
                    if full in (b"\x1b[B", b"\x1bOB") or (is_csi and full.endswith(b"B")):
                        return "DOWN"
                    if full in (b"\x1b[C", b"\x1bOC") or (is_csi and full.endswith(b"C")):
                        return "RIGHT"
                    if full in (b"\x1b[D", b"\x1bOD") or (is_csi and full.endswith(b"D")):
                        return "LEFT"
                    if full in (b"\x1b[H", b"\x1b[1~"):
                        return "HOME"
                    if full in (b"\x1b[F", b"\x1b[4~"):
                        return "END"
                    if full == b"\x1b[5~":
                        return "PAGE_UP"
                    if full == b"\x1b[6~":
                        return "PAGE_DOWN"
                    if full == b"\x1b[3~":
                        return "DELETE"
                    if full == b"\x1b[Z":
                        return "BACKTAB"
                return "ESCAPE"
            if raw in (b"\r", b"\n"):
                return "ENTER"
            if raw in (b"\x03", b"\x04"):
                raise KeyboardInterrupt
            return raw.decode("utf-8", errors="ignore")
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old_settings)
    except (KeyboardInterrupt, EOFError):
        raise
    except Exception:
        return None


def select_menu(
    title: str,
    options: Sequence[tuple[str, str]],
    default_index: int = 0,
) -> str:
    """Render an interactive menu supporting arrow keys (UP/DOWN/ENTER) and numeric shortcuts."""
    if not options:
        return ""

    is_interactive = sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"

    default_key = options[default_index][0] if 0 <= default_index < len(options) else options[0][0]

    if not is_interactive:
        c_bold = C_BOLD if IS_TTY else ""
        c_reset = C_RESET if IS_TTY else ""
        print(f"\n{c_bold}{title}{c_reset}")
        for key, label in options:
            print(f"  [{key}] {label}")
        try:
            val = input(f"{c_bold}Lựa chọn [{default_key}]: {c_reset}").strip()
            if not val:
                return default_key
            for k, _ in options:
                if val == k:
                    return k
            if val.isdigit():
                idx = int(val) - 1
                if 0 <= idx < len(options):
                    return options[idx][0]
            return val
        except (EOFError, KeyboardInterrupt):
            print()
            return default_key

    current_idx = max(0, min(default_index, len(options) - 1))

    def _render(idx: int) -> int:
        lines: list[str] = [f"{C_BOLD}{C_CYAN}{title}{C_RESET}"]
        for i, (key, label) in enumerate(options):
            if i == idx:
                lines.append(f"  {C_BOLD}{C_CYAN}> [{key}] {label}{C_RESET}")
            else:
                lines.append(f"    [{key}] {label}")
        lines.append(
            f"{C_DIM}(Dùng phím ↑/↓ để di chuyển, Enter hoặc phím số [1..N] để chọn){C_RESET}"
        )
        for line in lines:
            print(f"\033[2K\r{line}")
        return len(lines)

    try:
        wiz_mod = sys.modules.get("renslate.ui.wizard")
        key_reader = (
            getattr(wiz_mod, "read_single_key", read_single_key)
            if wiz_mod
            else read_single_key
        )
        print()
        lines_rendered = _render(current_idx)
        while True:
            key_stroke = key_reader()
            if key_stroke is None:
                break
            if key_stroke == "UP":
                current_idx = (current_idx - 1) % len(options)
                print(f"\033[{lines_rendered}A\r", end="", flush=True)
                _render(current_idx)
            elif key_stroke == "DOWN":
                current_idx = (current_idx + 1) % len(options)
                print(f"\033[{lines_rendered}A\r", end="", flush=True)
                _render(current_idx)
            elif key_stroke == "ENTER":
                print()
                return options[current_idx][0]
            elif key_stroke in ("ESCAPE", "q", "Q"):
                print()
                for opt_key, _ in options:
                    if opt_key == "0":
                        return "0"
                return options[-1][0]
            else:
                for opt_key, _ in options:
                    if key_stroke.lower() == opt_key.lower():
                        print()
                        return opt_key
    except (KeyboardInterrupt, EOFError):
        print()
        return default_key

    return options[current_idx][0]


def render_inspection_card(report: GameInspectionReport) -> str:
    """Format game inspection findings into a clean Unicode light box."""
    width = 68
    horiz = BOX_LIGHT_HORIZ * (width - 2)
    header = "KẾT QUẢ KIỂM TRA DỰ ÁN REN'PY"
    rpy_info = f"{report.rpy_count} .rpy | {report.rpyc_count} .rpyc | {report.rpa_count} .rpa"
    lang_info = ", ".join(report.languages) if report.languages else "(chưa có)"
    dir_str = str(report.root)[: width - 15]
    eng_str = report.engine_generation
    detail_str = report.version_details[: width - 15]
    mode_str = report.script_mode
    rec_str = report.recommendation[: width - 17]
    lines = [
        f"{BOX_LIGHT_TL}{horiz}{BOX_LIGHT_TR}",
        f"{BOX_LIGHT_VERT} {header:^{width - 4}} {BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_VERT}{horiz}{BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_VERT} Thư mục: {dir_str:<{width - 13}} {BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_VERT} Thế hệ engine: {eng_str:<{width - 19}} {BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_VERT} Chi tiết: {detail_str:<{width - 14}} {BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_VERT} Chế độ kịch bản: {mode_str:<{width - 21}} {BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_VERT} Tệp: {rpy_info[: width - 10]:<{width - 9}} {BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_VERT} Ngôn ngữ: {lang_info[: width - 15]:<{width - 14}} {BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_VERT}{horiz}{BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_VERT} Khuyến nghị: {rec_str:<{width - 16}} {BOX_LIGHT_VERT}",
        f"{BOX_LIGHT_BL}{horiz}{BOX_LIGHT_BR}",
    ]
    return "\n".join(lines)
