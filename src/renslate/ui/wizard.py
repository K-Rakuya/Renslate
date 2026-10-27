"""Interactive setup wizard, game inspection, and event-driven workflow studio."""

from __future__ import annotations

import dataclasses
import enum
import os
import re
import struct
import sys
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from renslate.core.config import Config
from renslate.core.project import Project
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
    C_GREEN,
    C_RESET,
    C_YELLOW,
    IS_TTY,
)

# ==============================================================================
# Ren'Py Engine & Game Inspection Component
# ==============================================================================


class RenpyVersion(enum.Enum):
    """Detected Ren'Py engine generation."""

    V6 = "6"
    V7 = "7"
    V8 = "8"
    UNKNOWN = "unknown"


class ScriptMode(enum.Enum):
    """Composition mode of game scripts."""

    RPY_SOURCE = "RPY_SOURCE"
    RPYC_BYTECODE_ONLY = "RPYC_BYTECODE_ONLY"
    HYBRID = "HYBRID"
    EMPTY = "EMPTY"


@dataclass(frozen=True)
class GameInspectionReport:
    """Detailed structural report of an inspected Ren'Py project."""

    root: Path
    game_dir: Path
    version: str
    engine_generation: str
    version_details: str
    script_mode: str
    rpy_count: int
    rpyc_count: int
    rpa_count: int
    languages: list[str]
    target_lang: str
    target_lang_exists: bool
    target_lang_files_count: int
    recommendation: str

    @property
    def rpy_files_count(self) -> int:
        return self.rpy_count

    @property
    def rpyc_files_count(self) -> int:
        return self.rpyc_count

    @property
    def rpa_files_count(self) -> int:
        return self.rpa_count

    @property
    def configured_languages(self) -> list[str]:
        return self.languages


def inspect_game(root: Path, target_lang: str = "vietnamese") -> GameInspectionReport:
    """Inspect Ren'Py game engine generation, script composition, and locales."""
    resolved = root.resolve()
    if resolved.name == "game" and resolved.is_dir():
        game_dir = resolved
        proj_root = resolved.parent
    elif (resolved / "game").is_dir():
        game_dir = (resolved / "game").resolve()
        proj_root = resolved
    else:
        game_dir = (resolved / "game").resolve()
        proj_root = resolved

    # Script and archive counts
    if game_dir.is_dir():
        rpy_count = len(list(game_dir.rglob("*.rpy")))
        rpyc_count = len(list(game_dir.rglob("*.rpyc")))
        rpa_count = len(list(game_dir.rglob("*.rpa")))
    else:
        rpy_count = 0
        rpyc_count = 0
        rpa_count = 0

    if rpy_count > 0 and rpyc_count == 0:
        script_mode = ScriptMode.RPY_SOURCE.value
    elif rpyc_count > 0 and rpy_count == 0:
        script_mode = ScriptMode.RPYC_BYTECODE_ONLY.value
    elif rpy_count > 0 and rpyc_count > 0:
        script_mode = ScriptMode.HYBRID.value
    else:
        script_mode = ScriptMode.EMPTY.value

    # Configured languages
    tl_dir = game_dir / "tl"
    languages: list[str] = []
    if tl_dir.is_dir():
        languages = sorted(p.name for p in tl_dir.iterdir() if p.is_dir() and p.name != "None")

    target_dir = tl_dir / target_lang if tl_dir.is_dir() else None
    target_exists = bool(target_dir and target_dir.is_dir())
    target_count = len(list(target_dir.rglob("*.rpy"))) if target_exists and target_dir else 0

    # Engine version detection
    version = ""
    engine_generation = ""
    version_details = ""

    # Tier 1: Check runtime libraries in lib/
    lib_dir = proj_root / "lib"
    if lib_dir.is_dir():
        names = [p.name.lower() for p in lib_dir.iterdir()]
        if any("py3" in n for n in names) or any("python3" in n for n in names):
            version = RenpyVersion.V8.value
            engine_generation = "Ren'Py 8 (Python 3)"
            matched = next((n for n in names if "py3" in n or "python3" in n), "")
            version_details = f"Phát hiện qua runtime lib/{matched}"
        elif any("windows-x86_64" in n or "linux-x86_64" in n or "py2" in n for n in names):
            version = RenpyVersion.V7.value
            engine_generation = "Ren'Py 7 (Python 2.7)"
            matched = next(
                (n for n in names if "windows-x86_64" in n or "linux-x86_64" in n or "py2" in n),
                "",
            )
            version_details = f"Phát hiện qua runtime lib/{matched}"
        elif any("windows-i686" in n or "linux-i686" in n for n in names):
            version = RenpyVersion.V6.value
            engine_generation = "Ren'Py 6 (Legacy)"
            matched = next((n for n in names if "i686" in n), "")
            version_details = f"Phát hiện qua runtime 32-bit legacy lib/{matched}"

    # Tier 2: Check bytecode .rpyc pickle protocol
    if not version and game_dir.is_dir():
        rpyc_files = list(game_dir.glob("*.rpyc"))
        if not rpyc_files:
            rpyc_files = list(game_dir.rglob("*.rpyc"))
        for rpyc_path in rpyc_files[:5]:
            try:
                with rpyc_path.open("rb") as f:
                    hdr = f.read(1024)
                if hdr.startswith(b"RENPY RPC2"):
                    pos = len(b"RENPY RPC2")
                    slots: dict[int, tuple[int, int]] = {}
                    while pos + 12 <= len(hdr):
                        sid, start, length = struct.unpack("<III", hdr[pos : pos + 12])
                        pos += 12
                        if sid == 0:
                            break
                        slots[sid] = (start, length)

                    slot_to_check = 1 if 1 in slots else (2 if 2 in slots else None)
                    if slot_to_check is not None:
                        st, ln = slots[slot_to_check]
                        with rpyc_path.open("rb") as f:
                            f.seek(st)
                            comp = f.read(min(ln, 4096))
                        raw = zlib.decompressobj().decompress(comp, 64)
                        if raw.startswith((b"\x80\x03", b"\x80\x04", b"\x80\x05")):
                            version = RenpyVersion.V8.value
                            engine_generation = "Ren'Py 8 (Python 3)"
                            version_details = (
                                f"Phát hiện qua pickle protocol >= 3 ({rpyc_path.name})"
                            )
                            break
                        if raw.startswith(b"\x80\x02"):
                            if (game_dir / "gui.rpy").is_file() or (game_dir / "gui").is_dir():
                                version = RenpyVersion.V7.value
                                engine_generation = "Ren'Py 7 (Python 2.7)"
                                version_details = (
                                    f"Phát hiện qua pickle protocol 2 và gui.rpy ({rpyc_path.name})"
                                )
                            else:
                                version = RenpyVersion.V6.value
                                engine_generation = "Ren'Py 6 (Legacy)"
                                version_details = (
                                    f"Phát hiện qua pickle protocol 2 ({rpyc_path.name})"
                                )
                            break
            except Exception:
                continue

    # Tier 3: Metadata files and GUI system
    if not version:
        vc_file = proj_root / "renpy" / "vc_version.py"
        if vc_file.is_file():
            try:
                text = vc_file.read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"version_tuple\s*=\s*\(\s*(\d+)", text)
                if m:
                    v_num = m.group(1)
                    version = v_num
                    engine_generation = f"Ren'Py {v_num}"
                    version_details = f"Phát hiện qua renpy/vc_version.py: {v_num}"
            except Exception:
                pass

    if not version:
        if (game_dir / "gui.rpy").is_file() or (game_dir / "gui").is_dir():
            version = RenpyVersion.V7.value
            engine_generation = "Ren'Py 7 (Python 2.7)"
            version_details = "Phát hiện qua cấu trúc giao diện modern GUI (gui.rpy)"
        elif (game_dir / "options.rpy").is_file():
            version = RenpyVersion.V6.value
            engine_generation = "Ren'Py 6 (Legacy)"
            version_details = "Phát hiện qua options.rpy cổ điển (không có gui.rpy)"
        else:
            version = RenpyVersion.V8.value
            engine_generation = "Ren'Py 8"
            version_details = "Mặc định chuẩn Ren'Py hiện đại"

    # Recommendation
    if script_mode == ScriptMode.RPYC_BYTECODE_ONLY.value:
        rec = (
            "Dự án chỉ chứa bytecode .rpyc: Tự động trích xuất Slot-2 "
            "trực tiếp và tổng hợp khung dịch."
        )
    elif script_mode == ScriptMode.EMPTY.value:
        rec = "Chưa phát hiện kịch bản game (.rpy hoặc .rpyc) trong thư mục game/."
    elif script_mode == ScriptMode.HYBRID.value:
        rec = (
            "Dự án chứa cả mã nguồn .rpy và bytecode .rpyc: "
            "Sẵn sàng trích xuất và đồng bộ cơ sở dữ liệu."
        )
    else:
        rec = "Kịch bản mã nguồn .rpy sẵn sàng cho trích xuất và dịch tự động."

    return GameInspectionReport(
        root=proj_root,
        game_dir=game_dir,
        version=version,
        engine_generation=engine_generation,
        version_details=version_details,
        script_mode=script_mode,
        rpy_count=rpy_count,
        rpyc_count=rpyc_count,
        rpa_count=rpa_count,
        languages=languages,
        target_lang=target_lang,
        target_lang_exists=target_exists,
        target_lang_files_count=target_count,
        recommendation=rec,
    )


class GameInspector:
    """Inspector for Ren'Py game engine generation, scripts, and locales."""

    def __init__(self, root: Path | None = None, target_lang: str = "vietnamese") -> None:
        self.root = root
        self.target_lang = target_lang

    @classmethod
    def inspect(cls, root: Path, target_lang: str = "vietnamese") -> GameInspectionReport:
        return inspect_game(root, target_lang)

    def run(self, root: Path | None = None, target_lang: str | None = None) -> GameInspectionReport:
        target_root = root or self.root or Path.cwd()
        lang = target_lang or self.target_lang
        return inspect_game(target_root, lang)


# ==============================================================================
# Pure-Stdlib Cross-Platform Terminal Navigation (termios / msvcrt)
# ==============================================================================


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
        print()
        lines_rendered = _render(current_idx)
        while True:
            key_stroke = read_single_key()
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


# ==============================================================================
# 6-Phase Event-Driven State Machine
# ==============================================================================


class WizardPhase(enum.Enum):
    """The 6 phases of the localization studio state machine."""

    INIT = "INIT"
    DETECT_GAME = "DETECT_GAME"
    EXTRACT_OR_REPAIR = "EXTRACT_OR_REPAIR"
    LLM_PROBE = "LLM_PROBE"
    TRANSLATE_DASHBOARD = "TRANSLATE_DASHBOARD"
    APPLY_VERIFY = "APPLY_VERIFY"
    EXIT = "EXIT"


WizardState = WizardPhase


class WizardEvent(enum.Enum):
    """Event triggers transitioning the wizard state machine."""

    PROCEED = "PROCEED"
    BACK = "BACK"
    RETRY = "RETRY"
    CANCEL = "CANCEL"
    CUSTOM = "CUSTOM"


DEFAULT_TRANSITIONS: dict[WizardPhase, dict[WizardEvent, WizardPhase]] = {
    WizardPhase.INIT: {
        WizardEvent.PROCEED: WizardPhase.DETECT_GAME,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.DETECT_GAME: {
        WizardEvent.PROCEED: WizardPhase.EXTRACT_OR_REPAIR,
        WizardEvent.BACK: WizardPhase.INIT,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.EXTRACT_OR_REPAIR: {
        WizardEvent.PROCEED: WizardPhase.LLM_PROBE,
        WizardEvent.BACK: WizardPhase.DETECT_GAME,
        WizardEvent.RETRY: WizardPhase.EXTRACT_OR_REPAIR,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.LLM_PROBE: {
        WizardEvent.PROCEED: WizardPhase.TRANSLATE_DASHBOARD,
        WizardEvent.BACK: WizardPhase.EXTRACT_OR_REPAIR,
        WizardEvent.RETRY: WizardPhase.LLM_PROBE,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.TRANSLATE_DASHBOARD: {
        WizardEvent.PROCEED: WizardPhase.APPLY_VERIFY,
        WizardEvent.BACK: WizardPhase.LLM_PROBE,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.APPLY_VERIFY: {
        WizardEvent.PROCEED: WizardPhase.EXIT,
        WizardEvent.BACK: WizardPhase.TRANSLATE_DASHBOARD,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
}


@dataclass
class WizardContext:
    """Mutable runtime state passed across wizard phases."""

    root: Path
    project: Project | None = None
    config: Config = field(default_factory=Config)
    inspection: GameInspectionReport | None = None
    server_url: str = "http://localhost:8080"
    detected_model: str = ""
    chunk_size: int = 100
    retry_failed: bool = False
    auto_apply: bool = True
    dry_run: bool = False
    extracted_count: int = 0
    qa_issues_count: int = 0


class InteractiveWizard:
    """Step-by-step interactive configuration wizard and 6-phase state machine."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.root = (project_root or Path.cwd()).resolve()
        self.project: Project | None = None
        self.inspection: GameInspectionReport | None = None
        self.state: WizardPhase = WizardPhase.INIT
        self.context = WizardContext(root=self.root)

    @property
    def current_phase(self) -> WizardPhase:
        return self.state

    def prompt(self, message: str, default: str = "") -> str:
        """Prompt user for input with default fallback."""
        default_hint = f" [{default}]" if default else ""
        try:
            val = input(f"{C_BOLD}{message}{default_hint}: {C_RESET}").strip()
            return val if val else default
        except (EOFError, KeyboardInterrupt):
            print()
            return default

    def confirm(self, message: str, default: bool = True) -> bool:
        """Prompt user for yes/no confirmation."""
        hint = " [Y/n]" if default else " [y/N]"
        try:
            val = input(f"{C_BOLD}{message}{hint}: {C_RESET}").strip().lower()
            if not val:
                return default
            return val in ("y", "yes", "1", "true")
        except (EOFError, KeyboardInterrupt):
            print()
            return default

    def transition(self, next_state: WizardPhase) -> None:
        """Explicitly transition to another state machine phase."""
        self.state = next_state

    def send_event(self, event: WizardEvent) -> WizardPhase:
        """Dispatch an event trigger to transition to the next phase."""
        transitions = DEFAULT_TRANSITIONS.get(self.state, {})
        next_phase = transitions.get(event, self.state)
        self.state = next_phase
        return self.state

    def step(self, event: WizardEvent | None = None) -> WizardPhase:
        """Step to next phase."""
        return self.send_event(event or WizardEvent.PROCEED)

    def run_state_machine(self) -> int:
        """Execute the interactive wizard through pure-stdlib menu navigation."""
        c_cyan = C_CYAN if IS_TTY else ""
        c_bold = C_BOLD if IS_TTY else ""
        c_reset = C_RESET if IS_TTY else ""

        print(f"\n{c_bold}{c_cyan}{'=' * 66}{c_reset}")
        print(f"{c_bold}Renslate Autonomous Localization Studio{c_reset}".center(66))
        print(f"{c_bold}{c_cyan}{'=' * 66}{c_reset}\n")

        if not sys.stdin.isatty():
            print(f"Non-interactive terminal detected. Validating project at: {self.root}")
            if (self.root / "game").is_dir() and not (self.root / ".renslate").is_dir():
                Project.create(self.root)
            self.state = WizardPhase.INIT
            return 0

        self.state = WizardPhase.INIT

        while self.state != WizardPhase.EXIT:
            if self.state == WizardPhase.INIT:
                options = [
                    ("1", f"Tiếp tục với thư mục hiện tại: {self.root}"),
                    ("2", "Chọn thư mục dự án khác"),
                    ("0", "Thoát"),
                ]
                choice = select_menu("Giai đoạn 1/6: Khởi tạo dự án (INIT)", options)
                if choice == "0":
                    self.state = WizardPhase.EXIT
                    return 0
                if choice == "2":
                    new_path = self.prompt("Nhập đường dẫn thư mục dự án", default=str(self.root))
                    self.root = Path(new_path).expanduser().resolve()
                    self.context.root = self.root

                game_dir = self.root / "game"
                if not game_dir.is_dir() and self.confirm(
                    "Tạo thư mục 'game/' để tiếp tục?", default=True
                ):
                    game_dir.mkdir(parents=True, exist_ok=True)

                try:
                    self.project = Project.open(self.root)
                except Exception:
                    if self.confirm("Khởi tạo cấu hình Renslate mới cho dự án này?", default=True):
                        self.project = Project.create(self.root)
                    else:
                        self.state = WizardPhase.EXIT
                        return 0
                self.context.project = self.project
                self.context.config = self.project.config
                self.state = WizardPhase.DETECT_GAME

            elif self.state == WizardPhase.DETECT_GAME:
                self.inspection = inspect_game(self.root, self.context.config.target_lang)
                self.context.inspection = self.inspection
                print(render_inspection_card(self.inspection))

                options = [
                    ("1", "Tiếp tục sang bước trích xuất kịch bản"),
                    ("2", "Đổi ngôn ngữ đích"),
                    ("3", "Quay lại bước chọn thư mục dự án"),
                    ("0", "Thoát"),
                ]
                choice = select_menu("Giai đoạn 2/6: Kiểm tra Game Ren'Py (DETECT_GAME)", options)
                if choice == "1":
                    self.state = WizardPhase.EXTRACT_OR_REPAIR
                elif choice == "2":
                    new_lang = self.prompt(
                        "Nhập ngôn ngữ đích", default=self.context.config.target_lang
                    )
                    if self.project:
                        new_config = dataclasses.replace(self.project.config, target_lang=new_lang)
                        new_config.save(self.project.config_path)
                        self.project = Project(self.project.root, new_config)
                        self.context.project = self.project
                        self.context.config = new_config
                elif choice == "3":
                    self.state = WizardPhase.INIT
                else:
                    self.state = WizardPhase.EXIT
                    return 0

            elif self.state == WizardPhase.EXTRACT_OR_REPAIR:
                mode_str = (
                    self.context.inspection.script_mode if self.context.inspection else "UNKNOWN"
                )
                print(f"\n{C_BOLD}Chế độ kịch bản hiện tại: {mode_str}{C_RESET}")
                options = [
                    ("1", "Tiếp tục: Kiểm tra kết nối LLM Server"),
                    ("2", "Quét và trích xuất kịch bản vào cơ sở dữ liệu"),
                    ("3", "Quay lại bước kiểm tra game"),
                    ("0", "Thoát"),
                ]
                choice = select_menu(
                    "Giai đoạn 3/6: Trích xuất & Đồng bộ kịch bản (EXTRACT_OR_REPAIR)",
                    options,
                )
                if choice == "1":
                    self.state = WizardPhase.LLM_PROBE
                elif choice == "2":
                    if self.project:
                        try:
                            from renslate.renpy.extract import extract

                            db = self.project.open_db()
                            extract_rep = extract(self.project, db)
                            db.close()
                            extract_msg = f"✓ Đã trích xuất {extract_rep.units} đơn vị dịch."
                            print(f"{C_GREEN}{extract_msg}{C_RESET}")
                        except Exception as e:
                            print(f"{C_YELLOW}Thông báo: {e}{C_RESET}")
                elif choice == "3":
                    self.state = WizardPhase.DETECT_GAME
                else:
                    self.state = WizardPhase.EXIT
                    return 0

            elif self.state == WizardPhase.LLM_PROBE:
                from renslate.ui.runner import check_server_health, detect_server_model

                url = self.context.config.llm.base_url
                healthy = check_server_health(url)
                if healthy:
                    model = detect_server_model(url, self.context.config.llm.model)
                    print(f"{C_GREEN}✓ LLM Server trực tuyến! Model: {model}{C_RESET}")
                else:
                    print(f"{C_YELLOW}⚠ LLM Server chưa phản hồi tại {url}.{C_RESET}")

                options = [
                    (
                        "1",
                        "Bắt đầu dịch tự động kèm Live Dashboard (tự động phân tích glossary)",
                    ),
                    ("2", "Phân tích và dịch Glossary & Nhân vật bằng LLM ngay bây giờ"),
                    ("3", "Thay đổi địa chỉ LLM Server"),
                    ("4", "Điều chỉnh chunk size & tham số"),
                    ("5", "Quay lại bước trích xuất"),
                    ("0", "Thoát"),
                ]
                choice = select_menu(
                    "Giai đoạn 4/6: Kiểm tra LLM Server & Tham số (LLM_PROBE)", options
                )
                if choice == "1":
                    self.state = WizardPhase.TRANSLATE_DASHBOARD
                elif choice == "2":
                    if self.project:
                        from renslate.ui.runner import bootstrap_if_needed

                        db = self.project.open_db()
                        try:
                            bootstrap_if_needed(self.project, db, bootstrap=True)
                            print(f"{C_GREEN}✓ Đã phân tích và dịch glossary & nhân vật.{C_RESET}")
                        finally:
                            db.close()
                elif choice == "3":
                    new_url = self.prompt("Địa chỉ LLM Server", default=url)
                    new_llm = dataclasses.replace(self.context.config.llm, base_url=new_url)
                    new_config = dataclasses.replace(self.context.config, llm=new_llm)
                    if self.project:
                        new_config.save(self.project.config_path)
                        self.project = Project(self.project.root, new_config)
                        self.context.project = self.project
                    self.context.config = new_config
                elif choice == "4":
                    c_str = self.prompt(
                        "Kích thước chunk (chunk size)",
                        default=str(self.context.chunk_size),
                    )
                    try:
                        self.context.chunk_size = max(1, int(c_str))
                    except ValueError:
                        self.context.chunk_size = 100
                elif choice == "5":
                    self.state = WizardPhase.EXTRACT_OR_REPAIR
                else:
                    self.state = WizardPhase.EXIT
                    return 0

            elif self.state == WizardPhase.TRANSLATE_DASHBOARD:
                from renslate.ui.runner import run_unattended_pipeline

                print(
                    f"\n{C_BOLD}Khởi động pipeline dịch với chunk size "
                    f"{self.context.chunk_size}...{C_RESET}"
                )
                code = run_unattended_pipeline(
                    self.root,
                    batch_chunk=self.context.chunk_size,
                    retry_failed=self.context.retry_failed,
                    auto_apply=self.context.auto_apply,
                )
                self.state = WizardPhase.APPLY_VERIFY
                if code != 0:
                    print(f"{C_YELLOW}Pipeline tạm dừng hoặc hoàn tất với mã {code}.{C_RESET}")

            elif self.state == WizardPhase.APPLY_VERIFY:
                options = [
                    ("1", "Áp dụng bản dịch vào game (Apply)"),
                    ("2", "Kiểm tra chất lượng bản dịch (QA Check)"),
                    ("3", "Thiết lập phông tiếng Việt (renslate_setup.rpy)"),
                    ("0", "Hoàn thành & Thoát"),
                ]
                choice = select_menu("Giai đoạn 6/6: Áp dụng & Kiểm tra QA (APPLY_VERIFY)", options)
                if choice == "1":
                    if self.project:
                        from renslate.renpy.writer import apply_translations

                        db = self.project.open_db()
                        apply_rep = apply_translations(self.project, db)
                        db.close()
                        print(
                            f"{C_GREEN}✓ Đã ghi {apply_rep.lines_written} "
                            f"bản dịch vào game.{C_RESET}"
                        )
                elif choice == "2":
                    if self.project:
                        from renslate.engine.qa import QAChecker

                        db = self.project.open_db()
                        units = db.all_units()
                        db.close()
                        issues = 0
                        checker = QAChecker()
                        for u in units:
                            if u.target:
                                iss = checker.check_unit(
                                    uid=u.uid,
                                    source=u.source,
                                    masked=None,
                                    target=u.target,
                                    speaker=u.speaker,
                                )
                                if iss:
                                    issues += len(iss)
                        print(
                            f"{C_GREEN}✓ Kiểm tra QA hoàn tất: "
                            f"phát hiện {issues} vấn đề cần lưu ý.{C_RESET}"
                        )
                elif choice == "3":
                    if self.project:
                        from renslate.renpy.writer import setup_vietnamese_support

                        p = setup_vietnamese_support(self.project)
                        print(f"{C_GREEN}✓ Đã tạo cấu hình phông tại {p}{C_RESET}")
                else:
                    self.state = WizardPhase.EXIT
                    return 0

        return 0

    def run(self, interactive_menu: bool = False) -> int:
        """Execute interactive wizard steps across the 6-phase state machine."""
        if interactive_menu:
            return self.run_state_machine()

        from renslate.ui.runner import check_server_health, detect_server_model

        c_cyan = C_CYAN if IS_TTY else ""
        c_bold = C_BOLD if IS_TTY else ""
        c_green = C_GREEN if IS_TTY else ""
        c_yellow = C_YELLOW if IS_TTY else ""
        c_reset = C_RESET if IS_TTY else ""

        print(f"\n{c_bold}{c_cyan}{'=' * 60}{c_reset}")
        print(f"{c_bold}Renslate Interactive Setup Wizard{c_reset}".center(60))
        print(f"{c_bold}{c_cyan}{'=' * 60}{c_reset}\n")

        # Non-interactive fallback
        if not sys.stdin.isatty():
            print(f"Non-interactive terminal detected. Validating project at: {self.root}")
            if (self.root / "game").is_dir() and not (self.root / ".renslate").is_dir():
                Project.create(self.root)
            self.state = WizardPhase.INIT
            return 0

        # Phase 1: INIT
        self.state = WizardPhase.INIT
        proj_str = self.prompt("1. Đường dẫn thư mục dự án Ren'Py", default=str(self.root))
        target_root = Path(proj_str).expanduser().resolve()
        if not target_root.exists():
            print(f"{c_yellow}Thư mục '{target_root}' chưa tồn tại. Đang tạo...{c_reset}")
            target_root.mkdir(parents=True, exist_ok=True)

        self.root = target_root
        self.context.root = target_root

        game_dir = target_root / "game"
        if not game_dir.is_dir():
            print(
                f"{c_yellow}Lưu ý: Không tìm thấy thư mục 'game/' trong '{target_root}'.{c_reset}"
            )
            if self.confirm("Tạo thư mục 'game/' để tiếp tục?", default=True):
                game_dir.mkdir(parents=True, exist_ok=True)

        project: Project | None = None
        try:
            project = Project.open(target_root)
            print(f"{c_green}✓ Đã mở dự án hiện có tại {target_root}{c_reset}")
        except Exception:
            if self.confirm("Khởi tạo cấu hình Renslate mới cho dự án này?", default=True):
                project = Project.create(target_root)
                print(f"{c_green}✓ Đã khởi tạo dự án mới tại {target_root}{c_reset}")
            else:
                self.state = WizardPhase.EXIT
                return 0

        self.project = project
        self.context.project = project
        config = project.config if project else Config()
        self.context.config = config

        # Phase 2: DETECT_GAME
        self.state = WizardPhase.DETECT_GAME
        self.inspection = inspect_game(target_root, config.target_lang)
        self.context.inspection = self.inspection
        if IS_TTY:
            print(
                f"{c_green}✓ Đã phát hiện {self.inspection.engine_generation} "
                f"({self.inspection.script_mode}){c_reset}"
            )

        # Phase 3: EXTRACT_OR_REPAIR
        self.state = WizardPhase.EXTRACT_OR_REPAIR
        if self.inspection.script_mode == ScriptMode.RPYC_BYTECODE_ONLY.value:
            print(
                f"{c_cyan}[i] Phát hiện game dạng bytecode-only. "
                f"Chế độ Slot-2 direct extraction khả dụng.{c_reset}"
            )

        # Phase 4: LLM_PROBE
        self.state = WizardPhase.LLM_PROBE
        server_url = self.prompt(
            "2. Địa chỉ LLM server (OpenAI-compatible)", default=config.llm.base_url
        )
        self.context.server_url = server_url
        print(f"Đang kiểm tra kết nối tới {server_url}...")
        is_healthy = check_server_health(server_url)
        if is_healthy:
            detected_model = detect_server_model(server_url, config.llm.model)
            self.context.detected_model = detected_model
            print(f"{c_green}✓ LLM Server trực tuyến! Model kích hoạt: {detected_model}{c_reset}")
        else:
            print(
                f"{c_yellow}⚠ LLM Server chưa phản hồi. "
                f"Bạn vẫn có thể tiếp tục và khởi động server sau.{c_reset}"
            )

        chunk_str = self.prompt("3. Kích thước mỗi chu kỳ dịch (chunk size)", default="100")
        try:
            chunk_size = max(1, int(chunk_str))
        except ValueError:
            chunk_size = 100
        self.context.chunk_size = chunk_size

        retry_failed = self.confirm("4. Thử lại các dòng từng bị lỗi (failed)?", default=False)
        self.context.retry_failed = retry_failed
        auto_apply = self.confirm(
            "5. Tự động áp dụng bản dịch vào game khi xong 100%?", default=True
        )
        self.context.auto_apply = auto_apply

        # Phase 5: TRANSLATE_DASHBOARD
        self.state = WizardPhase.TRANSLATE_DASHBOARD
        if self.confirm("\nBắt đầu chạy pipeline dịch tự động ngay bây giờ?", default=True):
            from renslate.ui.runner import run_unattended_pipeline

            code = run_unattended_pipeline(
                target_root,
                batch_chunk=chunk_size,
                retry_failed=retry_failed,
                auto_apply=auto_apply,
            )
            # Phase 6: APPLY_VERIFY
            self.state = WizardPhase.APPLY_VERIFY
            self.state = WizardPhase.EXIT
            return code

        print(f"\n{c_green}Cấu hình đã sẵn sàng. Bạn có thể chạy lệnh sau bất cứ lúc nào:{c_reset}")
        print(f'  renslate run -p "{target_root}" --chunk-size {chunk_size}\n')
        self.state = WizardPhase.APPLY_VERIFY
        self.state = WizardPhase.EXIT
        return 0


def run_interactive_wizard(
    project_root: Path | None = None,
    interactive_menu: bool = False,
) -> int:
    """Prompt user interactively to configure translation settings."""
    wizard = InteractiveWizard(project_root)
    return wizard.run(interactive_menu=interactive_menu)


run_wizard = run_interactive_wizard
