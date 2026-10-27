"""ANSI terminal styling, color palettes, TTY detection, and width calculation."""

from __future__ import annotations

import os
import re
import sys
import unicodedata

IS_TTY: bool = sys.stdout.isatty()


def supports_color() -> bool:
    """Determine whether color output should be enabled."""
    if "NO_COLOR" in os.environ and os.environ["NO_COLOR"] != "":
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.stdout.isatty()


_USE_COLOR: bool = supports_color()

# ANSI Color sequences (empty if color disabled or non-TTY)
C_RESET: str = "\033[0m" if _USE_COLOR else ""
C_BOLD: str = "\033[1m" if _USE_COLOR else ""
C_DIM: str = "\033[2m" if _USE_COLOR else ""
C_GREEN: str = "\033[0;32m" if _USE_COLOR else ""
C_CYAN: str = "\033[0;36m" if _USE_COLOR else ""
C_YELLOW: str = "\033[1;33m" if _USE_COLOR else ""
C_RED: str = "\033[0;31m" if _USE_COLOR else ""
C_GRAY: str = "\033[0;90m" if _USE_COLOR else ""
C_MAGENTA: str = "\033[0;35m" if _USE_COLOR else ""
C_BLUE: str = "\033[0;34m" if _USE_COLOR else ""

# Double-line box drawing glyphs
BOX_HORIZ = "═"
BOX_VERT = "║"
BOX_TOP_LEFT = "╔"
BOX_TOP_RIGHT = "╗"
BOX_BOTTOM_LEFT = "╚"
BOX_BOTTOM_RIGHT = "╝"
BOX_T_DOWN = "╦"
BOX_T_UP = "╩"
BOX_T_RIGHT = "╠"
BOX_T_LEFT = "╣"
BOX_CROSS = "╬"

# Single-line box drawing glyphs
BOX_LIGHT_HORIZ = "─"
BOX_LIGHT_VERT = "│"
BOX_LIGHT_TL = "┌"
BOX_LIGHT_TR = "┐"
BOX_LIGHT_BL = "└"
BOX_LIGHT_BR = "┘"
BOX_LIGHT_T_DOWN = "┬"
BOX_LIGHT_T_UP = "┴"
BOX_LIGHT_T_RIGHT = "├"
BOX_LIGHT_T_LEFT = "┤"
BOX_LIGHT_CROSS = "┼"

# Rounded box drawing glyphs
BOX_ROUND_TL = "╭"
BOX_ROUND_TR = "╮"
BOX_ROUND_BL = "╰"
BOX_ROUND_BR = "╯"

# Modern 24-bit / 256-color palette
C_EMERALD: str = "\033[38;2;34;197;94m" if _USE_COLOR else ""
C_SKY: str = "\033[38;2;56;189;248m" if _USE_COLOR else ""
C_AMBER: str = "\033[38;2;245;158;11m" if _USE_COLOR else ""
C_ROSE: str = "\033[38;2;244;63;94m" if _USE_COLOR else ""
C_INDIGO: str = "\033[38;2;129;140;248m" if _USE_COLOR else ""
C_SLATE: str = "\033[38;2;148;163;184m" if _USE_COLOR else ""
C_BRIGHT_CYAN: str = "\033[1;96m" if _USE_COLOR else ""
C_BRIGHT_GREEN: str = "\033[1;92m" if _USE_COLOR else ""

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text to get plain string."""
    return _ANSI_ESCAPE_RE.sub("", text)


def char_width(char: str) -> int:
    """Calculate terminal display cell width of a single character."""
    category = unicodedata.category(char)
    if category in ("Mn", "Me", "Cf"):
        return 0
    if unicodedata.east_asian_width(char) in ("W", "F"):
        return 2
    return 1


def visible_len(text: str) -> int:
    """Return visible terminal cell width excluding ANSI escape codes."""
    plain = strip_ansi(text)
    return sum(char_width(ch) for ch in plain)
