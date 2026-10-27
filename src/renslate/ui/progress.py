"""Progress meters, duration formatting, and speed calculations for renslate UI."""

from __future__ import annotations

import math

FRAC_BLOCKS: list[str] = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if math.isnan(seconds) or math.isinf(seconds) or seconds <= 0:
        return "--:--:--"
    sec_int = int(seconds)
    hours, rem = divmod(sec_int, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def render_progress_bar(pct: float, width: int = 20) -> str:
    """Render a Unicode progress bar with sub-character fractional resolution."""
    clamped_pct = max(0.0, min(100.0, pct))
    exact = width * clamped_pct / 100.0
    whole = int(exact)
    rem = exact - whole
    frac_idx = round(rem * 8)

    if frac_idx == 8:
        whole += 1
        frac_idx = 0

    filled = "█" * min(width, whole)
    if 0 < frac_idx < 8 and whole < width:
        frac_char = FRAC_BLOCKS[frac_idx]
        empty = "░" * max(0, width - whole - 1)
        bar = f"{filled}{frac_char}{empty}"
    else:
        empty = "░" * max(0, width - whole)
        bar = f"{filled}{empty}"

    return f"[{bar}] {clamped_pct:5.1f}%"


def format_speed(units_per_sec: float) -> str:
    """Format speed into units/s or units/min."""
    if units_per_sec <= 0:
        return "0.0 u/s"
    if units_per_sec < 0.2:
        return f"{units_per_sec * 60:4.1f} u/min"
    return f"{units_per_sec:4.1f} u/s"
