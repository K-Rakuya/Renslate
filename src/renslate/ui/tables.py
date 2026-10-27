"""Pure standard-library Unicode box-drawing tables for CLI previews."""

from __future__ import annotations

from collections.abc import Sequence

from renslate.ui.theme import (
    BOX_LIGHT_BL,
    BOX_LIGHT_BR,
    BOX_LIGHT_CROSS,
    BOX_LIGHT_HORIZ,
    BOX_LIGHT_T_DOWN,
    BOX_LIGHT_T_LEFT,
    BOX_LIGHT_T_RIGHT,
    BOX_LIGHT_T_UP,
    BOX_LIGHT_TL,
    BOX_LIGHT_TR,
    BOX_LIGHT_VERT,
    strip_ansi,
    visible_len,
)


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    max_widths: Sequence[int] | None = None,
    alignments: Sequence[str] | None = None,
) -> str:
    """Render a tabular dataset using clean Unicode light box borders."""
    if not headers and not rows:
        return ""

    num_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
    col_widths = [visible_len(h) for h in headers] if headers else [0] * num_cols

    for row in rows:
        for idx, cell in enumerate(row[:num_cols]):
            col_widths[idx] = max(col_widths[idx], visible_len(cell))

    if max_widths:
        for idx, mw in enumerate(max_widths[:num_cols]):
            if mw > 0:
                col_widths[idx] = min(col_widths[idx], mw)

    def _truncate_visible(text: str, max_w: int) -> str:
        if visible_len(text) <= max_w:
            return text
        plain = strip_ansi(text)
        if len(plain) <= max_w:
            return plain
        return plain[: max(0, max_w - 1)] + "…"

    def _pad(cell: str, width: int, align: str = "left") -> str:
        trimmed = _truncate_visible(cell, width)
        vlen = visible_len(trimmed)
        pad_size = max(0, width - vlen)
        if align == "right":
            return " " * pad_size + trimmed
        if align == "center":
            left_pad = pad_size // 2
            right_pad = pad_size - left_pad
            return " " * left_pad + trimmed + " " * right_pad
        return trimmed + " " * pad_size

    def _get_align(col_idx: int) -> str:
        if alignments and col_idx < len(alignments):
            return alignments[col_idx].lower()
        return "left"

    lines: list[str] = []
    # Top border
    top_sep = (
        BOX_LIGHT_TL
        + BOX_LIGHT_T_DOWN.join(BOX_LIGHT_HORIZ * (w + 2) for w in col_widths)
        + BOX_LIGHT_TR
    )
    lines.append(top_sep)

    # Header
    if headers:
        header_line = (
            BOX_LIGHT_VERT
            + BOX_LIGHT_VERT.join(
                f" {_pad(h, w, _get_align(idx))} "
                for idx, (h, w) in enumerate(zip(headers, col_widths, strict=False))
            )
            + BOX_LIGHT_VERT
        )
        lines.append(header_line)
        mid_sep = (
            BOX_LIGHT_T_RIGHT
            + BOX_LIGHT_CROSS.join(BOX_LIGHT_HORIZ * (w + 2) for w in col_widths)
            + BOX_LIGHT_T_LEFT
        )
        lines.append(mid_sep)

    # Rows
    for row in rows:
        row_cells = list(row[:num_cols]) + [""] * max(0, num_cols - len(row))
        row_line = (
            BOX_LIGHT_VERT
            + BOX_LIGHT_VERT.join(
                f" {_pad(c, w, _get_align(idx))} "
                for idx, (c, w) in enumerate(zip(row_cells, col_widths, strict=False))
            )
            + BOX_LIGHT_VERT
        )
        lines.append(row_line)

    # Bottom border
    bot_sep = (
        BOX_LIGHT_BL
        + BOX_LIGHT_T_UP.join(BOX_LIGHT_HORIZ * (w + 2) for w in col_widths)
        + BOX_LIGHT_BR
    )
    lines.append(bot_sep)

    return "\n".join(lines)
