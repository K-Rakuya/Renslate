"""Automatic Vietnamese font discovery, compatibility checking, and Ren'Py configuration."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from renslate.core.textio import write_atomic

if TYPE_CHECKING:
    from renslate.core.project import Project

VI_SAMPLE_CHARS = (
    "aáàảãạăắằẳẵặâấầẩẫậ"
    "eéèẻẽẹêếềểễệ"
    "iíìỉĩị"
    "oóòỏõọôốồổỗộơớờởỡợ"
    "uúùủũụưứừửữự"
    "yýỳỷỹỵ"
    "dđ"
    "AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ"
    "EÉÈẺẼẸÊẾỀỂỄỆ"
    "IÍÌỈĨỊ"
    "OÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "UÚÙỦŨỤƯỨỪỬỮỰ"
    "YÝỲỶỸỴ"
    "DĐ"
)

_SERIF_KEYWORDS = frozenset(
    {
        "serif",
        "times",
        "mincho",
        "georgia",
        "cormorant",
        "soul",
        "bridge",
        "antique",
        "roman",
        "book",
        "garamond",
        "baskerville",
        "playfair",
        "lora",
        "merriweather",
        "cinzel",
        "bodoni",
        "didot",
        "cambria",
    }
)

_DISPLAY_KEYWORDS = frozenset(
    {
        "script",
        "hand",
        "manatee",
        "chicken",
        "brush",
        "calligraphy",
        "comic",
        "pixel",
        "arcade",
        "gothic",
        "curvy",
        "cute",
        "doodle",
        "marker",
    }
)

_FONT_EXTS = frozenset({".ttf", ".otf", ".woff", ".woff2"})


@dataclass(frozen=True, slots=True)
class FontInfo:
    """Information about a detected font in the game."""

    rel_path: str
    filename: str
    style: str  # 'sans', 'serif', 'display'
    supports_vi: bool
    missing_count: int


def check_font_vietnamese_support(path: Path) -> tuple[bool, int, list[str]]:
    """Inspect TrueType/OpenType sfnt cmap table to check for complete Vietnamese Unicode coverage.

    Returns (supports_all, missing_count, missing_characters).
    """
    if not path.is_file():
        return False, len(VI_SAMPLE_CHARS), list(VI_SAMPLE_CHARS)
    try:
        data = path.read_bytes()
        if len(data) < 12:
            return False, len(VI_SAMPLE_CHARS), list(VI_SAMPLE_CHARS)

        num_tables = struct.unpack(">H", data[4:6])[0]
        cmap_offset = None
        for i in range(num_tables):
            idx = 12 + i * 16
            tag, _, offset, _ = struct.unpack(">4sIII", data[idx : idx + 16])
            if tag == b"cmap":
                cmap_offset = offset
                break

        if not cmap_offset or cmap_offset + 4 > len(data):
            return False, len(VI_SAMPLE_CHARS), list(VI_SAMPLE_CHARS)

        num_subtables = struct.unpack(">H", data[cmap_offset + 2 : cmap_offset + 4])[0]
        chars: set[int] = set()

        for i in range(num_subtables):
            sub_idx = cmap_offset + 4 + i * 8
            if sub_idx + 8 > len(data):
                break
            _, _, sub_offset = struct.unpack(">HHI", data[sub_idx : sub_idx + 8])
            pos = cmap_offset + sub_offset
            if pos + 2 > len(data):
                continue
            fmt = struct.unpack(">H", data[pos : pos + 2])[0]
            if fmt == 4 and pos + 16 <= len(data):
                seg_count = struct.unpack(">H", data[pos + 6 : pos + 8])[0] // 2
                end_pos = pos + 14
                start_pos = end_pos + seg_count * 2 + 2
                if start_pos + seg_count * 2 <= len(data):
                    end_code = struct.unpack(f">{seg_count}H", data[end_pos : end_pos + seg_count * 2])
                    start_code = struct.unpack(
                        f">{seg_count}H", data[start_pos : start_pos + seg_count * 2]
                    )
                    for s, e in zip(start_code, end_code, strict=False):
                        for cp in range(s, e + 1):
                            chars.add(cp)
            elif fmt == 12 and pos + 16 <= len(data):
                num_groups = struct.unpack(">I", data[pos + 12 : pos + 16])[0]
                g_pos = pos + 16
                for _ in range(num_groups):
                    if g_pos + 12 > len(data):
                        break
                    s, e, _ = struct.unpack(">III", data[g_pos : g_pos + 12])
                    for cp in range(s, e + 1):
                        chars.add(cp)
                    g_pos += 12

        missing = [ch for ch in VI_SAMPLE_CHARS if ord(ch) not in chars]
        return len(missing) == 0, len(missing), missing
    except Exception:
        return False, len(VI_SAMPLE_CHARS), list(VI_SAMPLE_CHARS)


def classify_font_style(name: str) -> str:
    """Classify a font into 'serif', 'display', or 'sans' based on its filename."""
    lower = name.lower()
    for kw in _SERIF_KEYWORDS:
        if kw in lower:
            return "serif"
    for kw in _DISPLAY_KEYWORDS:
        if kw in lower:
            return "display"
    return "sans"


def discover_game_fonts(project: Project) -> list[FontInfo]:
    """Scan game directory for font files and inspect their Vietnamese Unicode support."""
    game_dir = project.root / "game"
    if not game_dir.is_dir():
        return []

    detected: list[FontInfo] = []
    seen_files: set[str] = set()

    # 1. Scan filesystem for fonts
    for ext in _FONT_EXTS:
        for p in sorted(game_dir.rglob(f"*{ext}")):
            # Ignore translation folders of other languages
            if "tl" in p.parts:
                continue
            rel = str(p.relative_to(game_dir))
            if rel not in seen_files:
                seen_files.add(rel)
                style = classify_font_style(p.name)
                supports_vi, missing_count, _ = check_font_vietnamese_support(p)
                detected.append(
                    FontInfo(
                        rel_path=rel,
                        filename=p.name,
                        style=style,
                        supports_vi=supports_vi,
                        missing_count=missing_count,
                    )
                )

    # 2. Scan script files for font literals (e.g. fonts/Custom.ttf)
    font_literal_pattern = re.compile(
        r"""(?:["'])([a-zA-Z0-9_\-\s/]+\.(?:ttf|otf|woff2?))(?:["'])""",
        re.IGNORECASE,
    )
    for rpy in sorted(game_dir.glob("*.rpy")):
        if "tl" in rpy.parts:
            continue
        try:
            content = rpy.read_text(encoding="utf-8", errors="ignore")
            for m in font_literal_pattern.finditer(content):
                fpath_str = m.group(1).strip()
                if fpath_str not in seen_files and not fpath_str.startswith("common/"):
                    seen_files.add(fpath_str)
                    fname = Path(fpath_str).name
                    full_p = game_dir / fpath_str
                    supports_vi, missing_count, _ = check_font_vietnamese_support(full_p)
                    style = classify_font_style(fname)
                    detected.append(
                        FontInfo(
                            rel_path=fpath_str,
                            filename=fname,
                            style=style,
                            supports_vi=supports_vi,
                            missing_count=missing_count,
                        )
                    )
        except Exception:
            continue

    return detected


def setup_vietnamese_support(
    project: Project,
    *,
    font_file: str | None = None,
    serif_font: str | None = None,
    auto_replace_game_fonts: bool = True,
) -> Path:
    """Generate renslate_setup.rpy to configure Unicode font support and font mapping for Ren'Py.

    - Automatically detects all game fonts that lack Vietnamese diacritics.
    - Sets up config.font_replacement_map so {font=...} tags transparently render with VN fonts.
    - Configures gui.text_font, gui.name_text_font, gui.interface_text_font with smart fallbacks.
    - Cascades to high-quality VN fonts (Be Vietnam Pro, Nunito, Lora) or Ren'Py core DejaVuSans.
    """
    target_lang = project.config.target_lang
    detected_fonts = discover_game_fonts(project) if auto_replace_game_fonts else []

    incomplete_serif: list[str] = []
    incomplete_sans: list[str] = []

    for f in detected_fonts:
        if not f.supports_vi:
            # Add both full relative path and basename
            candidates = [f.rel_path, f.filename]
            if f.style == "serif":
                for c in candidates:
                    if c not in incomplete_serif:
                        incomplete_serif.append(c)
            else:
                for c in candidates:
                    if c not in incomplete_sans:
                        incomplete_sans.append(c)

    # Format font replacement code
    rules_lines: list[str] = []
    if incomplete_serif:
        rules_lines.append(f"    _incomplete_serif_fonts = {incomplete_serif!r}")
        rules_lines.append("    for _f in _incomplete_serif_fonts:")
        rules_lines.append("        for _b in (False, True):")
        rules_lines.append("            for _i in (False, True):")
        rules_lines.append(
            "                config.font_replacement_map[_f, _b, _i] = (_vi_serif, _b, _i)"
        )
    if incomplete_sans:
        rules_lines.append(f"    _incomplete_sans_fonts = {incomplete_sans!r}")
        rules_lines.append("    for _f in _incomplete_sans_fonts:")
        rules_lines.append("        for _b in (False, True):")
        rules_lines.append("            for _i in (False, True):")
        rules_lines.append(
            "                config.font_replacement_map[_f, _b, _i] = (_vi_sans, _b, _i)"
        )

    font_replacement_code = "\n".join(rules_lines) if rules_lines else "    pass"

    # User custom font overrides or candidate defaults
    user_sans = f'"{font_file}",' if font_file else ""
    user_serif = f'"{serif_font}",' if serif_font else ""

    content = f"""\
# renslate_setup.rpy — Cấu hình hiển thị tiếng Việt & Tự động đổi font cho Ren'Py
# Tự động tạo bởi renslate

init 10 python:
    # Bắt và dịch các chuỗi thay thế động khi đang ở ngôn ngữ tiếng Việt
    _orig_replace_text = getattr(config, "replace_text", None)
    def _vi_replace_text(s):
        if _orig_replace_text:
            s = _orig_replace_text(s)
        if getattr(renpy.game.preferences, "language", None) == "{target_lang}":
            translated = renpy.translation.translate_string(s)
            if translated != s:
                return translated
        return s
    config.replace_text = _vi_replace_text

translate {target_lang} python:
    # 1. Tự động chọn font tiếng Việt khả dụng
    # Danh sách font Sans-serif tối ưu cho Visual Novel (Be Vietnam Pro, Nunito, Lexend, Roboto):
    _vi_sans_candidates = [
        {user_sans}
        "tl/{target_lang}/fonts/BeVietnamPro-Regular.ttf",
        "fonts/BeVietnamPro-Regular.ttf",
        "tl/{target_lang}/fonts/Nunito-Regular.ttf",
        "fonts/Nunito-Regular.ttf",
        "tl/{target_lang}/fonts/LexendDeca-Regular.ttf",
        "fonts/LexendDeca-Regular.ttf",
        "DejaVuSans.ttf",
    ]
    _vi_sans = next((f for f in _vi_sans_candidates if f and renpy.loadable(f)), "DejaVuSans.ttf")

    # Danh sách font Serif tối ưu cho kỳ ảo, thần thoại, tiểu thuyết (Lora, Merriweather, Playfair):
    _vi_serif_candidates = [
        {user_serif}
        "tl/{target_lang}/fonts/Lora-Regular.ttf",
        "fonts/Lora-Regular.ttf",
        "tl/{target_lang}/fonts/Merriweather-Regular.ttf",
        "fonts/Merriweather-Regular.ttf",
        "DejaVuSerif.ttf",
        _vi_sans,
    ]
    _vi_serif = next((f for f in _vi_serif_candidates if f and renpy.loadable(f)), _vi_sans)

    # 2. Cập nhật phông chữ hệ thống giao diện Ren'Py GUI
    gui.text_font = _vi_sans
    gui.name_text_font = _vi_sans
    gui.interface_text_font = _vi_sans
    gui.button_text_font = _vi_sans
    gui.choice_button_text_font = _vi_sans

    # 3. Tự động thay thế các phông chữ gốc bị thiếu dấu tiếng Việt (Font Replacement Map)
{font_replacement_code}
"""
    dest = project.tl_root / "renslate_setup.rpy"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Respect BOM and CRLF if existing translation files use them
    use_bom = False
    use_crlf = False
    for existing_rpy in project.tl_root.glob("*.rpy"):
        if existing_rpy.name == "renslate_setup.rpy":
            continue
        try:
            raw = existing_rpy.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                use_bom = True
            if b"\r\n" in raw:
                use_crlf = True
            break
        except Exception:
            pass

    if use_crlf:
        content = content.replace("\r\n", "\n").replace("\n", "\r\n")
    raw_bytes = content.encode("utf-8")
    if use_bom:
        raw_bytes = b"\xef\xbb\xbf" + raw_bytes

    write_atomic(dest, raw_bytes)
    return dest


__all__ = [
    "FontInfo",
    "check_font_vietnamese_support",
    "classify_font_style",
    "discover_game_fonts",
    "setup_vietnamese_support",
]
