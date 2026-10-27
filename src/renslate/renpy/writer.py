"""Application of stored translations to the Ren'Py translation files.

Files are rewritten in place, one line per slot, leaving every other byte of the structure
alone. Before anything is written all targets are validated and every slot is checked against
its fingerprint, so a stale store or malformed markup aborts the whole operation.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from renslate.core.db import Database
from renslate.core.errors import ApplyError
from renslate.core.models import ApplyTarget, UnitKind
from renslate.core.project import Project
from renslate.core.textio import TextFile, read_text_file, write_atomic
from renslate.renpy.literals import encode_say_text, quote_python_text
from renslate.renpy.mask import find_markup_error
from renslate.renpy.parser import fingerprint, parse_say, parse_string_line

_MAX_LISTED = 20
_RERUN_HINT = "hãy chạy lại 'renslate extract'"
_STALE_MESSAGE = f"tệp đã thay đổi kể từ lần extract, {_RERUN_HINT}"


@dataclass(slots=True)
class ApplyReport:
    files_changed: int = 0
    lines_written: int = 0
    unchanged: int = 0
    backup_dir: Path | None = None
    dry_run: bool = False


def _render(kind: UnitKind, line: str, target: str) -> str | None:
    """Return the line carrying target, None if it already does, or raise on a bad line."""
    if kind is UnitKind.DIALOGUE:
        say = parse_say(line.strip())
        if say is None:
            raise ApplyError("không phải câu lời thoại")
        if say.text == target:
            return None
        indent = line[: len(line) - len(line.lstrip())]
        return f'{indent}{say.prefix}"{encode_say_text(target)}"{say.suffix}'
    parsed = parse_string_line(line, "new")
    if parsed is None:
        raise ApplyError("không phải dòng 'new'")
    head, current = parsed
    if current == target:
        return None
    return f'{head}"{quote_python_text(target)}"'


def _summarise(problems: list[str]) -> str:
    shown = problems[:_MAX_LISTED]
    extra = len(problems) - len(shown)
    return "\n".join(shown + ([f"... và {extra} mục khác"] if extra > 0 else []))


def _validate_targets(targets: list[ApplyTarget]) -> None:
    problems = []
    for item in targets:
        error = find_markup_error(item.target)
        if error is not None:
            problems.append(f"{item.slot.uid}: {error}")
    if problems:
        raise ApplyError("bản dịch có markup không hợp lệ:\n" + _summarise(problems))


def _plan_file(
    project: Project, relative: str, items: list[ApplyTarget]
) -> tuple[TextFile, list[tuple[str, int, str]], int]:
    """Compute the new content of one file; return it with fingerprint updates and unchanged."""
    path = project.root / relative
    try:
        text_file = read_text_file(path)
    except (FileNotFoundError, UnicodeDecodeError) as error:
        raise ApplyError(f"{relative}: không đọc được ({error}); {_RERUN_HINT}") from error

    lines = text_file.lines
    updates: list[tuple[str, int, str]] = []
    unchanged = 0
    for item in items:
        slot = item.slot
        if slot.line > len(lines) or slot.anchor_line > len(lines):
            raise ApplyError(f"{relative}:{slot.line}: {_STALE_MESSAGE}")
        anchor, current = lines[slot.anchor_line - 1], lines[slot.line - 1]
        if fingerprint(anchor, current) != slot.fingerprint:
            raise ApplyError(f"{relative}:{slot.line}: {_STALE_MESSAGE}")
        try:
            rendered = _render(item.kind, current, item.target)
        except ApplyError as error:
            raise ApplyError(f"{relative}:{slot.line}: {error}") from error
        if rendered is None:
            unchanged += 1
            continue
        lines[slot.line - 1] = rendered
        updates.append((relative, slot.line, fingerprint(anchor, rendered)))
    return text_file, updates, unchanged


def _backup_directory(root: Path, timestamp: str) -> Path:
    candidate = root / timestamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{timestamp}-{suffix}"
    return candidate


def apply_translations(
    project: Project,
    database: Database,
    *,
    dry_run: bool = False,
    timestamp: str | None = None,
) -> ApplyReport:
    """Write every stored translation into the game's translation files."""
    targets = database.apply_targets()
    _validate_targets(targets)

    by_file: dict[str, list[ApplyTarget]] = defaultdict(list)
    for item in targets:
        by_file[item.slot.file].append(item)

    report = ApplyReport(dry_run=dry_run)
    plans: dict[str, TextFile] = {}
    updates: list[tuple[str, int, str]] = []
    for relative, items in by_file.items():
        text_file, file_updates, unchanged = _plan_file(project, relative, items)
        report.unchanged += unchanged
        if file_updates:
            plans[relative] = text_file
            updates.extend(file_updates)
    report.files_changed = len(plans)
    report.lines_written = len(updates)
    if dry_run or not plans:
        return report

    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = _backup_directory(project.backup_root, stamp)
    for relative in plans:
        destination = backup_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project.root / relative, destination)
    report.backup_dir = backup_dir

    for relative, text_file in plans.items():
        write_atomic(project.root / relative, text_file.to_bytes())
    with database.transaction():
        database.update_fingerprints(updates)
    return report


def setup_vietnamese_support(project: Project, *, font_file: str = "DejaVuSans.ttf") -> Path:
    """Generate renslate_setup.rpy to configure Unicode font support for the target language."""
    target_lang = project.config.target_lang
    content = f"""\
# renslate_setup.rpy — Cấu hình tiếng Việt cho Ren'Py
# Tự động tạo bởi renslate

init 10 python:
    _orig_replace_text = getattr(config, "replace_text", None)
    def _vi_replace_text(s):
        if _orig_replace_text:
            s = _orig_replace_text(s)
        if renpy.game.preferences.language == "{target_lang}":
            translated = renpy.translation.translate_string(s)
            if translated != s:
                return translated
        return s
    config.replace_text = _vi_replace_text

translate {target_lang} python:
    # Thiết lập phông chữ hỗ trợ đầy đủ ký tự tiếng Việt
    _vi_font = "{font_file}"
    if renpy.loadable(_vi_font):
        gui.text_font = _vi_font
        gui.name_text_font = _vi_font
        gui.interface_text_font = _vi_font
        gui.button_text_font = _vi_font
        gui.choice_button_text_font = _vi_font
"""
    dest = project.tl_root / "renslate_setup.rpy"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    return dest
