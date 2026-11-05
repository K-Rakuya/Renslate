"""Centralized console output formatting, ANSI styling, and report presentation."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TextIO

from renslate.engine.translate import TranslationStats
from renslate.renpy.extract import ExtractReport
from renslate.renpy.parser import Issue
from renslate.renpy.writer import ApplyReport
from renslate.ui.theme import IS_TTY

# Raw ANSI escape sequences
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[0;32m"
ANSI_YELLOW = "\033[1;33m"
ANSI_RED = "\033[0;31m"
ANSI_CYAN = "\033[0;36m"
ANSI_SLATE = "\033[38;5;244m"


class ConsoleFormatter:
    """Manages clean, colored or plain text console output."""

    def __init__(
        self,
        plain: bool = False,
        force_color: bool = False,
        out_stream: TextIO | None = None,
        err_stream: TextIO | None = None,
    ) -> None:
        self.out_stream = out_stream if out_stream is not None else sys.stdout
        self.err_stream = err_stream if err_stream is not None else sys.stderr

        out_isatty = getattr(self.out_stream, "isatty", lambda: False)()
        err_isatty = getattr(self.err_stream, "isatty", lambda: False)()

        self.use_color = force_color or ((not plain) and (out_isatty or err_isatty or IS_TTY))

    def _color(self, code: str, text: str) -> str:
        if not self.use_color:
            return text
        return f"{code}{text}{ANSI_RESET}"

    def info(self, msg: str) -> None:
        """Print standard information message."""
        print(msg, file=self.out_stream)

    def success(self, msg: str) -> None:
        """Print green success message with checkmark."""
        prefix = self._color(ANSI_GREEN + ANSI_BOLD, "✓ ")
        print(f"{prefix}{msg}", file=self.out_stream)

    def warning(self, msg: str) -> None:
        """Print yellow warning message with triangle symbol."""
        prefix = self._color(ANSI_YELLOW + ANSI_BOLD, "▲ ")
        print(f"{prefix}{msg}", file=self.err_stream)

    def error(self, msg: str) -> None:
        """Print red error message with cross symbol."""
        prefix = self._color(ANSI_RED + ANSI_BOLD, "✗ ")
        print(f"{prefix}{msg}", file=self.err_stream)

    def section(self, title: str) -> None:
        """Print stylized section header."""
        header = f"=== {title} ==="
        print(f"\n{self._color(ANSI_CYAN + ANSI_BOLD, header)}", file=self.out_stream)


def print_issues(
    issues: Sequence[Issue],
    max_shown: int = 20,
    file: TextIO | None = None,
) -> None:
    """Print Ren'Py syntax and translation issues to target stream."""
    out = file if file is not None else sys.stderr
    for issue in issues[:max_shown]:
        sev = getattr(issue.severity, "value", str(issue.severity))
        print(
            f"  {issue.file}:{issue.line}: {sev} [{issue.code}] {issue.message}",
            file=out,
        )
    if len(issues) > max_shown:
        print(f"  ... và {len(issues) - max_shown} vấn đề khác", file=out)


def print_extract_report(report: ExtractReport, plain: bool = False) -> None:
    """Print standardized summary of script extraction."""
    print(
        f"Đã quét {report.files} tệp, {report.units} đơn vị dịch, {report.slots} vị trí ghi"
    )
    print(
        f"  mới {report.created}, đổi nguồn {report.source_changed}, "
        f"nhập bản dịch có sẵn {report.imported}, khôi phục {report.restored}, "
        f"lỗi thời {report.obsolete}"
    )
    if report.issues:
        print(f"  {report.errors} lỗi, {report.warnings} cảnh báo:", file=sys.stderr)
        print_issues(report.issues, max_shown=20, file=sys.stderr)


def print_apply_report(report: ApplyReport, plain: bool = False) -> None:
    """Print standardized summary of translation write-back."""
    verb = "Sẽ ghi" if report.dry_run else "Đã ghi"
    print(f"{verb} {report.lines_written} dòng vào {report.files_changed} tệp")
    print(f"  giữ nguyên {report.unchanged} dòng đã khớp")
    if report.backup_dir is not None:
        print(f"  bản sao lưu: {report.backup_dir}")


def print_translation_stats(stats: TranslationStats, plain: bool = False) -> None:
    """Print comprehensive summary of translation execution."""
    print("\nKết quả dịch:")
    print(f"  tổng số cần dịch       : {stats.total_pending}")
    print(f"  chuỗi UI có sẵn (seed) : {stats.seeded}")
    print(f"  tái sử dụng từ TM      : {stats.tm_reused}")
    print(f"  dịch mới qua LLM       : {stats.llm_translated}")
    if stats.failed:
        print(f"  thất bại / cần xem xét : {stats.failed}")
    print(f"  số lô thực hiện        : {stats.batches_count}")
    print(f"  token vào/ra           : {stats.tokens_in} / {stats.tokens_out}")
    if stats.errors:
        print(f"  {len(stats.errors)} lỗi trong quá trình dịch:", file=sys.stderr)
        for err in stats.errors[:5]:
            print(f"    - {err}", file=sys.stderr)
