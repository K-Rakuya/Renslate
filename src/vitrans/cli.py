"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from vitrans import __version__
from vitrans.errors import SdkError, VitransError
from vitrans.extract import ExtractReport, extract
from vitrans.parser import Issue
from vitrans.project import Project
from vitrans.sdk import generate_translations
from vitrans.writer import ApplyReport, apply_translations

_MAX_ISSUES_SHOWN = 20
_LOG = logging.getLogger("vitrans")

Handler = Callable[[argparse.Namespace], int]


def _root(args: argparse.Namespace) -> Path:
    return Path(args.project) if args.project else Path.cwd()


def _print_issues(issues: Sequence[Issue]) -> None:
    for issue in issues[:_MAX_ISSUES_SHOWN]:
        print(
            f"  {issue.file}:{issue.line}: {issue.severity.value} [{issue.code}] {issue.message}",
            file=sys.stderr,
        )
    if len(issues) > _MAX_ISSUES_SHOWN:
        print(f"  ... và {len(issues) - _MAX_ISSUES_SHOWN} vấn đề khác", file=sys.stderr)


def _cmd_init(args: argparse.Namespace) -> int:
    project = Project.create(_root(args), target_lang=args.lang, sdk=args.sdk, force=args.force)
    print(f"Đã khởi tạo dự án tại {project.root}")
    print(f"  cấu hình       : {project.relative(project.config_path)}")
    print(f"  cơ sở dữ liệu  : {project.relative(project.db_path)}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    sdk = args.sdk or project.config.sdk
    if not sdk:
        raise SdkError("thiếu đường dẫn Ren'Py SDK: dùng --sdk hoặc đặt 'sdk' trong cấu hình")
    output = generate_translations(Path(sdk).expanduser(), project.root, project.config.target_lang)
    _LOG.info("%s", output)
    print(f"Đã sinh khung dịch trong {project.relative(project.tl_root)}")
    return 0


def _print_extract(report: ExtractReport) -> None:
    print(f"Đã quét {report.files} tệp, {report.units} đơn vị dịch, {report.slots} vị trí ghi")
    print(
        f"  mới {report.created}, đổi nguồn {report.source_changed}, "
        f"nhập bản dịch có sẵn {report.imported}, khôi phục {report.restored}, "
        f"lỗi thời {report.obsolete}"
    )
    if report.issues:
        print(f"  {report.errors} lỗi, {report.warnings} cảnh báo:", file=sys.stderr)
        _print_issues(report.issues)


def _cmd_extract(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    with project.open_db() as database:
        report = extract(project, database)
    _print_extract(report)
    return 1 if report.errors else 0


def _print_apply(report: ApplyReport) -> None:
    verb = "Sẽ ghi" if report.dry_run else "Đã ghi"
    print(f"{verb} {report.lines_written} dòng vào {report.files_changed} tệp")
    print(f"  giữ nguyên {report.unchanged} dòng đã khớp")
    if report.backup_dir is not None:
        print(f"  bản sao lưu: {report.backup_dir}")


def _cmd_apply(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    with project.open_db() as database:
        report = apply_translations(project, database, dry_run=args.dry_run)
    _print_apply(report)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    with project.open_db() as database:
        rows = database.stats()
        slots = database.count_slots()
    print(f"Dự án: {project.root}")
    print(f"Ngôn ngữ đích: {project.config.target_lang}")
    if not rows:
        print("Chưa có dữ liệu, hãy chạy 'vitrans extract'.")
        return 0
    print(f"{'loại':<10} {'trạng thái':<12} {'số lượng':>8}")
    for kind, status, total in rows:
        print(f"{kind.value:<10} {status.value:<12} {total:>8}")
    print(f"Tổng: {sum(total for _, _, total in rows)} đơn vị, {slots} vị trí ghi")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-p", "--project", help="thư mục dự án Ren'Py (mặc định: thư mục hiện tại)")
    common.add_argument("-v", "--verbose", action="store_true", help="hiển thị nhật ký chi tiết")

    parser = argparse.ArgumentParser(
        prog="vitrans", description="Bộ công cụ dịch game Ren'Py bằng model local."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="lệnh")

    def add(name: str, handler: Handler, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, parents=[common], help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        return sub

    init = add("init", _cmd_init, "khởi tạo dự án dịch")
    init.add_argument("--lang", default="vietnamese", help="tên ngôn ngữ trong Ren'Py")
    init.add_argument("--sdk", help="đường dẫn Ren'Py SDK, lưu vào cấu hình")
    init.add_argument("--force", action="store_true", help="ghi đè cấu hình hiện có")

    generate = add("generate", _cmd_generate, "sinh khung dịch bằng Ren'Py SDK")
    generate.add_argument("--sdk", help="đường dẫn Ren'Py SDK (ghi đè cấu hình)")

    add("extract", _cmd_extract, "đọc tệp dịch vào cơ sở dữ liệu")
    apply = add("apply", _cmd_apply, "ghi bản dịch vào tệp dịch của game")
    apply.add_argument("--dry-run", action="store_true", help="chỉ báo cáo, không ghi tệp")
    add("status", _cmd_status, "thống kê tiến độ")
    return parser


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _configure_streams()
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s"
    )
    try:
        return int(args.handler(args))
    except VitransError as error:
        print(f"lỗi: {error}", file=sys.stderr)
        return 1
