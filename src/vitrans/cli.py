"""Command-line interface."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from vitrans import __version__
from vitrans.analyze import run_analysis
from vitrans.errors import SdkError, VitransError
from vitrans.extract import ExtractReport, extract
from vitrans.mask import mask
from vitrans.models import Status
from vitrans.parser import Issue
from vitrans.project import Project
from vitrans.qa import QAChecker
from vitrans.sdk import generate_translations
from vitrans.translate import TranslationPipeline
from vitrans.tsvio import export_tsv, import_tsv
from vitrans.writer import ApplyReport, apply_translations, setup_vietnamese_support

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


def _cmd_analyze(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    llm = None
    if getattr(args, "bootstrap", False):
        from vitrans.llm import LLMClient

        llm = LLMClient(project.config.llm)
    with project.open_db() as database:
        stats = run_analysis(
            project,
            database,
            mine_glossary=not args.no_glossary,
            enrich_llm=getattr(args, "bootstrap", False),
            llm=llm,
        )
    print(f"Đã phân tích dự án tại {project.root}:")
    print(f"  nhân vật phát hiện : {stats['characters']}")
    print(f"  phân đoạn cảnh     : {stats['scenes']}")
    print(f"  ứng viên glossary  : {stats['glossary_candidates']}")
    if stats.get("glossary_translated"):
        print(f"  thuật ngữ đã dịch  : {stats['glossary_translated']}")
    return 0


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    from vitrans.analyze import bootstrap_characters_with_llm
    from vitrans.glossary import Glossary, translate_glossary_with_llm
    from vitrans.llm import LLMClient

    llm = LLMClient(project.config.llm)
    print(f"Bắt đầu LLM bootstrapping cho dự án tại {project.root}...")
    with project.open_db() as database:
        # 1. Characters
        print(f"  Đang phân tích hồ sơ nhân vật (tối đa {args.top_chars} nhân vật)...")
        chars = bootstrap_characters_with_llm(project, database, llm, max_chars=args.top_chars)
        print(f"  -> Hoàn tất hồ sơ cho {len(chars)} nhân vật.")

        # 2. Glossary
        glossary_file = project.work_dir / "glossary.json"
        glossary = Glossary.load_db(database)
        if not glossary.entries and glossary_file.is_file():
            glossary = Glossary.load_file(glossary_file)

        from vitrans.glossary import clean_glossary
        from vitrans.models import GlossaryEntry

        char_names = {
            p.display_name.lower() for p in chars.values() if p.display_name
        } | set(chars.keys())
        cleaned_entries = clean_glossary(glossary.entries, exclude_terms=char_names)

        untranslated = [e for e in cleaned_entries if not e.translation.strip()]
        if untranslated:
            def _freq(e: GlossaryEntry) -> int:
                m = re.search(r"(\d+)\s*lần", e.notes)
                return int(m.group(1)) if m else 1

            untranslated.sort(key=_freq, reverse=True)
            to_trans = untranslated[: args.top_terms]
            total_untrans = len(untranslated)
            print(f"  Đang dịch thuật ngữ bằng LLM ({len(to_trans)}/{total_untrans} ứng viên)...")
            enriched = translate_glossary_with_llm(to_trans, llm)
            trans_map = {e.term: e for e in enriched if e.translation.strip()}
            final_entries = [trans_map.get(e.term, e) for e in cleaned_entries]
            g_final = Glossary(final_entries)
            g_final.save_file(glossary_file)
            g_final.save_db(database)
            translated_count = sum(1 for e in final_entries if e.translation.strip())
            print(f"  -> Đã dịch {translated_count}/{len(final_entries)} thuật ngữ.")
        else:
            print("  -> Bảng thuật ngữ đã có bản dịch đầy đủ hoặc chưa có ứng viên.")

    print("Hoàn tất bootstrap! characters.json và glossary.json đã được cập nhật thành công.")
    return 0


def _cmd_translate(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    config = project.config
    if args.batch_size:
        batch_cfg = dataclasses.replace(config.batch, max_lines=args.batch_size)
        config = dataclasses.replace(config, batch=batch_cfg)
    if args.temperature is not None:
        llm_cfg = dataclasses.replace(config.llm, temperature=args.temperature)
        config = dataclasses.replace(config, llm=llm_cfg)

    def _progress(current: int, total: int, msg: str) -> None:
        print(f"[{current}/{total}] {msg}")

    with project.open_db() as database:
        pipeline = TranslationPipeline(project, database, config)
        print(f"Bắt đầu dịch bằng backend {config.llm.base_url} ({config.llm.model})...")
        stats = pipeline.run(
            limit=args.limit,
            include_failed=args.retry_failed,
            progress_cb=_progress,
        )

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
        return 1
    return 0


def _cmd_qa(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    qa_checker = QAChecker(max_length_ratio=project.config.qa.max_length_ratio)
    issues_found = 0
    errors_found = 0

    with project.open_db() as database:
        units = database.all_units()
        glossary = database.get_glossary()

        for u in units:
            if not u.target or u.status.value == "obsolete":
                continue
            masked = mask(u.source) if u.source else None
            issues = qa_checker.check_unit(
                u.uid,
                u.source,
                masked,
                u.target,
                glossary_matches=glossary,
                speaker=u.speaker,
            )
            if issues:
                issues_found += len(issues)
                for issue in issues:
                    if issue.severity == "error":
                        errors_found += 1
                    print(
                        f"  [{issue.severity.upper()}] {issue.uid} ({issue.code}): {issue.message}",
                        file=sys.stderr if issue.severity == "error" else sys.stdout,
                    )

    if issues_found == 0:
        print("Tất cả bản dịch đều vượt qua kiểm tra chất lượng (QA pass).")
        return 0

    print(f"\nTổng kết: {issues_found} vấn đề QA ({errors_found} lỗi nghiêm trọng).")
    return 1 if errors_found > 0 else 0


def _cmd_export(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    out_path = Path(args.output) if args.output else project.work_dir / "review.tsv"
    with project.open_db() as database:
        count = export_tsv(
            database,
            out_path,
            qa_only=args.qa_only,
            status_filter=Status.PENDING if args.pending_only else None,
        )
    print(f"Đã xuất {count} dòng ra {project.relative(out_path)}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"lỗi: không tìm thấy tệp {in_path}", file=sys.stderr)
        return 1

    with project.open_db() as database:
        imported, skipped, errors = import_tsv(database, in_path)

    print(f"Đã nhập {imported} bản dịch mới/sửa đổi (bỏ qua {skipped} dòng không đổi).")
    if errors:
        print(f"  {len(errors)} lỗi khi nhập:", file=sys.stderr)
        for e in errors[:10]:
            print(f"    - {e}", file=sys.stderr)
        return 1
    return 0


def _print_apply(report: ApplyReport) -> None:
    verb = "Sẽ ghi" if report.dry_run else "Đã ghi"
    print(f"{verb} {report.lines_written} dòng vào {report.files_changed} tệp")
    print(f"  giữ nguyên {report.unchanged} dòng đã khớp")
    if report.backup_dir is not None:
        print(f"  bản sao lưu: {report.backup_dir}")


def _cmd_apply(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    if getattr(args, "setup_font", False):
        font_path = setup_vietnamese_support(project)
        print(f"Đã tạo cấu hình phông tiếng Việt: {project.relative(font_path)}")

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


def _cmd_preview(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    limit = args.limit or 10
    with project.open_db() as database:
        order = "DESC" if getattr(args, "latest", False) else "ASC"
        where_clauses = ["target IS NOT NULL", "target != ''"]
        params: list[Any] = []
        if getattr(args, "status", None):
            where_clauses.append("status = ?")
            params.append(args.status)

        where_sql = " AND ".join(where_clauses)
        query = (
            f"SELECT uid, speaker, source, target, origin, status FROM units "
            f"WHERE {where_sql} ORDER BY seq {order} LIMIT ?"
        )
        params.append(limit)
        rows = database.execute(query, params).fetchall()

        if not rows:
            print("Chưa có dòng nào được dịch (hoặc không khớp bộ lọc).")
            return 0

        direction_label = "mới nhất" if getattr(args, "latest", False) else "theo thứ tự kịch bản"
        print(
            f"\n=== Xem trước {len(rows)} dòng bản dịch ({direction_label}) "
            f"[{project.root.name}] ==="
        )
        for idx, row in enumerate(rows, 1):
            uid, speaker, source, target, origin, status = row
            spk_info = f"[{speaker}] " if speaker else ""
            print(f"\n({idx}) {uid} [{origin or 'manual'} / {status}]")
            print(f"  ENG: {spk_info}{source}")
            print(f"  VIE: {spk_info}{target}")
        print("\n" + "=" * 60)
    return 0


def _cmd_clean_db(args: argparse.Namespace) -> int:
    project = Project.open(_root(args))
    with project.open_db() as database:
        units_cleaned, tm_cleaned = database.sanitize_stored_translations()
        print(
            f"Đã làm sạch {units_cleaned} dòng thoại trong units "
            f"và {tm_cleaned} mục trong translation memory (TM)."
        )
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

    analyze = add("analyze", _cmd_analyze, "phân tích nhân vật, phân cảnh và thuật ngữ")
    analyze.add_argument(
        "--no-glossary", action="store_true", help="bỏ qua bước khai thác ứng viên thuật ngữ"
    )
    analyze.add_argument(
        "--bootstrap",
        action="store_true",
        help="tự động gọi LLM để phân tích nhân vật và dịch thuật ngữ",
    )

    bootstrap = add(
        "bootstrap", _cmd_bootstrap, "tự động làm giàu hồ sơ nhân vật và dịch thuật ngữ bằng LLM"
    )
    bootstrap.add_argument(
        "--top-chars", type=int, default=15, help="số nhân vật chính cần phân tích (mặc định: 15)"
    )
    bootstrap.add_argument(
        "--top-terms", type=int, default=100, help="số thuật ngữ cần dịch bằng LLM (mặc định: 100)"
    )

    translate = add("translate", _cmd_translate, "dịch các dòng thoại bằng model local")
    translate.add_argument("--limit", type=int, help="giới hạn số dòng cần dịch")
    translate.add_argument("--batch-size", type=int, help="số dòng tối đa mỗi lô (ghi đè config)")
    translate.add_argument("--temperature", type=float, help="temperature sinh mẫu (ghi đè config)")
    translate.add_argument(
        "--retry-failed", action="store_true", help="thử dịch lại các dòng từng bị lỗi"
    )

    add("qa", _cmd_qa, "kiểm tra chất lượng và phát hiện lỗi bản dịch")

    export = add("export", _cmd_export, "xuất bản dịch ra TSV để review hoặc chỉnh sửa")
    export.add_argument("-o", "--output", help="đường dẫn tệp TSV xuất ra")
    export.add_argument("--qa-only", action="store_true", help="chỉ xuất các dòng có gắn cờ QA")
    export.add_argument("--pending-only", action="store_true", help="chỉ xuất các dòng chưa dịch")

    import_cmd = add("import", _cmd_import, "nhập bản dịch đã sửa tay từ tệp TSV")
    import_cmd.add_argument("-i", "--input", required=True, help="đường dẫn tệp TSV cần nhập")

    apply = add("apply", _cmd_apply, "ghi bản dịch vào tệp dịch của game")
    apply.add_argument("--dry-run", action="store_true", help="chỉ báo cáo, không ghi tệp")
    apply.add_argument(
        "--setup-font",
        action="store_true",
        help="tạo tệp cấu hình phông chữ tiếng Việt vitrans_setup.rpy",
    )

    add("status", _cmd_status, "thống kê tiến độ")

    preview = add("preview", _cmd_preview, "xem trước các dòng thoại đã dịch song ngữ")
    preview.add_argument(
        "-n", "--limit", type=int, default=10, help="số dòng hiển thị (mặc định: 10)"
    )
    preview.add_argument(
        "--latest", action="store_true", help="hiển thị các dòng dịch gần đây nhất"
    )
    preview.add_argument(
        "--status", choices=["translated", "approved", "failed"], help="lọc theo trạng thái"
    )

    add("clean-db", _cmd_clean_db, "chuẩn hóa và làm sạch nhãn người nói / ngoặc kép trong DB")
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
