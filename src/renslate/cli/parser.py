"""Ergonomic command-line argument parser with 2026 literary quality controls."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from renslate import __version__
from renslate.cli.handlers import (
    cmd_analyze,
    cmd_apply,
    cmd_bootstrap,
    cmd_clean_db,
    cmd_dashboard,
    cmd_decompile,
    cmd_export,
    cmd_extract,
    cmd_font,
    cmd_generate,
    cmd_import,
    cmd_init,
    cmd_preview,
    cmd_qa,
    cmd_run,
    cmd_status,
    cmd_translate,
    cmd_wizard,
)

Handler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Build the root argument parser with hierarchical subcommand definitions."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-p", "--project", help="thư mục dự án Ren'Py (mặc định: thư mục hiện tại)")
    common.add_argument("-v", "--verbose", action="store_true", help="hiển thị nhật ký chi tiết")

    parser = argparse.ArgumentParser(
        prog="renslate",
        description="Bộ công cụ dịch game Ren'Py bằng model local và kiến trúc văn học 2026.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="lệnh")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    def add(name: str, handler: Handler, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, parents=[common], help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        return sub

    # -------------------------------------------------------------------------
    # 1. Project Lifecycle & Extraction
    # -------------------------------------------------------------------------
    init = add("init", cmd_init, "khởi tạo dự án dịch")
    init.add_argument("--lang", default="vietnamese", help="tên ngôn ngữ trong Ren'Py")
    init.add_argument("--sdk", help="đường dẫn Ren'Py SDK, lưu vào cấu hình")
    init.add_argument("--force", action="store_true", help="ghi đè cấu hình hiện có")

    generate = add("generate", cmd_generate, "sinh khung dịch bằng Ren'Py SDK")
    generate.add_argument("--sdk", help="đường dẫn Ren'Py SDK (ghi đè cấu hình)")

    add("extract", cmd_extract, "đọc tệp dịch vào cơ sở dữ liệu")

    decompile = add(
        "decompile", cmd_decompile, "dịch ngược tệp bytecode .rpyc thành mã nguồn .rpy"
    )
    decompile.add_argument(
        "target", nargs="?", help="tệp .rpyc hoặc thư mục chứa bytecode cần dịch ngược"
    )
    decompile.add_argument("-o", "--output", help="đường dẫn tệp .rpy hoặc thư mục lưu kết quả")
    decompile.add_argument(
        "--overwrite", action="store_true", help="ghi đè tệp .rpy nếu đã tồn tại"
    )

    # -------------------------------------------------------------------------
    # 2. Discourse Analysis & Bootstrapping
    # -------------------------------------------------------------------------
    analyze = add("analyze", cmd_analyze, "phân tích nhân vật và phân cảnh")
    analyze.add_argument(
        "--bootstrap",
        action="store_true",
        help="tự động gọi LLM để phân tích nhân vật",
    )

    bootstrap = add(
        "bootstrap", cmd_bootstrap, "tự động làm giàu hồ sơ nhân vật bằng LLM"
    )
    bootstrap.add_argument(
        "--top-chars", type=int, default=15, help="số nhân vật chính cần phân tích (mặc định: 15)"
    )
    bootstrap.add_argument(
        "--top-pairs",
        type=int,
        default=30,
        help="số cặp nhân vật cần xác định quan hệ/xưng hô bằng LLM (0 để bỏ qua; mặc định: 30)",
    )
    bootstrap.add_argument(
        "--refresh-relationships",
        action="store_true",
        help="làm lại cả những cặp đã được LLM xác định trước đó",
    )
    bootstrap.add_argument(
        "--top-scenes",
        type=int,
        default=80,
        help="số phân cảnh cần tóm tắt bằng LLM làm ngữ cảnh (0 để bỏ qua; mặc định: 80)",
    )
    bootstrap.add_argument(
        "--refresh-scenes", action="store_true", help="tóm tắt lại cả những cảnh đã có tóm tắt"
    )

    # -------------------------------------------------------------------------
    # 3. Translation Pipeline & 2026 Quality Flags
    # -------------------------------------------------------------------------
    translate = add("translate", cmd_translate, "dịch các dòng thoại bằng model local")
    translate.add_argument("--limit", type=int, help="giới hạn số dòng cần dịch")
    translate.add_argument("--batch-size", type=int, help="số dòng tối đa mỗi lô (ghi đè config)")
    translate.add_argument("--temperature", type=float, help="temperature sinh mẫu (ghi đè config)")
    translate.add_argument(
        "--retry-failed", action="store_true", help="thử dịch lại các dòng từng bị lỗi"
    )
    translate.add_argument(
        "--guard",
        action="store_true",
        help="thẩm định LLM Hallucination Guard liên tục (mặc định đã bật)",
    )
    translate.add_argument(
        "--no-guard",
        action="store_true",
        help="tắt giai đoạn LLM Hallucination Guard quét ảo giác liên tục để tăng tốc độ",
    )
    translate.add_argument(
        "--concise-tags",
        action="store_true",
        help="rút gọn nhãn thoại từng dòng (bỏ lặp lại quan hệ) để giảm 25%% token prompt",
    )
    # Modern 2026 Quality Flags
    translate.add_argument(
        "--deliberate",
        action="store_true",
        help="kích hoạt 2-pass deliberative reflection cho các câu thoại calque cao",
    )
    translate.add_argument(
        "--no-deliberate",
        action="store_true",
        help="tắt cơ chế 2-pass reflection, chỉ chạy 1-pass trực tiếp",
    )
    translate.add_argument(
        "--calque-threshold",
        type=float,
        default=0.35,
        help="ngưỡng calque density để kích hoạt deliberative reflection (mặc định: 0.35)",
    )
    translate.add_argument(
        "--exemplars",
        type=int,
        default=2,
        help="số lượng câu thoại mẫu tương đồng rút từ TM cho cặp nhân vật (mặc định: 2)",
    )
    translate.add_argument(
        "--narrative-memory",
        type=int,
        default=6,
        help="dung lượng bộ nhớ ring buffer ngữ cảnh phân cảnh (mặc định: 6 lượt thoại)",
    )
    translate.add_argument(
        "--transgraph",
        action="store_true",
        help="kích hoạt TransGraph micro-framing 4D (mặc định: bật)",
    )
    translate.add_argument(
        "--no-transgraph",
        action="store_true",
        help="tắt TransGraph micro-framing 4D",
    )
    translate.add_argument(
        "--studio",
        action="store_true",
        help="mở Live Translation Studio xem full câu thoại và sửa tại chỗ",
    )

    # -------------------------------------------------------------------------
    # 4. Quality Assurance & Audit
    # -------------------------------------------------------------------------
    qa = add("qa", cmd_qa, "kiểm tra chất lượng và phát hiện lỗi bản dịch")
    qa.add_argument(
        "--guard",
        action="store_true",
        help="chạy quét ảo giác sâu bằng LLM Hallucination Guard trên các dòng đã dịch",
    )
    qa.add_argument(
        "--limit",
        type=int,
        help="giới hạn số dòng thoại cần quét ảo giác bằng LLM",
    )

    export = add("export", cmd_export, "xuất bản dịch ra TSV để review hoặc chỉnh sửa")
    export.add_argument("-o", "--output", help="đường dẫn tệp TSV xuất ra")
    export.add_argument("--qa-only", action="store_true", help="chỉ xuất các dòng có gắn cờ QA")
    export.add_argument("--pending-only", action="store_true", help="chỉ xuất các dòng chưa dịch")

    import_cmd = add("import", cmd_import, "nhập bản dịch đã sửa tay từ tệp TSV")
    import_cmd.add_argument("-i", "--input", required=True, help="đường dẫn tệp TSV cần nhập")

    apply = add("apply", cmd_apply, "ghi bản dịch vào tệp dịch của game")
    apply.add_argument("--dry-run", action="store_true", help="chỉ báo cáo, không ghi tệp")
    apply.add_argument(
        "--setup-font",
        action="store_true",
        help="tạo tệp cấu hình phông chữ tiếng Việt renslate_setup.rpy",
    )
    apply.add_argument("--sans-font", help="đường dẫn phông Sans-serif tùy chỉnh")
    apply.add_argument("--serif-font", help="đường dẫn phông Serif tùy chỉnh")

    font_cmd = add("font", cmd_font, "quét phông chữ và thiết lập hiển thị tiếng Việt cho Ren'Py")
    font_cmd.add_argument(
        "--inspect",
        "--scan",
        dest="inspect",
        action="store_true",
        help="chỉ kiểm tra phông chữ mà không ghi tệp cấu hình",
    )
    font_cmd.add_argument("--sans-font", help="đường dẫn phông Sans-serif tùy chỉnh")
    font_cmd.add_argument("--serif-font", help="đường dẫn phông Serif tùy chỉnh")
    font_cmd.add_argument(
        "--no-replace",
        action="store_true",
        help="không tự động tạo font_replacement_map thay thế phông gốc",
    )

    status_cmd = add("status", cmd_status, "thống kê tiến độ")
    status_cmd.add_argument(
        "-d", "--dashboard", action="store_true", help="hiển thị TUI dashboard trực quan"
    )
    status_cmd.add_argument("--plain", action="store_true", help="không dùng màu ANSI")

    dash_cmd = add(
        "dashboard",
        cmd_dashboard,
        "bảng điều khiển CLI thời gian thực trực quan của dự án",
    )
    dash_cmd.add_argument("--plain", action="store_true", help="không dùng màu ANSI")
    dash_cmd.add_argument(
        "--once",
        "--snapshot",
        dest="once",
        action="store_true",
        help="chỉ hiển thị 1 ảnh chụp trạng thái rồi thoát ngay (dùng cho script/CI)",
    )
    dash_cmd.add_argument(
        "-i",
        "--interval",
        type=float,
        default=2.0,
        help="tần suất làm mới dữ liệu tính bằng giây (mặc định: 2.0s)",
    )
    dash_cmd.add_argument(
        "-n",
        "--limit",
        type=int,
        default=5,
        help="số lượng câu thoại gần đây hiển thị (mặc định: 5)",
    )
    dash_cmd.add_argument(
        "-f",
        "--filter",
        choices=["all", "dialogue", "flagged"],
        default="all",
        help="bộ lọc dòng thoại ban đầu: all, dialogue, hoặc flagged",
    )

    preview = add("preview", cmd_preview, "xem trước các dòng thoại đã dịch song ngữ")
    preview.add_argument(
        "-n", "--limit", type=int, default=10, help="số dòng hiển thị (mặc định: 10)"
    )
    preview.add_argument(
        "--latest", action="store_true", help="hiển thị các dòng dịch gần đây nhất"
    )
    preview.add_argument(
        "--status", choices=["translated", "approved", "failed"], help="lọc theo trạng thái"
    )

    add("clean-db", cmd_clean_db, "chuẩn hóa và làm sạch nhãn người nói / ngoặc kép trong DB")

    # -------------------------------------------------------------------------
    # 5. Autonomous Runner & Interactive Studio
    # -------------------------------------------------------------------------
    def _add_run_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--chunk-size",
            type=int,
            default=100,
            help="số dòng mỗi chu kỳ kiểm tra (mặc định: 100)",
        )
        sub.add_argument(
            "--retry-failed",
            action="store_true",
            help="thử lại các dòng từng bị lỗi (failed)",
        )
        sub.add_argument(
            "--retry-flagged",
            action="store_true",
            help="đặt lại các dòng có cờ QA cảnh báo (qa_flagged) để dịch lại",
        )
        sub.add_argument(
            "--auto-apply",
            action="store_true",
            help="tự động apply bản dịch vào game sau khi hoàn thành",
        )
        sub.add_argument(
            "--plain",
            "--no-dashboard",
            action="store_true",
            help="không sử dụng ANSI dashboard động",
        )
        sub.add_argument(
            "-i",
            "--interactive",
            action="store_true",
            help="chạy chế độ thiết lập wizard tương tác",
        )
        sub.add_argument(
            "--bootstrap",
            action="store_true",
            help="bắt buộc LLM bootstrapping lại cho hồ sơ nhân vật trước khi dịch",
        )
        sub.add_argument(
            "--no-bootstrap",
            action="store_true",
            help="bỏ qua bước LLM bootstrapping cho nhân vật",
        )
        sub.add_argument(
            "--no-guard",
            action="store_true",
            help="tắt giai đoạn LLM Hallucination Guard quét ảo giác liên tục",
        )
        sub.add_argument(
            "--concise-tags",
            action="store_true",
            help="rút gọn nhãn thoại từng dòng (bỏ lặp lại quan hệ) để giảm 25%% token prompt",
        )
        sub.add_argument(
            "--studio",
            action="store_true",
            help="chạy chế độ Real-Time Translation Studio xem và sửa trực tiếp",
        )

    run_cmd = add("run", cmd_run, "chạy dịch tự động liên tục với TUI dashboard")
    _add_run_args(run_cmd)

    auto_cmd = add("auto", cmd_run, "chạy dịch tự động liên tục (alias cho run)")
    _add_run_args(auto_cmd)

    def _add_wizard_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--menu",
            "--studio",
            dest="menu",
            action="store_true",
            help="chạy chế độ menu studio trực quan 6 pha",
        )
        p.add_argument(
            "--classic",
            dest="classic",
            action="store_true",
            help="chạy chế độ hỏi đáp tuần tự cổ điển",
        )

    wiz_cmd = add("wizard", cmd_wizard, "trình hướng dẫn thiết lập tương tác và menu studio")
    _add_wizard_args(wiz_cmd)
    interactive_cmd = add("interactive", cmd_wizard, "trình hướng dẫn thiết lập tương tác (alias)")
    _add_wizard_args(interactive_cmd)

    return parser
