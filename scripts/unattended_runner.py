#!/usr/bin/env python3
"""Unattended automated runner for ViTrans translation pipelines.

Standardized, low-overhead execution for Ren'Py visual novel translations.
Features:
- Real-time database progress tracking (accurate counts, percentage, ETA)
- Automatic active model detection from llama.cpp /props and /v1/models
- High-resilience loop with health monitoring and safe resume
- Clean, transparent CLI dashboard & batch logging
- Safe, instant Ctrl+C signal handling with zero database corruption
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Ensure src/ is in sys.path when executed directly
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from vitrans.analyze import bootstrap_characters_with_llm  # noqa: E402
from vitrans.db import Database  # noqa: E402
from vitrans.errors import ProjectError  # noqa: E402
from vitrans.llm import LLMClient  # noqa: E402
from vitrans.project import Project  # noqa: E402
from vitrans.translate import TranslationPipeline  # noqa: E402

# Configure standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_LOG = logging.getLogger("vitrans.runner")

# Color helpers for TTY terminals
IS_TTY = sys.stdout.isatty()
C_RESET = "\033[0m" if IS_TTY else ""
C_BOLD = "\033[1m" if IS_TTY else ""
C_GREEN = "\033[0;32m" if IS_TTY else ""
C_CYAN = "\033[0;36m" if IS_TTY else ""
C_YELLOW = "\033[1;33m" if IS_TTY else ""
C_RED = "\033[0;31m" if IS_TTY else ""
C_GRAY = "\033[0;90m" if IS_TTY else ""

_STOP_REQUESTED = False
_INTERRUPT_COUNT = 0


def _sig_handler(signum: int, frame: Any) -> None:
    global _STOP_REQUESTED, _INTERRUPT_COUNT
    _INTERRUPT_COUNT += 1
    if _INTERRUPT_COUNT >= 2:
        _LOG.warning("\n[BUỘC DỪNG] Đã nhận tín hiệu ngắt lần thứ 2. Thoát ngay lập tức!")
        os._exit(130)

    _STOP_REQUESTED = True
    _LOG.warning(
        "\n%s[DỪNG TIẾN TRÌNH]%s Đã nhận lệnh dừng (Ctrl+C). "
        "Đang hoàn tất lô hiện tại và lưu dữ liệu an toàn...",
        C_YELLOW,
        C_RESET,
    )


def check_server_health(base_url: str, timeout_s: float = 3.0) -> bool:
    """Check if the local llama-server is online and responsive."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ViTrans-Runner"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, TimeoutError, OSError):
        try:
            url_models = f"{base_url.rstrip('/')}/models"
            req_m = urllib.request.Request(url_models, headers={"User-Agent": "ViTrans-Runner"})
            with urllib.request.urlopen(req_m, timeout=timeout_s) as resp_m:
                return bool(resp_m.status == 200)
        except Exception:
            return False


def detect_server_model(base_url: str, default_name: str, timeout_s: float = 3.0) -> str:
    """Query the active model filename from llama-server's /props or /v1/models."""
    # 1. Query native /props endpoint
    try:
        props_url = f"{base_url.rstrip('/')}/props"
        req = urllib.request.Request(props_url, headers={"User-Agent": "ViTrans-Runner"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict):
                model_path = data.get("default_generation_settings", {}).get("model")
                if model_path:
                    return Path(model_path).name
    except Exception:
        pass

    # 2. Query OpenAI-compatible /v1/models endpoint
    try:
        models_url = f"{base_url.rstrip('/')}/models"
        req = urllib.request.Request(models_url, headers={"User-Agent": "ViTrans-Runner"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if (
                isinstance(data, dict)
                and "data" in data
                and isinstance(data["data"], list)
                and data["data"]
            ):
                m_id = data["data"][0].get("id")
                if m_id:
                    return Path(m_id).name
    except Exception:
        pass

    return default_name


def wait_for_server(base_url: str, check_interval: float = 4.0) -> bool:
    """Wait until the LLM server becomes responsive, respecting stop requests."""
    first = True
    while not _STOP_REQUESTED:
        if check_server_health(base_url):
            if not first:
                _LOG.info("%sLLM server đã phục hồi trực tuyến!%s", C_GREEN, C_RESET)
            return True
        if first:
            _LOG.warning("LLM server (%s) hiện không phản hồi. Đang đợi máy chủ...", base_url)
            first = False
        time.sleep(check_interval)
    return False


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
    """Render a Unicode progress bar: [████████░░░░░░░░░░░░]."""
    clamped_pct = max(0.0, min(100.0, pct))
    filled_chars = round(width * clamped_pct / 100.0)
    bar = "█" * filled_chars + "░" * (width - filled_chars)
    return f"[{bar}] {clamped_pct:5.1f}%"


def query_db_status(db: Database) -> dict[str, int]:
    """Retrieve accurate status counts across all units in the database."""
    cur = db.execute("SELECT status, COUNT(*) FROM units GROUP BY status")
    return {str(row[0]): int(row[1]) for row in cur.fetchall()}


def run_unattended_pipeline(
    project_root: Path,
    *,
    batch_chunk: int = 100,
    retry_failed: bool = False,
    retry_flagged: bool = False,
    auto_apply: bool = False,
) -> int:
    """Execute the translation process unattended in chunks until finished."""
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        project = Project.open(project_root)
    except ProjectError as err:
        if (project_root / "game").is_dir():
            _LOG.info("Phát hiện dự án Ren'Py chưa được khởi tạo. Đang tự động 'init'...")
            project = Project.create(project_root)
            _LOG.info("Đã khởi tạo dự án tại %s", project.root)
        else:
            raise err

    config = project.config

    if not wait_for_server(config.llm.base_url):
        _LOG.info("Đã dừng runner theo yêu cầu.")
        return 0

    active_model = detect_server_model(config.llm.base_url, config.llm.model)

    with project.open_db() as db:
        # Check if translation directory exists; auto-generate if possible
        if not project.tl_root.is_dir():
            _LOG.info(
                "Thư mục dịch %s chưa tồn tại. "
                "Đang tìm trình khởi chạy để tự động sinh khung dịch...",
                project.tl_root,
            )
            from vitrans.errors import SdkError
            from vitrans.sdk import find_launcher, generate_translations

            sdk_candidate = (
                Path(project.config.sdk).expanduser() if project.config.sdk else project.root
            )
            try:
                launcher = find_launcher(sdk_candidate)
                _LOG.info("Tìm thấy trình khởi chạy: %s. Đang sinh khung dịch...", launcher)
                generate_translations(launcher.parent, project.root, project.config.target_lang)
                _LOG.info("Đã sinh khung dịch trong %s", project.tl_root)
            except SdkError as s_err:
                _LOG.error(
                    "Không thể tự động sinh khung dịch: %s\n"
                    "Lý do: Ren'Py cần biên dịch script thành khung dịch trong 'game/tl/%s/'.\n"
                    "Giải pháp:\n"
                    "  1. Nếu máy đã có Ren'Py SDK:\n"
                    '     .venv/bin/vitrans generate -p "%s" --sdk /path/to/renpy-sdk\n'
                    "  2. Hoặc mở game bằng Ren'Py Launcher -> Chọn game -> "
                    "Nhấn 'Generate Translations' -> Chọn 'vietnamese'.",
                    s_err,
                    project.config.target_lang,
                    project.root,
                )
                return 1

        # Check if units exist; auto-extract if fresh project
        unit_count = db.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        if unit_count == 0:
            _LOG.info("Cơ sở dữ liệu chưa có đơn vị dịch. Đang tự động quét 'extract'...")
            from vitrans.extract import extract

            report = extract(project, db)
            _LOG.info("Đã quét %d tệp, trích xuất %d đơn vị dịch.", report.files, report.units)
            if report.units == 0:
                _LOG.warning(
                    "Không tìm thấy tệp bản dịch nào trong game/tl/%s hoặc game/.",
                    project.config.target_lang,
                )
                return 0

        # Check if analysis has been performed (characters & scenes); auto-analyze if empty
        char_count = db.execute("SELECT COUNT(*) FROM characters").fetchone()[0]
        if char_count == 0 and (project.root / "game").is_dir():
            _LOG.info("Đang tự động phân tích nhân vật, phân cảnh và khai thác glossary...")
            from vitrans.analyze import run_analysis

            analysis_rep = run_analysis(project, db)
            _LOG.info(
                "Phân tích hoàn tất: %d nhân vật, %d phân cảnh, %d thuật ngữ ứng viên.",
                analysis_rep.get("characters", 0),
                analysis_rep.get("scenes", 0),
                analysis_rep.get("glossary_candidates", 0),
            )

        # Autonomous LLM enrichment for character profiles and glossary terms
        chars = db.get_characters()
        glossary_entries = db.get_glossary()
        from vitrans.glossary import Glossary, clean_glossary, translate_glossary_with_llm

        char_names = {
            p.display_name.lower() for p in chars.values() if p.display_name
        } | set(chars.keys())
        cleaned_glossary = clean_glossary(glossary_entries, exclude_terms=char_names)
        if len(cleaned_glossary) != len(glossary_entries) or any(
            g.case_sensitive for g in glossary_entries
        ):
            # Clean and persist to DB/file immediately
            g_clean = Glossary(cleaned_glossary)
            g_clean.save_file(project.work_dir / "glossary.json")
            g_clean.save_db(db)
            glossary_entries = cleaned_glossary
        else:
            glossary_entries = cleaned_glossary

        untranslated_glossary = [g for g in glossary_entries if not g.translation.strip()]
        has_important_untranslated = any(
            (int(m.group(1)) if (m := re.search(r"(\d+)\s*lần", g.notes)) else 1) >= 5
            for g in untranslated_glossary
        )
        needs_char_enrichment = any(
            (c.gender == "unknown" or not c.role or not c.speech_style)
            for c in list(chars.values())[:10]
        )
        has_server = check_server_health(config.llm.base_url)

        if has_server and (
            needs_char_enrichment
            or has_important_untranslated
            or (glossary_entries and len(untranslated_glossary) == len(glossary_entries))
        ):
            _LOG.info("Phát hiện hồ sơ nhân vật hoặc thuật ngữ chưa được làm giàu bằng LLM.")
            _LOG.info(
                "Đang kích hoạt LLM bootstrapping tự động (hồ sơ nhân vật & dịch thuật ngữ)..."
            )
            llm_client = LLMClient(config.llm)
            if needs_char_enrichment:
                bootstrapped = bootstrap_characters_with_llm(project, db, llm_client, max_chars=25)
                enriched_count = sum(
                    1 for c in bootstrapped.values() if c.pronoun_map or c.speech_style
                )
                _LOG.info(
                    "Đã cập nhật hồ sơ và xưng hô chi tiết cho %d nhân vật chính tuyến.",
                    enriched_count,
                )

            if untranslated_glossary:
                def _freq(e: Any) -> int:
                    m = re.search(r"(\d+)\s*lần", e.notes)
                    return int(m.group(1)) if m else 1

                untranslated_glossary.sort(key=_freq, reverse=True)
                to_trans = untranslated_glossary[:150]
                _LOG.info("Đang dịch %d thuật ngữ phổ biến nhất bằng LLM...", len(to_trans))
                enriched = translate_glossary_with_llm(to_trans, llm_client)
                trans_map = {e.term: e for e in enriched if e.translation.strip()}
                final_entries = [trans_map.get(e.term, e) for e in glossary_entries]
                g_final = Glossary(final_entries)
                g_final.save_file(project.work_dir / "glossary.json")
                g_final.save_db(db)
                translated_count = sum(1 for e in final_entries if e.translation.strip())
                _LOG.info(
                    "Đã dịch tự động %d/%d thuật ngữ quan trọng bằng LLM.",
                    translated_count,
                    len(final_entries),
                )

        # Auto-sanitize any legacy corrupted speaker tags or wrapper quotes in database
        cleaned_units, cleaned_tm = db.sanitize_stored_translations()
        if cleaned_units > 0 or cleaned_tm > 0:
            _LOG.info(
                "Đã tự động chuẩn hóa định dạng: làm sạch %d dòng thoại "
                "và %d mục TM trong cơ sở dữ liệu.",
                cleaned_units,
                cleaned_tm,
            )

        if retry_flagged:
            sql_cnt = "SELECT COUNT(*) FROM units WHERE status = 'qa_flagged'"
            row_flagged = db.execute(sql_cnt).fetchone()
            count_flagged = row_flagged[0] if row_flagged else 0
            if count_flagged > 0:
                sql_reset = (
                    "UPDATE units SET status = 'pending', target = NULL, qa_flags = '[]' "
                    "WHERE status = 'qa_flagged'"
                )
                db.execute(sql_reset)
                db._connection.commit()
                _LOG.info(
                    "Đã đặt lại %d dòng cảnh báo QA (qa_flagged) về hàng đợi để dịch lại.",
                    count_flagged,
                )

        # Query baseline statistics from database
        initial_counts = query_db_status(db)
        total_units = sum(initial_counts.values())
        initial_translated = (
            initial_counts.get("translated", 0)
            + initial_counts.get("approved", 0)
            + initial_counts.get("qa_flagged", 0)
        )
        initial_pending = initial_counts.get("pending", 0)
        initial_failed = initial_counts.get("failed", 0)
        initial_pct = (initial_translated / total_units * 100.0) if total_units > 0 else 0.0

        # Display standardized startup banner
        banner_title = "ViTrans — Unattended Translation Runner".center(80)
        print(f"\n{C_BOLD}{C_CYAN}{'=' * 80}{C_RESET}")
        print(f"{C_BOLD}{banner_title}{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}{'=' * 80}{C_RESET}")
        print(f"  • Dự án          : {C_BOLD}{project.root}{C_RESET}")
        print(f"  • Ngôn ngữ đích  : {project.config.target_lang.upper()} (Tiếng Việt)")
        print(f"  • LLM Endpoint   : {config.llm.base_url}")
        print(f"  • Model kích hoạt: {C_GREEN}{C_BOLD}{active_model}{C_RESET}")
        print(f"  • Quy mô dự án   : {total_units:,} tổng số dòng")
        print(f"                     ├─ Đã dịch : {C_GREEN}{initial_translated:,}{C_RESET} dòng")
        print(f"                     ├─ Cần dịch: {C_YELLOW}{initial_pending:,}{C_RESET} dòng")
        print(f"                     └─ Lỗi cũ  : {initial_failed:,} dòng")
        print(f"  • Tiến độ hiện tại: {C_CYAN}{render_progress_bar(initial_pct)}{C_RESET}")
        print(f"  • Kích thước lô  : {batch_chunk} dòng / chu kỳ")
        print(
            f"{C_BOLD}{C_CYAN}================================================================================{C_RESET}\n"
        )

        pipeline = TranslationPipeline(project, db, config)

        session_start_time = time.time()
        session_translated = 0
        session_failed = 0
        batch_idx = 0

        try:
            while not _STOP_REQUESTED:
                # 1. Server availability check
                if not check_server_health(config.llm.base_url) and not wait_for_server(
                    config.llm.base_url
                ):
                    break

                batch_idx += 1
                batch_start_time = time.time()

                # 2. Run chunk
                stats = pipeline.run(limit=batch_chunk, include_failed=retry_failed)

                chunk_translated = stats.llm_translated + stats.seeded + stats.tm_reused
                chunk_duration = max(0.001, time.time() - batch_start_time)
                chunk_speed = chunk_translated / chunk_duration

                session_translated += chunk_translated
                session_failed += stats.failed

                # 3. Query current global database progress
                current_counts = query_db_status(db)
                cur_total = sum(current_counts.values())
                cur_translated = (
                    current_counts.get("translated", 0)
                    + current_counts.get("approved", 0)
                    + current_counts.get("qa_flagged", 0)
                )
                cur_pending = current_counts.get("pending", 0)
                cur_failed = current_counts.get("failed", 0)
                cur_pct = (cur_translated / cur_total * 100.0) if cur_total > 0 else 100.0

                session_elapsed = max(0.001, time.time() - session_start_time)
                session_avg_speed = session_translated / session_elapsed

                remaining_work = cur_pending + (cur_failed if retry_failed else 0)
                eta_seconds = (
                    (remaining_work / session_avg_speed) if session_avg_speed > 0.05 else 0.0
                )
                eta_str = format_duration(eta_seconds) if remaining_work > 0 else "00m 00s"

                # Check if completely finished
                if stats.total_pending == 0 or remaining_work == 0:
                    _LOG.info(
                        "%s[HOÀN TẤT]%s Tất cả các dòng thoại cần dịch đã hoàn tất 100%%!",
                        C_GREEN,
                        C_RESET,
                    )
                    break

                # 4. Standardized per-batch report
                progress_bar = render_progress_bar(cur_pct, width=20)
                fail_badge = f" | {C_RED}{stats.failed} lỗi{C_RESET}" if stats.failed > 0 else ""
                _LOG.info(
                    "Lô #%d hoàn tất (+%d dòng trong %.1fs • %.2f dòng/s%s):",
                    batch_idx,
                    chunk_translated,
                    chunk_duration,
                    chunk_speed,
                    fail_badge,
                )
                _LOG.info(
                    "  Tiến độ: %s%s%s (%s/%s dòng) | Còn lại: %s%s%s | ETA: ~%s (TB: %.2f dòng/s)",
                    C_CYAN,
                    progress_bar,
                    C_RESET,
                    f"{cur_translated:,}",
                    f"{cur_total:,}",
                    C_YELLOW,
                    f"{remaining_work:,}",
                    C_RESET,
                    eta_str,
                    session_avg_speed,
                )

                # Avoid busy-loop if all units in this batch failed with 0 progress
                if (
                    stats.llm_translated == 0
                    and stats.seeded == 0
                    and stats.tm_reused == 0
                    and stats.failed > 0
                ):
                    _LOG.warning(
                        "%d dòng trong lô đều thất bại liên tiếp. "
                        "Tạm dừng để tránh tiêu tốn GPU vô ích.",
                        stats.failed,
                    )
                    break

                time.sleep(1.0)
        except KeyboardInterrupt:
            _LOG.warning("Tiến trình đã được dừng ngay lập tức theo yêu cầu của bạn (Ctrl+C).")

        # Query final database status
        final_counts = query_db_status(db)
        final_total = sum(final_counts.values())
        final_translated = final_counts.get("translated", 0) + final_counts.get("approved", 0)
        final_pending = final_counts.get("pending", 0)
        final_pct = (final_translated / final_total * 100.0) if final_total > 0 else 100.0

        total_elapsed = max(0.001, time.time() - session_start_time)
        final_avg_speed = session_translated / total_elapsed
        speed_per_min = final_avg_speed * 60.0

        # Display standardized session summary
        status_label = (
            f"{C_YELLOW}Tạm dừng bởi người dùng (Ctrl+C){C_RESET}"
            if _STOP_REQUESTED
            else f"{C_GREEN}Hoàn tất phiên dịch{C_RESET}"
        )

        summary_title = "TỔNG KẾT PHIÊN DỊCH VITRANS".center(80)
        print(f"\n{C_BOLD}{C_CYAN}{'=' * 80}{C_RESET}")
        print(f"{C_BOLD}{summary_title}{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}{'=' * 80}{C_RESET}")
        print(f"  • Trạng thái       : {status_label}")
        print(f"  • Thời gian chạy   : {format_duration(total_elapsed)}")
        print(f"  • Dịch phiên này   : {C_GREEN}{session_translated:,}{C_RESET} dòng thành công")
        fail_c = C_RED if session_failed > 0 else C_RESET
        print(f"  • Thất bại         : {fail_c}{session_failed:,}{C_RESET} dòng")
        print(
            f"  • Tốc độ trung bình: {final_avg_speed:.2f} dòng/s (~{speed_per_min:.0f} dòng/phút)"
        )
        bar_str = render_progress_bar(final_pct)
        print(
            f"  • Tiến độ toàn game: {C_CYAN}{bar_str}{C_RESET} "
            f"({final_translated:,} / {final_total:,} dòng)"
        )
        print(f"  • Dòng còn lại     : {final_pending:,} dòng đang chờ")
        print(
            f"  • Cơ sở dữ liệu    : {C_GREEN}Đã lưu an toàn (ACID SQLite) — "
            f"Không mất mát dữ liệu.{C_RESET}"
        )
        print(f"{C_BOLD}{C_CYAN}{'=' * 80}{C_RESET}\n")

    # Optional auto-apply
    if auto_apply and not _STOP_REQUESTED and session_translated > 0:
        _LOG.info("Đang tự động áp dụng bản dịch vào tệp Ren'Py...")
        from vitrans.writer import apply_translations

        with project.open_db() as db_apply:
            rep = apply_translations(project, db_apply)
            _LOG.info("Đã ghi %d dòng vào %d tệp.", rep.lines_written, rep.files_changed)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy dịch tự động ngầm qua đêm cho ViTrans.")
    parser.add_argument(
        "command",
        nargs="?",
        default="translate",
        help="Lệnh chạy (mặc định: translate)",
    )
    parser.add_argument("-p", "--project", default=".", help="Đường dẫn thư mục dự án")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Số dòng mỗi chu kỳ kiểm tra (mặc định: 100)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Thử lại các dòng từng bị lỗi (failed)",
    )
    parser.add_argument(
        "--retry-flagged",
        action="store_true",
        help="Đặt lại các dòng có cờ QA cảnh báo (qa_flagged) để dịch lại",
    )
    parser.add_argument(
        "--auto-apply",
        action="store_true",
        help="Tự động apply bản dịch vào game sau khi hoàn thành",
    )
    args = parser.parse_args()

    return run_unattended_pipeline(
        Path(args.project),
        batch_chunk=args.chunk_size,
        retry_failed=args.retry_failed,
        retry_flagged=args.retry_flagged,
        auto_apply=args.auto_apply,
    )


if __name__ == "__main__":
    sys.exit(main())
