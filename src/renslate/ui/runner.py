"""Autonomous runner engine and local LLM server health monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from renslate.core.db import Database, utc_timestamp
from renslate.core.errors import ProjectError, SdkError
from renslate.core.models import CharacterProfile
from renslate.core.project import Project
from renslate.engine.analyze import (
    bootstrap_characters_with_llm,
    run_analysis,
    segment_scenes,
)
from renslate.engine.glossary import Glossary, clean_glossary, translate_glossary_with_llm
from renslate.engine.graph import RelationshipGraph
from renslate.engine.llm import LLMClient
from renslate.engine.translate import TranslationPipeline
from renslate.renpy.extract import extract
from renslate.renpy.sdk import find_launcher, generate_translations
from renslate.renpy.writer import apply_translations
from renslate.ui.dashboard import (
    DashboardState,
    LiveDashboard,
    render_banner,
    render_batch_card,
    render_summary_card,
)
from renslate.ui.progress import format_duration, render_progress_bar
from renslate.ui.theme import (
    C_GREEN,
    C_RESET,
    C_YELLOW,
    IS_TTY,
)

_LOG = logging.getLogger("renslate.runner")

_STOP_REQUESTED: bool = False
_INTERRUPT_COUNT: int = 0


def _sig_handler(signum: int, frame: Any) -> None:
    global _STOP_REQUESTED, _INTERRUPT_COUNT

    unattended: Any = sys.modules.get("scripts.unattended_runner")
    if unattended is not None and hasattr(unattended, "_INTERRUPT_COUNT"):
        _INTERRUPT_COUNT = int(unattended._INTERRUPT_COUNT)

    _INTERRUPT_COUNT += 1

    if unattended is not None:
        unattended._INTERRUPT_COUNT = _INTERRUPT_COUNT

    if _INTERRUPT_COUNT >= 2:
        _LOG.warning("\n[BUỘC DỪNG] Đã nhận tín hiệu ngắt lần thứ 2. Thoát ngay lập tức!")
        os._exit(130)

    _STOP_REQUESTED = True
    if unattended is not None:
        unattended._STOP_REQUESTED = True

    c_yellow = C_YELLOW if IS_TTY else ""
    c_reset = C_RESET if IS_TTY else ""
    _LOG.warning(
        "\n%s[DỪNG TIẾN TRÌNH]%s Đã nhận lệnh dừng (Ctrl+C). "
        "Đang hoàn tất lô hiện tại và lưu dữ liệu an toàn...",
        c_yellow,
        c_reset,
    )


def _is_stop_requested() -> bool:
    unattended = sys.modules.get("scripts.unattended_runner")
    return _STOP_REQUESTED or bool(unattended and getattr(unattended, "_STOP_REQUESTED", False))


def check_server_health(base_url: str, timeout_s: float = 3.0) -> bool:
    """Check if the local llama-server is online and responsive."""
    url = f"{base_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Renslate-Runner"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, TimeoutError, OSError):
        try:
            url_models = f"{base_url.rstrip('/')}/models"
            req_m = urllib.request.Request(url_models, headers={"User-Agent": "Renslate-Runner"})
            with urllib.request.urlopen(req_m, timeout=timeout_s) as resp_m:
                return bool(resp_m.status == 200)
        except Exception:
            return False


def detect_server_model(base_url: str, default_name: str, timeout_s: float = 3.0) -> str:
    """Query the active model filename from llama-server's /props or /v1/models."""
    try:
        props_url = f"{base_url.rstrip('/')}/props"
        req = urllib.request.Request(props_url, headers={"User-Agent": "Renslate-Runner"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict):
                model_path = data.get("default_generation_settings", {}).get("model")
                if model_path:
                    return Path(model_path).name
    except Exception:
        pass

    try:
        models_url = f"{base_url.rstrip('/')}/models"
        req = urllib.request.Request(models_url, headers={"User-Agent": "Renslate-Runner"})
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
    c_green = C_GREEN if IS_TTY else ""
    c_reset = C_RESET if IS_TTY else ""
    while not _is_stop_requested():
        if check_server_health(base_url):
            if not first:
                _LOG.info("%sLLM server đã phục hồi trực tuyến!%s", c_green, c_reset)
            return True
        if first:
            _LOG.warning("LLM server (%s) hiện không phản hồi. Đang đợi máy chủ...", base_url)
            first = False
        time.sleep(check_interval)
    return False


def query_db_status(db: Database) -> dict[str, int]:
    """Retrieve accurate status counts across all units in the database."""
    cur = db.execute("SELECT status, COUNT(*) FROM units GROUP BY status")
    return {str(row[0]): int(row[1]) for row in cur.fetchall()}


def bootstrap_if_needed(
    project: Project,
    db: Database,
    *,
    bootstrap: bool | None = None,
) -> None:
    """Ensure characters, scenes, and glossary are initialized idempotently."""
    char_row = db.execute("SELECT COUNT(*) FROM characters").fetchone()
    char_count = char_row[0] if char_row else 0
    chars_file = project.work_dir / "characters.json"

    if char_count == 0:
        if chars_file.is_file():
            try:
                raw_chars = json.loads(chars_file.read_text(encoding="utf-8"))
                with db.transaction():
                    for tag, data in raw_chars.items():
                        profile = CharacterProfile.from_dict(data, tag=tag)
                        db.save_character(profile)
                _LOG.info("Đã đồng bộ %d hồ sơ nhân vật từ %s.", len(raw_chars), chars_file.name)
                char_count = len(raw_chars)
            except Exception as e:
                _LOG.warning("Không thể đọc %s: %s", chars_file, e)

        if (
            char_count == 0
            and not db.get_meta("analysis_completed")
            and (project.root / "game").is_dir()
        ):
            _LOG.info("Đang tự động phân tích nhân vật, phân cảnh và glossary...")
            run_analysis(project, db, mine_glossary=True, enrich_llm=False)

    scene_row = db.execute("SELECT COUNT(*) FROM scenes").fetchone()
    scene_count = scene_row[0] if scene_row else 0
    if scene_count == 0:
        scenes = segment_scenes(db)
        if scenes:
            with db.transaction():
                for sc in scenes:
                    db.save_scene(sc)

    glossary_file = project.work_dir / "glossary.json"
    cur_glossary = db.get_glossary()
    if not cur_glossary and glossary_file.is_file():
        loaded_g = Glossary.load_file(glossary_file)
        if loaded_g.entries:
            loaded_g.save_db(db)
            cur_glossary = loaded_g.entries
            _LOG.info("Đã đồng bộ %d thuật ngữ từ %s.", len(cur_glossary), glossary_file.name)

    chars = db.get_characters()
    char_names = {p.display_name.lower() for p in chars.values() if p.display_name} | set(
        chars.keys()
    )
    cleaned_glossary = clean_glossary(cur_glossary, exclude_terms=char_names)
    if len(cleaned_glossary) != len(cur_glossary) or any(g.case_sensitive for g in cur_glossary):
        g_clean = Glossary(cleaned_glossary)
        g_clean.save_file(glossary_file)
        g_clean.save_db(db)
        cur_glossary = cleaned_glossary

    # Determine if characters or glossary need LLM bootstrapping
    force_bootstrap = bootstrap is True
    skip_bootstrap = bootstrap is False
    needs_char_bootstrap = bool(chars) and (
        force_bootstrap
        or (
            not skip_bootstrap
            and not db.get_meta("characters_bootstrapped")
            and any(not p.role and not p.speech_style for p in chars.values())
        )
    )
    needs_gloss_bootstrap = force_bootstrap or (
        not skip_bootstrap
        and not db.get_meta("glossary_bootstrapped")
        and any(not g.translation.strip() for g in cur_glossary)
    )

    if (needs_char_bootstrap or needs_gloss_bootstrap) and check_server_health(
        project.config.llm.base_url
    ):
        llm_client = LLMClient(project.config.llm)
        if needs_char_bootstrap:
            _LOG.info("Đang thực hiện LLM bootstrapping cho hồ sơ nhân vật...")
            bootstrap_characters_with_llm(project, db, llm_client, max_chars=30)
            db.set_meta("characters_bootstrapped", utc_timestamp())

        if needs_gloss_bootstrap:
            untranslated = [g for g in cur_glossary if not g.translation.strip()]
            if untranslated:

                def _freq(e: Any) -> int:
                    m = re.search(r"(\d+)\s*lần", e.notes)
                    return int(m.group(1)) if m else 1

                untranslated.sort(key=_freq, reverse=True)
                to_trans = untranslated[:150]
                _LOG.info("Đang dịch tự động %d thuật ngữ phổ biến nhất...", len(to_trans))
                enriched = translate_glossary_with_llm(to_trans, llm_client)
                trans_map = {e.term: e for e in enriched if e.translation.strip()}
                final_entries = [trans_map.get(e.term, e) for e in cur_glossary]
                g_final = Glossary(final_entries)
                g_final.save_file(glossary_file)
                g_final.save_db(db)
            db.set_meta("glossary_bootstrapped", utc_timestamp())

    # Sync character relationships with relationships.json if present
    rel_file = getattr(project, "relationships_path", None)
    if rel_file is None or not rel_file.is_file():
        root_rel = project.root / "relationships.json"
        if root_rel.is_file():
            rel_file = root_rel
        elif rel_file is None:
            rel_file = project.work_dir / "relationships.json"
    RelationshipGraph.sync(db, rel_file)


def run_unattended_pipeline(
    project_root: Path,
    *,
    batch_chunk: int = 100,
    retry_failed: bool = False,
    retry_flagged: bool = False,
    auto_apply: bool = False,
    plain: bool = False,
    bootstrap: bool | None = None,
) -> int:
    """Execute the autonomous translation process until 100% finished."""
    global _STOP_REQUESTED, _INTERRUPT_COUNT
    _STOP_REQUESTED = False
    _INTERRUPT_COUNT = 0
    unattended: Any = sys.modules.get("scripts.unattended_runner")
    if unattended is not None:
        unattended._STOP_REQUESTED = False
        unattended._INTERRUPT_COUNT = 0

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

    if not wait_for_server(config.llm.base_url, check_interval=0.1):
        _LOG.info("Đã dừng runner theo yêu cầu.")
        return 0

    active_model = detect_server_model(config.llm.base_url, config.llm.model)

    c_green = C_GREEN if (not plain and IS_TTY) else ""
    c_yellow = C_YELLOW if (not plain and IS_TTY) else ""
    c_reset = C_RESET if (not plain and IS_TTY) else ""

    with project.open_db() as db:
        # Check translation directory; auto-generate skeleton if missing
        if not project.tl_root.is_dir():
            _LOG.info(
                "Thư mục dịch %s chưa tồn tại. Đang tìm launcher sinh khung dịch...",
                project.tl_root,
            )
            sdk_candidate = Path(project.config.sdk).expanduser() if project.config.sdk else None
            try:
                launcher_parent = find_launcher(sdk_candidate).parent if sdk_candidate else None
                generate_translations(
                    launcher_parent,
                    project.root,
                    project.config.target_lang,
                    fallback_rpyc=True,
                )
            except SdkError as s_err:
                _LOG.error("Không thể tự động sinh khung dịch: %s", s_err)
                return 1

        # Check units; auto-extract if empty
        row = db.execute("SELECT COUNT(*) FROM units").fetchone()
        unit_count = row[0] if row else 0
        if unit_count == 0:
            _LOG.info("Cơ sở dữ liệu chưa có đơn vị dịch. Đang tự động quét 'extract'...")
            report = extract(project, db)
            _LOG.info("Đã quét %d tệp, trích xuất %d đơn vị dịch.", report.files, report.units)
            if report.units == 0:
                _LOG.warning(
                    "Không tìm thấy tệp bản dịch nào trong game/tl/%s hoặc game/.",
                    project.config.target_lang,
                )
                return 0

        # Bootstrap characters, scenes, and glossary idempotently
        bootstrap_if_needed(project, db, bootstrap=bootstrap)

        # Auto-sanitize database
        db.sanitize_stored_translations()

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
                _LOG.info("Đã đặt lại %d dòng cảnh báo QA về pending.", count_flagged)

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

        rel_file = getattr(project, "relationships_path", None)
        if rel_file is None or not rel_file.is_file():
            root_rel = project.root / "relationships.json"
            if root_rel.is_file():
                rel_file = root_rel
            elif rel_file is None:
                rel_file = project.work_dir / "relationships.json"
        rel_graph = RelationshipGraph.sync(db, rel_file)

        chars = db.get_characters()
        banner_info = [
            ("Dự án", str(project.root)),
            ("Ngôn ngữ đích", f"{project.config.target_lang.upper()} (Tiếng Việt)"),
            ("LLM Endpoint", config.llm.base_url),
            ("Model kích hoạt", active_model),
            ("Quy mô dự án", f"{total_units:,} tổng số dòng ({initial_translated:,} đã dịch)"),
        ]
        if chars or rel_graph.edge_count > 0:
            banner_info.append(
                ("Nhân vật & Quan hệ", f"{len(chars):,} nhân vật, {rel_graph.edge_count:,} quan hệ")
            )
        banner_info.extend([
            ("Tiến độ hiện tại", render_progress_bar(initial_pct)),
            ("Kích thước lô", f"{batch_chunk} dòng / chu kỳ"),
        ])
        banner_text = render_banner(
            "Renslate — Unattended Translation Runner",
            width=80,
            items=banner_info,
            plain=plain,
        )
        print("\n" + banner_text + "\n")

        pipeline = TranslationPipeline(project, db, config, graph=rel_graph)

        session_start_time = time.time()
        session_translated = 0
        session_failed = 0
        batch_idx = 0

        dashboard_state = DashboardState(
            model_name=active_model,
            server_status="Online",
            server_url=config.llm.base_url,
            total_units=total_units,
            completed_units=initial_translated,
            pending_units=initial_pending,
            failed_units=initial_failed,
        )
        dashboard = LiveDashboard(dashboard_state)

        try:
            while not _is_stop_requested():
                if not check_server_health(config.llm.base_url):
                    dashboard_state.server_status = "Offline"
                    if not wait_for_server(config.llm.base_url, check_interval=0.5):
                        break
                    dashboard_state.server_status = "Online"

                batch_idx += 1
                batch_start_time = time.time()

                stats = pipeline.run(limit=batch_chunk, include_failed=retry_failed)

                chunk_translated = stats.llm_translated + stats.seeded + stats.tm_reused
                chunk_duration = max(0.001, time.time() - batch_start_time)
                chunk_speed = chunk_translated / chunk_duration

                session_translated += chunk_translated
                session_failed += stats.failed

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

                dashboard_state.completed_units = cur_translated
                dashboard_state.pending_units = cur_pending
                dashboard_state.failed_units = cur_failed
                dashboard_state.speed_u_s = session_avg_speed
                dashboard_state.elapsed_s = session_elapsed
                dashboard_state.eta_s = eta_seconds
                dashboard_state.current_batch = batch_idx
                dashboard_state.batch_latency_s = chunk_duration
                dashboard_state.lookback_size = getattr(pipeline, "lookback_size", 0)

                if stats.total_pending == 0 or remaining_work == 0:
                    _LOG.info(
                        "%s[HOÀN TẤT]%s Tất cả các dòng thoại cần dịch đã hoàn tất 100%%!",
                        c_green,
                        c_reset,
                    )
                    break

                if not plain and IS_TTY:
                    print(dashboard.render(width=78, plain=False))
                else:
                    batch_msg = render_batch_card(
                        batch_idx,
                        chunk_translated,
                        chunk_duration,
                        chunk_speed,
                        stats.failed,
                        plain=plain,
                    )
                    _LOG.info("%s", batch_msg)
                    _LOG.info(
                        "  Tiến độ: %s (%s/%s) | Còn lại: %s%s%s | ETA: ~%s (TB: %.2f u/s)",
                        render_progress_bar(cur_pct, width=20),
                        f"{cur_translated:,}",
                        f"{cur_total:,}",
                        c_yellow,
                        f"{remaining_work:,}",
                        c_reset,
                        eta_str,
                        session_avg_speed,
                    )

                if (
                    stats.llm_translated == 0
                    and stats.seeded == 0
                    and stats.tm_reused == 0
                    and stats.failed > 0
                ):
                    _LOG.warning(
                        "%d dòng trong lô đều thất bại liên tiếp. Tạm dừng loop.", stats.failed
                    )
                    break

                time.sleep(0.01)
        except KeyboardInterrupt:
            _LOG.warning("Tiến trình đã được dừng ngay lập tức theo yêu cầu của bạn (Ctrl+C).")

        final_counts = query_db_status(db)
        final_total = sum(final_counts.values())
        final_translated = final_counts.get("translated", 0) + final_counts.get("approved", 0)
        final_pending = final_counts.get("pending", 0)
        final_failed = final_counts.get("failed", 0)

        total_elapsed = max(0.001, time.time() - session_start_time)
        final_avg_speed = session_translated / total_elapsed

        summary_state = DashboardState(
            model_name=active_model,
            total_units=final_total,
            completed_units=final_translated,
            pending_units=final_pending,
            failed_units=final_failed,
            speed_u_s=final_avg_speed,
            elapsed_s=total_elapsed,
        )
        print("\n" + render_summary_card(summary_state, width=80, plain=plain) + "\n")

    has_trans = session_translated > 0 or initial_translated > 0
    if auto_apply and not _is_stop_requested() and has_trans:
        _LOG.info("Đang tự động áp dụng bản dịch vào tệp Ren'Py...")
        with project.open_db() as db_apply:
            rep = apply_translations(project, db_apply)
            _LOG.info("Đã ghi %d dòng vào %d tệp.", rep.lines_written, rep.files_changed)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for standalone runner execution."""
    parser = argparse.ArgumentParser(description="Chạy dịch tự động ngầm qua đêm cho Renslate.")
    parser.add_argument(
        "command", nargs="?", default="translate", help="Lệnh chạy (mặc định: translate)"
    )
    parser.add_argument("-p", "--project", default=".", help="Đường dẫn thư mục dự án")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Số dòng mỗi chu kỳ kiểm tra (mặc định: 100)",
    )
    parser.add_argument(
        "--retry-failed", action="store_true", help="Thử lại các dòng từng bị lỗi (failed)"
    )
    parser.add_argument(
        "--retry-flagged",
        action="store_true",
        help="Đặt lại các dòng có cờ QA cảnh báo để dịch lại",
    )
    parser.add_argument(
        "--auto-apply",
        action="store_true",
        help="Tự động apply bản dịch vào game sau khi hoàn thành",
    )
    parser.add_argument(
        "--plain", "--no-dashboard", action="store_true", help="Không sử dụng ANSI dashboard động"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Chạy chế độ thiết lập wizard tương tác"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bắt buộc LLM bootstrapping lại cho hồ sơ nhân vật và thuật ngữ trước khi dịch",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Bỏ qua bước LLM bootstrapping cho nhân vật và glossary",
    )
    args = parser.parse_args(argv)

    if args.interactive:
        from renslate.ui.wizard import run_interactive_wizard

        return run_interactive_wizard(Path(args.project))

    bootstrap_arg: bool | None = None
    if args.bootstrap:
        bootstrap_arg = True
    elif args.no_bootstrap:
        bootstrap_arg = False

    return run_unattended_pipeline(
        Path(args.project),
        batch_chunk=args.chunk_size,
        retry_failed=args.retry_failed,
        retry_flagged=args.retry_flagged,
        auto_apply=args.auto_apply,
        plain=args.plain,
        bootstrap=bootstrap_arg,
    )
