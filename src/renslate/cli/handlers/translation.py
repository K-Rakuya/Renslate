"""Translation workflow command handlers: translate, run, preview, live studio."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from typing import Any

from renslate.cli.formatter import print_translation_stats
from renslate.cli.handlers.project import get_project_root
from renslate.core.project import Project
from renslate.engine.context.exemplars import TMDyadExemplarRetriever
from renslate.engine.context.narrative_memory import NarrativeMemory
from renslate.engine.deliberate import DeliberativeGate
from renslate.engine.translate import TranslationPipeline

_LOG = logging.getLogger("renslate.cli.translation")


def cmd_translate(args: argparse.Namespace) -> int:
    """Execute batch translation with 2026 literary quality deliberator."""
    project = Project.open(get_project_root(args))
    config = project.config

    if getattr(args, "batch_size", None):
        batch_cfg = dataclasses.replace(config.batch, max_lines=args.batch_size)
        config = dataclasses.replace(config, batch=batch_cfg)
    if getattr(args, "temperature", None) is not None:
        llm_cfg = dataclasses.replace(config.llm, temperature=args.temperature)
        config = dataclasses.replace(config, llm=llm_cfg)
    if getattr(args, "concise_tags", False):
        batch_cfg = dataclasses.replace(config.batch, concise_tags=True)
        config = dataclasses.replace(config, batch=batch_cfg)

    def _progress(current: int, total: int, msg: str) -> None:
        print(f"[{current}/{total}] {msg}")

    guard_enabled = not getattr(args, "no_guard", False)
    guard = None
    if guard_enabled:
        try:
            from renslate.engine.hallucination import HallucinationGuard

            guard = HallucinationGuard(config=config.llm)
        except Exception as exc:
            _LOG.warning("Không thể khởi tạo HallucinationGuard: %s", exc)

    # Configure 2026 Literary Quality Components
    use_deliberation = not getattr(args, "no_deliberate", False)
    calque_thresh = float(getattr(args, "calque_threshold", 0.35))
    gate = DeliberativeGate(calque_threshold=calque_thresh) if use_deliberation else None

    mem_cap = int(getattr(args, "narrative_memory", 6))
    narrative_mem = NarrativeMemory(capacity=mem_cap)

    use_transgraph = not getattr(args, "no_transgraph", False)

    with project.open_db() as database:
        retriever = TMDyadExemplarRetriever(database)
        cli_mod = sys.modules.get("renslate.cli")
        pipeline_cls = getattr(cli_mod, "TranslationPipeline", TranslationPipeline)
        pipeline = pipeline_cls(
            project,
            database,
            config,
            hallucination_guard=guard,
            use_transgraph=use_transgraph,
            narrative_memory=narrative_mem,
            exemplar_retriever=retriever,
            deliberative_gate=gate,
        )
        print(f"Bắt đầu dịch bằng backend {config.llm.base_url} ({config.llm.model})...")
        if guard:
            print("  [BẢO VỆ] Continuous LLM Hallucination Guard đang hoạt động.")
        if use_deliberation:
            print(f"  [2-PASS] Deliberative Reflection kích hoạt (ngưỡng calque: {calque_thresh}).")
        if use_transgraph:
            print("  [TRANSGRAPH] Micro-Framing quan hệ 4D & ngữ cảnh phân cảnh đang kích hoạt.")

        if getattr(args, "studio", False) and sys.stdin.isatty():
            import threading

            from renslate.engine.translate import TranslationStats
            from renslate.ui.studio import StudioState, run_studio_session

            state = StudioState()
            pipeline.unit_cb = state.add_or_update_unit
            worker_error: list[Exception] = []

            def _worker() -> None:
                try:
                    pipeline.run(
                        limit=args.limit,
                        include_failed=getattr(args, "retry_failed", False),
                    )
                except Exception as exc:
                    worker_error.append(exc)

            worker_thread = threading.Thread(target=_worker, daemon=True)
            worker_thread.start()
            run_studio_session(project, state=state, db=database)
            worker_thread.join(timeout=1.0)
            stats = TranslationStats()
        else:
            stats = pipeline.run(
                limit=args.limit,
                include_failed=getattr(args, "retry_failed", False),
                progress_cb=_progress,
            )

    print_translation_stats(stats)
    return 1 if stats.errors else 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run autonomous translation pipeline or launch interactive TUI studio."""
    if getattr(args, "interactive", False):
        from renslate.ui.wizard import run_interactive_wizard

        return run_interactive_wizard(get_project_root(args))

    from renslate.ui.runner import run_unattended_pipeline

    bootstrap_arg: bool | None = None
    if getattr(args, "bootstrap", False):
        bootstrap_arg = True
    elif getattr(args, "no_bootstrap", False):
        bootstrap_arg = False

    return run_unattended_pipeline(
        get_project_root(args),
        batch_chunk=args.chunk_size,
        retry_failed=args.retry_failed,
        retry_flagged=args.retry_flagged,
        auto_apply=args.auto_apply,
        plain=getattr(args, "plain", False),
        bootstrap=bootstrap_arg,
        guard=not getattr(args, "no_guard", False),
        concise_tags=getattr(args, "concise_tags", False),
    )


def cmd_preview(args: argparse.Namespace) -> int:
    """Preview translated dialogue lines in chronological or recent order."""
    project = Project.open(get_project_root(args))
    limit = args.limit or 10
    with project.open_db() as database:
        is_latest = getattr(args, "latest", False)
        order_col = "updated_at" if is_latest else "seq"
        order = "DESC" if is_latest else "ASC"
        where_clauses = ["target IS NOT NULL", "target != ''"]
        params: list[Any] = []
        if getattr(args, "status", None):
            where_clauses.append("status = ?")
            params.append(args.status)

        where_sql = " AND ".join(where_clauses)
        query = (
            f"SELECT uid, speaker, source, target, origin, status FROM units "
            f"WHERE {where_sql} ORDER BY {order_col} {order} LIMIT ?"
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
