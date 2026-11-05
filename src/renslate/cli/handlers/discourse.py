"""Discourse analysis and LLM bootstrapping command handlers."""

from __future__ import annotations

import argparse
import logging

from renslate.cli.handlers.project import get_project_root
from renslate.core.project import Project
from renslate.engine.analyze import run_analysis

_LOG = logging.getLogger("renslate.cli.discourse")


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze script AST to discover characters, scenes, and co-occurrences."""
    project = Project.open(get_project_root(args))
    llm = None
    if getattr(args, "bootstrap", False):
        from renslate.engine.llm import LLMClient

        llm = LLMClient(project.config.llm)
    with project.open_db() as database:
        stats = run_analysis(
            project,
            database,
            enrich_llm=getattr(args, "bootstrap", False),
            llm=llm,
        )
    print(f"Đã phân tích dự án tại {project.root}:")
    print(f"  nhân vật phát hiện : {stats['characters']}")
    print(f"  phân đoạn cảnh     : {stats['scenes']}")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """Bootstrap character profiles, relationships, and scene summaries with LLM."""
    project = Project.open(get_project_root(args))
    from renslate.engine.analyze import bootstrap_characters_with_llm
    from renslate.engine.llm import LLMClient
    from renslate.engine.relationships import bootstrap_relationships_with_llm
    from renslate.engine.scenes import summarize_scenes_with_llm

    llm = LLMClient(project.config.llm)
    print(f"Bắt đầu LLM bootstrapping cho dự án tại {project.root}...")
    with project.open_db() as database:
        # 1. Characters
        print(f"  Đang phân tích hồ sơ nhân vật (tối đa {args.top_chars} nhân vật)...")
        chars = bootstrap_characters_with_llm(project, database, llm, max_chars=args.top_chars)
        print(f"  -> Hoàn tất hồ sơ cho {len(chars)} nhân vật.")

        # 2. Relationships and address forms from real script dialogues
        if args.top_pairs > 0:
            print(f"  Đang xác định quan hệ và cách xưng hô (tối đa {args.top_pairs} cặp)...")
            rel_stats = bootstrap_relationships_with_llm(
                project,
                database,
                llm,
                max_pairs=args.top_pairs,
                force=args.refresh_relationships,
            )
            print(
                f"  -> {rel_stats['resolved']}/{rel_stats['pairs']} cặp hợp lệ, "
                f"{rel_stats['edges']} quan hệ được ghi."
            )

        # 3. Scene summaries for micro-framing narrative context
        if args.top_scenes > 0:
            print(f"  Đang tóm tắt phân cảnh (tối đa {args.top_scenes} cảnh)...")
            sc_stats = summarize_scenes_with_llm(
                project, database, llm, max_scenes=args.top_scenes, force=args.refresh_scenes
            )
            print(f"  -> Đã tóm tắt {sc_stats['summarized']}/{sc_stats['scenes']} cảnh.")

    print("Hoàn tất bootstrap! characters.json đã được cập nhật thành công.")
    return 0
