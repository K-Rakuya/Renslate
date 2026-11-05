"""Quality assurance and database hygiene command handlers."""

from __future__ import annotations

import argparse
import logging
import sys

from renslate.cli.handlers.project import get_project_root
from renslate.core.models import UnitKind
from renslate.core.project import Project
from renslate.engine.qa import QAChecker
from renslate.renpy.mask import mask

_LOG = logging.getLogger("renslate.cli.qa")


def cmd_qa(args: argparse.Namespace) -> int:
    """Run structural quality checks and optional LLM hallucination scans."""
    project = Project.open(get_project_root(args))
    qa_checker = QAChecker(max_length_ratio=project.config.qa.max_length_ratio)
    issues_found = 0
    errors_found = 0

    with project.open_db() as database:
        units = database.all_units()

        for u in units:
            if not u.target or u.status.value == "obsolete":
                continue
            masked = mask(u.source) if u.source else None
            issues = qa_checker.check_unit(
                u.uid,
                u.source,
                masked,
                u.target,
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

        if getattr(args, "guard", False):
            from renslate.engine.hallucination import (
                HallucinationGuard,
                persist_hallucination_audit,
            )

            guard = HallucinationGuard(config=project.config.llm)
            limit = getattr(args, "limit", None)
            candidates = [
                u
                for u in units
                if u.target
                and u.kind == UnitKind.DIALOGUE
                and any(c.isalnum() for c in u.source)
                and not (
                    u.source.strip().startswith("[")
                    and u.source.strip().endswith("]")
                    and ":" in u.source
                )
            ]
            if limit:
                candidates = candidates[:limit]

            if candidates:
                print(
                    f"\nBắt đầu thẩm định ảo giác bằng LLM Hallucination Guard ({len(candidates)} dòng)..."
                )
                batch_size = max(1, project.config.batch.max_lines)
                total_batches = (len(candidates) + batch_size - 1) // batch_size
                for idx, i in enumerate(range(0, len(candidates), batch_size), start=1):
                    chunk = candidates[i : i + batch_size]
                    print(f"  Đang quét lô {idx}/{total_batches} ({len(chunk)} dòng)...")
                    scan_results = guard.scan_batch(chunk, context_window="")
                    persist_hallucination_audit(database, scan_results)
                    for r in scan_results:
                        if r.has_hallucination:
                            sugg = (
                                f" -> Gợi ý: {r.suggested_correction}"
                                if r.suggested_correction
                                else ""
                            )
                            print(
                                f"  [HALLUCINATION] {r.unit_id} ({r.category}): {r.explanation}{sugg}",
                                file=sys.stderr,
                            )
                            issues_found += 1
                            if r.severity == "error":
                                errors_found += 1

        try:
            audit_logs = database.execute(
                "SELECT uid, issue_type, explanation, suggested_correction "
                "FROM qa_audit_log WHERE verdict = 'FAIL' ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
            if audit_logs and not getattr(args, "guard", False):
                print(
                    f"\nPhát hiện {len(audit_logs)} cảnh báo từ LLM Hallucination Guard (Audit Log):"
                )
                for a_uid, a_type, a_exp, a_sugg in audit_logs:
                    sugg_str = f" -> Gợi ý: {a_sugg}" if a_sugg else ""
                    print(f"  [HALLUCINATION] {a_uid} ({a_type}): {a_exp}{sugg_str}")
                issues_found += len(audit_logs)
        except Exception:
            pass

    if issues_found == 0:
        print("Tất cả bản dịch đều vượt qua kiểm tra chất lượng (QA pass).")
        return 0

    print(f"\nTổng kết: {issues_found} vấn đề QA ({errors_found} lỗi nghiêm trọng).")
    return 1 if errors_found > 0 else 0


def cmd_clean_db(args: argparse.Namespace) -> int:
    """Normalize speaker tags, clean duplicate quotes, and sanitize DB/TM."""
    project = Project.open(get_project_root(args))
    from renslate.renpy.extract import deduplicate_tl_strings

    dups_removed = deduplicate_tl_strings(project.tl_root, project.config.target_lang)
    with project.open_db() as database:
        units_cleaned, tm_cleaned = database.sanitize_stored_translations()
        msg = (
            f"Đã làm sạch {units_cleaned} dòng thoại trong units "
            f"và {tm_cleaned} mục trong translation memory (TM)."
        )
        if dups_removed:
            msg += f" Đã loại bỏ {dups_removed} chuỗi trùng lặp trong tệp dịch."
        print(msg)
    return 0
