"""Data exchange command handlers: TSV export and import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from renslate.cli.handlers.project import get_project_root
from renslate.core.models import Status
from renslate.core.project import Project
from renslate.engine.tsvio import export_tsv, import_tsv


def cmd_export(args: argparse.Namespace) -> int:
    """Export translations to TSV format for human review or offline editing."""
    project = Project.open(get_project_root(args))
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


def cmd_import(args: argparse.Namespace) -> int:
    """Import reviewed translations from TSV file back into database."""
    project = Project.open(get_project_root(args))
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
