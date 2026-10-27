"""Export and import translation units via TSV for human review."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from vitrans.db import Database, utc_timestamp
from vitrans.mask import reconcile_translation
from vitrans.models import Origin, Status
from vitrans.tm import TranslationMemory

_TSV_HEADER = ["uid", "file", "line", "speaker", "source", "target", "status", "qa_flags"]


def export_tsv(
    db: Database,
    out_path: Path,
    *,
    status_filter: Status | None = None,
    qa_only: bool = False,
) -> int:
    """Export translation units to a TSV file with UTF-8 BOM for spreadsheet compatibility."""
    query = (
        "SELECT u.uid, s.file, s.line, u.speaker, u.source, u.target, u.status, u.qa_flags "
        "FROM units u LEFT JOIN slots s ON u.uid = s.uid "
        "WHERE u.status != 'obsolete' "
    )
    params: list[str] = []
    if status_filter:
        query += "AND u.status = ? "
        params.append(status_filter.value)
    if qa_only:
        query += "AND u.status = 'qa_flagged' "
    query += "GROUP BY u.uid ORDER BY u.seq"

    rows = db.execute(query, params).fetchall()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(_TSV_HEADER)
        for r in rows:
            flags = ", ".join(json.loads(r["qa_flags"])) if r["qa_flags"] else ""
            writer.writerow(
                [
                    r["uid"],
                    r["file"] or "",
                    r["line"] or "",
                    r["speaker"] or "",
                    r["source"],
                    r["target"] or "",
                    r["status"],
                    flags,
                ]
            )

    return len(rows)


def import_tsv(db: Database, in_path: Path) -> tuple[int, int, list[str]]:
    """Import edited translations from a TSV file into the database.

    Translations with edits are validated, assigned origin='manual',
    and recorded into the Translation Memory.
    Returns (imported_count, skipped_count, errors).
    """
    if not in_path.is_file():
        return 0, 0, [f"Tệp TSV không tồn tại: {in_path}"]

    content = in_path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(content), delimiter="\t")

    stored_units = db.units()
    tm = TranslationMemory(db)
    now = utc_timestamp()

    imported = 0
    skipped = 0
    errors: list[str] = []

    with db.transaction():
        for line_num, row in enumerate(reader, start=2):
            uid = row.get("uid", "").strip()
            if not uid or uid not in stored_units:
                continue

            target = (row.get("target") or "").strip()
            unit = stored_units[uid]

            # If unchanged or empty
            if not target or target == unit.target:
                skipped += 1
                continue

            # Reconcile translation against source (supports natural Ren'Py text and masked tokens)
            final_target, issues = reconcile_translation(unit.source, target)
            if issues:
                err_desc = "; ".join(f"{i.code} ({i.detail})" for i in issues)
                errors.append(f"Dòng {line_num} ({uid}): Lỗi markup/placeholder: {err_desc}")
                continue

            # Update DB with reconciled natural target
            db.update_translation(
                uid=uid,
                target=final_target,
                origin=Origin.MANUAL,
                status=Status.REVIEWED,
                qa_flags=(),
                updated_at=now,
            )
            # Sync to TM
            tm.record(unit.source, final_target, origin=Origin.MANUAL)
            imported += 1

    return imported, skipped, errors
