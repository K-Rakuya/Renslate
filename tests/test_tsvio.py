"""Tests for renslate.tsvio."""

from __future__ import annotations

from pathlib import Path

from renslate.db import Database
from renslate.models import Origin, Slot, Status, Unit, UnitKind
from renslate.tsvio import export_tsv, import_tsv


def test_export_and_import_tsv(tmp_path: Path) -> None:
    db = Database.open(tmp_path / "test.db", create=True)
    db.save_unit(Unit("d:1", UnitKind.DIALOGUE, 1, "Hello [player]!", Status.PENDING, speaker="e"))
    db.save_unit(Unit("d:2", UnitKind.DIALOGUE, 2, "Good morning.", Status.PENDING, speaker="e"))
    db.replace_slots(
        [
            Slot("d:1", "script.rpy", 5, 4, "fp1"),
            Slot("d:2", "script.rpy", 8, 7, "fp2"),
        ]
    )

    tsv_file = tmp_path / "review.tsv"
    exported = export_tsv(db, tsv_file)
    assert exported == 2
    assert tsv_file.is_file()

    # Read exported content, edit target for d:1
    content = tsv_file.read_text(encoding="utf-8-sig")
    lines = content.splitlines()
    header = lines[0]
    # Replace target column in row 1
    row1 = lines[1].split("\t")
    row1[5] = "Xin chào ⟦V1⟧!"  # target column
    row2 = lines[2].split("\t")
    row2[5] = "Chào buổi sáng."  # target column

    new_content = "\n".join([header, "\t".join(row1), "\t".join(row2)])
    tsv_file.write_text(new_content, encoding="utf-8-sig")

    imported, skipped, errors = import_tsv(db, tsv_file)
    assert imported == 2
    assert skipped == 0
    assert not errors

    # Check units in DB
    u1 = db.get_unit("d:1")
    assert u1 is not None
    assert u1.target == "Xin chào [player]!"
    assert u1.origin is Origin.MANUAL
    assert u1.status is Status.REVIEWED

    u2 = db.get_unit("d:2")
    assert u2 is not None
    assert u2.target == "Chào buổi sáng."


def test_import_tsv_rejects_bad_placeholders(tmp_path: Path) -> None:
    db = Database.open(tmp_path / "test.db", create=True)
    db.save_unit(Unit("d:1", UnitKind.DIALOGUE, 1, "Hello [player]!", Status.PENDING))
    db.replace_slots([Slot("d:1", "script.rpy", 5, 4, "fp")])

    tsv_file = tmp_path / "bad.tsv"
    export_tsv(db, tsv_file)

    content = tsv_file.read_text(encoding="utf-8-sig")
    lines = content.splitlines()
    row = lines[1].split("\t")
    row[5] = "Xin chào bạn!"  # Missing ⟦V1⟧!
    tsv_file.write_text("\n".join([lines[0], "\t".join(row)]), encoding="utf-8-sig")

    imported, _skipped, errors = import_tsv(db, tsv_file)
    assert imported == 0
    assert errors
    assert "placeholder" in errors[0].lower()


def test_import_tsv_accepts_natural_renpy_markup(tmp_path: Path) -> None:
    db = Database.open(tmp_path / "test.db", create=True)
    unit = Unit("d:1", UnitKind.DIALOGUE, 1, "Hello [player], meet {b}[npc]{/b}!", Status.PENDING)
    db.save_unit(unit)
    db.replace_slots([Slot("d:1", "script.rpy", 5, 4, "fp")])

    tsv_file = tmp_path / "natural.tsv"
    export_tsv(db, tsv_file)

    content = tsv_file.read_text(encoding="utf-8-sig")
    lines = content.splitlines()
    row = lines[1].split("\t")
    row[5] = "Xin chào [player], hãy gặp {b}[npc]{/b}!"  # Natural markup!
    tsv_file.write_text("\n".join([lines[0], "\t".join(row)]), encoding="utf-8-sig")

    imported, _skipped, errors = import_tsv(db, tsv_file)
    assert imported == 1
    assert not errors

    u1 = db.get_unit("d:1")
    assert u1 is not None
    assert u1.target == "Xin chào [player], hãy gặp {b}[npc]{/b}!"
    assert u1.status is Status.REVIEWED
