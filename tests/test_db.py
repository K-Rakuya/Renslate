from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from renslate.db import MIGRATIONS, Database
from renslate.errors import ProjectError
from renslate.models import Origin, Slot, Status, TMEntry, Unit, UnitKind


def make_unit(uid: str = "d:one", **overrides: object) -> Unit:
    fields: dict[str, object] = {
        "uid": uid,
        "kind": UnitKind.DIALOGUE,
        "seq": 0,
        "source": "Hello",
        "status": Status.PENDING,
        "updated_at": "2026-10-23T00:00:00Z",
    }
    fields.update(overrides)
    return Unit(**fields)  # type: ignore[arg-type]


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database.open(tmp_path / ".renslate" / "project.db", create=True)


def test_open_requires_existing_file_unless_creating(tmp_path: Path) -> None:
    with pytest.raises(ProjectError):
        Database.open(tmp_path / "missing.db")
    Database.open(tmp_path / "new" / "project.db", create=True).close()
    Database.open(tmp_path / "new" / "project.db").close()


def test_migration_sets_schema_version(db: Database) -> None:
    assert db.schema_version == len(MIGRATIONS)


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "project.db"
    Database.open(path, create=True).close()
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")
    connection.close()
    with pytest.raises(ProjectError):
        Database.open(path)


def test_unit_round_trip_and_upsert(db: Database) -> None:
    unit = make_unit(
        ref="game/script.rpy:1",
        speaker="e",
        label="start",
        masked_source="Hello",
        target="Xin chào",
        origin=Origin.MANUAL,
        qa_flags=("a", "b"),
    )
    db.save_unit(unit)
    assert db.units() == {"d:one": unit}

    changed = make_unit(source="Hi", status=Status.TRANSLATED)
    db.save_unit(changed)
    assert db.units() == {"d:one": changed}


def test_units_are_listed_in_sequence_order(db: Database) -> None:
    db.save_unit(make_unit("b", seq=2))
    db.save_unit(make_unit("a", seq=1))
    assert list(db.units()) == ["a", "b"]


def test_check_constraints_reject_unknown_values(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db._connection.execute(
            "INSERT INTO units (uid, kind, seq, source, status, updated_at) "
            "VALUES ('x', 'bogus', 0, '', 'pending', '')"
        )


def test_slots_replacement_and_obsolescence(db: Database) -> None:
    db.save_unit(make_unit("a", seq=0))
    db.save_unit(make_unit("b", seq=1))
    db.replace_slots([Slot("a", "f.rpy", 5, 3, "fp"), Slot("b", "f.rpy", 9, 7, "fp2")])
    assert db.count_slots() == 2
    assert db.mark_orphans_obsolete("t") == 0

    db.replace_slots([Slot("a", "f.rpy", 5, 3, "fp")])
    assert db.mark_orphans_obsolete("t1") == 1
    assert db.mark_orphans_obsolete("t2") == 0
    assert db.units()["b"].status is Status.OBSOLETE
    assert db.units()["a"].status is Status.PENDING


def test_slot_positions_are_unique(db: Database) -> None:
    db.save_unit(make_unit("a"))
    with pytest.raises(sqlite3.IntegrityError):
        db.replace_slots([Slot("a", "f.rpy", 5, 3, "x"), Slot("a", "f.rpy", 5, 3, "y")])


def test_apply_targets_exclude_untranslated_and_obsolete(db: Database) -> None:
    db.save_unit(make_unit("a", seq=0, target="A", status=Status.TRANSLATED))
    db.save_unit(make_unit("b", seq=1))
    db.save_unit(make_unit("c", seq=2, target="C", status=Status.OBSOLETE))
    db.replace_slots(
        [
            Slot("a", "f.rpy", 5, 3, "x"),
            Slot("b", "f.rpy", 8, 6, "y"),
            Slot("c", "f.rpy", 9, 7, "z"),
        ]
    )
    targets = db.apply_targets()
    assert [(t.slot.uid, t.target, t.kind) for t in targets] == [("a", "A", UnitKind.DIALOGUE)]


def test_fingerprint_update(db: Database) -> None:
    db.save_unit(make_unit("a", target="A", status=Status.TRANSLATED))
    db.replace_slots([Slot("a", "f.rpy", 5, 3, "old")])
    db.update_fingerprints([("f.rpy", 5, "new")])
    assert db.apply_targets()[0].slot.fingerprint == "new"


def test_statistics_group_by_kind_and_status(db: Database) -> None:
    db.save_unit(make_unit("a", seq=0))
    db.save_unit(make_unit("b", seq=1, kind=UnitKind.UI, status=Status.TRANSLATED, target="B"))
    db.save_unit(make_unit("c", seq=2, target="C", status=Status.TRANSLATED))
    assert db.stats() == [
        (UnitKind.DIALOGUE, Status.PENDING, 1),
        (UnitKind.DIALOGUE, Status.TRANSLATED, 1),
        (UnitKind.UI, Status.TRANSLATED, 1),
    ]


def test_transaction_rolls_back_on_error(db: Database) -> None:
    with pytest.raises(RuntimeError), db.transaction():
        db.save_unit(make_unit("a"))
        raise RuntimeError
    assert db.units() == {}
    with db.transaction():
        db.save_unit(make_unit("a"))
    assert list(db.units()) == ["a"]


def test_meta_values(db: Database) -> None:
    assert db.get_meta("k") is None
    db.set_meta("k", "v1")
    db.set_meta("k", "v2")
    assert db.get_meta("k") == "v2"


def test_save_units_bulk(db: Database) -> None:
    # 1. Empty sequence does not fail
    db.save_units_bulk([])
    assert db.units() == {}

    # 2. Insert multiple units
    u1 = make_unit("u1", seq=0, source="Line 1")
    u2 = make_unit("u2", seq=1, source="Line 2")
    db.save_units_bulk([u1, u2])
    stored = db.units()
    assert len(stored) == 2
    assert stored["u1"].source == "Line 1"
    assert stored["u2"].source == "Line 2"

    # 3. Upsert update on conflict
    u1_updated = make_unit("u1", seq=0, source="Line 1", target="Dòng 1", status=Status.TRANSLATED)
    u3 = make_unit("u3", seq=2, source="Line 3")
    db.save_units_bulk([u1_updated, u3])
    stored = db.units()
    assert len(stored) == 3
    assert stored["u1"].target == "Dòng 1"
    assert stored["u1"].status == Status.TRANSLATED
    assert stored["u3"].source == "Line 3"


def test_recent_translated_units(db: Database) -> None:
    units = [
        make_unit("u0", seq=0, source="S0", target="T0", status=Status.TRANSLATED),
        make_unit("u1", seq=1, source="S1", status=Status.PENDING),
        make_unit("u2", seq=2, source="S2", target="T2", status=Status.TRANSLATED),
        make_unit("u3", seq=3, source="S3", target="T3", status=Status.REVIEWED),
        make_unit("u4", seq=4, source="S4", target="T4", status=Status.QA_FLAGGED),
        make_unit("u5", seq=5, source="S5", target="T5", status=Status.FAILED),
        make_unit("u6", seq=6, source="S6", target="T6", status=Status.TRANSLATED),
    ]
    db.save_units_bulk(units)

    # Without before_seq: returns most recent translated units in chronological order
    recent = db.recent_translated_units(limit=3)
    assert [u.uid for u in recent] == ["u3", "u4", "u6"]

    # With before_seq=5 (before seq 5, candidates seq 0, 2, 3, 4)
    recent_before = db.recent_translated_units(limit=2, before_seq=5)
    assert [u.uid for u in recent_before] == ["u3", "u4"]

    # With before_seq=0 (nothing before 0)
    assert db.recent_translated_units(limit=5, before_seq=0) == []


def test_sanitize_stored_translations(db: Database) -> None:
    # 1. Units with speaker tags and quotes
    u1 = make_unit("u1", seq=0, source="Hello", target='[MC] "Xin chào!"', speaker="MC")
    u2 = make_unit("u2", seq=1, source="Look", target="Nhìn kìa", speaker="Dyna")
    db.save_units_bulk([u1, u2])

    # 2. TM entry with speaker tags and quotes
    db.save_tm(TMEntry(src_norm="huh", target='[Dyna] "Hả?"', uses=1))
    db.save_tm(TMEntry(src_norm="ok", target="Được rồi", uses=1))

    cleaned_units, cleaned_tm = db.sanitize_stored_translations()
    assert cleaned_units == 1
    assert cleaned_tm == 1

    # Verify unit sanitized
    u1_fresh = db.get_unit("u1")
    assert u1_fresh is not None
    assert u1_fresh.target == "Xin chào!"

    # Verify TM sanitized
    tm_fresh = db.get_tm("huh")
    assert tm_fresh is not None
    assert tm_fresh.target == "Hả?"
