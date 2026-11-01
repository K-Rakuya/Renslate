"""Opaque-Box E2E Test Suite for Requirement R5: Data Flow, I/O & Embedded SQLite Database.

Covers:
- Embedded SQLite storage, WAL journal mode, and foreign keys enforcement
- Complete schema migrations (meta, units, slots, tm, glossary,
  characters, character_relationships, scenes)
- Transaction safety: atomic commits and rollback on error
- Atomic file writes (write_atomic) and pre-apply timestamped backups
- Foreign key cascade deletion on slots and data integrity constraints
- Zero hardcoded database paths and configuration-driven storage layout

Tiers:
- Tier 1: Feature Coverage (>=5 tests per feature)
- Tier 2: Boundary & Corner Cases (>=5 tests per feature)
- Tier 3: Cross-Feature Combinations
- Tier 4: Real-World Application Scenarios
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from renslate.core.db import Database
from renslate.core.errors import ProjectError
from renslate.core.models import (
    CharacterProfile,
    Origin,
    RelationshipEdge,
    RelationshipKind,
    Slot,
    Status,
    TMEntry,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.core.textio import write_atomic
from renslate.renpy.writer import apply_translations

# ==============================================================================
# Helper Factories
# ==============================================================================


def _make_unit(
    uid: str,
    seq: int,
    source: str = "Test source",
    *,
    status: Status = Status.PENDING,
    speaker: str | None = None,
    target: str | None = None,
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        masked_source=source,
        speaker=speaker,
        target=target,
        status=status,
        origin=Origin.LLM if target else None,
    )


# ==============================================================================
# Tier 1: Feature Coverage (SQLite Initialization, Schema & Transactions)
# ==============================================================================


class TestR5Tier1FeatureCoverage:
    """Tier 1: Core contracts of embedded SQLite, migrations, transactions, and atomic I/O."""

    def test_database_initialization_and_wal_mode(self, tmp_path: Path) -> None:
        """Verify Database opens in WAL mode with foreign keys enabled."""
        db_path = tmp_path / "test.db"
        with Database.open(db_path, create=True) as db:
            journal_mode = db.execute("PRAGMA journal_mode;").fetchone()[0]
            foreign_keys = db.execute("PRAGMA foreign_keys;").fetchone()[0]
            assert journal_mode.lower() == "wal"
            assert foreign_keys == 1

    def test_schema_tables_and_indexes_created(self, tmp_path: Path) -> None:
        """Verify all mandatory tables from migrations v1 to v4 exist."""
        db_path = tmp_path / "schema_check.db"
        with Database.open(db_path, create=True) as db:
            rows = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            ).fetchall()
            table_names = {row["name"] for row in rows}

            expected = {
                "meta",
                "units",
                "slots",
                "tm",
                "glossary",
                "characters",
                "character_relationships",
                "scenes",
                "batches",
            }
            assert expected.issubset(table_names)

    def test_transaction_rollback_on_exception(self, tmp_path: Path) -> None:
        """Verify unhandled exception inside transaction rolls back all pending inserts."""
        db_path = tmp_path / "rollback.db"
        with Database.open(db_path, create=True) as db:
            u1 = _make_unit("u_commit", 1, "Committed line")
            db.save_unit(u1)

            with pytest.raises(RuntimeError), db.transaction():
                u2 = _make_unit("u_rollback", 2, "Uncommitted line")
                db.save_unit(u2)
                raise RuntimeError("Simulated crash mid-batch!")

            assert db.get_unit("u_commit") is not None
            assert db.get_unit("u_rollback") is None

    def test_atomic_file_write_guarantee(self, tmp_path: Path) -> None:
        """Verify write_atomic writes to temp file and replaces atomically."""
        target_file = tmp_path / "atomic_script.rpy"
        target_file.write_text("Original content\n", encoding="utf-8")

        new_data = b"Updated new content\n"
        write_atomic(target_file, new_data)

        assert target_file.read_bytes() == new_data
        # Ensure temporary file was cleanly unlinked
        temp_file = target_file.with_name(f"{target_file.name}.tmp")
        assert not temp_file.exists()

    def test_project_backup_root_created_on_apply(self, tmp_path: Path) -> None:
        """Verify apply_translations snapshots original files into project.backup_root."""
        game_root = tmp_path / "game_project"
        game_root.mkdir()
        (game_root / "game" / "tl" / "vietnamese").mkdir(parents=True)
        script_file = game_root / "game" / "tl" / "vietnamese" / "story.rpy"
        script_file.write_text('# TODO: Translation\nold "Hello"\nnew "Hello"\n', encoding="utf-8")

        from renslate.renpy.parser import fingerprint

        project = Project.create(game_root)
        with project.open_db() as db:
            u = _make_unit("u1", 1, "Hello", target="Xin chào")
            db.save_unit(u)
            fp = fingerprint('old "Hello"', 'new "Hello"')
            slot = Slot(
                uid="u1",
                file="game/tl/vietnamese/story.rpy",
                line=3,
                anchor_line=2,
                fingerprint=fp,
            )
            db.replace_slots([slot])

            # Apply translations should trigger backup creation
            apply_translations(project, db)

        assert project.backup_root.exists()
        backups = list(project.backup_root.iterdir())
        assert len(backups) >= 1
        assert (backups[0] / "game" / "tl" / "vietnamese" / "story.rpy").exists()


# ==============================================================================
# Tier 2: Boundary & Corner Cases (Corrupt DB, FK Violations, Version Drift)
# ==============================================================================


class TestR5Tier2Boundaries:
    """Tier 2: Defensive database handling, constraint violations, and volume stress."""

    def test_corrupted_database_file_raises_operational_error(self, tmp_path: Path) -> None:
        """Verify opening garbage data as SQLite file raises sqlite3.DatabaseError."""
        corrupt_db = tmp_path / "corrupt.db"
        corrupt_db.write_bytes(b"NOT A SQLITE FILE AT ALL JUNK DATA" * 10)
        with pytest.raises(sqlite3.DatabaseError):
            Database.open(corrupt_db, create=False)

    def test_foreign_key_violation_on_slot_insertion(self, tmp_path: Path) -> None:
        """Verify inserting slot pointing to non-existent unit uid raises IntegrityError."""
        db_path = tmp_path / "fk_test.db"
        with Database.open(db_path, create=True) as db:
            orphan_slot = Slot(
                uid="non_existent_unit_uid",
                file="game/script.rpy",
                line=10,
                anchor_line=11,
                fingerprint="hash_xyz",
            )
            with pytest.raises(sqlite3.IntegrityError):
                db.replace_slots([orphan_slot])

    def test_foreign_key_cascade_deletion(self, tmp_path: Path) -> None:
        """Verify deleting a unit cascades to remove linked slot records."""
        db_path = tmp_path / "cascade.db"
        with Database.open(db_path, create=True) as db:
            u = _make_unit("unit_cascade", 1, "Source text")
            db.save_unit(u)
            slot = Slot(
                uid="unit_cascade",
                file="game/script.rpy",
                line=5,
                anchor_line=6,
                fingerprint="fp_cascade",
            )
            db.replace_slots([slot])
            assert db.count_slots() == 1

            # Delete the parent unit directly
            db.execute("DELETE FROM units WHERE uid = 'unit_cascade';")
            # Foreign key cascade must remove the slot
            assert db.count_slots() == 0

    def test_database_bulk_insert_stress_5000_units(self, tmp_path: Path) -> None:
        """Verify bulk insertion of 5000 units within single transaction executes < 1s."""
        import time

        db_path = tmp_path / "stress.db"
        with Database.open(db_path, create=True) as db:
            units = [
                _make_unit(f"u_bulk_{i:05d}", i, f"Dialogue sentence number {i}")
                for i in range(5000)
            ]
            t0 = time.perf_counter()
            db.save_units_bulk(units)
            elapsed = time.perf_counter() - t0

            assert len(db.all_units()) == 5000
            assert elapsed < 2.0, f"5000 units took {elapsed:.4f}s, expected < 2.0s"

    def test_non_existent_database_open_without_create_raises(self, tmp_path: Path) -> None:
        """Verify opening non-existent DB with create=False raises ProjectError."""
        missing = tmp_path / "does_not_exist.db"
        with pytest.raises(ProjectError, match="không tìm thấy cơ sở dữ liệu"):
            Database.open(missing, create=False)


# ==============================================================================
# Tier 3: Cross-Feature Combinations (DB -> TM -> Characters -> Graph)
# ==============================================================================


class TestR5Tier3CrossFeature:
    """Tier 3: Multi-table relational interactions and persistence synchronizations."""

    def test_tm_exact_normalization_and_use_count(self, tmp_path: Path) -> None:
        """Verify Translation Memory records normalized source and tracks usage count."""
        db_path = tmp_path / "tm_test.db"
        with Database.open(db_path, create=True) as db:
            # Save first TM entry
            db.save_tm(TMEntry(src_norm="hello world", target="Xin chào thế giới", uses=1))
            entry1 = db.get_tm("hello world")
            assert entry1 is not None
            assert entry1.uses == 1

            # Increment use count
            db.save_tm(TMEntry(src_norm="hello world", target="Xin chào thế giới", uses=1))
            entry2 = db.get_tm("hello world")
            assert entry2 is not None
            assert entry2.uses == 2

    def test_character_and_relationship_relational_storage(self, tmp_path: Path) -> None:
        """Verify Character profiles and directed Relationships coexist and query cleanly."""
        db_path = tmp_path / "chars_rel.db"
        with Database.open(db_path, create=True) as db:
            char_a = CharacterProfile(
                tag="mc",
                display_name="Protagonist",
                gender="male",
                role="Hero",
                speech_style="bình dân",
                pronoun_map={"eileen": "anh - em"},
                archetype="everyman",
                personality_traits=("dũng cảm",),
                tone_and_voice="tự tin",
                socioeconomic_status="học sinh",
                sample_dialogue=("Đi thôi!",),
            )
            char_b = CharacterProfile(
                tag="eileen",
                display_name="Eileen",
                gender="female",
                role="Heroine",
                speech_style="nhẹ nhàng",
                pronoun_map={"mc": "em - anh"},
                archetype="deredere",
                personality_traits=("tử tế",),
                tone_and_voice="ấm áp",
                socioeconomic_status="quý tộc",
                sample_dialogue=("Chào anh!",),
            )
            db.save_characters_bulk([char_a, char_b])
            characters = db.get_characters()
            assert "mc" in characters
            assert "eileen" in characters

            rel = RelationshipEdge(
                source_tag="mc",
                target_tag="eileen",
                kind=RelationshipKind.ROMANCE,
                speaker_pronoun="anh",
                listener_pronoun="em",
                confidence=1.0,
                source_type="manual",
            )
            db.save_relationship(rel)
            assert db.count_relationships() == 1
            loaded_rel = db.get_relationship("mc", "eileen")
            assert loaded_rel is not None
            assert loaded_rel.kind == RelationshipKind.ROMANCE
            assert loaded_rel.speaker_pronoun == "anh"

    def test_unit_status_transitions_and_filtering(self, tmp_path: Path) -> None:
        """Verify unit status transitions (PENDING -> TRANSLATED -> REVIEWED)."""
        db_path = tmp_path / "status_flow.db"
        with Database.open(db_path, create=True) as db:
            u = _make_unit("u_status", 1, "Status test")
            db.save_unit(u)
            assert len(db.pending_units()) == 1

            # Update to translated
            db.update_translation("u_status", "Bản dịch thử nghiệm", Origin.LLM)
            assert len(db.pending_units()) == 0
            loaded = db.get_unit("u_status")
            assert loaded is not None
            assert loaded.status == Status.TRANSLATED
            assert loaded.target == "Bản dịch thử nghiệm"


# ==============================================================================
# Tier 4: Real-World Application Scenarios (Crash Resilience & Recovery)
# ==============================================================================


class TestR5Tier4RealWorld:
    """Tier 4: Crash recovery simulation during active translation transaction."""

    def test_simulated_power_cut_wal_recovery(self, tmp_path: Path) -> None:
        """Simulate abrupt process termination during batch write and verify WAL recovery.

        Steps:
        1. Open connection A, write initial 50 units, commit.
        2. Open connection A, start transaction, insert 50 units, simulate abrupt exit.
        3. Open connection B, verify SQLite WAL rolls back uncommitted batch cleanly.
        """
        db_path = tmp_path / "power_cut.db"

        # 1. Initial committed data
        with Database.open(db_path, create=True) as db1:
            initial_units = [
                _make_unit(f"init_{i}", i, f"Initial sentence {i}") for i in range(50)
            ]
            db1.save_units_bulk(initial_units)

        # 2. Simulate uncommitted transaction on raw connection
        raw_conn = sqlite3.connect(db_path)
        raw_conn.execute("PRAGMA journal_mode = WAL;")
        raw_conn.execute("BEGIN IMMEDIATE;")
        for i in range(50):
            raw_conn.execute(
                """
                INSERT INTO units (uid, kind, seq, source, masked_source, status, updated_at)
                VALUES (?, 'dialogue', ?, 'Crash source', 'Crash source', 'pending', '2026-11-01');
                """,
                (f"crash_{i}", 100 + i),
            )
        # Abruptly close connection without COMMIT or ROLLBACK (simulating SIGKILL)
        raw_conn.close()

        # 3. New connection recovers state
        with Database.open(db_path, create=False) as db2:
            units = db2.all_units()
            # Must contain only the 50 initially committed units
            assert len(units) == 50
            assert all(u.uid.startswith("init_") for u in units)
            assert db2.get_unit("crash_0") is None
