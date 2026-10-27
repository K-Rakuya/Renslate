"""Empirical adversarial stress tests for Renslate Milestone 1 persistence and synchronization.

Stress tests:
1. SQLite Migration 3 on fresh database and pre-existing v2 database with rich legacy data.
2. Bulk insertion of 5000+ relationships in a single atomic transaction.
3. Bidirectional synchronization between SQLite and relationships.json.
4. Adversarial examination of sync edge cases:
   - Modifying DB when relationships.json exists.
   - Overwrite protection asymmetry between DB and JSON.
5. Corrupted JSON resilience: malformed syntax, schema violations, missing fields.
6. Human curation overwrite protection in RelationshipGraph.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from renslate.core.db import MIGRATIONS, Database
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
)
from renslate.engine.graph import RelationshipGraph

# ─────────────────────────────────────────────────────────────────────────────
# 1. SQLite Migration 3 Stress Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMigration3Stress:
    """Stress tests for SQLite Migration 3."""

    def test_fresh_database_migration(self, tmp_path: Path) -> None:
        """A fresh database cleanly migrates to schema v3 with all tables and columns."""
        db_path = tmp_path / "fresh.db"
        with Database.open(db_path, create=True) as db:
            assert db.schema_version == 3

            # Verify character_relationships table and index
            conn = db._connection
            table_info = conn.execute(
                "PRAGMA table_info(character_relationships)"
            ).fetchall()
            col_names = {row["name"] for row in table_info}
            expected_cols = {
                "source_tag",
                "target_tag",
                "kind",
                "speaker_pronoun",
                "listener_pronoun",
                "confidence",
                "source_type",
                "notes",
            }
            assert expected_cols.issubset(col_names)

            indices = conn.execute(
                "PRAGMA index_list(character_relationships)"
            ).fetchall()
            idx_names = {row["name"] for row in indices}
            assert "idx_character_relationships_target" in idx_names

            # Verify characters table has 5 new persona columns
            char_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(characters)").fetchall()
            }
            expected_persona_cols = {
                "archetype",
                "personality_traits",
                "tone_and_voice",
                "socioeconomic_status",
                "sample_dialogue",
            }
            assert expected_persona_cols.issubset(char_cols)

            # Test CRUD on fresh database
            char = CharacterProfile(
                tag="liana",
                display_name="Liana",
                gender="female",
                role="heroine",
                archetype="Tsundere Mage",
                personality_traits=("proud", "loyal", "stubborn"),
                tone_and_voice="Sharp, feisty, softens when flustered",
                socioeconomic_status="Noble Archmage",
                sample_dialogue=("B-Baka! It's not like I wanted to help you!",),
            )
            db.save_character(char)
            fetched = db.get_characters()
            assert len(fetched) == 1
            assert fetched["liana"].tag == "liana"
            assert fetched["liana"].archetype == "Tsundere Mage"
            assert fetched["liana"].personality_traits == ("proud", "loyal", "stubborn")
            assert fetched["liana"].sample_dialogue == (
                "B-Baka! It's not like I wanted to help you!",
            )

            edge = RelationshipEdge(
                source_tag="hero",
                target_tag="liana",
                kind=RelationshipKind.ROMANCE,
                speaker_pronoun="anh",
                listener_pronoun="em",
                confidence=0.95,
                source_type="manual",
                notes="Childhood promise",
            )
            db.save_relationship(edge)
            fetched_edge = db.get_relationship("hero", "liana")
            assert fetched_edge is not None
            assert fetched_edge.kind == RelationshipKind.ROMANCE
            assert fetched_edge.speaker_pronoun == "anh"
            assert fetched_edge.listener_pronoun == "em"
            assert fetched_edge.confidence == 0.95

    def test_legacy_v2_to_v3_migration_with_full_dataset(self, tmp_path: Path) -> None:
        """A pre-existing legacy v2 database migrates to v3 preserving all legacy data."""
        db_path = tmp_path / "legacy_v2.db"

        # 1. Build authentic v2 database using migration scripts 1 and 2 directly
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(
            f"BEGIN;\n{MIGRATIONS[0]}\nPRAGMA user_version = 1;\nCOMMIT;"
        )
        conn.executescript(
            f"BEGIN;\n{MIGRATIONS[1]}\nPRAGMA user_version = 2;\nCOMMIT;"
        )

        # 2. Seed extensive legacy data
        # Insert 100 units
        for i in range(100):
            conn.execute(
                "INSERT INTO units (uid, kind, seq, ref, speaker, label, source, masked_source, "
                "target, status, origin, qa_flags, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"uid_{i:04d}",
                    "dialogue" if i % 2 == 0 else "string",
                    i,
                    f"game/script.rpy:{i+10}",
                    f"char_{i % 5}",
                    f"lbl_{i // 20}",
                    f"Source sentence {i}",
                    f"Source sentence {i}",
                    f"Target sentence {i}" if i % 3 == 0 else None,
                    "translated" if i % 3 == 0 else "pending",
                    "llm" if i % 3 == 0 else None,
                    json.dumps(["tag_check"] if i % 5 == 0 else []),
                    "2026-10-25T10:00:00Z",
                ),
            )

        # Insert 15 legacy characters (without new persona columns)
        for i in range(15):
            conn.execute(
                "INSERT INTO characters "
                "(tag, display_name, gender, role, speech_style, pronoun_map) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    f"char_{i}",
                    f"Character {i}",
                    "female" if i % 2 == 0 else "male",
                    f"Role {i}",
                    f"Style {i}",
                    json.dumps({f"char_{(i+1)%15}": "ngươi"}),
                ),
            )

        # Insert TM entries, glossary entries, scenes, batches
        for i in range(30):
            conn.execute(
                "INSERT INTO tm (src_norm, target, uses, origin, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"norm_{i}", f"dịch_{i}", i + 1, "manual", "2026-10-25T10:00:00Z"),
            )
            conn.execute(
                "INSERT INTO glossary (term, translation, kind, case_sensitive, aliases, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (f"term_{i}", f"thuật ngữ {i}", "general", 0, "[]", f"note {i}"),
            )

        conn.commit()
        conn.close()

        # 3. Open through Database.open() to trigger Migration 3
        with Database.open(db_path, create=False) as db:
            assert db.schema_version == 3

            # Verify all 100 units intact
            units = list(db.iter_units())
            assert len(units) == 100
            assert units[0].uid == "uid_0000"
            assert units[99].uid == "uid_0099"

            # Verify all 15 characters migrated with default persona fields
            chars = db.get_characters()
            assert len(chars) == 15
            for c in chars.values():
                assert c.tag.startswith("char_")
                assert c.display_name.startswith("Character ")
                assert c.archetype == ""
                assert c.personality_traits == ()
                assert c.tone_and_voice == ""
                assert c.socioeconomic_status == ""
                assert c.sample_dialogue == ()

            # 4. Update an existing character with new persona fields
            updated_char = CharacterProfile(
                tag="char_0",
                display_name="Character 0",
                gender="female",
                role="Main Mentor",
                speech_style="Style 0",
                pronoun_map={"char_1": "ngươi"},
                archetype="Wise Elder",
                personality_traits=("calm", "perceptive"),
                tone_and_voice="Gentle and archaic",
                socioeconomic_status="Hermit Master",
                sample_dialogue=("Con đường phía trước còn dài...",),
            )
            db.save_character(updated_char)

            char_0 = db.get_characters()["char_0"]
            assert char_0.archetype == "Wise Elder"
            assert char_0.personality_traits == ("calm", "perceptive")
            assert char_0.sample_dialogue == ("Con đường phía trước còn dài...",)

            # 5. Insert relationship edges into migrated DB
            db.save_relationship(
                RelationshipEdge(
                    "char_0",
                    "char_1",
                    RelationshipKind.MENTOR_DISCIPLE,
                    speaker_pronoun="ta",
                    listener_pronoun="con",
                )
            )
            assert db.count_relationships() == 1
            rel = db.get_relationship("char_0", "char_1")
            assert rel is not None
            assert rel.kind == RelationshipKind.MENTOR_DISCIPLE

        # 6. Re-open to confirm idempotency and clean re-access
        with Database.open(db_path, create=False) as db2:
            assert db2.schema_version == 3
            assert db2.count_relationships() == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bulk Insertion of 5000+ Relationships
# ─────────────────────────────────────────────────────────────────────────────


class TestBulkInsertionStress:
    """Adversarial stress test for bulk inserting 5000+ relationships."""

    def test_bulk_insert_5500_relationships_throughput_and_correctness(
        self, tmp_path: Path
    ) -> None:
        """Bulk insertion of 5500 relationships in a single transaction executes rapidly."""
        db_path = tmp_path / "bulk_stress.db"
        kinds = list(RelationshipKind)

        edges: list[RelationshipEdge] = []
        # Generate 5,500 unique directed pairs (110 sources x 50 targets)
        for s_idx in range(110):
            src = f"char_src_{s_idx:03d}"
            for t_idx in range(50):
                tgt = f"char_tgt_{t_idx:03d}"
                kind = kinds[(s_idx + t_idx) % len(kinds)]
                speaker_pronoun = "anh" if s_idx % 2 == 0 else "tôi"
                listener_pronoun = "em" if s_idx % 2 == 0 else "bạn"
                confidence = round(0.5 + ((s_idx * 50 + t_idx) % 500) / 1000.0, 3)
                source_type = "manual" if (s_idx + t_idx) % 3 == 0 else "llm"
                notes = f"Auto-generated edge between {src} and {tgt} (id {s_idx}_{t_idx})"

                edges.append(
                    RelationshipEdge(
                        source_tag=src,
                        target_tag=tgt,
                        kind=kind,
                        speaker_pronoun=speaker_pronoun,
                        listener_pronoun=listener_pronoun,
                        confidence=confidence,
                        source_type=source_type,
                        notes=notes,
                    )
                )

        assert len(edges) == 5500

        with Database.open(db_path, create=True) as db:
            start_time = time.perf_counter()
            db.save_relationships_bulk(edges)
            duration = time.perf_counter() - start_time

            # Performance check: 5,500 rows in SQLite WAL should comfortably complete in < 2.0s
            assert duration < 2.0, f"Bulk insertion took {duration:.2f}s, exceeding 2.0s limit"

            # Integrity checks
            assert db.count_relationships() == 5500

            # Retrieve all edges
            retrieved = db.get_relationships()
            assert len(retrieved) == 5500

            # Query by source tag
            src_edges = db.get_relationships("char_src_042")
            assert len(src_edges) == 50
            assert all(e.source_tag == "char_src_042" for e in src_edges)

            # Query by target tag
            tgt_edges = db.get_relationships(target_tag="char_tgt_015")
            assert len(tgt_edges) == 110
            assert all(e.target_tag == "char_tgt_015" for e in tgt_edges)

            # Single point query
            point_edge = db.get_relationship("char_src_042", "char_tgt_015")
            assert point_edge is not None
            assert point_edge.source_tag == "char_src_042"
            assert point_edge.target_tag == "char_tgt_015"

    def test_bulk_insert_duplicate_pairs_within_same_batch(self, tmp_path: Path) -> None:
        """Duplicate (source_tag, target_tag) pairs inside the same batch upsert cleanly."""
        db_path = tmp_path / "bulk_dups.db"
        with Database.open(db_path, create=True) as db:
            batch = [
                RelationshipEdge(
                    "hero",
                    "villain",
                    RelationshipKind.RIVAL_HOSTILE,
                    speaker_pronoun="ta",
                    listener_pronoun="ngươi",
                    confidence=0.5,
                    notes="initial guess",
                ),
                RelationshipEdge(
                    "hero",
                    "heroine",
                    RelationshipKind.FRIENDSHIP_CLOSE,
                    speaker_pronoun="tôi",
                    listener_pronoun="cậu",
                    confidence=0.7,
                ),
                # Duplicate with updated confidence, pronouns, and notes
                RelationshipEdge(
                    "hero",
                    "villain",
                    RelationshipKind.RIVAL_HOSTILE,
                    speaker_pronoun="ta",
                    listener_pronoun="kẻ thù",
                    confidence=0.99,
                    notes="confirmed rival",
                ),
            ]
            db.save_relationships_bulk(batch)

            assert db.count_relationships() == 2
            edge = db.get_relationship("hero", "villain")
            assert edge is not None
            assert edge.listener_pronoun == "kẻ thù"
            assert edge.confidence == 0.99
            assert edge.notes == "confirmed rival"

    def test_bulk_insert_empty_list_is_noop(self, tmp_path: Path) -> None:
        """Inserting empty list of relationships does not crash or corrupt database."""
        db_path = tmp_path / "bulk_empty.db"
        with Database.open(db_path, create=True) as db:
            db.save_relationships_bulk([])
            assert db.count_relationships() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. JSON Bidirectional Synchronization & Edge Case Stress
# ─────────────────────────────────────────────────────────────────────────────


class TestBidirectionalSyncStress:
    """Stress tests for bidirectional synchronization between SQLite and JSON."""

    def test_sync_loads_updates_from_modified_disk_json_into_db(
        self, tmp_path: Path
    ) -> None:
        """Modifying relationships.json on disk loads new and updated edges into DB."""
        db_path = tmp_path / "sync_test.db"
        json_path = tmp_path / "relationships.json"

        with Database.open(db_path, create=True) as db:
            # 1. Initial graph in JSON
            initial_data = {
                "version": 1,
                "nodes": ["mc", "liana"],
                "relationships": [
                    {
                        "source_tag": "mc",
                        "target_tag": "liana",
                        "kind": "FRIENDSHIP_CASUAL",
                        "speaker_pronoun": "tôi",
                        "listener_pronoun": "bạn",
                        "confidence": 0.8,
                        "source_type": "manual",
                        "notes": "initial",
                    }
                ],
            }
            json_path.write_text(json.dumps(initial_data), encoding="utf-8")

            # 2. Sync into DB
            RelationshipGraph.sync(db, json_path)
            assert db.count_relationships() == 1
            rel = db.get_relationship("mc", "liana")
            assert rel is not None
            assert rel.kind == RelationshipKind.FRIENDSHIP_CASUAL

            # 3. Modify JSON on disk: update existing edge and add a new edge
            modified_data = {
                "version": 1,
                "nodes": ["mc", "liana", "elena"],
                "relationships": [
                    {
                        "source_tag": "mc",
                        "target_tag": "liana",
                        "kind": "ROMANCE",
                        "speaker_pronoun": "anh",
                        "listener_pronoun": "em",
                        "confidence": 0.95,
                        "source_type": "manual",
                        "notes": "developed feelings",
                    },
                    {
                        "source_tag": "mc",
                        "target_tag": "elena",
                        "kind": "MENTOR_DISCIPLE",
                        "speaker_pronoun": "ta",
                        "listener_pronoun": "đồ nhi",
                        "confidence": 0.9,
                        "source_type": "manual",
                        "notes": "apprentice",
                    },
                ],
            }
            json_path.write_text(json.dumps(modified_data), encoding="utf-8")

            # 4. Re-sync and verify DB updated
            RelationshipGraph.sync(db, json_path)
            assert db.count_relationships() == 2

            updated_liana = db.get_relationship("mc", "liana")
            assert updated_liana is not None
            assert updated_liana.kind == RelationshipKind.ROMANCE
            assert updated_liana.speaker_pronoun == "anh"
            assert updated_liana.listener_pronoun == "em"
            assert updated_liana.confidence == 0.95

            new_elena = db.get_relationship("mc", "elena")
            assert new_elena is not None
            assert new_elena.kind == RelationshipKind.MENTOR_DISCIPLE

    def test_sync_exports_db_to_json_when_json_is_missing(
        self, tmp_path: Path
    ) -> None:
        """Sync creates relationships.json when it does not exist on disk."""
        db_path = tmp_path / "export_test.db"
        json_path = tmp_path / "relationships.json"
        assert not json_path.exists()

        with Database.open(db_path, create=True) as db:
            db.save_relationship(
                RelationshipEdge(
                    "hero",
                    "ally",
                    RelationshipKind.FRIENDSHIP_CLOSE,
                    speaker_pronoun="tôi",
                    listener_pronoun="cậu",
                    confidence=0.88,
                )
            )

            RelationshipGraph.sync(db, json_path)
            assert json_path.is_file()

            exported = json.loads(json_path.read_text(encoding="utf-8"))
            assert len(exported["relationships"]) == 1
            assert exported["relationships"][0]["source_tag"] == "hero"
            assert exported["relationships"][0]["target_tag"] == "ally"
            assert exported["relationships"][0]["kind"] == "FRIENDSHIP_CLOSE"

    def test_sync_propagates_new_db_edge_into_existing_json(
        self, tmp_path: Path
    ) -> None:
        """Adding a brand-new edge in DB propagates into an existing relationships.json."""
        db_path = tmp_path / "sync_new_db.db"
        json_path = tmp_path / "relationships.json"

        with Database.open(db_path, create=True) as db:
            # Step 1: Initial sync with 1 edge
            db.save_relationship(
                RelationshipEdge(
                    "alice",
                    "bob",
                    RelationshipKind.FORMAL_ACQUAINTANCE,
                    speaker_pronoun="tôi",
                    listener_pronoun="anh",
                )
            )
            RelationshipGraph.sync(db, json_path)

            # Step 2: Add second edge directly to DB
            db.save_relationship(
                RelationshipEdge(
                    "alice",
                    "charlie",
                    RelationshipKind.SUPERIOR_SUBORDINATE,
                    speaker_pronoun="tôi",
                    listener_pronoun="cậu",
                )
            )

            # Step 3: Sync
            RelationshipGraph.sync(db, json_path)

            # Step 4: Verify JSON now contains both edges
            data = json.loads(json_path.read_text(encoding="utf-8"))
            pairs = {(r["source_tag"], r["target_tag"]) for r in data["relationships"]}
            assert pairs == {("alice", "bob"), ("alice", "charlie")}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Adversarial Challenges on Sync & Human Overwrite Protection
# ─────────────────────────────────────────────────────────────────────────────


class TestAdversarialSyncAndOverwriteChallenges:
    """Targeted empirical stress tests probing specific edge-case failure modes."""

    def test_challenge_modify_db_overwritten_by_stale_json_when_both_manual(
        self, tmp_path: Path
    ) -> None:
        """EMPIRICAL CHALLENGE: Verify whether modifying an existing manual edge in DB

        is preserved or overwritten by stale JSON when calling sync().
        """
        db_path = tmp_path / "stale_json_challenge.db"
        json_path = tmp_path / "relationships.json"

        with Database.open(db_path, create=True) as db:
            # 1. Establish initial edge in both DB and JSON (source_type='manual', confidence=1.0)
            initial_edge = RelationshipEdge(
                source_tag="hero",
                target_tag="heroine",
                kind=RelationshipKind.FRIENDSHIP_CASUAL,
                speaker_pronoun="tôi",
                listener_pronoun="bạn",
                confidence=1.0,
                source_type="manual",
                notes="Initial meeting",
            )
            db.save_relationship(initial_edge)
            RelationshipGraph.sync(db, json_path)

            # Confirm JSON and DB have initial state
            edge_init = db.get_relationship("hero", "heroine")
            assert edge_init is not None
            assert edge_init.kind == RelationshipKind.FRIENDSHIP_CASUAL

            # 2. User modifies the relationship in SQLite DB (e.g. through Renslate command or API)
            # The relationship progressed to ROMANCE
            db_updated_edge = RelationshipEdge(
                source_tag="hero",
                target_tag="heroine",
                kind=RelationshipKind.ROMANCE,
                speaker_pronoun="anh",
                listener_pronoun="em",
                confidence=1.0,
                source_type="manual",
                notes="Progressed to romance via DB update",
            )
            db.save_relationship(db_updated_edge)

            # Verify DB has ROMANCE before sync
            edge_before_sync = db.get_relationship("hero", "heroine")
            assert edge_before_sync is not None
            assert edge_before_sync.kind == RelationshipKind.ROMANCE

            # 3. Call sync(db, json_path)
            # Mission requirement: "Modify DB, verify sync() updates JSON."
            RelationshipGraph.sync(db, json_path)

            # 4. EMPIRICAL VERIFICATION:
            # What does DB and JSON hold now?
            db_edge_after_sync = db.get_relationship("hero", "heroine")
            assert db_edge_after_sync is not None
            json_data = json.loads(json_path.read_text(encoding="utf-8"))
            json_edge_after_sync = json_data["relationships"][0]

            print(f"\n[EMPIRICAL TEST] DB edge kind after sync: {db_edge_after_sync.kind}")
            print(f"[EMPIRICAL TEST] JSON edge kind after sync: {json_edge_after_sync['kind']}")

            # If stale JSON overwrote DB's modification:
            # db_edge_after_sync.kind is FRIENDSHIP_CASUAL instead of ROMANCE!
            # Let's assert what the contract required:
            is_db_preserved = db_edge_after_sync.kind == RelationshipKind.ROMANCE
            is_json_updated = json_edge_after_sync["kind"] == "ROMANCE"

            assert is_db_preserved and is_json_updated, (
                f"FAILURE: DB modification was overwritten by stale JSON! "
                f"(db={db_edge_after_sync.kind}, json={json_edge_after_sync['kind']})"
            )

    def test_challenge_sync_direction_parameter_is_ignored(
        self, tmp_path: Path
    ) -> None:
        """EMPIRICAL CHALLENGE: Verify whether direction='db_to_json' is respected by sync()."""
        db_path = tmp_path / "sync_dir_challenge.db"
        json_path = tmp_path / "relationships.json"

        with Database.open(db_path, create=True) as db:
            # Seed edge in both DB and JSON
            edge_v1 = RelationshipEdge("x", "y", RelationshipKind.FORMAL_ACQUAINTANCE)
            db.save_relationship(edge_v1)
            RelationshipGraph.sync(db, json_path)

            # Update DB
            edge_v2 = RelationshipEdge("x", "y", RelationshipKind.FRIENDSHIP_CLOSE)
            db.save_relationship(edge_v2)

            # Explicitly request direction='db_to_json'
            RelationshipGraph.sync(db, json_path, direction="db_to_json")

            # Check if JSON was updated to edge_v2
            data = json.loads(json_path.read_text(encoding="utf-8"))
            json_kind = data["relationships"][0]["kind"]
            edge_db = db.get_relationship("x", "y")
            assert edge_db is not None
            db_kind = edge_db.kind

            print(
                f"\n[EMPIRICAL TEST] direction='db_to_json' result: db={db_kind}, json={json_kind}"
            )
            assert json_kind == "FRIENDSHIP_CLOSE", (
                f"FAILURE: sync(direction='db_to_json') was ignored! JSON still has {json_kind}"
            )
            assert db_kind == RelationshipKind.FRIENDSHIP_CLOSE, (
                f"FAILURE: sync(direction='db_to_json') reverted DB to {db_kind}!"
            )

    def test_challenge_human_curation_overwritten_by_llm_json_edge(
        self, tmp_path: Path
    ) -> None:
        """EMPIRICAL CHALLENGE: Verify whether a human manual edge in DB is protected.

        LLM edges in relationships.json with equal confidence must not overwrite manual DB edges.
        """
        db_path = tmp_path / "human_curation_challenge.db"
        json_path = tmp_path / "relationships.json"

        with Database.open(db_path, create=True) as db:
            # 1. Human editor curated a manual edge in DB
            human_edge = RelationshipEdge(
                source_tag="master",
                target_tag="servant",
                kind=RelationshipKind.SERVANT_MASTER,
                speaker_pronoun="ta",
                listener_pronoun="ngươi",
                confidence=1.0,
                source_type="manual",
                notes="Human curated master-servant relationship",
            )
            db.save_relationship(human_edge)

            # 2. Automated LLM extraction pipeline wrote to relationships.json
            # LLM hallucinated or classified them as FRIENDSHIP_CLOSE
            llm_json_data = {
                "version": 1,
                "nodes": ["master", "servant"],
                "relationships": [
                    {
                        "source_tag": "master",
                        "target_tag": "servant",
                        "kind": "FRIENDSHIP_CLOSE",
                        "speaker_pronoun": "mình",
                        "listener_pronoun": "cậu",
                        "confidence": 1.0,
                        "source_type": "llm",
                        "notes": "LLM extraction",
                    }
                ],
            }
            json_path.write_text(json.dumps(llm_json_data), encoding="utf-8")

            # 3. Call sync without forced overwrite
            RelationshipGraph.sync(db, json_path)

            # 4. Check whether human manual edge in DB was preserved
            db_edge = db.get_relationship("master", "servant")
            assert db_edge is not None

            print(f"\n[EMPIRICAL TEST] DB source_type: {db_edge.source_type}, kind: {db_edge.kind}")

            assert db_edge.source_type == "manual", (
                f"CRITICAL OVERWRITE: Human manual edge was overwritten by LLM edge! "
                f"source_type is '{db_edge.source_type}' instead of 'manual', kind='{db_edge.kind}'"
            )
            assert db_edge.kind == RelationshipKind.SERVANT_MASTER, (
                f"CRITICAL OVERWRITE: Kind changed to {db_edge.kind} by automated LLM edge"
            )

    def test_human_curation_protection_in_graph_add_edge(self) -> None:
        """RelationshipGraph.add_edge rejects automated overwrites on manual edges unless forced."""
        graph = RelationshipGraph()

        # Add manual edge
        manual_edge = RelationshipEdge(
            "alice",
            "bob",
            RelationshipKind.ROMANCE,
            speaker_pronoun="em",
            listener_pronoun="anh",
            confidence=1.0,
            source_type="manual",
        )
        assert graph.add_edge(manual_edge) is True

        # Attempt unforced overwrite with LLM edge -> rejected
        llm_edge = RelationshipEdge(
            "alice",
            "bob",
            RelationshipKind.FRIENDSHIP_CLOSE,
            speaker_pronoun="tôi",
            listener_pronoun="cậu",
            confidence=1.0,
            source_type="llm",
        )
        assert graph.add_edge(llm_edge, overwrite_manual=False) is False
        edge_check1 = graph.get_edge("alice", "bob")
        assert edge_check1 is not None
        assert edge_check1.kind == RelationshipKind.ROMANCE
        assert edge_check1.source_type == "manual"

        # Attempt unforced overwrite with heuristic edge -> rejected
        heuristic_edge = RelationshipEdge(
            "alice",
            "bob",
            RelationshipKind.FORMAL_ACQUAINTANCE,
            confidence=0.5,
            source_type="heuristic",
        )
        assert graph.add_edge(heuristic_edge, overwrite_manual=False) is False
        edge_check2 = graph.get_edge("alice", "bob")
        assert edge_check2 is not None
        assert edge_check2.kind == RelationshipKind.ROMANCE

        # Forced overwrite -> allowed
        assert graph.add_edge(llm_edge, overwrite_manual=True) is True
        edge_check3 = graph.get_edge("alice", "bob")
        assert edge_check3 is not None
        assert edge_check3.kind == RelationshipKind.FRIENDSHIP_CLOSE
        assert edge_check3.source_type == "llm"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Corrupted JSON Resilience Stress Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCorruptedJsonResilienceStress:
    """Stress tests for malformed and invalid JSON inputs."""

    def test_corrupted_json_syntax_error_protects_database_from_corruption(
        self, tmp_path: Path
    ) -> None:
        """Syntax errors in relationships.json raise JSONDecodeError without corrupting DB."""
        db_path = tmp_path / "corrupt_syntax.db"
        bad_json = tmp_path / "relationships.json"

        with Database.open(db_path, create=True) as db:
            # Seed 5 valid edges in DB
            for i in range(5):
                db.save_relationship(
                    RelationshipEdge(
                        f"src_{i}",
                        f"tgt_{i}",
                        RelationshipKind.FRIENDSHIP_CLOSE,
                        speaker_pronoun="tôi",
                        listener_pronoun="bạn",
                    )
                )
            assert db.count_relationships() == 5

            # Write corrupted JSON on disk
            bad_json.write_text("{'invalid_json': incomplete bracket,", encoding="utf-8")

            # Call from_json_file
            with pytest.raises(json.JSONDecodeError):
                RelationshipGraph.from_json_file(bad_json)

            # Call sync with corrupted JSON
            with pytest.raises(json.JSONDecodeError):
                RelationshipGraph.sync(db, bad_json)

            # Verify Database is completely unharmed and intact
            assert db.count_relationships() == 5
            for i in range(5):
                edge = db.get_relationship(f"src_{i}", f"tgt_{i}")
                assert edge is not None
                assert edge.kind == RelationshipKind.FRIENDSHIP_CLOSE

    def test_corrupted_json_schema_violations_raise_gracefully(
        self, tmp_path: Path
    ) -> None:
        """Various schema corruptions raise TypeError/ValueError/KeyError without corrupting DB."""
        db_path = tmp_path / "corrupt_schema.db"
        json_file = tmp_path / "relationships.json"

        with Database.open(db_path, create=True) as db:
            db.save_relationship(
                RelationshipEdge(
                    "a", "b", RelationshipKind.ROMANCE, speaker_pronoun="anh", listener_pronoun="em"
                )
            )

            # Case 1: Array of non-dicts
            json_file.write_text("[1, 2, 3, 'string']", encoding="utf-8")
            with pytest.raises((TypeError, ValueError)):
                RelationshipGraph.from_json_file(json_file)

            # Case 2: Relationship item is not a dict
            json_file.write_text(
                json.dumps({"relationships": ["not_a_dict"]}), encoding="utf-8"
            )
            with pytest.raises(TypeError):
                RelationshipGraph.from_json_file(json_file)

            # Case 3: Missing required key 'source_tag'
            json_file.write_text(
                json.dumps({"relationships": [{"target_tag": "b", "kind": "ROMANCE"}]}),
                encoding="utf-8",
            )
            with pytest.raises(KeyError):
                RelationshipGraph.from_json_file(json_file)

            # Case 4: Invalid relationship kind
            json_file.write_text(
                json.dumps(
                    {
                        "relationships": [
                            {
                                "source_tag": "a",
                                "target_tag": "b",
                                "kind": "COMPLETELY_INVALID_KIND_XYZ",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with pytest.raises(ValueError):
                RelationshipGraph.from_json_file(json_file)

            # Database remains 100% intact after all failure cases
            assert db.count_relationships() == 1
            rel = db.get_relationship("a", "b")
            assert rel is not None
            assert rel.kind == RelationshipKind.ROMANCE
