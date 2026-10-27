"""Comprehensive unit tests for RelationshipGraph, RelationshipKind, RelationshipEdge,
extended CharacterProfile, and database operations.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from renslate.core.db import MIGRATIONS, Database
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
)
from renslate.engine.graph import RelationshipGraph

# ==============================================================================
# 1. RelationshipKind Semantic Enum Tests
# ==============================================================================


class TestRelationshipKindUnit:
    def test_all_ten_values_present(self) -> None:
        expected = {
            "FAMILY_ELDER",
            "FAMILY_SIBLING",
            "ROMANCE",
            "MENTOR_DISCIPLE",
            "SUPERIOR_SUBORDINATE",
            "FRIENDSHIP_CLOSE",
            "FRIENDSHIP_CASUAL",
            "FORMAL_ACQUAINTANCE",
            "RIVAL_HOSTILE",
            "SERVANT_MASTER",
        }
        assert {k.value for k in RelationshipKind} == expected
        assert len(RelationshipKind) == 10

    def test_string_subclassing_and_equality(self) -> None:
        assert issubclass(RelationshipKind, str)
        assert RelationshipKind.ROMANCE == "ROMANCE"
        assert RelationshipKind.ROMANCE.value == "ROMANCE"
        assert f"{RelationshipKind.ROMANCE}" in ("RelationshipKind.ROMANCE", "ROMANCE")

    def test_from_str_case_and_formatting_resilience(self) -> None:
        assert RelationshipKind.from_str("romance") == RelationshipKind.ROMANCE
        assert RelationshipKind.from_str("FAMILY_SIBLING") == RelationshipKind.FAMILY_SIBLING
        assert RelationshipKind.from_str("family sibling") == RelationshipKind.FAMILY_SIBLING
        assert RelationshipKind.from_str("mentor-disciple") == RelationshipKind.MENTOR_DISCIPLE
        assert RelationshipKind.from_str("  servant_master  ") == RelationshipKind.SERVANT_MASTER

    def test_from_str_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown relationship kind"):
            RelationshipKind.from_str("INVALID_KIND_TYPE")


# ==============================================================================
# 2. RelationshipEdge Domain Model Tests
# ==============================================================================


class TestRelationshipEdgeUnit:
    def test_edge_creation_and_defaults(self) -> None:
        edge = RelationshipEdge(
            source_tag="hero",
            target_tag="mentor",
            kind=RelationshipKind.MENTOR_DISCIPLE,
        )
        assert edge.source_tag == "hero"
        assert edge.target_tag == "mentor"
        assert edge.kind == RelationshipKind.MENTOR_DISCIPLE
        assert edge.speaker_pronoun == ""
        assert edge.listener_pronoun == ""
        assert edge.confidence == 1.0
        assert edge.source_type == "manual"
        assert edge.notes == ""

    def test_edge_immutability_and_slots(self) -> None:
        edge = RelationshipEdge(
            source_tag="ren",
            target_tag="liana",
            kind=RelationshipKind.ROMANCE,
            speaker_pronoun="anh",
            listener_pronoun="em",
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            edge.speaker_pronoun = "tôi"  # type: ignore[misc]

    def test_edge_hashability_and_set_equality(self) -> None:
        e1 = RelationshipEdge("a", "b", RelationshipKind.FRIENDSHIP_CLOSE, "tớ", "cậu")
        e2 = RelationshipEdge("a", "b", RelationshipKind.FRIENDSHIP_CLOSE, "tớ", "cậu")
        assert e1 == e2
        assert hash(e1) == hash(e2)
        assert len({e1, e2}) == 1

    def test_edge_serialization_roundtrip(self) -> None:
        edge = RelationshipEdge(
            source_tag="a",
            target_tag="b",
            kind=RelationshipKind.RIVAL_HOSTILE,
            speaker_pronoun="ta",
            listener_pronoun="ngươi",
            confidence=0.9,
            source_type="llm",
            notes="Childhood rivalry",
        )
        d = edge.to_dict()
        assert d["source_tag"] == "a"
        assert d["kind"] == "RIVAL_HOSTILE"
        restored = RelationshipEdge.from_dict(d)
        assert restored == edge

    def test_edge_from_dict_validation(self) -> None:
        with pytest.raises(TypeError, match="Expected dict or Mapping"):
            RelationshipEdge.from_dict("invalid_payload")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Expected RelationshipKind or str"):
            RelationshipEdge.from_dict({"source_tag": "a", "target_tag": "b", "kind": 12345})


# ==============================================================================
# 3. Extended CharacterProfile Model Tests
# ==============================================================================


class TestCharacterProfileUnit:
    def test_positional_backward_compatibility(self) -> None:
        p = CharacterProfile("mc", "Protagonist")
        assert p.tag == "mc"
        assert p.display_name == "Protagonist"
        assert p.gender == "unknown"
        assert p.role == ""
        assert p.speech_style == ""
        assert p.pronoun_map == {}
        assert p.archetype == ""
        assert p.personality_traits == ()
        assert p.tone_and_voice == ""
        assert p.socioeconomic_status == ""
        assert p.sample_dialogue == ()

    def test_extended_fields_instantiation(self) -> None:
        p = CharacterProfile(
            tag="rin",
            display_name="Rin Tohsaka",
            gender="female",
            role="heroine",
            archetype="tsundere",
            personality_traits=("proud", "capable", "caring"),
            tone_and_voice="sharp but gentle deep down",
            socioeconomic_status="noble mage",
            sample_dialogue=("Don't get the wrong idea, it's not like I did this for you!",),
        )
        assert p.archetype == "tsundere"
        assert p.personality_traits == ("proud", "capable", "caring")
        assert p.tone_and_voice == "sharp but gentle deep down"
        assert p.socioeconomic_status == "noble mage"
        assert len(p.sample_dialogue) == 1

    def test_tuple_coercion_from_lists(self) -> None:
        p = CharacterProfile(
            tag="artoria",
            display_name="Saber",
            personality_traits=["chivalrous", "loyal"],  # type: ignore[arg-type]
            sample_dialogue=["I ask of you, are you my Master?"],  # type: ignore[arg-type]
        )
        assert isinstance(p.personality_traits, tuple)
        assert isinstance(p.sample_dialogue, tuple)
        assert p.personality_traits == ("chivalrous", "loyal")

    def test_profile_to_dict_and_from_dict_roundtrip(self) -> None:
        p = CharacterProfile(
            tag="liana",
            display_name="Liana",
            archetype="childhood friend",
            personality_traits=("sweet", "supportive"),
            sample_dialogue=("Welcome home, Ren!",),
        )
        d = p.to_dict()
        assert d["tag"] == "liana"
        assert d["archetype"] == "childhood friend"
        restored = CharacterProfile.from_dict(d)
        assert restored == p

    def test_profile_from_dict_explicit_tag(self) -> None:
        d = {
            "display_name": "Sakura",
            "gender": "female",
            "archetype": "yandere",
            "personality_traits": "quiet, devoted",
            "sample_dialogue": "Senpai...",
        }
        p = CharacterProfile.from_dict(d, tag="sakura")
        assert p.tag == "sakura"
        assert p.display_name == "Sakura"
        assert p.archetype == "yandere"
        assert p.personality_traits == ("quiet", "devoted")
        assert p.sample_dialogue == ("Senpai...",)


# ==============================================================================
# 4. RelationshipGraph Engine Unit Tests
# ==============================================================================


class TestRelationshipGraphUnit:
    def test_empty_graph(self) -> None:
        graph = RelationshipGraph()
        assert graph.node_count == 0
        assert graph.edge_count == 0
        assert graph.nodes == frozenset()
        assert graph.all_edges() == []
        assert graph.edges == []
        assert graph.get_edge("a", "b") is None
        assert graph.get_bilateral("a", "b") == (None, None)
        assert graph.get_neighbors("a") == set()
        assert graph.find_path("a", "b") == []

    def test_add_and_get_edge_dual_adjacency(self) -> None:
        graph = RelationshipGraph()
        edge = RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em")
        assert graph.add_edge(edge) is True

        assert graph.has_node("ren")
        assert graph.has_node("liana")
        assert graph.has_edge("ren", "liana")
        assert not graph.has_edge("liana", "ren")
        assert graph.get_edge("ren", "liana") == edge
        assert graph.edge_count == 1
        assert graph.node_count == 2

        # Check internal dual adjacency maps
        assert graph._out_edges["ren"]["liana"] is edge
        assert graph._in_edges["liana"]["ren"] is edge

    def test_get_bilateral(self) -> None:
        graph = RelationshipGraph()
        e_ab = RelationshipEdge("A", "B", RelationshipKind.SUPERIOR_SUBORDINATE, "ta", "ngươi")
        e_ba = RelationshipEdge("B", "A", RelationshipKind.SUPERIOR_SUBORDINATE, "thần", "bệ hạ")
        graph.add_edge(e_ab)
        graph.add_edge(e_ba)

        forward, reverse = graph.get_bilateral("A", "B")
        assert forward == e_ab
        assert reverse == e_ba

        # Asymmetric query where only one direction exists
        graph.remove_edge("B", "A")
        f2, r2 = graph.get_bilateral("A", "B")
        assert f2 == e_ab
        assert r2 is None

    def test_remove_edge_and_remove_node(self) -> None:
        graph = RelationshipGraph()
        e1 = RelationshipEdge("A", "B", RelationshipKind.FRIENDSHIP_CLOSE)
        e2 = RelationshipEdge("B", "C", RelationshipKind.FRIENDSHIP_CLOSE)
        graph.add_edge(e1)
        graph.add_edge(e2)

        # Remove edge
        assert graph.remove_edge("A", "B") is True
        assert not graph.has_edge("A", "B")
        assert "B" not in graph._out_edges.get("A", {})
        assert "A" not in graph._in_edges.get("B", {})
        assert graph.remove_edge("A", "B") is False

        # Remove node
        assert graph.remove_node("B") is True
        assert not graph.has_node("B")
        assert not graph.has_edge("B", "C")
        assert graph.remove_node("B") is False

    def test_manual_curation_protection(self) -> None:
        graph = RelationshipGraph()
        manual_edge = RelationshipEdge(
            "hero", "princess", RelationshipKind.ROMANCE, source_type="manual"
        )
        llm_edge = RelationshipEdge(
            "hero", "princess", RelationshipKind.FORMAL_ACQUAINTANCE, source_type="llm"
        )

        assert graph.add_edge(manual_edge) is True
        # Attempt overwrite by automated source should fail
        assert graph.add_edge(llm_edge, overwrite_manual=False) is False
        assert graph.get_edge("hero", "princess") == manual_edge

        # Forced overwrite should succeed
        assert graph.add_edge(llm_edge, overwrite_manual=True) is True
        assert graph.get_edge("hero", "princess") == llm_edge

    def test_neighbors_by_direction(self) -> None:
        graph = RelationshipGraph()
        graph.add_edge(RelationshipEdge("A", "B", RelationshipKind.ROMANCE))
        graph.add_edge(RelationshipEdge("A", "C", RelationshipKind.FRIENDSHIP_CLOSE))
        graph.add_edge(RelationshipEdge("D", "A", RelationshipKind.SUPERIOR_SUBORDINATE))

        assert graph.get_neighbors("A", direction="out") == {"B", "C"}
        assert graph.get_neighbors("A", direction="in") == {"D"}
        assert graph.get_neighbors("A", direction="both") == {"B", "C", "D"}

        with pytest.raises(ValueError, match="Invalid direction"):
            graph.get_neighbors("A", direction="diagonal")

    def test_interaction_tracking(self) -> None:
        graph = RelationshipGraph()
        graph.record_interaction("hero", "liana", count=5)
        graph.record_interaction("liana", "hero", count=3)
        graph.record_interaction("hero", "villain", count=2)

        assert graph.get_interaction_count("hero", "liana") == 5
        assert graph.get_interaction_count("liana", "hero") == 3
        assert graph.get_bilateral_interaction_count("hero", "liana") == 8
        assert graph.get_total_interactions("hero") == 10  # 5 + 2 out, 3 in

        top = graph.top_interlocutors("hero", limit=2)
        assert top == [("liana", 8), ("villain", 2)]

        # Zero or negative counts are ignored
        graph.record_interaction("hero", "liana", count=0)
        graph.record_interaction("hero", "liana", count=-2)
        assert graph.get_interaction_count("hero", "liana") == 5

    def test_bfs_shortest_pathfinding(self) -> None:
        graph = RelationshipGraph()
        e_ab = RelationshipEdge("A", "B", RelationshipKind.FRIENDSHIP_CLOSE)
        e_bc = RelationshipEdge("B", "C", RelationshipKind.FRIENDSHIP_CLOSE)
        e_cd = RelationshipEdge("C", "D", RelationshipKind.FRIENDSHIP_CLOSE)
        e_ad = RelationshipEdge("A", "D", RelationshipKind.RIVAL_HOSTILE)

        graph.add_edge(e_ab)
        graph.add_edge(e_bc)
        graph.add_edge(e_cd)
        graph.add_edge(e_ad)

        # Direct shortcut chosen
        path = graph.find_path("A", "D")
        assert path == [e_ad]

        # Multi-hop path when shortcut removed
        graph.remove_edge("A", "D")
        path2 = graph.find_path("A", "D")
        assert path2 == [e_ab, e_bc, e_cd]

        # max_depth limit
        assert graph.find_path("A", "D", max_depth=2) == []

        # Disconnected
        assert graph.find_path("A", "Z") == []

        # Self loop
        assert graph.find_path("A", "A") == []

        # Undirected search
        # D has incoming from C, but no outgoing to C
        assert graph.find_path("D", "C", directed=True) == []
        undirected_path = graph.find_path("D", "C", directed=False)
        assert len(undirected_path) == 1
        assert undirected_path[0] == e_cd

    def test_subgraph_extraction(self) -> None:
        graph = RelationshipGraph()
        e1 = RelationshipEdge("A", "B", RelationshipKind.ROMANCE)
        e2 = RelationshipEdge("B", "C", RelationshipKind.ROMANCE)
        e3 = RelationshipEdge("C", "D", RelationshipKind.ROMANCE)
        graph.add_edge(e1)
        graph.add_edge(e2)
        graph.add_edge(e3)
        graph.record_interaction("A", "B", 10)
        graph.record_interaction("C", "D", 20)

        sub = graph.subgraph(["A", "B", "C"])
        assert sub.node_count == 3
        assert sub.edge_count == 2
        assert sub.has_edge("A", "B")
        assert sub.has_edge("B", "C")
        assert not sub.has_edge("C", "D")
        assert sub.get_interaction_count("A", "B") == 10
        assert sub.get_interaction_count("C", "D") == 0


# ==============================================================================
# 5. Dual Persistence & Synchronization Tests
# ==============================================================================


class TestDualPersistenceUnit:
    def test_json_roundtrip_file(self, tmp_path: Path) -> None:
        json_path = tmp_path / "relationships.json"
        graph = RelationshipGraph()
        e1 = RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em")
        e2 = RelationshipEdge("liana", "ren", RelationshipKind.ROMANCE, "em", "anh")
        graph.add_edge(e1)
        graph.add_edge(e2)
        graph.record_interaction("ren", "liana", 42)

        graph.save_to_json_file(json_path)
        assert json_path.is_file()

        loaded = RelationshipGraph.from_json_file(json_path)
        assert loaded.get_edge("ren", "liana") == e1
        assert loaded.get_edge("liana", "ren") == e2
        assert loaded.get_interaction_count("ren", "liana") == 42

    def test_json_parse_error_handling(self, tmp_path: Path) -> None:
        # Syntax error
        syntax_bad = tmp_path / "syntax_bad.json"
        syntax_bad.write_text("{broken: json,", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            RelationshipGraph.from_json_file(syntax_bad)

        # Invalid schema (list of ints)
        schema_bad = tmp_path / "schema_bad.json"
        schema_bad.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(TypeError, match="Expected dict for relationship item"):
            RelationshipGraph.from_json_file(schema_bad)

        # Non-existent file returns empty graph
        missing = tmp_path / "missing.json"
        assert RelationshipGraph.from_json_file(missing).edge_count == 0

    def test_db_save_and_load(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        db = Database.open(db_path, create=True)
        try:
            assert db.schema_version == len(MIGRATIONS)
            assert db.count_relationships() == 0

            graph = RelationshipGraph()
            e1 = RelationshipEdge("hero", "villain", RelationshipKind.RIVAL_HOSTILE, "ta", "ngươi")
            graph.add_edge(e1)
            graph.save_to_db(db)

            assert db.count_relationships() == 1
            retrieved = db.get_relationship("hero", "villain")
            assert retrieved == e1

            reconstructed = RelationshipGraph.from_db(db)
            assert reconstructed.get_edge("hero", "villain") == e1
        finally:
            db.close()

    def test_db_crud_methods(self, tmp_path: Path) -> None:
        db_path = tmp_path / "crud.db"
        db = Database.open(db_path, create=True)
        try:
            edges = [
                RelationshipEdge("A", "B", RelationshipKind.ROMANCE),
                RelationshipEdge("A", "C", RelationshipKind.FRIENDSHIP_CLOSE),
                RelationshipEdge("D", "B", RelationshipKind.RIVAL_HOSTILE),
            ]
            db.save_relationships_bulk(edges)
            assert db.count_relationships() == 3

            # Filter by source
            by_src = db.get_relationships("A")
            assert len(by_src) == 2
            assert {e.target_tag for e in by_src} == {"B", "C"}

            # Filter by target
            by_tgt = db.get_relationships(target_tag="B")
            assert len(by_tgt) == 2
            assert {e.source_tag for e in by_tgt} == {"A", "D"}

            # Delete single
            assert db.delete_relationship("A", "B") is True
            assert db.count_relationships() == 2
            assert db.get_relationship("A", "B") is None
            assert db.delete_relationship("A", "B") is False

            # Clear all
            deleted_count = db.clear_relationships()
            assert deleted_count == 2
            assert db.count_relationships() == 0
        finally:
            db.close()

    def test_character_profile_db_persistence(self, tmp_path: Path) -> None:
        db_path = tmp_path / "chars.db"
        db = Database.open(db_path, create=True)
        try:
            p = CharacterProfile(
                tag="rin",
                display_name="Rin Tohsaka",
                gender="female",
                role="heroine",
                speech_style="refined",
                pronoun_map={"mc": "anh"},
                archetype="tsundere",
                personality_traits=("proud", "capable"),
                tone_and_voice="sharp",
                socioeconomic_status="noble",
                sample_dialogue=("Baka!",),
            )
            db.save_character(p)
            chars = db.get_characters()
            assert "rin" in chars
            loaded = chars["rin"]
            assert loaded.tag == "rin"
            assert loaded.display_name == "Rin Tohsaka"
            assert loaded.archetype == "tsundere"
            assert loaded.personality_traits == ("proud", "capable")
            assert loaded.sample_dialogue == ("Baka!",)
        finally:
            db.close()

    def test_sync_reconciliation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "sync.db"
        json_path = tmp_path / "relationships.json"
        db = Database.open(db_path, create=True)
        try:
            # 1. DB has data, JSON does not -> sync exports DB to JSON
            g_db = RelationshipGraph()
            e_db = RelationshipEdge(
                "hero", "mentor", RelationshipKind.MENTOR_DISCIPLE, source_type="manual"
            )
            g_db.add_edge(e_db)
            g_db.save_to_db(db)

            synced = RelationshipGraph.sync(db, json_path)
            assert json_path.is_file()
            assert synced.get_edge("hero", "mentor") == e_db

            # 2. Modify JSON externally by adding new edge
            g_json = RelationshipGraph.from_json_file(json_path)
            e_manual = RelationshipEdge(
                "hero", "liana", RelationshipKind.ROMANCE, source_type="manual"
            )
            g_json.add_edge(e_manual)
            g_json.save_to_json_file(json_path)

            # Sync merges both into DB and JSON
            synced2 = RelationshipGraph.sync(db, json_path)
            assert synced2.edge_count == 2
            assert db.count_relationships() == 2
            assert db.get_relationship("hero", "liana") == e_manual
            assert db.get_relationship("hero", "mentor") == e_db
        finally:
            db.close()
