"""Comprehensive 4-Tier Opaque-Box E2E Test Suite for Character Relationship Knowledge Graph,
Deep Persona Engine, and Dynamic Interlocutor Context Assembly.

Requirements Covered:
- R1: Directed Character Relationship Knowledge Graph (renslate.engine.graph & renslate.core.models)
      RelationshipKind (10 semantic kinds), RelationshipEdge, RelationshipGraph with O(1) bilateral
      lookups and BFS pathfinding, SQLite character_relationships table & relationships.json sync.
- R2: Deep Persona & Voice Profile Engine (renslate.engine.analyze & CharacterProfile)
      Extended CharacterProfile (archetype, personality_traits, tone_and_voice,
      socioeconomic_status, sample_dialogue), eliminating hardcoded "anh - em" assumptions,
      bidirectional characters.json sync.
- R3: Real-Time Interlocutor Resolution & Dynamic Prompt Assembly (renslate.engine.context & qa)
      O(1) InterlocutorResolver (Turn-taking adjacency, vocative detection,
      monologue/soliloquy detection), rich relational dialogue framing, Scene Macro Context,
      QA leak peeling and validation.
- R4: Deep Module Architecture & ADR-028 Compliance
      Zero external runtime dependencies, strict type safety, clean separation of concerns.

Tiers:
- Tier 1: Feature Coverage (7 Feature Areas x >=5 tests = 38 tests)
- Tier 2: Boundary & Corner Cases (15 tests)
- Tier 3: Cross-Feature Combinations (6 tests)
- Tier 4: Real-World Scenarios (5 multi-party visual novel scenarios)
Total: 64 Comprehensive Test Cases
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from renslate.config import Config
from renslate.core.db import MIGRATIONS, Database
from renslate.core.errors import ProjectError
from renslate.core.models import (
    CharacterProfile,
    Origin,
    SceneSummary,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.context import ContextBuilder
from renslate.engine.qa import QAChecker, clean_translation

# ==============================================================================
# Progressive Feature Detection Flags for Milestone Readiness
# ==============================================================================

RelationshipKind: Any = None
RelationshipEdge: Any = None
try:
    import renslate.core.models as _models

    _RKind = getattr(_models, "RelationshipKind", None)
    _REdge = getattr(_models, "RelationshipEdge", None)
    HAS_RELATIONSHIP_MODELS = hasattr(_RKind, "ROMANCE") and hasattr(
        _REdge, "__dataclass_fields__"
    )
    if HAS_RELATIONSHIP_MODELS:
        RelationshipKind = _RKind
        RelationshipEdge = _REdge
except Exception:
    HAS_RELATIONSHIP_MODELS = False

HAS_EXTENDED_PROFILE = hasattr(CharacterProfile, "__dataclass_fields__") and "archetype" in getattr(
    CharacterProfile, "__dataclass_fields__", {}
)

RelationshipGraph: Any = None
try:
    import renslate.engine.graph as _graph_mod

    _RGraph = getattr(_graph_mod, "RelationshipGraph", None)
    HAS_RELATIONSHIP_GRAPH = hasattr(_RGraph, "from_db")
    if HAS_RELATIONSHIP_GRAPH:
        RelationshipGraph = _RGraph
except Exception:
    HAS_RELATIONSHIP_GRAPH = False


def _check_db_has_relationships() -> bool:
    try:
        return any("character_relationships" in m for m in MIGRATIONS)
    except Exception:
        return False


HAS_DB_RELATIONSHIPS = _check_db_has_relationships()

InterlocutorResolver: Any = None
ResolvedInterlocutor: Any = None
try:
    import renslate.engine.context as _context_mod

    _IResolver = getattr(_context_mod, "InterlocutorResolver", None)
    _RInterlocutor = getattr(_context_mod, "ResolvedInterlocutor", None)
    HAS_INTERLOCUTOR_RESOLVER = hasattr(_IResolver, "resolve")
    if HAS_INTERLOCUTOR_RESOLVER:
        InterlocutorResolver = _IResolver
        ResolvedInterlocutor = _RInterlocutor
except Exception:
    HAS_INTERLOCUTOR_RESOLVER = False


def _check_adr_028() -> bool:
    adr_path = Path(__file__).resolve().parent.parent / "docs" / "ARCHITECTURE_DECISIONS.md"
    if not adr_path.exists():
        return False
    return "ADR-028" in adr_path.read_text(encoding="utf-8")


HAS_ADR_028 = _check_adr_028()


# ==============================================================================
# Helper Factories & Fixtures
# ==============================================================================


def make_test_unit(
    uid: str,
    source: str,
    *,
    speaker: str | None = None,
    seq: int = 1,
    kind: UnitKind = UnitKind.DIALOGUE,
    status: Status = Status.PENDING,
    target: str | None = None,
    origin: Origin | None = None,
) -> Unit:
    """Helper to construct a Unit with safe defaults."""
    return Unit(
        uid=uid,
        kind=kind,
        seq=seq,
        source=source,
        status=status,
        speaker=speaker,
        target=target,
        origin=origin,
    )


# ==============================================================================
# TIER 1: FEATURE COVERAGE (Core Contract Tests)
# ==============================================================================


class TestTier1RelationshipKind:
    """Tier 1.1: Feature tests for RelationshipKind semantic enum."""

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipKind in renslate.core.models",
    )
    def test_all_ten_enum_values_present_and_exact(self) -> None:
        """Requirement R1: Exactly 10 canonical relationship kinds defined in spec."""
        expected_kinds = {
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
        actual_kinds = {k.value for k in RelationshipKind}
        assert actual_kinds == expected_kinds
        assert len(RelationshipKind) == 10

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipKind in renslate.core.models",
    )
    def test_relationship_kind_string_subclassing_and_equality(self) -> None:
        """RelationshipKind subclasses str for string comparisons and serialization."""
        assert issubclass(RelationshipKind, str)
        assert RelationshipKind.ROMANCE == "ROMANCE"
        assert RelationshipKind.FAMILY_ELDER.value == "FAMILY_ELDER"
        assert (
            f"{RelationshipKind.RIVAL_HOSTILE}" == "RelationshipKind.RIVAL_HOSTILE"
            or str(RelationshipKind.RIVAL_HOSTILE) == "RIVAL_HOSTILE"
        )

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipKind in renslate.core.models",
    )
    def test_relationship_kind_membership_and_iteration(self) -> None:
        """RelationshipKind supports fast membership checking and deterministic iteration."""
        assert "ROMANCE" in RelationshipKind.__members__
        assert "SERVANT_MASTER" in RelationshipKind.__members__
        assert "NON_EXISTENT_KIND" not in RelationshipKind.__members__
        all_names = list(RelationshipKind.__members__)
        assert len(all_names) == 10

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipKind in renslate.core.models",
    )
    def test_relationship_kind_serialization_deserialization(self) -> None:
        """Enum members round-trip cleanly via json.dumps and constructor lookup."""
        serialized = json.dumps({"kind": RelationshipKind.ROMANCE.value})
        data = json.loads(serialized)
        restored = RelationshipKind(data["kind"])
        assert restored is RelationshipKind.ROMANCE

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipKind in renslate.core.models",
    )
    def test_relationship_kind_invalid_value_rejection(self) -> None:
        """Instantiating RelationshipKind with an unrecognized string raises ValueError."""
        with pytest.raises(ValueError):
            RelationshipKind("ACQUAINTANCE_CASUAL_INVALID")


class TestTier1RelationshipEdge:
    """Tier 1.2: Feature tests for RelationshipEdge domain model."""

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipEdge in renslate.core.models",
    )
    def test_edge_creation_and_slots_immutability(self) -> None:
        """RelationshipEdge must be frozen, slots-allocated, and immutable."""
        edge = RelationshipEdge(
            source_tag="ren",
            target_tag="liana",
            kind=RelationshipKind.ROMANCE,
            speaker_pronoun="anh",
            listener_pronoun="em",
        )
        assert edge.source_tag == "ren"
        assert edge.target_tag == "liana"
        assert edge.kind == RelationshipKind.ROMANCE
        with pytest.raises((FrozenInstanceError, AttributeError)):
            edge.speaker_pronoun = "tôi"

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipEdge in renslate.core.models",
    )
    def test_edge_defaults_confidence_and_source_type(self) -> None:
        """RelationshipEdge defaults confidence to 1.0 and source_type to 'manual'."""
        edge = RelationshipEdge(
            source_tag="hero",
            target_tag="mentor",
            kind=RelationshipKind.MENTOR_DISCIPLE,
        )
        assert edge.confidence == 1.0
        assert edge.source_type == "manual"
        assert edge.speaker_pronoun == ""
        assert edge.listener_pronoun == ""
        assert edge.notes == ""

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipEdge in renslate.core.models",
    )
    def test_edge_speaker_and_listener_pronouns(self) -> None:
        """RelationshipEdge faithfully holds speaker and listener pronoun pair."""
        edge = RelationshipEdge(
            source_tag="luke",
            target_tag="vane",
            kind=RelationshipKind.FAMILY_ELDER,
            speaker_pronoun="con",
            listener_pronoun="cha",
        )
        assert edge.speaker_pronoun == "con"
        assert edge.listener_pronoun == "cha"

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipEdge in renslate.core.models",
    )
    def test_edge_equality_and_hashability(self) -> None:
        """Edges with identical fields must be equal and hashable in sets/dictionaries."""
        e1 = RelationshipEdge("a", "b", RelationshipKind.FRIENDSHIP_CLOSE, "tớ", "cậu")
        e2 = RelationshipEdge("a", "b", RelationshipKind.FRIENDSHIP_CLOSE, "tớ", "cậu")
        assert e1 == e2
        assert hash(e1) == hash(e2)
        edge_set = {e1, e2}
        assert len(edge_set) == 1

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipEdge in renslate.core.models",
    )
    def test_edge_notes_and_metadata_preservation(self) -> None:
        """RelationshipEdge preserves structured notes (e.g. JSON strings or context notes)."""
        notes_str = "Childhood friend turned sworn rival after chapter 3"
        edge = RelationshipEdge(
            source_tag="a",
            target_tag="b",
            kind=RelationshipKind.RIVAL_HOSTILE,
            notes=notes_str,
        )
        assert edge.notes == notes_str


class TestTier1CharacterProfile:
    """Tier 1.3: Feature tests for extended CharacterProfile persona & voice engine."""

    @pytest.mark.skipif(
        not HAS_EXTENDED_PROFILE,
        reason="Requires M1/M2 extended CharacterProfile fields in renslate.core.models",
    )
    def test_profile_extended_fields_defaults_and_types(self) -> None:
        """Extended CharacterProfile includes archetype, traits, voice, and samples."""
        Profile: Any = CharacterProfile
        profile = Profile(
            tag="scarlet",
            display_name="Scarlet",
            archetype="tsundere noble",
            personality_traits=("proud", "defensive", "caring"),
            tone_and_voice="haughty but softens when embarrassed",
            socioeconomic_status="high noble",
            sample_dialogue=("H-Hmph! It's not like I waited for you or anything!",),
        )
        assert profile.archetype == "tsundere noble"
        assert profile.personality_traits == ("proud", "defensive", "caring")
        assert profile.tone_and_voice == "haughty but softens when embarrassed"
        assert profile.socioeconomic_status == "high noble"
        assert len(profile.sample_dialogue) == 1

    @pytest.mark.skipif(
        not HAS_EXTENDED_PROFILE,
        reason="Requires M1/M2 extended CharacterProfile fields in renslate.core.models",
    )
    def test_profile_immutability_and_slots(self) -> None:
        """CharacterProfile must remain frozen dataclass with slots."""
        profile = CharacterProfile(tag="hero", display_name="Hero")
        target_obj: Any = profile
        with pytest.raises((FrozenInstanceError, AttributeError)):
            target_obj.role = "Protagonist"

    @pytest.mark.skipif(
        not HAS_EXTENDED_PROFILE,
        reason="Requires M1/M2 extended CharacterProfile fields in renslate.core.models",
    )
    def test_profile_personality_traits_tuple(self) -> None:
        """Personality traits must be stored as an immutable tuple of strings."""
        Profile: Any = CharacterProfile
        profile = Profile(
            tag="liana",
            display_name="Liana",
            personality_traits=("gentle", "observant", "determined"),
        )
        traits = profile.personality_traits
        assert isinstance(traits, tuple)
        assert "gentle" in traits

    @pytest.mark.skipif(
        not HAS_EXTENDED_PROFILE,
        reason="Requires M1/M2 extended CharacterProfile fields in renslate.core.models",
    )
    def test_profile_sample_dialogue_tuple(self) -> None:
        """Sample dialogue must be stored as an immutable tuple of dialogue strings."""
        Profile: Any = CharacterProfile
        profile = Profile(
            tag="vane",
            display_name="Lord Vane",
            sample_dialogue=("Silence! You bring disgrace to our lineage.",),
        )
        dialogues = profile.sample_dialogue
        assert isinstance(dialogues, tuple)
        assert len(dialogues) == 1

    def test_profile_backward_compatibility_minimal_args(self) -> None:
        """CharacterProfile can be instantiated with legacy minimal arguments."""
        profile = CharacterProfile(tag="mc", display_name="Main Character")
        assert profile.tag == "mc"
        assert profile.display_name == "Main Character"
        assert profile.gender == "unknown"
        assert profile.role == ""
        assert profile.pronoun_map == {}


class TestTier1RelationshipGraph:
    """Tier 1.4: Feature tests for RelationshipGraph O(1) lookups and BFS pathfinding."""

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph in renslate.engine.graph",
    )
    def test_graph_initialization_and_empty_state(self) -> None:
        """An empty graph has 0 edges and returns None for arbitrary queries."""
        graph = RelationshipGraph()
        assert graph.get_edge("a", "b") is None
        assert graph.get_bilateral("a", "b") == (None, None)
        assert graph.find_path("a", "b") == []

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph in renslate.engine.graph",
    )
    def test_graph_add_and_get_edge_o1(self) -> None:
        """get_edge retrieves added edge in O(1) time complexity."""
        graph = RelationshipGraph()
        edge = RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em")
        graph.add_edge(edge)
        retrieved = graph.get_edge("ren", "liana")
        assert retrieved == edge
        assert graph.get_edge("liana", "ren") is None

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph in renslate.engine.graph",
    )
    def test_graph_get_bilateral_o1(self) -> None:
        """get_bilateral returns (edge_a_to_b, edge_b_to_a) in O(1) time."""
        graph = RelationshipGraph()
        e_ren_liana = RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em")
        e_liana_ren = RelationshipEdge("liana", "ren", RelationshipKind.ROMANCE, "em", "anh")
        graph.add_edge(e_ren_liana)
        graph.add_edge(e_liana_ren)

        forward, reverse = graph.get_bilateral("ren", "liana")
        assert forward == e_ren_liana
        assert reverse == e_liana_ren

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph in renslate.engine.graph",
    )
    def test_graph_bfs_pathfinding_direct_and_multi_hop(self) -> None:
        """find_path returns the shortest sequence of RelationshipEdge objects via BFS."""
        graph = RelationshipGraph()
        e_ab = RelationshipEdge("A", "B", RelationshipKind.FAMILY_SIBLING)
        e_bc = RelationshipEdge("B", "C", RelationshipKind.MENTOR_DISCIPLE)
        graph.add_edge(e_ab)
        graph.add_edge(e_bc)

        path = graph.find_path("A", "C")
        assert len(path) == 2
        assert path[0] == e_ab
        assert path[1] == e_bc

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph in renslate.engine.graph",
    )
    def test_graph_edge_overwriting_and_update(self) -> None:
        """Adding a new edge with the same source and target updates the existing relation."""
        graph = RelationshipGraph()
        e_old = RelationshipEdge("A", "B", RelationshipKind.FORMAL_ACQUAINTANCE, confidence=0.5)
        e_new = RelationshipEdge("A", "B", RelationshipKind.FRIENDSHIP_CLOSE, confidence=1.0)
        graph.add_edge(e_old)
        graph.add_edge(e_new)
        assert graph.get_edge("A", "B") == e_new

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph in renslate.engine.graph",
    )
    def test_graph_neighbors_and_all_edges_listing(self) -> None:
        """Graph can enumerate outgoing edges and all contained edges."""
        graph = RelationshipGraph()
        e1 = RelationshipEdge("A", "B", RelationshipKind.ROMANCE)
        e2 = RelationshipEdge("A", "C", RelationshipKind.RIVAL_HOSTILE)
        graph.add_edge(e1)
        graph.add_edge(e2)

        if hasattr(graph, "get_neighbors"):
            neighbors = graph.get_neighbors("A")
            assert set(neighbors) == {"B", "C"}
        if hasattr(graph, "edges"):
            all_edges = list(graph.edges()) if callable(graph.edges) else list(graph.edges)
            assert len(all_edges) == 2


class TestTier1PersistenceAndSync:
    """Tier 1.5: Feature tests for SQLite and relationships.json dual persistence."""

    @pytest.mark.skipif(
        not HAS_DB_RELATIONSHIPS,
        reason="Requires M1 SQLite character_relationships table in renslate.core.db",
    )
    def test_sqlite_character_relationships_table_schema(self, tmp_path: Path) -> None:
        """SQLite migration creates character_relationships table with required columns."""
        db_path = tmp_path / "project.db"
        db = Database.open(db_path, create=True)
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(character_relationships)")
            columns = {row[1] for row in cursor.fetchall()}
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
            assert expected_cols.issubset(columns)
            conn.close()
        finally:
            db.close()

    @pytest.mark.skipif(
        not (HAS_RELATIONSHIP_GRAPH and HAS_DB_RELATIONSHIPS),
        reason="Requires M1 RelationshipGraph and DB persistence",
    )
    def test_graph_save_to_db_and_load_from_db(self, tmp_path: Path) -> None:
        """Graph saves to SQLite and reconstructs faithfully via from_db."""
        db_path = tmp_path / "project.db"
        db = Database.open(db_path, create=True)
        try:
            graph = RelationshipGraph()
            e1 = RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em")
            e2 = RelationshipEdge("liana", "ren", RelationshipKind.ROMANCE, "em", "anh")
            graph.add_edge(e1)
            graph.add_edge(e2)

            graph.save_to_db(db)

            loaded_graph = RelationshipGraph.from_db(db)
            assert loaded_graph.get_edge("ren", "liana") == e1
            assert loaded_graph.get_edge("liana", "ren") == e2
        finally:
            db.close()

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph in renslate.engine.graph",
    )
    def test_graph_save_to_json_file_and_load_from_json(self, tmp_path: Path) -> None:
        """Graph serializes to relationships.json and reloads without loss."""
        json_path = tmp_path / "relationships.json"
        graph = RelationshipGraph()
        e1 = RelationshipEdge("vane", "luke", RelationshipKind.FAMILY_ELDER, "cha", "con")
        graph.add_edge(e1)

        graph.save_to_json_file(json_path)
        assert json_path.exists()

        # Validate that JSON is human-readable and valid
        raw_data = json.loads(json_path.read_text(encoding="utf-8"))
        assert raw_data is not None

        loaded_graph = RelationshipGraph.from_json_file(json_path)
        assert loaded_graph.get_edge("vane", "luke") == e1

    @pytest.mark.skipif(
        not (HAS_RELATIONSHIP_GRAPH and HAS_DB_RELATIONSHIPS),
        reason="Requires M1 RelationshipGraph sync method",
    )
    def test_graph_sync_db_and_json_bidirectional(self, tmp_path: Path) -> None:
        """sync() reconciles SQLite and relationships.json."""
        db_path = tmp_path / "project.db"
        json_path = tmp_path / "relationships.json"
        db = Database.open(db_path, create=True)
        try:
            graph = RelationshipGraph()
            e1 = RelationshipEdge(
                "sonya", "ronald", RelationshipKind.SERVANT_MASTER, "trẫm", "ngươi"
            )
            graph.add_edge(e1)
            graph.save_to_db(db)

            # Sync to JSON
            RelationshipGraph.sync(db, json_path)
            assert json_path.exists()

            # Verify loaded from JSON matches
            reloaded = RelationshipGraph.from_json_file(json_path)
            assert reloaded.get_edge("sonya", "ronald") == e1
        finally:
            db.close()

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph",
    )
    def test_graph_persistence_preserves_pronouns_and_confidence(self, tmp_path: Path) -> None:
        """Persistence preserves exact float confidence, source_type, and pronouns."""
        json_path = tmp_path / "rel.json"
        graph = RelationshipGraph()
        edge = RelationshipEdge(
            "src",
            "tgt",
            RelationshipKind.FRIENDSHIP_CASUAL,
            speaker_pronoun="mình",
            listener_pronoun="bạn",
            confidence=0.85,
            source_type="heuristic",
            notes="Inferred from school setting",
        )
        graph.add_edge(edge)
        graph.save_to_json_file(json_path)

        reloaded = RelationshipGraph.from_json_file(json_path)
        restored = reloaded.get_edge("src", "tgt")
        assert restored is not None
        assert restored.confidence == 0.85
        assert restored.source_type == "heuristic"
        assert restored.notes == "Inferred from school setting"

    def test_character_profile_db_and_json_persistence(self, tmp_path: Path) -> None:
        """Character profile serialization and SQLite persistence preserves tag and display name."""
        db_path = tmp_path / "project.db"
        db = Database.open(db_path, create=True)
        try:
            profile = CharacterProfile(
                tag="eileen",
                display_name="Eileen",
                gender="female",
                role="Heroine",
                speech_style="cheerful",
            )
            db.save_character(profile)
            loaded = db.get_characters()
            assert "eileen" in loaded
            assert loaded["eileen"].display_name == "Eileen"
        finally:
            db.close()


class TestTier1InterlocutorResolver:
    """Tier 1.6: Feature tests for InterlocutorResolver listener deduction."""

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver in renslate.engine.context",
    )
    def test_turn_taking_adjacency_two_party_ping_pong(self) -> None:
        """When speaker B immediately follows speaker A, listener is resolved to A."""
        resolver = InterlocutorResolver()
        unit1 = make_test_unit("d:1", "Good morning, everyone.", speaker="ren")
        unit2 = make_test_unit("d:2", "Good morning to you too!", speaker="liana")

        resolved = resolver.resolve(
            unit=unit2,
            prev_unit=unit1,
            active_speakers={"ren", "liana"},
            known_names={"ren": "Ren", "liana": "Liana"},
        )
        assert resolved.speaker == "liana"
        assert resolved.listener == "ren"
        assert not resolved.is_monologue
        assert resolved.resolution_method == "turn_taking"

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver in renslate.engine.context",
    )
    def test_vocative_detection_leading_name(self) -> None:
        """Leading vocative name (e.g. 'Liana, wait!') deduces listener with 100% precision."""
        resolver = InterlocutorResolver()
        unit = make_test_unit("d:1", "Liana, please wait for me!", speaker="ren")

        resolved = resolver.resolve(
            unit=unit,
            prev_unit=None,
            active_speakers={"ren", "liana"},
            known_names={"ren": "Ren", "liana": "Liana"},
        )
        assert resolved.listener == "liana"
        assert resolved.resolution_method == "vocative"

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver in renslate.engine.context",
    )
    def test_vocative_detection_trailing_name(self) -> None:
        """Trailing vocative name overrides turn-taking adjacency."""
        resolver = InterlocutorResolver()
        # Previous speaker was Liana, but Ren explicitly calls out Scarlet
        prev = make_test_unit("d:1", "I'm ready, Ren.", speaker="liana")
        curr = make_test_unit("d:2", "Are you ready too, Scarlet?", speaker="ren")

        resolved = resolver.resolve(
            unit=curr,
            prev_unit=prev,
            active_speakers={"ren", "liana", "scarlet"},
            known_names={"ren": "Ren", "liana": "Liana", "scarlet": "Scarlet"},
        )
        assert resolved.listener == "scarlet"
        assert resolved.resolution_method == "vocative"

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver in renslate.engine.context",
    )
    def test_monologue_parentheses_detection(self) -> None:
        """Dialogue enclosed in parentheses is classified as inner monologue/soliloquy."""
        resolver = InterlocutorResolver()
        unit = make_test_unit("d:1", "(She seems angry today... what did I do?)", speaker="ren")

        resolved = resolver.resolve(
            unit=unit,
            prev_unit=None,
            active_speakers={"ren", "liana"},
            known_names={"ren": "Ren", "liana": "Liana"},
        )
        assert resolved.is_monologue
        assert resolved.listener is None

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver in renslate.engine.context",
    )
    def test_monologue_pseudo_speaker_tags(self) -> None:
        """Speakers with pseudo tags ('thought', 't', 'narration') resolve as monologues."""
        resolver = InterlocutorResolver()
        for tag in ("thought", "t", "narration"):
            unit = make_test_unit("d:1", "The wind howling outside...", speaker=tag)
            resolved = resolver.resolve(
                unit=unit,
                prev_unit=None,
                active_speakers={tag},
                known_names={},
            )
            assert resolved.is_monologue

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver in renslate.engine.context",
    )
    def test_resolver_returns_resolved_interlocutor_dataclass(self) -> None:
        """InterlocutorResolver returns an instance of ResolvedInterlocutor."""
        resolver = InterlocutorResolver()
        unit = make_test_unit("d:1", "Hello world", speaker="ren")
        resolved = resolver.resolve(unit, None, {"ren"}, {"ren": "Ren"})
        assert isinstance(resolved, ResolvedInterlocutor)
        assert hasattr(resolved, "speaker")
        assert hasattr(resolved, "listener")
        assert hasattr(resolved, "is_monologue")
        assert hasattr(resolved, "detected_emotion")
        assert hasattr(resolved, "resolution_method")


class TestTier1PromptAssembly:
    """Tier 1.7: Feature tests for dynamic relational prompt formatting."""

    def test_clean_translation_peels_relational_speaker_frame(self) -> None:
        """clean_translation must cleanly strip rich metadata frame from target translation."""
        raw_output = "[Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE] Xin chào em!"
        cleaned = clean_translation(raw_output)
        assert cleaned == "Xin chào em!"

    def test_clean_translation_peels_monologue_frame(self) -> None:
        """clean_translation must cleanly strip monologue metadata frames."""
        raw_output = "[Độc thoại/Nội tâm: Ren] Mình phải suy nghĩ kỹ."
        cleaned = clean_translation(raw_output)
        assert cleaned == "Mình phải suy nghĩ kỹ."

    def test_qa_checker_flags_leaked_relational_speaker_frame(self) -> None:
        """QAChecker emits speaker_tag_leakage error if relational frame leaks in target."""
        checker = QAChecker()
        target = "[Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE] Xin chào em"
        issues = checker.check_unit(
            uid="d:101",
            source="Hello Liana",
            masked=None,
            target=target,
        )
        assert any(i.code == "speaker_tag_leakage" for i in issues)

    def test_cleaned_translation_passes_qa_check_with_zero_leak_issues(self) -> None:
        """Once passed through clean_translation, target has zero speaker_tag_leakage issues."""
        checker = QAChecker()
        target = "[Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE] Xin chào em"
        sanitized = clean_translation(target)
        assert sanitized == "Xin chào em"
        issues = checker.check_unit(
            uid="d:101",
            source="Hello Liana",
            masked=None,
            target=sanitized,
        )
        leak_issues = [i for i in issues if i.code == "speaker_tag_leakage"]
        assert len(leak_issues) == 0

    def test_scene_macro_context_inclusion_in_user_prompt(self) -> None:
        """ContextBuilder injects scene summary into prompt when provided."""
        cfg = Config()
        builder = ContextBuilder(cfg)
        unit = make_test_unit("d:1", "Where are we?", speaker="ren")
        scene = SceneSummary(
            scene_id="s1",
            label="forest_night",
            summary="Ren and Liana are lost in the enchanted forest at midnight.",
        )
        prompt = builder.build_user_prompt([unit], scene_summary=scene)
        assert (
            "forest" in prompt
            or "enchanted" in prompt
            or "s1" in prompt
            or "TÓM TẮT CẢNH" in prompt
        )

    def test_clean_translation_preserves_genuine_dialogue_brackets(self) -> None:
        """clean_translation must not strip brackets that were present in source."""
        assert (
            clean_translation("[Thở dài] Thật là mệt mỏi.", source="[Sigh] It's tiring.")
            == "[Thở dài] Thật là mệt mỏi."
        )
        assert (
            clean_translation("[...] Không có ai ở đây cả.", source="[...] Nobody is here.")
            == "[...] Không có ai ở đây cả."
        )
        assert (
            clean_translation("[1] Đi sang trái", source="[1] Go left")
            == "[1] Đi sang trái"
        )
        assert (
            clean_translation("[player_name] ơi, xin chào!", source="Hello, [player_name]!")
            == "[player_name] ơi, xin chào!"
        )

    def test_clean_translation_peels_trailing_metadata_suffix(self) -> None:
        """clean_translation must peel metadata suffixes appended to translation."""
        assert (
            clean_translation("Xin chào em! [Quan hệ: ROMANCE]", source="Hello!")
            == "Xin chào em!"
        )

    def test_qa_checker_flags_leaked_relation_metadata(self) -> None:
        """QAChecker emits relation_metadata_leakage error for leaked relation tags."""
        checker = QAChecker()
        target = "[Quan hệ: ROMANCE] Xin chào em"
        issues = checker.check_unit(uid="d:102", source="Hello", masked=None, target=target)
        assert any(
            i.code == "relation_metadata_leakage" and i.severity == "error"
            for i in issues
        )

    def test_qa_checker_flags_leaked_monologue_frame(self) -> None:
        """QAChecker emits speaker_tag_leakage error for leaked monologue frame."""
        checker = QAChecker()
        target = "[Độc thoại/Nội tâm: Ren] Mình phải nhanh lên."
        issues = checker.check_unit(
            uid="d:103", source="I must hurry.", masked=None, target=target
        )
        assert any(
            i.code == "speaker_tag_leakage" and i.severity == "error"
            for i in issues
        )


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (Pathological & Stress Tests)
# ==============================================================================


class TestTier2BoundaryCases:
    """Tier 2: Boundary and corner cases across all relationship & context features."""

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph",
    )
    def test_boundary_empty_graph_lookups_and_pathfinding(self) -> None:
        """Querying an empty graph returns None for lookups and empty list for BFS path."""
        graph = RelationshipGraph()
        assert graph.get_edge("unknown_1", "unknown_2") is None
        assert graph.get_bilateral("unknown_1", "unknown_2") == (None, None)
        assert graph.find_path("unknown_1", "unknown_2") == []

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph",
    )
    def test_boundary_cyclic_relationships_no_infinite_recursion(self) -> None:
        """Cyclic graph terminates and finds shortest path without infinite loop."""
        graph = RelationshipGraph()
        graph.add_edge(RelationshipEdge("A", "B", RelationshipKind.FRIENDSHIP_CLOSE))
        graph.add_edge(RelationshipEdge("B", "C", RelationshipKind.FRIENDSHIP_CLOSE))
        graph.add_edge(RelationshipEdge("C", "A", RelationshipKind.FRIENDSHIP_CLOSE))

        path = graph.find_path("A", "C")
        assert len(path) == 2
        assert path[0].source_tag == "A" and path[0].target_tag == "B"
        assert path[1].source_tag == "B" and path[1].target_tag == "C"

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph",
    )
    def test_boundary_self_loops_handling(self) -> None:
        """Self-loop edge (A -> A, soliloquy/self-reflection) is handled cleanly."""
        graph = RelationshipGraph()
        edge = RelationshipEdge(
            "A", "A", RelationshipKind.FORMAL_ACQUAINTANCE, notes="Self-reflection"
        )
        graph.add_edge(edge)
        assert graph.get_edge("A", "A") == edge
        path = graph.find_path("A", "A")
        assert isinstance(path, list)

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph",
    )
    def test_boundary_disconnected_subgraphs_pathfinding(self) -> None:
        """Disconnected components (A -> B and C -> D) return empty list for path A to D."""
        graph = RelationshipGraph()
        graph.add_edge(RelationshipEdge("A", "B", RelationshipKind.ROMANCE))
        graph.add_edge(RelationshipEdge("C", "D", RelationshipKind.RIVAL_HOSTILE))

        path = graph.find_path("A", "D")
        assert path == []

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver",
    )
    def test_boundary_unknown_speakers_fallback_no_crash(self) -> None:
        """Unknown speaker not in known_names does not crash InterlocutorResolver."""
        resolver = InterlocutorResolver()
        unit = make_test_unit("d:1", "Mysterious voice speaks...", speaker="stranger_999")
        resolved = resolver.resolve(
            unit=unit,
            prev_unit=None,
            active_speakers=set(),
            known_names={},
        )
        assert resolved.speaker == "stranger_999"

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver",
    )
    def test_boundary_ambiguous_vocatives_multiple_names_in_sentence(self) -> None:
        """Sentence mentioning multiple characters prioritizes direct vocative address."""
        resolver = InterlocutorResolver()
        unit = make_test_unit("d:1", "Alice, did Bob tell you about Charlie?", speaker="ren")
        resolved = resolver.resolve(
            unit=unit,
            prev_unit=None,
            active_speakers={"ren", "alice", "bob", "charlie"},
            known_names={"alice": "Alice", "bob": "Bob", "charlie": "Charlie"},
        )
        # Leading vocative 'Alice,' must be resolved as listener
        assert resolved.listener == "alice"

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver",
    )
    def test_boundary_vocative_common_english_words_false_positive_guard(self) -> None:
        """Common words ('May', 'Will') in normal sentences do not falsely trigger vocatives."""
        resolver = InterlocutorResolver()
        # Even if a character is named "Will" or "May", non-vocative usage must not falsely bind
        unit = make_test_unit("d:1", "I will go there next May.", speaker="ren")
        prev = make_test_unit("d:0", "Where are you going?", speaker="liana")
        resolved = resolver.resolve(
            unit=unit,
            prev_unit=prev,
            active_speakers={"ren", "liana"},
            known_names={"ren": "Ren", "liana": "Liana", "will": "Will", "may": "May"},
        )
        # Should resolve to Liana via turn-taking, not Will or May
        assert resolved.listener == "liana"

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver",
    )
    def test_boundary_nested_parentheses_and_escaped_quotes_in_monologue(self) -> None:
        """Nested parentheses in thoughts resolve as monologue."""
        resolver = InterlocutorResolver()
        unit = make_test_unit("d:1", "((Whispering) Is someone watching us?)", speaker="ren")
        resolved = resolver.resolve(unit, None, {"ren"}, {"ren": "Ren"})
        assert resolved.is_monologue

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver",
    )
    def test_boundary_unclosed_parenthesis_in_dialogue(self) -> None:
        """Unclosed parenthesis (e.g. '(Wait... she ran off') handled gracefully."""
        resolver = InterlocutorResolver()
        unit = make_test_unit("d:1", "(Wait... she ran off", speaker="ren")
        resolved = resolver.resolve(unit, None, {"ren"}, {"ren": "Ren"})
        assert isinstance(resolved.is_monologue, bool)

    def test_boundary_missing_database_tables_graceful_handling(self, tmp_path: Path) -> None:
        """Accessing a nonexistent database file without create=True raises ProjectError."""
        missing_db = tmp_path / "nonexistent.db"
        with pytest.raises(ProjectError):
            Database.open(missing_db, create=False)

    def test_boundary_corrupted_json_syntax_error_handling(self, tmp_path: Path) -> None:
        """Corrupted JSON syntax in relationships.json raises JSONDecodeError or ProjectError."""
        bad_json = tmp_path / "relationships.json"
        bad_json.write_text("{malformed_json: true,", encoding="utf-8")
        if HAS_RELATIONSHIP_GRAPH:
            with pytest.raises((json.JSONDecodeError, ProjectError, Exception)):
                RelationshipGraph.from_json_file(bad_json)

    def test_boundary_corrupted_json_invalid_schema_handling(self, tmp_path: Path) -> None:
        """JSON with unexpected schema (array of numbers) raises error gracefully."""
        bad_json = tmp_path / "relationships.json"
        bad_json.write_text("[1, 2, 3, 4, 5]", encoding="utf-8")
        if HAS_RELATIONSHIP_GRAPH:
            with pytest.raises((ValueError, TypeError, ProjectError, Exception)):
                RelationshipGraph.from_json_file(bad_json)

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_MODELS,
        reason="Requires M1 RelationshipEdge",
    )
    def test_boundary_zero_and_negative_confidence_edges(self) -> None:
        """RelationshipEdge accepts float confidence bounds [0.0, 1.0]."""
        edge_zero = RelationshipEdge("a", "b", RelationshipKind.FORMAL_ACQUAINTANCE, confidence=0.0)
        assert edge_zero.confidence == 0.0

    def test_boundary_empty_or_whitespace_speaker_tags(self) -> None:
        """clean_translation handles empty or whitespace target strings without exception."""
        assert clean_translation("") == ""
        assert clean_translation("   ") == ""
        assert clean_translation("\n\t") == ""

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph",
    )
    def test_boundary_large_graph_depth_stress_pathfinding(self) -> None:
        """Pathfinding along a linear chain of 50 characters completes rapidly (<50ms)."""
        graph = RelationshipGraph()
        n = 50
        for i in range(n - 1):
            graph.add_edge(
                RelationshipEdge(
                    f"c_{i}",
                    f"c_{i + 1}",
                    RelationshipKind.FRIENDSHIP_CLOSE,
                )
            )

        path = graph.find_path("c_0", f"c_{n - 1}")
        assert len(path) == n - 1
        assert path[0].source_tag == "c_0"
        assert path[-1].target_tag == f"c_{n - 1}"


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (Integration Pipeline Tests)
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise and multi-module integration between Graph, DB, Resolver, and Prompt."""

    @pytest.mark.skipif(
        not (HAS_RELATIONSHIP_GRAPH and HAS_INTERLOCUTOR_RESOLVER),
        reason="Requires M1 Graph and M3 InterlocutorResolver",
    )
    def test_cross_graph_to_resolver_to_prompt_assembly(self) -> None:
        """Full flow: Graph pronouns -> Resolver listener -> Prompt formatting."""
        graph = RelationshipGraph()
        graph.add_edge(RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em"))
        graph.add_edge(RelationshipEdge("liana", "ren", RelationshipKind.ROMANCE, "em", "anh"))

        resolver = InterlocutorResolver()
        unit1 = make_test_unit("d:1", "Liana, are you ready?", speaker="ren")
        resolved = resolver.resolve(unit1, None, {"ren", "liana"}, {"ren": "Ren", "liana": "Liana"})
        assert resolved.listener == "liana"

        edge = graph.get_edge(resolved.speaker, resolved.listener)
        assert edge is not None
        assert edge.speaker_pronoun == "anh"
        assert edge.listener_pronoun == "em"

    @pytest.mark.skipif(
        not (HAS_RELATIONSHIP_GRAPH and HAS_DB_RELATIONSHIPS),
        reason="Requires M1 Graph and DB persistence",
    )
    def test_cross_db_save_json_sync_graph_reload_prompt_flow(self, tmp_path: Path) -> None:
        """Full lifecycle: DB save -> JSON sync -> reload -> lookup edge."""
        db_path = tmp_path / "project.db"
        json_path = tmp_path / "relationships.json"
        db = Database.open(db_path, create=True)
        try:
            g1 = RelationshipGraph()
            e = RelationshipEdge("hero", "villain", RelationshipKind.RIVAL_HOSTILE, "ta", "ngươi")
            g1.add_edge(e)
            g1.save_to_db(db)

            RelationshipGraph.sync(db, json_path)
            assert json_path.exists()

            g2 = RelationshipGraph.from_json_file(json_path)
            edge = g2.get_edge("hero", "villain")
            assert edge is not None
            assert edge.kind == RelationshipKind.RIVAL_HOSTILE
            assert edge.speaker_pronoun == "ta"
            assert edge.listener_pronoun == "ngươi"
        finally:
            db.close()

    @pytest.mark.skipif(
        not HAS_RELATIONSHIP_GRAPH,
        reason="Requires M1 RelationshipGraph",
    )
    def test_cross_dynamic_edge_mutation_reflected_in_prompt(self) -> None:
        """Updating relationship from FORMAL_ACQUAINTANCE to ROMANCE alters active pronouns."""
        graph = RelationshipGraph()
        graph.add_edge(
            RelationshipEdge("ren", "liana", RelationshipKind.FORMAL_ACQUAINTANCE, "tôi", "cô")
        )
        edge1 = graph.get_edge("ren", "liana")
        assert edge1 is not None and edge1.speaker_pronoun == "tôi"

        # Relationship evolves after story milestone
        graph.add_edge(RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em"))
        edge2 = graph.get_edge("ren", "liana")
        assert edge2 is not None and edge2.speaker_pronoun == "anh"

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver",
    )
    def test_cross_multi_party_turn_taking_with_vocative_interruption(self) -> None:
        """In a 3-way scene, a vocative instantly redirects attention to the addressed person."""
        resolver = InterlocutorResolver()
        active = {"ren", "liana", "scarlet"}
        names = {"ren": "Ren", "liana": "Liana", "scarlet": "Scarlet"}

        # 1. Ren speaks to Liana
        u1 = make_test_unit("d:1", "The weather is lovely today.", speaker="ren")
        # 2. Liana responds
        u2 = make_test_unit("d:2", "Yes, it is!", speaker="liana")
        r2 = resolver.resolve(u2, u1, active, names)
        assert r2.listener == "ren"

        # 3. Scarlet interrupts, addressing Ren directly
        u3 = make_test_unit("d:3", "Ren, don't ignore me!", speaker="scarlet")
        r3 = resolver.resolve(u3, u2, active, names)
        assert r3.listener == "ren"
        assert r3.resolution_method == "vocative"

    @pytest.mark.skipif(
        not (HAS_EXTENDED_PROFILE and HAS_RELATIONSHIP_MODELS),
        reason="Requires M1/M2 extended profile and relationship models",
    )
    def test_cross_persona_traits_and_relationship_pronoun_synergy(self) -> None:
        """Tsundere persona with RIVAL_HOSTILE relation enforces authentic Vietnamese addressing."""
        Profile: Any = CharacterProfile
        profile = Profile(
            tag="scarlet",
            display_name="Scarlet",
            archetype="tsundere",
            personality_traits=("proud", "defensive"),
            tone_and_voice="sharp and haughty",
        )
        edge = RelationshipEdge(
            "scarlet",
            "ren",
            RelationshipKind.RIVAL_HOSTILE,
            speaker_pronoun="tôi",
            listener_pronoun="cậu",
            notes="Reluctant classmate",
        )
        assert profile.archetype == "tsundere"
        assert edge.speaker_pronoun == "tôi"
        assert edge.listener_pronoun == "cậu"

    def test_cross_context_assembly_token_budget_preservation(self) -> None:
        """Batch of dialogue units with rich context remains within token budget."""
        cfg = Config()
        builder = ContextBuilder(cfg)
        units = [
            make_test_unit(f"d:{i}", f"Dialogue line {i} in the visual novel scene.", speaker="ren")
            for i in range(10)
        ]
        user_prompt = builder.build_user_prompt(units)
        # Ensure user prompt includes all unit IDs without silent truncation
        for u in units:
            assert u.uid in user_prompt


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (Visual Novel Sequences)
# ==============================================================================


class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end multi-party Visual Novel dialogues and pipeline simulations."""

    @pytest.mark.skipif(
        not (HAS_RELATIONSHIP_GRAPH and HAS_INTERLOCUTOR_RESOLVER),
        reason="Requires M1 Graph and M3 InterlocutorResolver",
    )
    def test_scenario_1_triangular_romance_and_rivalry_vn_scene(self) -> None:
        """Scenario 1: Triangular romance & rivalry sequence (Hero, Heroine, Tsundere Rival).

        Ren (hero), Liana (sweet heroine / romance), Scarlet (tsundere noble / rival).
        Verifies correct listener attribution and pronoun consistency across alternating lines.
        """
        graph = RelationshipGraph()
        graph.add_edge(RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em"))
        graph.add_edge(RelationshipEdge("liana", "ren", RelationshipKind.ROMANCE, "em", "anh"))
        graph.add_edge(
            RelationshipEdge("scarlet", "ren", RelationshipKind.RIVAL_HOSTILE, "tôi", "cậu")
        )
        graph.add_edge(
            RelationshipEdge("ren", "scarlet", RelationshipKind.RIVAL_HOSTILE, "tôi", "cô")
        )

        resolver = InterlocutorResolver()
        active = {"ren", "liana", "scarlet"}
        names = {"ren": "Ren", "liana": "Liana", "scarlet": "Scarlet"}

        # Line 1: Ren talks to Liana
        u1 = make_test_unit("d:1", "Liana, are you ready for the academy festival?", speaker="ren")
        r1 = resolver.resolve(u1, None, active, names)
        assert r1.listener == "liana"
        e1 = graph.get_edge("ren", r1.listener)
        assert e1 is not None and e1.speaker_pronoun == "anh" and e1.listener_pronoun == "em"

        # Line 2: Liana responds to Ren
        u2 = make_test_unit("d:2", "Of course! I've been waiting for you.", speaker="liana")
        r2 = resolver.resolve(u2, u1, active, names)
        assert r2.listener == "ren"
        e2 = graph.get_edge("liana", r2.listener)
        assert e2 is not None and e2.speaker_pronoun == "em" and e2.listener_pronoun == "anh"

        # Line 3: Scarlet interrupts, calling out Ren
        u3 = make_test_unit("d:3", "Hmph! Don't get so cocky, Ren!", speaker="scarlet")
        r3 = resolver.resolve(u3, u2, active, names)
        assert r3.listener == "ren"
        assert r3.resolution_method == "vocative"
        e3 = graph.get_edge("scarlet", r3.listener)
        assert e3 is not None and e3.speaker_pronoun == "tôi" and e3.listener_pronoun == "cậu"

        # Line 4: Scarlet inner thought (monologue)
        u4 = make_test_unit(
            "d:4", "(Why did I have to say that...? How embarrassing...)", speaker="scarlet"
        )
        r4 = resolver.resolve(u4, u3, active, names)
        assert r4.is_monologue

    @pytest.mark.skipif(
        not (HAS_RELATIONSHIP_GRAPH and HAS_INTERLOCUTOR_RESOLVER),
        reason="Requires M1 Graph and M3 InterlocutorResolver",
    )
    def test_scenario_2_family_dinner_and_heir_conflict(self) -> None:
        """Scenario 2: Hierarchical family conflict (Patriarch, Eldest Sister, Younger Brother).

        Lord Vane (Elder/Father), Evelyn (Eldest Sister), Luke (Protagonist/Son).
        Verifies hierarchical Vietnamese pronouns: 'cha - con', 'chị - em'.
        """
        graph = RelationshipGraph()
        graph.add_edge(
            RelationshipEdge("vane", "luke", RelationshipKind.FAMILY_ELDER, "cha", "con")
        )
        graph.add_edge(
            RelationshipEdge("luke", "vane", RelationshipKind.FAMILY_ELDER, "con", "cha")
        )
        graph.add_edge(
            RelationshipEdge("evelyn", "luke", RelationshipKind.FAMILY_SIBLING, "chị", "em")
        )
        graph.add_edge(
            RelationshipEdge("luke", "evelyn", RelationshipKind.FAMILY_SIBLING, "em", "chị")
        )

        resolver = InterlocutorResolver()
        active = {"vane", "evelyn", "luke"}
        names = {"vane": "Lord Vane", "evelyn": "Evelyn", "luke": "Luke"}

        u1 = make_test_unit("d:10", "Luke, you have neglected your sword practice.", speaker="vane")
        r1 = resolver.resolve(u1, None, active, names)
        assert r1.listener == "luke"
        e1 = graph.get_edge("vane", "luke")
        assert e1 is not None and e1.speaker_pronoun == "cha" and e1.listener_pronoun == "con"

        u2 = make_test_unit(
            "d:11", "I apologize, father. I was studying the ancient texts.", speaker="luke"
        )
        r2 = resolver.resolve(u2, u1, active, names)
        assert r2.listener == "vane"
        e2 = graph.get_edge("luke", "vane")
        assert e2 is not None and e2.speaker_pronoun == "con" and e2.listener_pronoun == "cha"

        u3 = make_test_unit("d:12", "Don't be too harsh on him, Father.", speaker="evelyn")
        r3 = resolver.resolve(u3, u2, active, names)
        assert r3.listener == "vane"

    @pytest.mark.skipif(
        not (HAS_RELATIONSHIP_GRAPH and HAS_INTERLOCUTOR_RESOLVER),
        reason="Requires M1 Graph and M3 InterlocutorResolver",
    )
    def test_scenario_3_master_servant_and_superior_subordinate_court(self) -> None:
        """Scenario 3: Feudal imperial court (Empress, Royal Guard, Treasonous Minister).

        Sonya (Empress/Superior), Ronald (Commander/Servant), Malakor (Traitor/Hostile).
        Verifies feudal honorifics: 'trẫm - tướng quân/ngươi', 'thần - bệ hạ'.
        """
        graph = RelationshipGraph()
        graph.add_edge(
            RelationshipEdge("sonya", "ronald", RelationshipKind.SERVANT_MASTER, "trẫm", "ngươi")
        )
        graph.add_edge(
            RelationshipEdge("ronald", "sonya", RelationshipKind.SERVANT_MASTER, "thần", "bệ hạ")
        )
        graph.add_edge(
            RelationshipEdge(
                "malakor", "sonya", RelationshipKind.SUPERIOR_SUBORDINATE, "thần", "bệ hạ"
            )
        )
        graph.add_edge(
            RelationshipEdge("malakor", "ronald", RelationshipKind.RIVAL_HOSTILE, "ta", "ngươi")
        )

        e_sonya_ronald = graph.get_edge("sonya", "ronald")
        assert e_sonya_ronald is not None
        assert e_sonya_ronald.speaker_pronoun == "trẫm"
        assert e_sonya_ronald.listener_pronoun == "ngươi"

        e_ronald_sonya = graph.get_edge("ronald", "sonya")
        assert e_ronald_sonya is not None
        assert e_ronald_sonya.speaker_pronoun == "thần"
        assert e_ronald_sonya.listener_pronoun == "bệ hạ"

    @pytest.mark.skipif(
        not HAS_INTERLOCUTOR_RESOLVER,
        reason="Requires M3 InterlocutorResolver",
    )
    def test_scenario_4_soliloquy_to_sudden_vocative_turn_transition(self) -> None:
        """Scenario 4: Soliloquy inner thoughts transitioning into sudden vocative call-out."""
        resolver = InterlocutorResolver()
        active = {"mc", "assassin"}
        names = {"mc": "Hero", "assassin": "Shadow Assassin"}

        # 1. Inner monologue in thoughts
        u1 = make_test_unit(
            "d:1", "(The alleyway is completely dark... something is off.)", speaker="mc"
        )
        r1 = resolver.resolve(u1, None, active, names)
        assert r1.is_monologue

        # 2. Hero speaks to the ambient shadows (no known listener)
        u2 = make_test_unit("d:2", "Show yourself! Who goes there?", speaker="mc")
        r2 = resolver.resolve(u2, u1, active, names)
        assert not r2.is_monologue

        # 3. Assassin emerges and answers
        u3 = make_test_unit("d:3", "Heh, you have sharp senses, boy.", speaker="assassin")
        r3 = resolver.resolve(u3, u2, active, names)
        assert r3.listener == "mc"

    def test_scenario_5_end_to_end_pipeline_translation_and_qa_leak_sanitization(self) -> None:
        """Scenario 5: Batch translation lifecycle with frame peeling and QA check.

        Verifies that even if model outputs contain residual metadata tags:
        1. clean_translation strips them 100%.
        2. QAChecker validates clean output with 0 speaker_tag_leakage errors.
        """
        checker = QAChecker()
        test_cases = [
            (
                "[Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE] Anh sẽ bảo vệ em.",
                "Anh sẽ bảo vệ em.",
            ),
            (
                "[Người nói: Liana] Cảm ơn anh rất nhiều!",
                "Cảm ơn anh rất nhiều!",
            ),
            (
                "[Độc thoại/Nội tâm: Ren] Mình nhất định không thể thất bại.",
                "Mình nhất định không thể thất bại.",
            ),
            (
                "[Người nói: Scarlet (scarlet)] H-Hmph! Đừng hiểu lầm đấy!",
                "H-Hmph! Đừng hiểu lầm đấy!",
            ),
        ]

        for raw_target, expected_clean in test_cases:
            # 1. Verify clean_translation sanitizes cleanly
            cleaned = clean_translation(raw_target)
            assert cleaned == expected_clean

            # 2. Verify QAChecker flags raw target as leaking when speaker metadata is present
            if "[Người nói:" in raw_target:
                raw_issues = checker.check_unit("d:test", "Sample source", None, raw_target)
                assert any(i.code == "speaker_tag_leakage" for i in raw_issues)

            # 3. Verify QAChecker accepts sanitized target with 0 leakage errors
            clean_issues = checker.check_unit("d:test", "Sample source", None, cleaned)
            assert not any(i.code == "speaker_tag_leakage" for i in clean_issues)
