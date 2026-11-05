"""Comprehensive property-based and unit test suite for 4D relational vector algebra.

Verifies:
1. 4D Relational Vector instantiation, bounding clamping, and immutability.
2. Vector inversion anti-symmetry and algebraic involution: (R^-1)^-1 == R.
3. Deterministic algebraic projection across all kinship and non-kinship partitions.
4. Reciprocal Invariant Closure Theorem across thousands of randomized permutations.
5. SQLite schema migration v7, relational columns, view, and repository roundtrips.
6. RelationshipGraph integration, bilateral lookup, and reciprocal synchronization.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from renslate.core.db import Database
from renslate.core.db.repositories import RelationshipRepository
from renslate.core.db.schema import MIGRATIONS
from renslate.core.models import CharacterProfile, RelationshipEdge, RelationshipKind
from renslate.domain.relational import KinshipKind, RelationalVector, project_relational_vector
from renslate.engine.graph import RelationshipGraph
from renslate.engine.relationships import PairEvidence, _apply_pair, _ParsedPair


class TestRelationalVectorFundamentals:
    """Unit tests for 4D relational state vector instantiation and boundary guarantees."""

    def test_default_vector_values(self) -> None:
        vec = RelationalVector()
        assert vec.kinship == KinshipKind.NONE
        assert vec.delta_hierarchy == 0.0
        assert vec.affinity == 0.0
        assert vec.register == 0.5

    def test_clamping_out_of_bounds_hierarchy(self) -> None:
        vec_pos = RelationalVector(delta_hierarchy=999.0)
        assert vec_pos.delta_hierarchy == 1.0
        vec_neg = RelationalVector(delta_hierarchy=-999.0)
        assert vec_neg.delta_hierarchy == -1.0

    def test_clamping_out_of_bounds_affinity(self) -> None:
        vec_pos = RelationalVector(affinity=5.5)
        assert vec_pos.affinity == 1.0
        vec_neg = RelationalVector(affinity=-5.5)
        assert vec_neg.affinity == -1.0

    def test_clamping_out_of_bounds_register(self) -> None:
        vec_high = RelationalVector(register=2.0)
        assert vec_high.register == 1.0
        vec_low = RelationalVector(register=-1.0)
        assert vec_low.register == 0.0

    def test_string_kinship_coercion(self) -> None:
        vec = RelationalVector(kinship="romantic_couple")  # type: ignore[arg-type]
        assert vec.kinship == KinshipKind.ROMANTIC_COUPLE
        vec_invalid = RelationalVector(kinship="INVALID_BOND")  # type: ignore[arg-type]
        assert vec_invalid.kinship == KinshipKind.NONE

    def test_immutability_frozen_dataclass(self) -> None:
        vec = RelationalVector()
        with pytest.raises(FrozenInstanceError):
            vec.delta_hierarchy = 0.5  # type: ignore[misc]

    def test_serialization_dict_roundtrip(self) -> None:
        vec = RelationalVector(
            kinship=KinshipKind.MENTOR_DISCIPLE,
            delta_hierarchy=0.75,
            affinity=0.85,
            register=0.9,
        )
        d = vec.to_dict()
        assert d == {
            "kinship": "MENTOR_DISCIPLE",
            "delta_hierarchy": 0.75,
            "affinity": 0.85,
            "register": 0.9,
        }
        reconstructed = RelationalVector.from_dict(d)
        assert reconstructed == vec

    def test_involution_theorem(self) -> None:
        """Theorem: (R^-1)^-1 == R for all relational state vectors."""
        for _ in range(50):
            dh = random.uniform(-1.0, 1.0)
            aff = random.uniform(-1.0, 1.0)
            reg = random.uniform(0.0, 1.0)
            kin = random.choice(list(KinshipKind))
            vec = RelationalVector(kinship=kin, delta_hierarchy=dh, affinity=aff, register=reg)
            inv = vec.inverted()
            assert inv.delta_hierarchy == pytest.approx(-vec.delta_hierarchy)
            assert inv.affinity == pytest.approx(vec.affinity)
            assert inv.register == pytest.approx(vec.register)
            assert inv.kinship == vec.kinship.inverse()
            assert inv.inverted().delta_hierarchy == pytest.approx(vec.delta_hierarchy)
            assert inv.inverted().affinity == pytest.approx(vec.affinity)
            assert inv.inverted().register == pytest.approx(vec.register)


class TestAlgebraicProjectionPartitions:
    """Unit tests for piecewise projection operator across all relational partitions."""

    def test_projection_romantic_anh_em(self) -> None:
        vec = RelationalVector(kinship=KinshipKind.ROMANTIC_COUPLE, delta_hierarchy=0.2, affinity=0.8, register=0.4)
        assert project_relational_vector(vec, "male", "female") == ("anh", "em")
        assert project_relational_vector(vec.inverted(), "female", "male") == ("em", "anh")

    def test_projection_romantic_courtly_chang_nang(self) -> None:
        vec = RelationalVector(kinship=KinshipKind.ROMANTIC_COUPLE, register=0.85)
        assert project_relational_vector(vec, "male", "female") == ("chàng", "nàng")
        assert project_relational_vector(vec.inverted(), "female", "male") == ("nàng", "chàng")

    def test_projection_parent_child(self) -> None:
        vec = RelationalVector(kinship=KinshipKind.PARENT_CHILD, delta_hierarchy=0.8)
        assert project_relational_vector(vec, "male", "female") == ("cha", "con")
        assert project_relational_vector(vec.inverted(), "female", "male") == ("con", "cha")

        assert project_relational_vector(vec, "female", "male") == ("mẹ", "con")
        assert project_relational_vector(vec.inverted(), "male", "female") == ("con", "mẹ")

    def test_projection_sibling(self) -> None:
        vec = RelationalVector(kinship=KinshipKind.SIBLING, delta_hierarchy=0.3)
        assert project_relational_vector(vec, "male", "female") == ("anh", "em")
        assert project_relational_vector(vec.inverted(), "female", "male") == ("em", "anh")

        assert project_relational_vector(vec, "female", "male") == ("chị", "em")
        assert project_relational_vector(vec.inverted(), "male", "female") == ("em", "chị")

    def test_projection_mentor_disciple(self) -> None:
        vec_formal = RelationalVector(kinship=KinshipKind.MENTOR_DISCIPLE, delta_hierarchy=0.6, register=0.8)
        assert project_relational_vector(vec_formal, "male", "male") == ("thầy", "trò")
        assert project_relational_vector(vec_formal.inverted(), "male", "male") == ("trò", "thầy")

        vec_informal = RelationalVector(kinship=KinshipKind.MENTOR_DISCIPLE, delta_hierarchy=0.6, register=0.4)
        assert project_relational_vector(vec_informal, "male", "male") == ("ta", "con")
        assert project_relational_vector(vec_informal.inverted(), "male", "male") == ("con", "ta")

    def test_projection_sovereign_subject(self) -> None:
        vec_imperial = RelationalVector(kinship=KinshipKind.SOVEREIGN_SUBJECT, delta_hierarchy=0.9, register=0.9)
        assert project_relational_vector(vec_imperial, "male", "male") == ("trẫm", "khanh")
        assert project_relational_vector(vec_imperial.inverted(), "male", "male") == ("khanh", "trẫm")

        vec_archaic = RelationalVector(kinship=KinshipKind.SOVEREIGN_SUBJECT, delta_hierarchy=0.9, register=0.5)
        assert project_relational_vector(vec_archaic, "male", "male") == ("ta", "ngươi")
        assert project_relational_vector(vec_archaic.inverted(), "male", "male") == ("ngươi", "ta")

    def test_projection_avuncular_and_grand(self) -> None:
        vec_av = RelationalVector(kinship=KinshipKind.AVUNCULAR, delta_hierarchy=0.7)
        assert project_relational_vector(vec_av, "male", "male") == ("chú", "cháu")
        assert project_relational_vector(vec_av.inverted(), "male", "male") == ("cháu", "chú")

        vec_gr = RelationalVector(kinship=KinshipKind.GRAND_RELATION, delta_hierarchy=0.95)
        assert project_relational_vector(vec_gr, "female", "male") == ("bà", "cháu")
        assert project_relational_vector(vec_gr.inverted(), "male", "female") == ("cháu", "bà")

    def test_projection_sworn_alliance(self) -> None:
        vec_sw = RelationalVector(kinship=KinshipKind.SWORN_ALLIANCE, delta_hierarchy=0.3, register=0.8)
        assert project_relational_vector(vec_sw, "male", "male") == ("anh", "hiền đệ")
        assert project_relational_vector(vec_sw.inverted(), "male", "male") == ("hiền đệ", "anh")

    def test_projection_hostility_tao_may_and_ta_nguoi(self) -> None:
        vec_vulgar = RelationalVector(affinity=-0.9, register=0.1)
        assert project_relational_vector(vec_vulgar, "male", "male") == ("tao", "mày")
        assert project_relational_vector(vec_vulgar.inverted(), "male", "male") == ("tao", "mày")

        vec_archaic = RelationalVector(affinity=-0.8, register=0.6)
        assert project_relational_vector(vec_archaic, "male", "male") == ("ta", "ngươi")

    def test_projection_street_intimacy(self) -> None:
        vec = RelationalVector(affinity=0.5, register=0.2, delta_hierarchy=0.0)
        assert project_relational_vector(vec, "male", "male") == ("tao", "mày")

    def test_projection_peer_friendship(self) -> None:
        vec = RelationalVector(affinity=0.6, register=0.3, delta_hierarchy=0.0)
        assert project_relational_vector(vec, "female", "female") == ("tớ", "cậu")

    def test_projection_hierarchical_non_kinship(self) -> None:
        vec_senior_male = RelationalVector(delta_hierarchy=0.5, affinity=0.1, register=0.5)
        assert project_relational_vector(vec_senior_male, "male", "female") == ("anh", "em")
        assert project_relational_vector(vec_senior_male.inverted(), "female", "male") == ("em", "anh")

        vec_senior_female = RelationalVector(delta_hierarchy=0.5, affinity=0.1, register=0.5)
        assert project_relational_vector(vec_senior_female, "female", "male") == ("chị", "em")
        assert project_relational_vector(vec_senior_female.inverted(), "male", "female") == ("em", "chị")

    def test_projection_formal_acquaintance_neutral(self) -> None:
        vec = RelationalVector()
        assert project_relational_vector(vec, "unknown", "unknown") == ("tôi", "bạn")
        assert project_relational_vector(vec, "male", "female") == ("tôi", "bạn")
        vec_formal = RelationalVector(register=0.8)
        assert project_relational_vector(vec_formal, "male", "female") == ("tôi", "cô")
        assert project_relational_vector(vec_formal.inverted(), "female", "male") == ("tôi", "anh")


class TestPropertyBasedReciprocalInvariantClosure:
    """Property-based verification of reciprocal invariant closure across random permutations."""

    def test_thousands_of_hierarchical_vector_permutations(self) -> None:
        """Prove that for all non-zero hierarchy vectors, Pi(R_BA, G_B, G_A) == sigma(Pi(R_AB, G_A, G_B))."""
        rng = random.Random(42)
        kinship_choices = list(KinshipKind)
        genders = ["male", "female", "unknown"]

        trials = 2500
        for _ in range(trials):
            kin = rng.choice(kinship_choices)
            # Pick non-zero delta_hierarchy
            dh = rng.uniform(-1.0, 1.0)
            while abs(dh) < 0.001:
                dh = rng.uniform(-1.0, 1.0)
            aff = rng.uniform(-1.0, 1.0)
            reg = rng.uniform(0.0, 1.0)

            vec_ab = RelationalVector(kinship=kin, delta_hierarchy=dh, affinity=aff, register=reg)
            vec_ba = vec_ab.inverted()

            ga = rng.choice(genders)
            gb = rng.choice(genders)

            p1, p2 = project_relational_vector(vec_ab, ga, gb)
            q1, q2 = project_relational_vector(vec_ba, gb, ga)

            assert (q1, q2) == (p2, p1), (
                f"Reciprocal invariant closure violated!\n"
                f"Vector AB: {vec_ab}, ga={ga}, gb={gb} -> ({p1!r}, {p2!r})\n"
                f"Vector BA: {vec_ba}, gb={gb}, ga={ga} -> ({q1!r}, {q2!r})"
            )

    def test_peer_symmetric_vector_permutations(self) -> None:
        """Prove that for zero hierarchy peer vectors with identical genders, Pi(R_BA) == Pi(R_AB)."""
        rng = random.Random(1337)
        trials = 1000
        for _ in range(trials):
            aff = rng.uniform(-1.0, 1.0)
            reg = rng.uniform(0.0, 1.0)
            g = rng.choice(["male", "female", "unknown"])

            vec_ab = RelationalVector(kinship=KinshipKind.NONE, delta_hierarchy=0.0, affinity=aff, register=reg)
            vec_ba = vec_ab.inverted()

            p1, p2 = project_relational_vector(vec_ab, g, g)
            q1, q2 = project_relational_vector(vec_ba, g, g)

            assert (q1, q2) == (p1, p2), (
                f"Symmetric baseline violated for peer:\n"
                f"Vector: {vec_ab}, g={g} -> ({p1!r}, {p2!r}) vs ({q1!r}, {q2!r})"
            )


class TestDatabaseMigrationV7AndPersistence:
    """Tests for SQLite schema migration v7, 4D column persistence, and legacy upgrade."""

    def test_schema_migration_v7_adds_columns_and_view(self, tmp_path: Path) -> None:
        db_path = tmp_path / "migration_v7.db"
        with Database.open(db_path, create=True) as db:
            assert db.schema_version >= 6

            cols = {
                r["name"]
                for r in db.execute("PRAGMA table_info(character_relationships);").fetchall()
            }
            assert "kinship" in cols
            assert "delta_hierarchy" in cols
            assert "affinity" in cols
            assert "register" in cols

            # Check view
            views = {
                r["name"]
                for r in db.execute("SELECT name FROM sqlite_master WHERE type='view';").fetchall()
            }
            assert "relationships" in views

    def test_migration_v7_backfills_legacy_rows(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy_v6.db"
        conn = sqlite3.connect(str(db_path))

        # Manually run v1 to v6
        for script in MIGRATIONS[:6]:
            conn.executescript(script)
        conn.execute("PRAGMA user_version = 6;")

        # Insert legacy rows in character_relationships
        conn.execute(
            """
            INSERT INTO character_relationships
            (source_tag, target_tag, kind, speaker_pronoun, listener_pronoun, confidence, source_type, notes)
            VALUES
            ('mc', 'yua', 'ROMANCE', 'anh', 'em', 0.9, 'manual', 'lovers'),
            ('mc', 'mentor', 'MENTOR_DISCIPLE', 'con', 'thầy', 0.85, 'llm', 'training'),
            ('sister', 'mc', 'FAMILY_SIBLING', 'chị', 'em', 0.9, 'manual', 'family'),
            ('villain', 'mc', 'RIVAL_HOSTILE', 'ta', 'ngươi', 0.8, 'llm', 'enemy');
            """
        )
        conn.commit()
        conn.close()

        # Open with Database to trigger migration v7 and backfill
        with Database.open(db_path) as db:
            repo = RelationshipRepository(db._connection)
            edge_romance = repo.get_relationship("mc", "yua")
            assert edge_romance is not None
            assert edge_romance.vector.kinship == KinshipKind.ROMANTIC_COUPLE
            assert edge_romance.vector.delta_hierarchy == 0.2
            assert edge_romance.vector.affinity == 0.8

            edge_mentor = repo.get_relationship("mc", "mentor")
            assert edge_mentor is not None
            assert edge_mentor.vector.kinship == KinshipKind.MENTOR_DISCIPLE
            assert edge_mentor.vector.delta_hierarchy == 0.6

            edge_sibling = repo.get_relationship("sister", "mc")
            assert edge_sibling is not None
            assert edge_sibling.vector.kinship == KinshipKind.SIBLING
            assert edge_sibling.vector.delta_hierarchy == 0.3

            edge_rival = repo.get_relationship("villain", "mc")
            assert edge_rival is not None
            assert edge_rival.vector.affinity == -0.8

    def test_repository_save_and_load_4d_vector(self, tmp_path: Path) -> None:
        db_path = tmp_path / "repo_vector.db"
        with Database.open(db_path, create=True) as db:
            repo = RelationshipRepository(db._connection)
            vec = RelationalVector(
                kinship=KinshipKind.SOVEREIGN_SUBJECT,
                delta_hierarchy=0.95,
                affinity=0.15,
                register=0.9,
            )
            edge = RelationshipEdge(
                source_tag="emperor",
                target_tag="general",
                kind=RelationshipKind.SERVANT_MASTER,
                speaker_pronoun="trẫm",
                listener_pronoun="khanh",
                confidence=0.98,
                source_type="manual",
                notes="courtly sovereign",
                vector=vec,
            )
            repo.save_relationship(edge)

            loaded = repo.get_relationship("emperor", "general")
            assert loaded is not None
            assert loaded.source_tag == "emperor"
            assert loaded.target_tag == "general"
            assert loaded.speaker_pronoun == "trẫm"
            assert loaded.listener_pronoun == "khanh"
            assert loaded.vector == vec


class TestGraphEngineIntegration:
    """Tests for RelationshipGraph integration with 4D vectors and reciprocal closure."""

    def test_graph_add_and_retrieve_relational_vector(self) -> None:
        graph = RelationshipGraph()
        vec = RelationalVector(
            kinship=KinshipKind.ROMANTIC_COUPLE,
            delta_hierarchy=0.2,
            affinity=0.9,
            register=0.4,
        )
        edge = RelationshipEdge(
            source_tag="hero",
            target_tag="heroine",
            kind=RelationshipKind.ROMANCE,
            speaker_pronoun="anh",
            listener_pronoun="em",
            vector=vec,
        )
        graph.add_edge(edge)

        retrieved_vec = graph.get_relational_vector("hero", "heroine")
        assert retrieved_vec == vec
        assert graph.get_relational_vector("nonexistent", "heroine") is None

    def test_graph_project_address_method(self) -> None:
        graph = RelationshipGraph()
        vec = RelationalVector(
            kinship=KinshipKind.SIBLING,
            delta_hierarchy=0.4,
            affinity=0.7,
            register=0.3,
        )
        edge = RelationshipEdge(
            source_tag="big_brother",
            target_tag="little_sister",
            kind=RelationshipKind.FAMILY_SIBLING,
            speaker_pronoun="anh",
            listener_pronoun="em",
            vector=vec,
        )
        graph.add_edge(edge)

        spk, lis = graph.project_address("big_brother", "little_sister")
        assert (spk, lis) == ("anh", "em")

        # Reverse on unregistered reverse edge projects from default
        default_spk, default_lis = graph.project_address("stranger", "other")
        assert (default_spk, default_lis) == ("tôi", "bạn")

    def test_graph_serialization_preserves_vector(self) -> None:
        graph = RelationshipGraph()
        vec = RelationalVector(
            kinship=KinshipKind.MENTOR_DISCIPLE,
            delta_hierarchy=0.6,
            affinity=0.5,
            register=0.7,
        )
        edge = RelationshipEdge(
            source_tag="mentor",
            target_tag="pupil",
            kind=RelationshipKind.MENTOR_DISCIPLE,
            speaker_pronoun="thầy",
            listener_pronoun="trò",
            vector=vec,
        )
        graph.add_edge(edge)

        d = graph.to_dict()
        reconstructed = RelationshipGraph.from_dict(d)
        edge_rec = reconstructed.get_edge("mentor", "pupil")
        assert edge_rec is not None
        assert edge_rec.vector == vec

    def test_apply_pair_enforces_algebraic_reciprocity_in_llm_bootstrap(self) -> None:
        """Verify _apply_pair normalizes asymmetric LLM outputs into reciprocal 4D pairs."""
        graph = RelationshipGraph()
        profiles = {
            "mc": CharacterProfile(tag="mc", display_name="Hero", gender="male"),
            "liana": CharacterProfile(tag="liana", display_name="Liana", gender="female"),
        }
        ev = PairEvidence(a="mc", b="liana", turns=10)

        # Mismatched asymmetric input from hypothetical hallucinating LLM
        parsed = _ParsedPair(
            kind=RelationshipKind.ROMANCE,
            a_to_b=("anh", "em"),
            b_to_a=("tôi", "cậu"),  # Asymmetric mismatch!
            confidence=0.85,
            evidence="Romantic interest",
        )

        written, _changed = _apply_pair(graph, profiles, ev, parsed)
        assert written == 2

        edge_ab, edge_ba = graph.get_bilateral("mc", "liana")
        assert edge_ab is not None
        assert edge_ba is not None

        # Symmetrical algebraic reciprocal closure is strictly enforced!
        assert edge_ab.speaker_pronoun == "anh" and edge_ab.listener_pronoun == "em"
        assert edge_ba.speaker_pronoun == "em" and edge_ba.listener_pronoun == "anh"

        # Relational vectors are mutually inverted
        assert edge_ba.vector == edge_ab.vector.inverted()
        assert edge_ab.vector.delta_hierarchy == pytest.approx(-edge_ba.vector.delta_hierarchy)
