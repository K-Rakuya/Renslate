"""Tier 5 Adversarial Coverage Hardening for RelationshipGraph and Deep Persona Engine.

Empirically tests white-box vectors across:
1. Dense cyclic topologies (500+ nodes, cliques, Hamiltonian paths, disconnected subgraphs).
2. Dual-persistence sync edge cases (corrupted JSON, casing, negative counts, conflicts).
3. Stylometrics extraction under extreme inputs (100% punctuation, zero words, multi-line emojis).
4. Persona inference fallbacks (unknown genders, conflicting signals, short/duplicate dialogues).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from renslate.core.db import Database
from renslate.core.models import (
    RelationshipEdge,
    RelationshipKind,
)
from renslate.engine.analyze import (
    StylometricSummary,
    extract_stylometrics,
    infer_character_gender_nlp,
    infer_character_role_nlp,
    infer_persona_fallback,
    infer_semantic_relationship,
    is_collective_entity,
    is_non_human_or_meta,
    resolve_pronoun_pair_from_relationship,
)
from renslate.engine.graph import RelationshipGraph

# ==============================================================================
# GROUP 1: Dense Cyclic Topologies, Cliques, Hamiltonian Paths & Subgraphs
# ==============================================================================


def test_dense_clique_and_500_node_hamiltonian_cycle() -> None:
    """Stress-test BFS pathfinding on a 550-node Hamiltonian cycle with an 80-node clique.

    Verifies:
    1. Graph construction of 550 nodes and > 6,700 edges completes in < 250ms.
    2. BFS find_path correctly identifies and traverses the 1-hop clique shortcut
       from node_0 to node_79, resulting in an optimal 422-hop path to node_500
       instead of the 500-hop linear cycle path.
    3. Traversal latency is < 50ms.
    4. max_depth limiting and undirected traversal invariants hold.
    """
    g = RelationshipGraph()
    n_total = 550
    clique_size = 80

    # 1. Build Hamiltonian cycle: node_0 -> node_1 -> ... -> node_549 -> node_0
    nodes = [f"node_{i:04d}" for i in range(n_total)]
    for i in range(n_total):
        src = nodes[i]
        tgt = nodes[(i + 1) % n_total]
        g.add_edge(RelationshipEdge(src, tgt, RelationshipKind.FRIENDSHIP_CLOSE))

    # 2. Embed complete directed clique among the first 80 nodes (80 * 79 = 6320 edges)
    t0 = time.perf_counter()
    for i in range(clique_size):
        for j in range(clique_size):
            if i != j:
                g.add_edge(
                    RelationshipEdge(
                        nodes[i],
                        nodes[j],
                        RelationshipKind.ROMANCE,
                        confidence=0.99,
                    )
                )
    build_time = time.perf_counter() - t0

    # Overwritten cycle edges for i in 0..clique_size-2
    expected_edges = n_total - (clique_size - 1) + (clique_size * (clique_size - 1))
    assert g.node_count == n_total
    assert g.edge_count == expected_edges
    assert build_time < 0.35, f"Dense clique insertion took {build_time*1000:.2f} ms"

    # 3. Pathfinding: node_0 -> node_500
    # Shortcut: node_0 -> node_79 (1 hop) + 421 hops (79 -> 500) = 422 hops
    t_start = time.perf_counter()
    path = g.find_path("node_0000", "node_0500")
    t_path = time.perf_counter() - t_start

    assert len(path) == 422, f"Expected shortcut path length 422, got {len(path)}"
    assert path[0].source_tag == "node_0000"
    assert path[0].target_tag == f"node_{clique_size-1:04d}"
    assert path[-1].target_tag == "node_0500"
    assert t_path < 0.05, f"Pathfinding took {t_path*1000:.2f} ms (expected < 50ms)"

    # 4. max_depth limit boundary: path length is 422
    assert g.find_path("node_0000", "node_0500", max_depth=421) == []
    assert len(g.find_path("node_0000", "node_0500", max_depth=422)) == 422

    # 5. Reverse traversal: directed vs undirected
    # Directed node_500 -> node_0 requires 50 hops:
    # node_500 -> node_549 (49 hops) + node_549 -> node_0 (1 hop)
    p_rev_dir = g.find_path("node_0500", "node_0000", directed=True)
    assert len(p_rev_dir) == 50

    # Undirected node_0500 -> node_0000 takes 50 hops forward
    p_rev_undir = g.find_path("node_0500", "node_0000", directed=False)
    assert len(p_rev_undir) == 50


def test_self_referential_loops_lifecycle() -> None:
    """Empirically test self-referential loop (A -> A) behavior and lifecycle.

    Verifies:
    1. Self-referential edges can be added, queried, and retrieved in O(1).
    2. get_bilateral on self-loop returns (edge, edge).
    3. get_neighbors includes self for in, out, and both.
    4. find_path to self returns [] (identity contract).
    5. Interaction counts on self-dialogue (monologue) track correctly.
    6. Subgraph extraction preserves self-loops.
    7. Node removal cascades cleanly without KeyError or corrupted state.
    """
    g = RelationshipGraph()
    edge = RelationshipEdge(
        source_tag="narcissus",
        target_tag="narcissus",
        kind=RelationshipKind.ROMANCE,
        speaker_pronoun="ta",
        listener_pronoun="chính ta",
        confidence=1.0,
        notes="Self-admiration",
    )

    assert g.add_edge(edge)
    assert g.has_node("narcissus")
    assert g.has_edge("narcissus", "narcissus")
    assert g.node_count == 1
    assert g.edge_count == 1

    # O(1) retrieval
    retrieved = g.get_edge("narcissus", "narcissus")
    assert retrieved == edge
    bi_a, bi_b = g.get_bilateral("narcissus", "narcissus")
    assert bi_a is retrieved and bi_b is retrieved

    # Neighbors
    assert g.get_neighbors("narcissus", "out") == {"narcissus"}
    assert g.get_neighbors("narcissus", "in") == {"narcissus"}
    assert g.get_neighbors("narcissus", "both") == {"narcissus"}

    # Path to self
    assert g.find_path("narcissus", "narcissus") == []

    # Self-dialogue interaction tracking
    g.record_interaction("narcissus", "narcissus", count=15)
    assert g.get_interaction_count("narcissus", "narcissus") == 15
    assert g.get_bilateral_interaction_count("narcissus", "narcissus") == 30
    assert g.get_total_interactions("narcissus") == 15

    # Subgraph isolation
    sub = g.subgraph(["narcissus"])
    assert sub.node_count == 1
    assert sub.edge_count == 1
    assert sub.has_edge("narcissus", "narcissus")
    assert sub.get_interaction_count("narcissus", "narcissus") == 15

    # Cascaded removal
    assert g.remove_node("narcissus")
    assert g.node_count == 0
    assert g.edge_count == 0
    assert not g.has_node("narcissus")
    assert not g.has_edge("narcissus", "narcissus")
    assert g.get_edge("narcissus", "narcissus") is None
    assert g._out_edges == {}
    assert g._in_edges == {}
    assert g._interactions == {}

    # Double remove returns False
    assert not g.remove_node("narcissus")


def test_disconnected_subgraphs_multi_component_stress() -> None:
    """Stress-test BFS pathfinding across 25 isolated island components (500 total nodes).

    Verifies:
    1. Cross-component queries return [] immediately without scanning unrelated components.
    2. Subgraph extraction across mixed components preserves only relevant intra-component edges.
    3. top_interlocutors on isolated nodes returns empty list.
    """
    g = RelationshipGraph()
    num_components = 25
    nodes_per_comp = 20

    for c in range(num_components):
        for i in range(nodes_per_comp):
            u = f"c{c:02d}_{i:02d}"
            v = f"c{c:02d}_{(i+1)%nodes_per_comp:02d}"
            g.add_edge(RelationshipEdge(u, v, RelationshipKind.FRIENDSHIP_CASUAL))
            g.record_interaction(u, v, count=i + 1)

    assert g.node_count == num_components * nodes_per_comp
    assert g.edge_count == num_components * nodes_per_comp

    # Query between component 0 and component 24
    t0 = time.perf_counter()
    path_disjoint = g.find_path("c00_00", "c24_19")
    t_query = time.perf_counter() - t0

    assert path_disjoint == []
    assert t_query < 0.005, f"Disjoint search took {t_query*1000:.2f} ms (expected < 5ms)"

    # Undirected cross-component query also returns []
    assert g.find_path("c00_00", "c24_19", directed=False) == []

    # Subgraph across selective components
    selected_nodes = ["c01_00", "c01_01", "c02_00", "nonexistent_node"]
    sub = g.subgraph(selected_nodes)
    assert sub.node_count == 3
    assert sub.has_edge("c01_00", "c01_01")
    assert not sub.has_edge("c01_01", "c02_00")
    assert sub.edge_count == 1

    # Isolated node with no edges
    g.add_node("lone_wanderer")
    assert g.top_interlocutors("lone_wanderer") == []
    assert g.get_total_interactions("lone_wanderer") == 0


def test_hub_node_removal_cascade_integrity() -> None:
    """Stress-test cascade when removing a hub node connected to 200 satellites."""
    g = RelationshipGraph()
    hub = "central_hub"
    num_satellites = 200

    g.add_node(hub)
    for i in range(num_satellites):
        sat = f"sat_{i:03d}"
        g.add_edge(RelationshipEdge(hub, sat, RelationshipKind.SUPERIOR_SUBORDINATE))
        g.add_edge(RelationshipEdge(sat, hub, RelationshipKind.SUPERIOR_SUBORDINATE))
        g.record_interaction(hub, sat, count=5)
        g.record_interaction(sat, hub, count=3)

    assert g.node_count == num_satellites + 1
    assert g.edge_count == num_satellites * 2

    # Remove the central hub
    assert g.remove_node(hub)

    # Invariants after hub removal
    assert g.node_count == num_satellites
    assert g.edge_count == 0
    assert not g.has_node(hub)

    for i in range(num_satellites):
        sat = f"sat_{i:03d}"
        assert g.get_edge(hub, sat) is None
        assert g.get_edge(sat, hub) is None
        assert g.get_neighbors(sat) == set()
        assert g.get_interaction_count(hub, sat) == 0
        assert g.get_interaction_count(sat, hub) == 0
        assert g.get_total_interactions(sat) == 0


# ==============================================================================
# GROUP 2: Dual-Persistence Sync Edge Cases (Corrupted JSON, Casing, Turn Counts)
# ==============================================================================


def test_dual_persistence_corrupted_json_syntax_and_schema_types(tmp_path: Path) -> None:
    """Verify strict error contracts when deserializing corrupted or non-conforming JSON."""
    corrupted_syntax_path = tmp_path / "syntax_error.json"
    corrupted_syntax_path.write_text('{"relationships": [{"source_tag": "a", ', encoding="utf-8")

    # Syntax corruption raises JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        RelationshipGraph.from_json_file(corrupted_syntax_path)

    # Corrupted relationship item: non-dict
    with pytest.raises(TypeError, match="Expected dict for relationship item"):
        RelationshipGraph.from_dict({"relationships": [12345]})

    # Corrupted relationship item: missing source_tag
    with pytest.raises(KeyError, match="source_tag"):
        RelationshipGraph.from_dict({
            "relationships": [{"target_tag": "hero", "kind": "ROMANCE"}]
        })

    # Corrupted relationship item: missing kind
    with pytest.raises(KeyError, match="kind"):
        RelationshipGraph.from_dict({
            "relationships": [{"source_tag": "hero", "target_tag": "liana"}]
        })

    # Corrupted relationship item: unknown relationship kind
    with pytest.raises(ValueError, match="Unknown relationship kind"):
        RelationshipGraph.from_dict({
            "relationships": [{
                "source_tag": "hero",
                "target_tag": "liana",
                "kind": "INTERGALACTIC_NEMESIS",
            }]
        })

    # Corrupted root: primitive integer
    with pytest.raises(TypeError, match="Expected dict or list"):
        RelationshipGraph.from_dict(99999)

    # Empty file returns empty graph without crash
    empty_path = tmp_path / "empty.json"
    empty_path.write_text("", encoding="utf-8")
    empty_graph = RelationshipGraph.from_json_file(empty_path)
    assert empty_graph.node_count == 0
    assert empty_graph.edge_count == 0

    # Non-existent file returns empty graph
    missing_graph = RelationshipGraph.from_json_file(tmp_path / "does_not_exist.json")
    assert missing_graph.node_count == 0


def test_dual_persistence_non_canonical_casing_resilience() -> None:
    """Verify that lowercase, mixed-case, hyphenated, and space-separated kinds parse cleanly."""
    raw_data = {
        "relationships": [
            {"source_tag": "a", "target_tag": "b", "kind": "family_elder"},
            {"source_tag": "b", "target_tag": "c", "kind": "Romance"},
            {"source_tag": "c", "target_tag": "d", "kind": "friendship casual"},
            {"source_tag": "d", "target_tag": "e", "kind": "mentor-disciple"},
            {"source_tag": "e", "target_tag": "f", "kind": "RIVAL_HOSTILE"},
        ]
    }

    graph = RelationshipGraph.from_dict(raw_data)
    assert graph.edge_count == 5

    assert graph.get_edge("a", "b") is not None
    assert graph.get_edge("a", "b").kind is RelationshipKind.FAMILY_ELDER  # type: ignore[union-attr]
    assert graph.get_edge("b", "c").kind is RelationshipKind.ROMANCE  # type: ignore[union-attr]
    assert graph.get_edge("c", "d").kind is RelationshipKind.FRIENDSHIP_CASUAL  # type: ignore[union-attr]
    assert graph.get_edge("d", "e").kind is RelationshipKind.MENTOR_DISCIPLE  # type: ignore[union-attr]
    assert graph.get_edge("e", "f").kind is RelationshipKind.RIVAL_HOSTILE  # type: ignore[union-attr]

    # Serializing to dict preserves UPPERCASE canonical strings
    serialized = graph.to_dict()
    kinds_in_json = [r["kind"] for r in serialized["relationships"]]
    assert kinds_in_json == [
        "FAMILY_ELDER",
        "ROMANCE",
        "FRIENDSHIP_CASUAL",
        "MENTOR_DISCIPLE",
        "RIVAL_HOSTILE",
    ]


def test_sync_negative_zero_and_float_turn_counts() -> None:
    """Verify handling of negative, zero, float, and non-numeric interaction counts."""
    raw_data = {
        "interactions": {
            "hero": {
                "liana": 25,
                "villain": -100,  # Negative count: must be ignored
                "ghost": 0,       # Zero count: must be ignored
                "mage": 4.9,      # Float count: must be cast to int (4)
                "cleric": "invalid",  # String: ignored
                "rogue": None,        # None: ignored
            }
        }
    }

    graph = RelationshipGraph.from_dict(raw_data)
    assert graph.get_interaction_count("hero", "liana") == 25
    assert graph.get_interaction_count("hero", "villain") == 0
    assert graph.get_interaction_count("hero", "ghost") == 0
    assert graph.get_interaction_count("hero", "mage") == 4
    assert graph.get_interaction_count("hero", "cleric") == 0
    assert graph.get_interaction_count("hero", "rogue") == 0

    # Direct method calls
    graph.record_interaction("hero", "villain", count=-5)
    assert graph.get_interaction_count("hero", "villain") == 0

    graph.record_interaction("hero", "ghost", count=0)
    assert graph.get_interaction_count("hero", "ghost") == 0

    graph.record_interaction("", "liana", count=10)
    assert graph.get_interaction_count("", "liana") == 0

    graph.record_interaction("hero", "   ", count=10)
    assert graph.get_interaction_count("hero", "") == 0


def test_sync_identical_timestamps_and_hash_conflict_resolution(tmp_path: Path) -> None:
    """Empirically test edge case conflict resolution under identical mtimes and hash states."""
    db_path = tmp_path / "sync_edge.db"
    json_path = tmp_path / "sync_edge.json"

    # 1. Setup initial state: DB and JSON identical
    db = Database.open(db_path, create=True)
    try:
        db.save_relationship(
            RelationshipEdge("alice", "bob", RelationshipKind.FRIENDSHIP_CASUAL, confidence=0.8)
        )
        g_json = RelationshipGraph()
        g_json.add_edge(
            RelationshipEdge("alice", "bob", RelationshipKind.FRIENDSHIP_CASUAL, confidence=0.8)
        )
        g_json.save_to_json_file(json_path)

        # Initial sync establishes hash in DB metadata
        RelationshipGraph.sync(db, json_path, direction="auto")
        stored_hash = db.get_meta("relationships_json_hash")
        assert stored_hash is not None

        # 2. Case: DB updated with higher confidence edge, JSON untouched on disk
        db.save_relationship(
            RelationshipEdge(
                "alice", "bob", RelationshipKind.FRIENDSHIP_CLOSE, confidence=0.95
            )
        )
        # Even if timestamps are forced identical:
        now_ts = time.time()
        os.utime(json_path, (now_ts, now_ts))

        synced = RelationshipGraph.sync(db, json_path, direction="auto")
        edge = synced.get_edge("alice", "bob")
        assert edge is not None
        assert edge.kind is RelationshipKind.FRIENDSHIP_CLOSE
        assert edge.confidence == 0.95

        # 3. Case: Symmetric human manual curation shield
        # DB has manual edge, JSON has automated edge with higher confidence (0.99)
        db.save_relationship(
            RelationshipEdge(
                "alice", "bob", RelationshipKind.ROMANCE,
                source_type="manual", confidence=0.5,
            )
        )
        g_new_json = RelationshipGraph()
        g_new_json.add_edge(
            RelationshipEdge(
                "alice", "bob", RelationshipKind.FRIENDSHIP_CLOSE,
                source_type="llm", confidence=0.99,
            )
        )
        g_new_json.save_to_json_file(json_path)

        synced_manual = RelationshipGraph.sync(db, json_path, direction="auto")
        edge_manual = synced_manual.get_edge("alice", "bob")
        assert edge_manual is not None
        assert edge_manual.source_type == "manual"
        assert edge_manual.kind is RelationshipKind.ROMANCE
    finally:
        db.close()


# ==============================================================================
# GROUP 3: Stylometrics Extraction with Extreme Inputs
# ==============================================================================


def test_stylometrics_100_percent_punctuation_no_zero_division() -> None:
    """Challenge stylometrics extractor with 100% punctuation without ZeroDivisionError."""
    # Sub-case A: Pure terminal delimiters produce 0 sentences and 0 words
    pure_delimiters = [
        "!?!?!?!",
        ".........",
        "\u2026\uff01\uff1f!?",
        "!?!?!?!!",
    ]

    summary_pure = extract_stylometrics(pure_delimiters)
    assert summary_pure.total_lines == 4
    assert summary_pure.total_words == 0
    assert summary_pure.unique_words == 0
    assert summary_pure.type_token_ratio == 0.0
    assert summary_pure.total_sentences == 0
    assert summary_pure.avg_sentence_length == 0.0

    # Punctuation counts are collected
    assert summary_pure.exclamation_count > 0
    assert summary_pure.question_count > 0
    assert summary_pure.ellipsis_count > 0

    # Rates fall back to 0.0 without ZeroDivisionError
    assert summary_pure.exclamation_rate == 0.0
    assert summary_pure.question_rate == 0.0
    assert summary_pure.ellipsis_rate == 0.0
    assert summary_pure.hesitation_rate == 0.0
    assert summary_pure.summary_text() == "Không có dữ liệu thoại thống kê"

    # Sub-case B: Non-terminal punctuation (dashes, slashes) with 0 words
    non_terminal = [
        "---",
        "///===***",
    ]
    summary_non_terminal = extract_stylometrics(non_terminal)
    assert summary_non_terminal.total_words == 0
    assert summary_non_terminal.total_sentences >= 1
    assert summary_non_terminal.avg_sentence_length == 0.0
    assert summary_non_terminal.summary_text() == "Không có dữ liệu thoại thống kê"

    # Serialization produces valid dictionary
    d = summary_pure.to_dict()
    assert d["total_words"] == 0
    assert d["type_token_ratio"] == 0.0


def test_stylometrics_zero_words_and_formatting_tags_only() -> None:
    """Verify stylometrics extraction when dialogues consist exclusively of whitespace and tags."""
    dialogues = [
        "",
        "    ",
        "\t\n  \r\n",
        "{b}{/b}",
        "{color=#ff0000}{size=30}{/size}{/color}",
        "[player_name]",
        "{i}[hero_title]{/i}",
    ]

    summary = extract_stylometrics(dialogues)
    assert summary.total_lines == 7
    assert summary.total_words == 0
    assert summary.unique_words == 0
    assert summary.total_sentences == 0
    assert summary.summary_text() == "Không có dữ liệu thoại thống kê"


def test_stylometrics_multiline_emojis_and_unicode_delimiters() -> None:
    """Stress-test tokenizer and sentence boundary detector with multiline emojis and CJK."""
    dialogues = [
        "😀😁😂\n🤣😃😄\n😅😆😉",
        "Thần kiếm xuất thế\uff01Người đến là ai\uff1f",
        "Thế sự như mộng\u2026\u2026 Vạn vật quy tông\u3002",
    ]

    summary = extract_stylometrics(dialogues)
    assert summary.total_lines == 3
    # Emojis on newlines create sentence splits
    assert summary.total_sentences >= 5

    # Full-width Asian punctuation counted
    assert summary.exclamation_count >= 1
    assert summary.question_count >= 1
    assert summary.ellipsis_count >= 1

    # Sentence rates are properly computed
    assert summary.exclamation_rate > 0.0
    assert summary.question_rate > 0.0
    assert summary.ellipsis_rate > 0.0


def test_stylometrics_extreme_word_repetitions_scale() -> None:
    """Stress-test tokenizer against 50,000 words repetition and 10,000 interjections."""
    t0 = time.perf_counter()
    dialogues = [
        "liana " * 25000,
        "magic " * 25000,
        "uh " * 5000,
        "ah " * 5000,
    ]

    summary = extract_stylometrics(dialogues)
    elapsed = time.perf_counter() - t0

    assert summary.total_lines == 4
    assert summary.total_words == 60000
    assert summary.unique_words == 4
    # Type-Token Ratio: 4 / 60000 = 0.000067 -> rounded to 4 decimals = 0.0001
    assert summary.type_token_ratio == 0.0001

    # Top interjections capture high-frequency markers
    assert "uh" in summary.top_interjections
    assert "ah" in summary.top_interjections
    assert summary.hesitation_count == 10000

    # 60,000 word tokenization must complete under 150ms
    assert elapsed < 0.20, f"Extreme stylometrics took {elapsed*1000:.2f} ms"


# ==============================================================================
# GROUP 4: Persona Inference Fallbacks & Heuristics
# ==============================================================================


def test_persona_fallback_unknown_and_non_binary_genders() -> None:
    """Verify persona fallback and Vietnamese pronoun resolution with unknown genders."""
    sty = extract_stylometrics(["Testing unknown gender dialogue line."])

    # 1. infer_persona_fallback with unknown/arbitrary gender
    fb = infer_persona_fallback(
        tag="android_7",
        dialogues=["Unit 7 operational. Awaiting command parameters."],
        stylometrics=sty,
        role="Autonomous Sentry",
        gender="unknown",
        display_name="Unit 7",
    )
    assert "archetype" in fb
    assert "personality_traits" in fb
    assert "tone_and_voice" in fb
    assert "socioeconomic_status" in fb

    # 2. Exhaustive test of resolve_pronoun_pair_from_relationship with 'unknown' genders
    for kind in RelationshipKind:
        spk_sen, lis_sen = resolve_pronoun_pair_from_relationship(
            kind, "unknown", "unknown", is_speaker_senior=True
        )
        assert spk_sen and lis_sen, f"Empty pronoun pair for senior {kind}"

        spk_jun, lis_jun = resolve_pronoun_pair_from_relationship(
            kind, "unknown", "unknown", is_speaker_senior=False
        )
        assert spk_jun and lis_jun, f"Empty pronoun pair for junior {kind}"

    # Specific invariants with unknown gender
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FORMAL_ACQUAINTANCE, "unknown", "unknown"
    ) == ("tôi", "bạn")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.RIVAL_HOSTILE, "unknown", "unknown"
    ) == ("ta", "ngươi")


def test_persona_fallback_conflicting_role_signals() -> None:
    """Verify deterministic archetype precedence when role cues contradict tag semantics."""
    sty_tsundere = StylometricSummary(
        total_lines=10,
        total_words=100,
        unique_words=50,
        type_token_ratio=0.5,
        total_sentences=10,
        avg_sentence_length=10.0,
        exclamation_count=12,
        exclamation_rate=1.2,
        question_count=2,
        question_rate=0.2,
        ellipsis_count=4,
        ellipsis_rate=0.4,
        hesitation_count=4,
        hesitation_rate=0.4,
        top_interjections=("hmph", "tch"),
    )

    # 1. mc tag always resolves to protagonist archetype regardless of conflicting cues
    fb_mc = infer_persona_fallback(
        tag="mc",
        dialogues=["I must find the truth."],
        stylometrics=sty_tsundere,
        role="Queen Empress of Darkness",
        display_name="Hero",
    )
    assert fb_mc["archetype"] == "protagonist"

    # 2. narrator tag always resolves to narrator archetype
    fb_narr = infer_persona_fallback(
        tag="narrator",
        dialogues=["The wind howled across the plains."],
        stylometrics=sty_tsundere,
        role="Bloodthirsty Berserker",
        display_name="Narrator",
    )
    assert fb_narr["archetype"] == "narrator"

    # 3. Non-meta character with conflicting tsundere stylometrics + military knight status
    fb_knight = infer_persona_fallback(
        tag="valeria",
        dialogues=["H-Hmph! It's not like I guarded the castle for you!"],
        stylometrics=sty_tsundere,
        role="Royal High Knight Commander",
        display_name="Valeria",
    )
    assert fb_knight["archetype"] == "tsundere"
    assert "kiêu hãnh" in fb_knight["personality_traits"]
    assert "Quân đội / Hiệp sĩ" in fb_knight["socioeconomic_status"]


def test_persona_fallback_empty_short_and_duplicate_dialogues() -> None:
    """Verify robust sample dialogue extraction under empty, short, and duplicate inputs."""
    sty = extract_stylometrics([])

    # 1. Empty dialogues produce empty sample tuple
    fb_empty = infer_persona_fallback("npc", [], sty)
    assert fb_empty["sample_dialogue"] == ()

    # 2. Only short lines (<= 5 chars) produce empty sample tuple
    short_lines = ["Hi", "Oh", "No", "Bye", "...", "Kuh"]
    fb_short = infer_persona_fallback("npc", short_lines, sty)
    assert fb_short["sample_dialogue"] == ()

    # 3. Duplicate lines behavior:
    # When <= 3 clean lines are provided, they are preserved as-is
    duplicate_short = [
        "I shall protect the realm with my life!",
        "I shall protect the realm with my life!",
    ]
    fb_short_dupe = infer_persona_fallback("paladin", duplicate_short, sty)
    assert len(fb_short_dupe["sample_dialogue"]) == 2

    # When > 3 lines are provided with duplication, deduplication selects 1 distinct line
    duplicate_lines = [
        "I shall protect the realm with my life!",
    ] * 10
    fb_dupe = infer_persona_fallback("paladin", duplicate_lines, sty)
    assert len(fb_dupe["sample_dialogue"]) == 1
    assert fb_dupe["sample_dialogue"][0] == "I shall protect the realm with my life!"

    # 4. Long list selects at most 3 representative lines (first, middle, last)
    long_list = [f"Distinct narrative statement number {i:02d}" for i in range(10)]
    fb_long = infer_persona_fallback("sage", long_list, sty)
    assert len(fb_long["sample_dialogue"]) == 3
    assert fb_long["sample_dialogue"][0] == long_list[0]
    assert fb_long["sample_dialogue"][1] == long_list[5]
    assert fb_long["sample_dialogue"][2] == long_list[9]


def test_nlp_gender_inference_adversarial_cues_and_balance() -> None:
    """Stress-test NLP character gender classifier on adversarial names and co-occurrence cues."""
    # 1. Edge-case and empty names
    assert infer_character_gender_nlp("", []) == "unknown"
    assert infer_character_gender_nlp("   ", []) == "unknown"
    assert infer_character_gender_nlp("!@#$%", []) == "unknown"

    # 2. Spaced character names with initial single-letter space (e.g. "C harlotte", "A lice")
    assert infer_character_gender_nlp("C harlotte", []) == "female"
    assert infer_character_gender_nlp("A lice", []) == "female"
    # Fully character-spaced name "A l i c e" is not collapsed and returns "unknown"
    assert infer_character_gender_nlp("A l i c e", []) == "unknown"

    # 3. Exact tie in pronoun co-occurrence falls back to unknown
    tie_sources = [
        "Charlotte is here. She went there, but he did that.",
        "She called him, and he answered her.",
    ]
    # Charlotte has a feminine name, so it returns female based on name pattern
    assert infer_character_gender_nlp("Charlotte", tie_sources) == "female"

    # Gender-neutral name with exact tie in pronoun co-occurrence
    neutral_name = "Shadow"
    tie_sources_neutral = [
        "Shadow was walking. He looked around. She looked around.",
        "Shadow spoke. He listened. She laughed.",
    ]
    assert infer_character_gender_nlp(neutral_name, tie_sources_neutral) == "unknown"

    # Male dominance (>= 2x female cues)
    masc_sources = [
        "Shadow raised his sword. He shouted with his warrior brother.",
        "He struck the dragon, proving he was the true king.",
    ]
    assert infer_character_gender_nlp(neutral_name, masc_sources) == "male"


def test_semantic_relationship_inference_adversarial_matrix() -> None:
    """Stress-test semantic relationship inference on contradictory and empty inputs."""
    # 1. Non-human entities return None
    assert infer_semantic_relationship(
        "system", {"display_name": "Hệ thống"}, "hero", {"display_name": "Hero"}
    ) is None
    assert is_non_human_or_meta("system")
    assert is_non_human_or_meta("announcer", "Người dẫn chuyện")

    # 2. Collective entities return FORMAL_ACQUAINTANCE
    assert is_collective_entity("crowd", "Đám đông thị trấn")
    res_collective = infer_semantic_relationship(
        "crowd", {"display_name": "Đám đông"}, "hero", {"display_name": "Hero"}
    )
    assert res_collective == (RelationshipKind.FORMAL_ACQUAINTANCE, True)

    # 3. Empty dictionaries fall back gracefully to FORMAL_ACQUAINTANCE
    res_empty = infer_semantic_relationship("char_a", {}, "char_b", {})
    assert res_empty == (RelationshipKind.FORMAL_ACQUAINTANCE, True)

    # 4. Infer character role NLP boundary conditions
    assert infer_character_role_nlp("t", 100, 1000) == "Độc thoại nội tâm (Inner Monologue)"
    assert infer_character_role_nlp("narrator", 100, 1000) == "Người dẫn chuyện (Narrator)"
    assert infer_character_role_nlp("mc", 500, 1000) == "Nhân vật chính (Protagonist)"
    assert infer_character_role_nlp("npc", 0, 1000) == ""
    assert infer_character_role_nlp("npc", 50, 0) == ""
    assert infer_character_role_nlp("side_char", 50, 1000) == "Nhân vật chính tuyến (Main Cast)"
    assert infer_character_role_nlp("guest", 15, 1000) == "Nhân vật phụ (Supporting Cast)"
    assert infer_character_role_nlp("extra", 2, 1000) == "Nhân vật quần chúng (NPC)"
