"""Comprehensive tests for Milestone 4: 2-Pass Hierarchical Bootstrap Engine.

Covers:
- Pass 1 AST Topology Scanning & Co-occurrence Tensor
- Dynamic Protagonist Discovery (centrality, turn ratio, dynamic names, zero hardcoding)
- Multi-Factor Salience Window Isolation
- Pass 2 Structured LLM Distillation with 4D Relational Vectors
- Algebraic Reciprocal Pronoun Projection and SQLite Persistence
- End-to-end Integration in run_analysis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from renslate.core.models import (
    CharacterProfile,
    KinshipKind,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.analyze import (
    SalienceWindow,
    apply_distillation_results,
    build_topology_analysis,
    compute_cooccurrence_matrix,
    detect_dynamic_name_patterns,
    discover_topological_protagonist,
    isolate_dyad_salience_windows,
    isolate_salience_windows,
    parse_distillation_json,
    run_analysis,
    run_pass2_distillation,
)
from renslate.engine.graph import RelationshipGraph
from renslate.renpy.grounding import (
    GroundingUnit,
    UtteranceKind,
    parse_grounded_script,
)


class FakeLLM:
    """Mock LLM returning deterministic completion output for testing."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def complete_json(
        self, system: str, user: str, *args: Any, **kwargs: Any
    ) -> tuple[Any, int, int]:
        self.system_prompts.append(system)
        self.user_prompts.append(user)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response, 100, 50


# ==============================================================================
# 1. Pass 1 AST Topology & Co-Occurrence Tensor
# ==============================================================================


class TestPass1Topology:
    """Tests for AST topological scanning and network co-occurrence."""

    def test_cooccurrence_matrix_symmetric_tensor(self) -> None:
        script = """
scene bg living_room
alex "Good morning, Bob."
bob "Morning, Alex!"
scene bg kitchen
bob "Clara, have you seen the tea?"
clara "Here it is, Bob."
"""
        chunks = parse_grounded_script(script)
        speakers, matrix = compute_cooccurrence_matrix(chunks)

        assert set(speakers) == {"alex", "bob", "clara"}
        # Self-cooccurrence is zero
        assert matrix["alex"]["alex"] == 0
        assert matrix["bob"]["bob"] == 0
        assert matrix["clara"]["clara"] == 0
        # Co-occurrence is symmetric and dialogue-weighted
        assert matrix["alex"]["bob"] == 2
        assert matrix["bob"]["alex"] == 2
        assert matrix["bob"]["clara"] == 2
        assert matrix["clara"]["bob"] == 2
        assert matrix["alex"]["clara"] == 0
        assert matrix["clara"]["alex"] == 0

    def test_cooccurrence_matrix_single_character(self) -> None:
        script = 'scene bg room\nsolo "Just me, completely alone."'
        chunks = parse_grounded_script(script)
        speakers, matrix = compute_cooccurrence_matrix(chunks)

        assert speakers == ["solo"]
        assert matrix["solo"]["solo"] == 0

    def test_cooccurrence_matrix_empty_script(self) -> None:
        speakers, matrix = compute_cooccurrence_matrix([])
        assert speakers == []
        assert matrix == {}

    def test_dynamic_name_pattern_detection(self) -> None:
        script = """
define p = Character("[player_name]")
define d = DynamicCharacter("pov_name")
define hero = Character(None)
define np = Character("Static NPC")
"""
        detected = detect_dynamic_name_patterns(script)
        assert "p" in detected
        assert "d" in detected
        assert "hero" in detected
        assert "np" not in detected

    def test_discover_topological_protagonist_degree_centrality(self) -> None:
        # Central hub interacts with multiple characters
        matrix = {
            "node_a": {"node_a": 0, "hub": 5, "node_c": 0},
            "hub": {"node_a": 5, "hub": 0, "node_c": 10},
            "node_c": {"node_a": 0, "hub": 10, "node_c": 0},
        }
        protagonist = discover_topological_protagonist(matrix)
        assert protagonist == "hub"

    def test_discover_topological_protagonist_turn_ratio_and_dynamic_name(self) -> None:
        # Player has fewer turns than mentor, but has dynamic name pattern
        matrix = {
            "player": {"player": 0, "mentor": 4},
            "mentor": {"player": 4, "mentor": 0},
        }
        turns = {"player": 10, "mentor": 20}
        dynamic_tags = {"player"}

        protagonist = discover_topological_protagonist(
            matrix,
            speaker_turns=turns,
            dynamic_name_tags=dynamic_tags,
            alpha=0.3,
            beta=0.3,
            gamma=0.4,
        )
        assert protagonist == "player"

    def test_discover_topological_protagonist_disconnected_graph(self) -> None:
        matrix = {
            "iso_1": {"iso_1": 0, "iso_2": 0},
            "iso_2": {"iso_1": 0, "iso_2": 0},
        }
        protagonist = discover_topological_protagonist(matrix)
        assert protagonist in ("iso_1", "iso_2")

    def test_zero_hardcoding_arbitrary_alien_names(self) -> None:
        script = """
scene bg cosmos
xenon_9 "Greetings, organic entity."
unit_7 "Acknowledged, synthetic prime."
scene bg chamber
xenon_9 "Commence energy transfer."
archon_x "Transfer authorized."
"""
        analysis = build_topology_analysis(script)
        assert set(analysis.speakers) == {"archon_x", "unit_7", "xenon_9"}
        assert analysis.protagonist_tag == "xenon_9"
        assert analysis.total_turns == 4
        assert analysis.turn_matrix["xenon_9"]["unit_7"] >= 1


# ==============================================================================
# 2. Multi-Factor Salience Window Isolation
# ==============================================================================


class TestPass1SalienceWindows:
    """Tests for multi-factor dialogue window salience scoring."""

    def test_salience_window_scoring_vocatives_and_emotions(self) -> None:
        units = [
            GroundingUnit("u1", 1, "k1", "Quiet night.", UtteranceKind.SPOKEN),
            GroundingUnit("u2", 2, "k1", "The stars are distant.", UtteranceKind.SPOKEN),
            GroundingUnit("u3", 3, "k1", "Hey Val! Look at the anomaly?!", UtteranceKind.SPOKEN),
            GroundingUnit("u4", 4, "val", "Captain! Shields offline!!", UtteranceKind.SPOKEN),
        ]
        windows = isolate_salience_windows(units, window_size=2, max_windows=3)
        assert len(windows) > 0
        top_window = windows[0]
        # Tuple unpacking contract: score, units = window
        score, window_units = top_window
        assert score >= 8.0
        assert any(u.uid in ("u3", "u4") for u in window_units)

    def test_salience_window_first_meeting_bonus(self) -> None:
        units = [
            GroundingUnit("u1", 1, "a", "Welcome to the academy.", UtteranceKind.SPOKEN),
            GroundingUnit("u2", 2, "b", "Thank you, senior.", UtteranceKind.SPOKEN),
        ]
        # Without first meeting
        w_normal = isolate_salience_windows(units, window_size=2)
        # With first meeting at turn 1
        w_first = isolate_salience_windows(units, window_size=2, first_meeting_turn=1)

        assert len(w_first) > 0 and len(w_normal) > 0
        assert w_first[0].score > w_normal[0].score
        assert w_first[0].first_meeting is True

    def test_salience_window_budget_cap(self) -> None:
        units = [
            GroundingUnit(f"u{i}", i, f"spk_{i%2}", f"Dialogue {i}!", UtteranceKind.SPOKEN)
            for i in range(50)
        ]
        windows = isolate_salience_windows(units, window_size=4, max_windows=3)
        assert len(windows) == 3

    def test_isolate_dyad_salience_windows(self) -> None:
        script = """
scene bg arena
show fighter_a at left
show fighter_b at right
fighter_a "Hey you! Prepare yourself!"
fighter_b "Hmph! You talk too much!!"
scene bg hall
show fighter_a at left
show mentor_c at right
mentor_c "Well fought, student."
fighter_a "Thank you, master."
"""
        chunks = parse_grounded_script(script)
        dyad_windows = isolate_dyad_salience_windows(chunks, max_windows_per_dyad=2)

        # Expected dyad pairs
        assert ("fighter_a", "fighter_b") in dyad_windows
        assert ("fighter_a", "mentor_c") in dyad_windows
        assert len(dyad_windows[("fighter_a", "fighter_b")]) >= 1
        assert dyad_windows[("fighter_a", "fighter_b")][0].score >= 6.0


# ==============================================================================
# 3. Pass 2 Structured LLM Distillation
# ==============================================================================


class TestPass2StructuredDistillation:
    """Tests for structured 4D vector extraction and algebraic reciprocal projection."""

    def test_parse_distillation_json_strict_schema(self) -> None:
        raw_json = json.dumps(
            {
                "characters": [
                    {
                        "tag": "hero",
                        "display_name_vi": "Anh Hùng",
                        "gender": "male",
                        "role": "Nhân vật chính",
                        "archetype": "protagonist",
                        "personality_traits": ["quả cảm", "trung hậu"],
                        "tone_and_voice": "Trầm ấm, quyết đoán",
                        "socioeconomic_status": "Hiệp sĩ tự do",
                        "speech_style": "Thẳng thắn",
                    }
                ],
                "relationships": [
                    {
                        "source": "hero",
                        "target": "aria",
                        "kinship": "ROMANTIC_COUPLE",
                        "delta_hierarchy": 0.1,
                        "affinity": 0.85,
                        "register": 0.3,
                        "confidence": 0.95,
                        "evidence": "Tình cảm lãng mạn gắn bó thân thiết.",
                    }
                ],
            }
        )
        result = parse_distillation_json(raw_json)
        assert len(result.characters) == 1
        assert len(result.relationships) == 1

        ch = result.characters[0]
        assert ch.tag == "hero"
        assert ch.gender == "male"
        assert ch.role == "Nhân vật chính"

        rel = result.relationships[0]
        assert rel.source == "hero"
        assert rel.target == "aria"
        assert rel.vector.kinship == KinshipKind.ROMANTIC_COUPLE
        assert rel.vector.delta_hierarchy == 0.1
        assert rel.vector.affinity == 0.85
        assert rel.vector.register == 0.3
        assert rel.confidence == 0.95

    def test_parse_distillation_json_markdown_fence_recovery(self) -> None:
        fenced = """```json
{
  "characters": [{"tag": "valkyrie", "gender": "female", "role": "Warrior"}],
  "relationships": []
}
```"""
        result = parse_distillation_json(fenced)
        assert len(result.characters) == 1
        assert result.characters[0].tag == "valkyrie"
        assert result.characters[0].gender == "female"

    def test_parse_distillation_json_preamble_and_postamble_resilience(self) -> None:
        raw_text = """Here is the extracted character and relationship JSON:
```json
{
  "characters": [{"tag": "valkyrie", "gender": "female", "role": "Warrior"}],
  "relationships": []
}
```
I hope this structured distillation is helpful for your translation!"""
        result = parse_distillation_json(raw_text)
        assert len(result.characters) == 1
        assert result.characters[0].tag == "valkyrie"
        assert result.characters[0].gender == "female"

    def test_parse_distillation_json_clamping_and_validation(self) -> None:
        malformed = {
            "characters": [{"tag": "test", "gender": "unknown_gender"}],
            "relationships": [
                {
                    "source": "a",
                    "target": "b",
                    "kinship": "INVALID_KINSHIP",
                    "delta_hierarchy": 5.0,  # Must clamp to 1.0
                    "affinity": -10.0,      # Must clamp to -1.0
                    "register": 2.5,        # Must clamp to 1.0
                    "confidence": 1.5,      # Must clamp to 1.0
                }
            ],
        }
        result = parse_distillation_json(malformed)
        assert result.characters[0].gender == "unknown"
        rel = result.relationships[0]
        assert rel.vector.kinship == KinshipKind.NONE
        assert rel.vector.delta_hierarchy == 1.0
        assert rel.vector.affinity == -1.0
        assert rel.vector.register == 1.0
        assert rel.confidence == 1.0

    def test_algebraic_projection_and_reciprocal_edge_creation(self, tmp_path: Path) -> None:
        root = tmp_path / "game_project"
        (root / "game").mkdir(parents=True)
        proj = Project.create(root)

        with proj.open_db() as db:
            db.save_character(CharacterProfile(tag="brother", display_name="Brother", gender="male"))
            db.save_character(CharacterProfile(tag="sister", display_name="Sister", gender="female"))

            # Sibling vector with hierarchy delta +0.5 (elder brother)
            raw_distill = {
                "characters": [],
                "relationships": [
                    {
                        "source": "brother",
                        "target": "sister",
                        "kinship": "SIBLING",
                        "delta_hierarchy": 0.5,
                        "affinity": 0.7,
                        "register": 0.3,
                        "confidence": 0.9,
                        "evidence": "Anh trai và em gái thân thiết.",
                    }
                ],
            }
            parsed = parse_distillation_json(raw_distill)
            _chars_updated, edges_written = apply_distillation_results(parsed, db, proj)

            assert edges_written == 2

            graph = RelationshipGraph.load_from_db(db)
            edge_fwd = graph.get_edge("brother", "sister")
            edge_rev = graph.get_edge("sister", "brother")

            assert edge_fwd is not None and edge_rev is not None
            # Reciprocal pronoun pair guarantee
            assert (edge_fwd.speaker_pronoun, edge_fwd.listener_pronoun) == ("anh", "em")
            assert (edge_rev.speaker_pronoun, edge_rev.listener_pronoun) == ("em", "anh")
            # Reciprocal vector inversion
            assert edge_fwd.vector is not None and edge_rev.vector is not None
            assert edge_rev.vector == edge_fwd.vector.inverted()

            # Check character profiles pronoun_map updated
            p_bro = db.get_character("brother")
            p_sis = db.get_character("sister")
            assert p_bro is not None and p_sis is not None
            assert p_bro.pronoun_map["sister"] == "anh - em"
            assert p_sis.pronoun_map["brother"] == "em - anh"

    def test_apply_distillation_results_canonical_tag_casing(self, tmp_path: Path) -> None:
        root = tmp_path / "game_project_casing"
        (root / "game").mkdir(parents=True)
        proj = Project.create(root)

        with proj.open_db() as db:
            db.save_character(CharacterProfile(tag="hero", display_name="Hero", gender="male"))
            db.save_character(CharacterProfile(tag="aria", display_name="Aria", gender="female"))

            # LLM output uses non-canonical casing: "HERO" and "Aria"
            raw_distill = {
                "characters": [
                    {"tag": "HERO", "display_name_vi": "Hero", "gender": "male", "role": "Protagonist"}
                ],
                "relationships": [
                    {
                        "source": "HERO",
                        "target": "Aria",
                        "kinship": "ROMANTIC_COUPLE",
                        "delta_hierarchy": 0.1,
                        "affinity": 0.85,
                        "register": 0.3,
                        "confidence": 0.95,
                        "evidence": "Tình yêu đôi lứa.",
                    }
                ],
            }
            parsed = parse_distillation_json(raw_distill)
            apply_distillation_results(parsed, db, proj)

            graph = RelationshipGraph.load_from_db(db)
            edge = graph.get_edge("hero", "aria")
            assert edge is not None
            assert edge.source_tag == "hero"
            assert edge.target_tag == "aria"

            p_hero = db.get_character("hero")
            assert p_hero is not None
            assert "aria" in p_hero.pronoun_map

    def test_run_pass2_distillation_with_fake_llm(self, tmp_path: Path) -> None:
        root = tmp_path / "game_project"
        (root / "game").mkdir(parents=True)
        proj = Project.create(root)

        fake_response = {
            "characters": [
                {
                    "tag": "jin",
                    "display_name_vi": "Jin",
                    "gender": "male",
                    "role": "Master",
                    "archetype": "elder_mentor",
                },
                {
                    "tag": "tomo",
                    "display_name_vi": "Tomo",
                    "gender": "male",
                    "role": "Apprentice",
                    "archetype": "inquisitive_apprentice",
                },
            ],
            "relationships": [
                {
                    "source": "jin",
                    "target": "tomo",
                    "kinship": "MENTOR_DISCIPLE",
                    "delta_hierarchy": 0.8,
                    "affinity": 0.6,
                    "register": 0.5,
                    "confidence": 0.95,
                    "evidence": "Sư đồ truyền thụ kiếm pháp.",
                }
            ],
        }
        fake_llm = FakeLLM(fake_response)

        with proj.open_db() as db:
            windows = [
                SalienceWindow(
                    score=10.0,
                    units=(
                        GroundingUnit("u1", 1, "jin", "Focus your breath, Tomo!", UtteranceKind.SPOKEN),
                        GroundingUnit("u2", 2, "tomo", "Yes, Master Jin!", UtteranceKind.SPOKEN),
                    ),
                    pair=("jin", "tomo"),
                )
            ]
            distilled = run_pass2_distillation(proj, db, fake_llm, windows, protagonist_tag="tomo")

            assert len(distilled.characters) == 2
            assert len(distilled.relationships) == 1

            graph = RelationshipGraph.load_from_db(db)
            edge = graph.get_edge("jin", "tomo")
            assert edge is not None
            assert edge.speaker_pronoun == "ta"
            assert edge.listener_pronoun == "con"


# ==============================================================================
# 4. End-to-End Analysis Pipeline with 2-Pass Bootstrap
# ==============================================================================


class TestHierarchicalBootstrapEndToEnd:
    """Tests verifying the full run_analysis integration."""

    def test_run_analysis_pass1_topology_discovery(self, tmp_path: Path) -> None:
        root = tmp_path / "game_e2e"
        game_dir = root / "game"
        game_dir.mkdir(parents=True)

        script_content = """
define p = Character("[player_name]")
define s = Character("Sophia")
define g = Character("Guard")

label start:
    scene bg town
    p "Hello Sophia! Nice day, isn't it?"
    s "Good morning! Are you heading to the castle?"
    p "Yes, the captain requested me."
    scene bg gates
    g "Halt! State your business!"
    p "I have an appointment."
    g "Pass through."
"""
        (game_dir / "script.rpy").write_text(script_content, encoding="utf-8")
        project = Project.create(root)

        with project.open_db() as db:
            # Seed units into DB corresponding to the script
            units = [
                Unit("u1", UnitKind.DIALOGUE, 1, "Hello Sophia! Nice day, isn't it?", Status.PENDING, speaker="p", label="start"),
                Unit("u2", UnitKind.DIALOGUE, 2, "Good morning! Are you heading to the castle?", Status.PENDING, speaker="s", label="start"),
                Unit("u3", UnitKind.DIALOGUE, 3, "Yes, the captain requested me.", Status.PENDING, speaker="p", label="start"),
                Unit("u4", UnitKind.DIALOGUE, 4, "Halt! State your business!", Status.PENDING, speaker="g", label="start"),
                Unit("u5", UnitKind.DIALOGUE, 5, "I have an appointment.", Status.PENDING, speaker="p", label="start"),
                Unit("u6", UnitKind.DIALOGUE, 6, "Pass through.", Status.PENDING, speaker="g", label="start"),
            ]
            db.save_units_bulk(units)

            stats = run_analysis(project, db)
            assert stats["characters"] >= 3

            chars = db.get_characters()
            assert "p" in chars
            # Protagonist detected via dynamic name [player_name] and topology
            assert chars["p"].role == "Nhân vật chính (Protagonist)"

    def test_run_analysis_pass2_enrich_llm_integration(self, tmp_path: Path) -> None:
        root = tmp_path / "game_pass2"
        game_dir = root / "game"
        game_dir.mkdir(parents=True)

        script_content = """
define hero = Character("Hero")
define aria = Character("Aria")

label start:
    scene bg room
    hero "Aria! Be careful!"
    aria "I'm with you, Hero!"
"""
        (game_dir / "script.rpy").write_text(script_content, encoding="utf-8")
        project = Project.create(root)

        fake_distill_response = {
            "characters": [
                {
                    "tag": "hero",
                    "display_name_vi": "Hero",
                    "gender": "male",
                    "role": "Nhân vật chính",
                    "archetype": "protagonist",
                },
                {
                    "tag": "aria",
                    "display_name_vi": "Aria",
                    "gender": "female",
                    "role": "Bạn đồng hành",
                    "archetype": "kuudere",
                },
            ],
            "relationships": [
                {
                    "source": "hero",
                    "target": "aria",
                    "kinship": "ROMANTIC_COUPLE",
                    "delta_hierarchy": 0.1,
                    "affinity": 0.8,
                    "register": 0.4,
                    "confidence": 0.9,
                    "evidence": "Cặp đôi tình cảm thân thiết.",
                }
            ],
        }
        fake_llm = FakeLLM(fake_distill_response)

        with project.open_db() as db:
            units = [
                Unit("u1", UnitKind.DIALOGUE, 1, "Aria! Be careful!", Status.PENDING, speaker="hero", label="start"),
                Unit("u2", UnitKind.DIALOGUE, 2, "I'm with you, Hero!", Status.PENDING, speaker="aria", label="start"),
            ]
            db.save_units_bulk(units)

            stats = run_analysis(project, db, enrich_llm=True, llm=fake_llm)
            assert stats["characters"] >= 2

            # Check that 4D relational vectors were saved to DB
            graph = RelationshipGraph.load_from_db(db)
            edge_hero = graph.get_edge("hero", "aria")
            edge_aria = graph.get_edge("aria", "hero")

            assert edge_hero is not None and edge_aria is not None
            assert edge_hero.speaker_pronoun == "anh"
            assert edge_hero.listener_pronoun == "em"
            assert edge_aria.speaker_pronoun == "em"
            assert edge_aria.listener_pronoun == "anh"
            assert edge_hero.vector is not None
            assert edge_hero.vector.kinship == KinshipKind.ROMANTIC_COUPLE

    def test_topology_analysis_ui_statement_and_screen_exclusion(self, tmp_path: Path) -> None:
        root = tmp_path / "game_ui"
        game_dir = root / "game"
        game_dir.mkdir(parents=True)

        screens_content = """
screen main_menu:
    add "gui/main_menu.png"
    textbutton "Start" action Start()
    add "gui/overlay.png"
"""
        (game_dir / "screens.rpy").write_text(screens_content, encoding="utf-8")

        script_content = """
define MC = Character("[player_name]")
define Alice = Character("Alice")

label start:
    scene bg room
    MC "Hello Alice!"
    Alice "Hi MC!"
    MC "Let's explore together."
    Alice "Sounds like a plan!"
"""
        (game_dir / "script.rpy").write_text(script_content, encoding="utf-8")

        analysis = build_topology_analysis(root)
        assert "add" not in analysis.speakers
        assert "textbutton" not in analysis.speakers
        assert "MC" in analysis.speakers
        assert "Alice" in analysis.speakers
        assert analysis.protagonist_tag == "MC"

    def test_grounding_autoload_rpyc_fallback(self, tmp_path: Path) -> None:
        import pickle

        from renslate.renpy.rpyc import (
            SLOT_STATEMENTS_POST,
            SLOT_STATEMENTS_PRE,
            Label,
            Say,
            Scene,
            create_rpyc,
        )

        root = tmp_path / "game_autoload"
        game_dir = root / "game"
        game_dir.mkdir(parents=True)

        # Autoload hook in .rpy contains no dialogue
        (game_dir / "00_renslate_autoload.rpy").write_text(
            "# Autoload hook\ninit python:\n    pass\n", encoding="utf-8"
        )

        lbl = Label()
        lbl.name = "start"

        sc = Scene()
        sc.imspec = (("bg", "room"), None, None, None)

        sy1 = Say()
        sy1.who = "hero"
        sy1.what = "A journey begins."
        sy1.filename = "game/script.rpy"
        sy1.linenumber = 10

        sy2 = Say()
        sy2.who = "companion"
        sy2.what = "I will follow you."
        sy2.filename = "game/script.rpy"
        sy2.linenumber = 11

        lbl.block = [sc, sy1, sy2]
        payload = pickle.dumps([lbl], protocol=2)
        rpyc_bytes = create_rpyc({SLOT_STATEMENTS_PRE: payload, SLOT_STATEMENTS_POST: payload})
        (game_dir / "script.rpyc").write_bytes(rpyc_bytes)

        chunks = parse_grounded_script(root)
        assert len(chunks) > 0
        dialogue_chunks = [c for c in chunks if c.dialogue_count > 0]
        assert len(dialogue_chunks) > 0

        analysis = build_topology_analysis(root)
        assert len(analysis.speakers) >= 2
        assert "hero" in analysis.speakers
        assert "companion" in analysis.speakers
        assert analysis.protagonist_tag is not None
