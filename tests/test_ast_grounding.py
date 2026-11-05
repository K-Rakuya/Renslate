"""Comprehensive unit and integration test suite for Milestone 2: Ren'Py Structural AST Grounding Engine.

Covers:
1. Unified AST Grounding Parser (.rpy text, .rpy files, .rpyc bytecode, and Project).
2. Scene Boundary Chunking delimited strictly by Scene/scene bg and label transitions.
3. Hybrid Presence Set: ActiveSprites | SceneSpeakers.
4. 3-Way Utterance Classification: SpokenDialogue, InternalThought, AmbientNarration.
5. Deterministic O(1) Dyad Turn Resolution.
6. Real-world verification against CommercialVN-Alpha and CommercialVN-Beta.
7. StageTracker AST node processing and rpyc index integration.
8. InterlocutorResolver grounded dyad integration.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from renslate.core.models import Status, Unit, UnitKind
from renslate.engine.context.resolver import InterlocutorResolver, resolve_dyad
from renslate.renpy.grounding import (
    GroundingUnit,
    UtteranceKind,
    classify_utterance,
    parse_grounded_script,
)
from renslate.renpy.rpyc import (
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    Hide,
    Label,
    Say,
    Scene,
    Show,
    create_rpyc,
)
from renslate.renpy.stage import StageTracker, build_stage_index

# ==============================================================================
# 1. 3-Way Utterance Classification Tests
# ==============================================================================


class TestUtteranceClassification:
    def test_spoken_dialogue_standard(self) -> None:
        kind, spk = classify_utterance("alex", "Good morning, everyone!")
        assert kind is UtteranceKind.SPOKEN
        assert spk == "alex"

    def test_thought_via_parentheses(self) -> None:
        kind, spk = classify_utterance(None, "(I hope she doesn't think I'm weird...)")
        assert kind is UtteranceKind.THOUGHT
        assert spk is None

    def test_thought_via_italic_parentheses(self) -> None:
        kind, spk = classify_utterance("mc", "{i}(This guy is trouble...){/i}")
        assert kind is UtteranceKind.THOUGHT
        assert spk == "mc"

    def test_thought_via_fullwidth_unicode_parentheses(self) -> None:
        kind, spk = classify_utterance(None, "\uff08Tôi hy vọng cô ấy không để ý...\uff09")
        assert kind is UtteranceKind.THOUGHT
        assert spk is None

        kind2, spk2 = classify_utterance("alex", "\uff08Chúng ta cần cẩn thận hơn.\uff09")
        assert kind2 is UtteranceKind.THOUGHT
        assert spk2 == "alex"

    def test_thought_via_italic_fullwidth_parentheses(self) -> None:
        kind, spk = classify_utterance("mc", "{i}\uff08Kẻ địch đã tiến vào phạm vi...\uff09{/i}")
        assert kind is UtteranceKind.THOUGHT
        assert spk == "mc"

    def test_thought_via_japanese_brackets(self) -> None:
        kind, spk = classify_utterance("hero", "『What could that noise have been?』")
        assert kind is UtteranceKind.THOUGHT
        assert spk == "hero"

    def test_thought_via_thought_tags(self) -> None:
        for tag in ("t", "th", "thought", "thinking"):
            kind, spk = classify_utterance(tag, "This should be the right place.")
            assert kind is UtteranceKind.THOUGHT
            assert spk == tag

    def test_thought_via_CommercialVN-Beta_size_reduction(self) -> None:
        kind, spk = classify_utterance(None, "{size=-6}Nice to meet you! I'm Mayu!")
        assert kind is UtteranceKind.THOUGHT
        assert spk is None

    def test_ambient_narration_unattributed(self) -> None:
        kind, spk = classify_utterance(None, "The rain tapped rhythmically against the tall window.")
        assert kind is UtteranceKind.NARRATION
        assert spk is None

    def test_ambient_narration_via_narrator_tag(self) -> None:
        for tag in ("narrator", "narration", "none", "system"):
            kind, spk = classify_utterance(tag, "Some moments later, you arrived at the Astral Station.")
            assert kind is UtteranceKind.NARRATION
            assert spk is None


# ==============================================================================
# 2. Deterministic O(1) Dyad Turn Resolution Tests
# ==============================================================================


class TestDeterministicDyadResolution:
    def test_dyad_exact_two_speakers(self) -> None:
        speakers = frozenset({"alex", "aria"})
        assert resolve_dyad("alex", speakers) == "aria"
        assert resolve_dyad("aria", speakers) == "alex"

    def test_dyad_speaker_not_in_set(self) -> None:
        speakers = frozenset({"alex", "aria"})
        assert resolve_dyad("stranger", speakers) is None

    def test_dyad_non_dialogue_filtered(self) -> None:
        speakers = frozenset({"alex", "aria", "narrator", "system"})
        assert resolve_dyad("alex", speakers) == "aria"

    def test_dyad_single_speaker_returns_none(self) -> None:
        assert resolve_dyad("alex", frozenset({"alex"})) is None

    def test_dyad_triad_returns_none(self) -> None:
        assert resolve_dyad("alex", frozenset({"alex", "aria", "boris"})) is None


# ==============================================================================
# 3. Scene Boundary Chunking & Script Parsing (.rpy Text)
# ==============================================================================


class TestSceneBoundaryChunking:
    def test_scene_boundary_delimit_discourse_chunks(self) -> None:
        script = """
scene bg classroom with dissolve
alex "Good morning!"
scene bg park with fade
aria "Nice breeze today."
"""
        chunks = parse_grounded_script(script)
        assert len(chunks) == 2
        assert chunks[0].scene_spec == "bg classroom"
        assert chunks[0].dialogue_count == 1
        assert chunks[0].units[0].text == "Good morning!"

        assert chunks[1].scene_spec == "bg park"
        assert chunks[1].dialogue_count == 1
        assert chunks[1].units[0].text == "Nice breeze today."

    def test_empty_scene_zero_dialogues(self) -> None:
        script = """
scene bg dark_void
scene bg bright_morning
"""
        chunks = parse_grounded_script(script)
        assert len(chunks) == 2
        assert chunks[0].scene_spec == "bg dark_void"
        assert chunks[0].dialogue_count == 0
        assert chunks[0].units == ()
        assert chunks[1].scene_spec == "bg bright_morning"
        assert chunks[1].dialogue_count == 0
        assert chunks[1].units == ()

    def test_label_transition_boundary(self) -> None:
        script = """
label chapter_1:
    scene bg house
    alex "Welcome home."

label chapter_2:
    aria "A new day begins."
"""
        chunks = parse_grounded_script(script)
        assert len(chunks) == 2
        assert chunks[0].label == "chapter_1"
        assert chunks[0].scene_spec == "bg house"
        assert chunks[0].units[0].speaker == "alex"

        assert chunks[1].label == "chapter_2"
        assert chunks[1].units[0].speaker == "aria"

    def test_malformed_rpy_syntax_graceful_recovery(self) -> None:
        script = """
    # Indented comment

scene     bg   spaced   with   Dissolve(0.5)
alex    "  Spaced dialogue  "
"""
        chunks = parse_grounded_script(script)
        assert len(chunks) == 1
        assert chunks[0].scene_spec == "bg spaced"
        assert chunks[0].units[0].text == "  Spaced dialogue  "


# ==============================================================================
# 4. Hybrid Presence Tracking Tests
# ==============================================================================


class TestHybridPresenceTracking:
    def test_2d_sprite_visibility_show_hide(self) -> None:
        script = """
scene bg courtyard
show alex normal at left
alex "Where is everyone?"
show aria happy at right
aria "I'm right here!"
hide alex
aria "Alex? Where did he go?"
"""
        chunks = parse_grounded_script(script)
        assert len(chunks) == 1
        units = chunks[0].units
        assert len(units) == 3

        # Unit 1: alex on stage
        assert units[0].active_presence == frozenset({"alex"})
        assert units[0].target is None  # Only 1 on stage -> soliloquy

        # Unit 2: both on stage -> dyad resolution
        assert units[1].active_presence == frozenset({"alex", "aria"})
        assert units[1].target == "alex"

        # Unit 3: alex hidden, aria on stage
        assert units[2].active_presence == frozenset({"aria"})
        assert units[2].target is None

    def test_3d_camera_cut_fallback_to_conversational_speakers(self) -> None:
        """Verify 3D camera cut VN pattern (e.g. CommercialVN-Alpha) where no sprites are shown."""
        script = """
scene bg_3d_courtyard_sunset
hero "The castle looks deserted."
valeria "It won't stay quiet for long."
"""
        chunks = parse_grounded_script(script)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.visual_sprites == ()
        assert chunk.conversational_speakers == frozenset({"hero", "valeria"})

        # Turn 1: hero addresses valeria
        assert chunk.units[0].speaker == "hero"
        assert chunk.units[0].target == "valeria"
        assert chunk.units[0].active_presence == frozenset({"hero", "valeria"})

        # Turn 2: valeria addresses hero
        assert chunk.units[1].speaker == "valeria"
        assert chunk.units[1].target == "hero"
        assert chunk.units[1].active_presence == frozenset({"hero", "valeria"})

    def test_rapid_sprite_transitions_same_turn(self) -> None:
        script = """
scene bg hall
show a
show b
hide a
show c
hide b
show d
d "I arrived last!"
"""
        chunks = parse_grounded_script(script)
        assert len(chunks) == 1
        unit = chunks[0].units[0]
        assert unit.active_presence == frozenset({"c", "d"})
        assert unit.target == "c"

    def test_triad_crowd_scene_non_dyad(self) -> None:
        script = """
scene bg plaza
show a
show b
show c
a "Attention everyone!"
"""
        chunks = parse_grounded_script(script)
        unit = chunks[0].units[0]
        assert len(unit.active_presence) == 3
        # In a triad, target cannot be resolved via simple dyad logic
        assert unit.target is None


# ==============================================================================
# 5. Compiled .rpyc Bytecode AST Grounding Tests
# ==============================================================================


class TestRpycBytecodeGrounding:
    def test_synthetic_rpyc_ast_grounding(self, tmp_path: Path) -> None:
        """Create a synthetic .rpyc file with AST nodes and parse grounded chunks."""
        lbl = Label()
        lbl.name = "synthetic_chapter"

        sc = Scene()
        sc.imspec = (("bg", "tavern"), None, None, None)

        sh1 = Show()
        sh1.imspec = (("barman", "neutral"), None, None, None)

        sh2 = Show()
        sh2.imspec = (("rogue", "smirk"), None, None, None)

        sy1 = Say()
        sy1.who = "barman"
        sy1.what = "What can I get you?"

        sy2 = Say()
        sy2.who = "rogue"
        sy2.what = "Something strong."

        sc2 = Scene()
        sc2.imspec = (("bg", "street"), None, None, None)

        sy3 = Say()
        sy3.who = None
        sy3.what = "Outside, the rain began to pour."

        lbl.block = [sc, sh1, sh2, sy1, sy2, sc2, sy3]

        ast_payload = pickle.dumps([lbl], protocol=2)
        rpyc_data = create_rpyc({
            SLOT_STATEMENTS_PRE: ast_payload,
            SLOT_STATEMENTS_POST: ast_payload,
        })
        rpyc_file = tmp_path / "game" / "tavern.rpyc"
        rpyc_file.parent.mkdir(parents=True, exist_ok=True)
        rpyc_file.write_bytes(rpyc_data)

        chunks = parse_grounded_script(rpyc_file)
        assert len(chunks) == 2

        # Chunk 1: Tavern scene
        c1 = chunks[0]
        assert c1.scene_spec == "bg tavern"
        assert c1.dialogue_count == 2
        assert c1.conversational_speakers == frozenset({"barman", "rogue"})
        assert c1.units[0].target == "rogue"
        assert c1.units[1].target == "barman"

        # Chunk 2: Street scene (narration)
        c2 = chunks[1]
        assert c2.scene_spec == "bg street"
        assert c2.dialogue_count == 1
        assert c2.units[0].kind is UtteranceKind.NARRATION
        assert c2.units[0].speaker is None
        assert c2.units[0].target is None

    def test_stage_tracker_process_node(self) -> None:
        tracker = StageTracker()

        sc = Scene()
        sc.imspec = (("bg", "castle"), None, None, None)
        assert tracker.process_node(sc)
        assert tracker.scene_bg == "bg castle"
        assert len(tracker.active_sprites) == 0

        sh = Show()
        sh.imspec = (("knight", "brave"), None, None, None)
        assert tracker.process_node(sh)
        assert "knight" in tracker.active_sprites
        assert tracker.active_sprites["knight"].expression == "brave"

        hd = Hide()
        hd.imspec = (("knight",), None, None, None)
        assert tracker.process_node(hd)
        assert "knight" not in tracker.active_sprites

    def test_build_stage_index_rpyc_fallback(self, tmp_path: Path) -> None:
        """build_stage_index extracts snapshots from .rpyc bytecode when no .rpy exists."""
        lbl = Label()
        lbl.name = "intro"
        sc = Scene()
        sc.imspec = (("bg", "throne"), None, None, None)
        sh = Show()
        sh.imspec = (("king", "regal"), None, None, None)
        sy = Say()
        sy.who = "king"
        sy.what = "Kneel, hero."
        sy.filename = "game/throne.rpy"
        sy.linenumber = 15
        lbl.block = [sc, sh, sy]

        payload = pickle.dumps([lbl], protocol=2)
        rpyc_data = create_rpyc({SLOT_STATEMENTS_PRE: payload, SLOT_STATEMENTS_POST: payload})
        game_dir = tmp_path / "game"
        game_dir.mkdir(parents=True)
        (game_dir / "throne.rpyc").write_bytes(rpyc_data)

        # Indexing game directory without any .rpy files
        stage_idx = build_stage_index(game_dir)
        assert "throne.rpy:15" in stage_idx or "game/throne.rpy:15" in stage_idx
        snap = stage_idx.get("throne.rpy:15") or stage_idx.get("game/throne.rpy:15")
        assert snap is not None
        assert snap.scene_bg == "bg throne"
        assert "king" in snap.active_tags


# ==============================================================================
# 6. InterlocutorResolver Integration Tests
# ==============================================================================


class TestInterlocutorResolverGroundedIntegration:
    def test_resolver_with_conversational_speakers(self) -> None:
        resolver = InterlocutorResolver()
        unit = Unit(
            uid="d:1",
            kind=UnitKind.DIALOGUE,
            seq=1,
            source="Are you ready?",
            status=Status.PENDING,
            speaker="alex",
        )
        res = resolver.resolve(unit, conversational_speakers={"alex", "aria"})
        assert res.listener == "aria"
        assert res.resolution_method == "dyad"

    def test_resolver_with_grounded_unit_active_presence(self) -> None:
        resolver = InterlocutorResolver()
        unit = GroundingUnit(
            uid="d:2",
            seq=2,
            speaker="aria",
            text="Right behind you.",
            kind=UtteranceKind.SPOKEN,
            active_presence=frozenset({"alex", "aria"}),
        )
        res = resolver.resolve(unit)
        assert res.listener == "alex"
        assert res.resolution_method == "dyad"

    def test_resolver_with_triad_active_presence_falls_back(self) -> None:
        resolver = InterlocutorResolver()
        unit = GroundingUnit(
            uid="d:3",
            seq=3,
            speaker="alex",
            text="Boris, take the vanguard.",
            kind=UtteranceKind.SPOKEN,
            active_presence=frozenset({"alex", "aria", "boris"}),
        )
        res = resolver.resolve(unit)
        # Resolves via vocative since active presence is a triad
        assert res.listener == "boris"
        assert res.resolution_method == "vocative"

    def test_resolver_multi_turn_grounding_units_dyad_sequence(self) -> None:
        """Verify seamless sequential resolution of >= 3 consecutive GroundingUnits without AttributeError."""
        resolver = InterlocutorResolver()
        presence = frozenset({"alex", "aria"})
        u1 = GroundingUnit(
            uid="d:1",
            seq=1,
            speaker="alex",
            text="Are you ready?",
            kind=UtteranceKind.SPOKEN,
            active_presence=presence,
        )
        u2 = GroundingUnit(
            uid="d:2",
            seq=2,
            speaker="aria",
            text="Right behind you.",
            kind=UtteranceKind.SPOKEN,
            active_presence=presence,
        )
        u3 = GroundingUnit(
            uid="d:3",
            seq=3,
            speaker="alex",
            text="Keep your eyes on the ridge.",
            kind=UtteranceKind.SPOKEN,
            active_presence=presence,
        )
        u4 = GroundingUnit(
            uid="d:4",
            seq=4,
            speaker="aria",
            text="Understood.",
            kind=UtteranceKind.SPOKEN,
            active_presence=presence,
        )

        r1 = resolver.resolve(u1)
        assert r1.speaker == "alex"
        assert r1.listener == "aria"
        assert r1.resolution_method == "dyad"

        r2 = resolver.resolve(u2)
        assert r2.speaker == "aria"
        assert r2.listener == "alex"
        assert r2.resolution_method == "dyad"

        r3 = resolver.resolve(u3)
        assert r3.speaker == "alex"
        assert r3.listener == "aria"
        assert r3.resolution_method == "dyad"

        r4 = resolver.resolve(u4)
        assert r4.speaker == "aria"
        assert r4.listener == "alex"
        assert r4.resolution_method == "dyad"

    def test_resolver_multi_turn_grounding_units_turn_taking_and_question_adjacency(self) -> None:
        """Verify turn-taking adjacency and question adjacency fallbacks with GroundingUnit objects."""
        resolver = InterlocutorResolver()
        u1 = GroundingUnit(
            uid="d:10",
            seq=10,
            speaker="alex",
            text="Did you find the artifacts?",
            kind=UtteranceKind.SPOKEN,
        )
        u2 = GroundingUnit(
            uid="d:11",
            seq=11,
            speaker="aria",
            text="Yes, both of them.",
            kind=UtteranceKind.SPOKEN,
        )
        u3 = GroundingUnit(
            uid="d:12",
            seq=12,
            speaker="alex",
            text="Excellent news.",
            kind=UtteranceKind.SPOKEN,
        )

        r1 = resolver.resolve(u1)
        assert r1.speaker == "alex"
        assert r1.listener is None

        # Turn 2: Question adjacency or turn-taking attributing to alex
        r2 = resolver.resolve(u2)
        assert r2.speaker == "aria"
        assert r2.listener == "alex"
        assert r2.resolution_method in {"question_adjacency", "turn_taking"}

        # Turn 3: Turn-taking adjacency attributing to aria
        r3 = resolver.resolve(u3)
        assert r3.speaker == "alex"
        assert r3.listener == "aria"
        assert r3.resolution_method == "turn_taking"

    def test_resolver_multi_turn_grounding_units_across_scene_boundary(self) -> None:
        """Verify sequence transition across scene boundaries resets state cleanly without error."""
        resolver = InterlocutorResolver()
        scene1_presence = frozenset({"alex", "aria"})
        u1 = GroundingUnit(
            uid="d:1",
            seq=1,
            speaker="alex",
            text="Farewell, Aria.",
            kind=UtteranceKind.SPOKEN,
            active_presence=scene1_presence,
        )
        u2 = GroundingUnit(
            uid="d:2",
            seq=2,
            speaker="aria",
            text="Safe travels, Alex.",
            kind=UtteranceKind.SPOKEN,
            active_presence=scene1_presence,
        )
        # Scene transition with sequence jump > 30
        scene2_presence = frozenset({"boris", "alex"})
        u3 = GroundingUnit(
            uid="d:40",
            seq=40,
            speaker="boris",
            text="Welcome to the northern outpost.",
            kind=UtteranceKind.SPOKEN,
            active_presence=scene2_presence,
        )
        u4 = GroundingUnit(
            uid="d:41",
            seq=41,
            speaker="alex",
            text="Good to be here.",
            kind=UtteranceKind.SPOKEN,
            active_presence=scene2_presence,
        )

        r1 = resolver.resolve(u1)
        assert r1.listener == "aria"
        r2 = resolver.resolve(u2)
        assert r2.listener == "alex"

        # Across scene boundary: u3 resolves dyad with boris and alex
        r3 = resolver.resolve(u3)
        assert r3.speaker == "boris"
        assert r3.listener == "alex"

        r4 = resolver.resolve(u4)
        assert r4.speaker == "alex"
        assert r4.listener == "boris"


# ==============================================================================
# 7. Real-World Benchmark Verification
# ==============================================================================


class TestRealWorldBenchmarkGrounding:
    def test_CommercialVN-Alpha_real_world_scene_grounding(self) -> None:
        """Verify AST grounding against real-world CommercialVN-Alpha."""
        cr_path = Path("/path/to/commercial_vn/")
        if not cr_path.exists():
            pytest.skip("CommercialVN-Alpha not found on disk")

        chunks = parse_grounded_script(cr_path)

        assert len(chunks) >= 50, f"Expected >= 50 chunks, found {len(chunks)}"

        # Verify dyadic scene properties in 3D VN
        dyad_chunks = [
            c for c in chunks
            if len(c.conversational_speakers) == 2 and c.dialogue_count >= 2
        ]
        assert len(dyad_chunks) >= 10, f"Expected >= 10 dyad scenes, found {len(dyad_chunks)}"

        # Verify deterministic dyad target attribution
        sample_dyad = dyad_chunks[0]
        spk_list = sorted(sample_dyad.conversational_speakers)
        for u in sample_dyad.units:
            if u.kind is UtteranceKind.SPOKEN and u.speaker in sample_dyad.conversational_speakers:
                expected_target = spk_list[1] if u.speaker == spk_list[0] else spk_list[0]
                assert u.target == expected_target

    def test_CommercialVN-Beta_real_world_rpyc_grounding(self) -> None:
        """Verify AST grounding against real-world CommercialVN-Beta (.rpyc distribution)."""
        gw_path = Path("/path/to/commercial_vn/'_Whim-0.4.7-pc")
        if not gw_path.exists():
            pytest.skip("CommercialVN-Beta not found on disk")

        mayu_rpyc = gw_path / "game" / "mayu_events.rpyc"
        if not mayu_rpyc.exists():
            pytest.skip("mayu_events.rpyc not found")

        chunks = parse_grounded_script(mayu_rpyc)
        assert len(chunks) >= 50, f"Expected >= 50 chunks in mayu_events.rpyc, found {len(chunks)}"

        # Check thought tags classified as THOUGHT (e.g. who == 't')
        thought_units = [
            u for c in chunks for u in c.units
            if u.kind is UtteranceKind.THOUGHT
        ]
        assert len(thought_units) >= 10, "Expected thought units to be disambiguated"

        # Check dyad resolution in 2-speaker scenes
        dyad_chunks = [
            c for c in chunks
            if len(c.conversational_speakers) == 2 and c.dialogue_count >= 2
        ]
        assert len(dyad_chunks) >= 10, "Expected dyad chunks in mayu_events"
        sample_dyad = dyad_chunks[0]
        spk_list = sorted(sample_dyad.conversational_speakers)
        for u in sample_dyad.units:
            if u.kind is UtteranceKind.SPOKEN and u.speaker in sample_dyad.conversational_speakers:
                expected_target = spk_list[1] if u.speaker == spk_list[0] else spk_list[0]
                assert u.target == expected_target
