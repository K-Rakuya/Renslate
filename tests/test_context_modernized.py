"""Comprehensive unit tests for the modernized renslate.engine.context modular architecture."""

from __future__ import annotations

from dataclasses import dataclass

from renslate.core.config import BatchConfig, Config
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
    SceneMacroContext,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.context import (
    ContextBuilder,
    InterlocutorResolver,
    ResolvedInterlocutor,
    assemble_scene_macro_context,
    estimate_tokens,
    format_rich_dialogue_line,
    infer_scene_cues,
    update_scene_macro_from_flow,
)
from renslate.engine.context.builder import ContextBuilder as ModularContextBuilder
from renslate.engine.context.macro import (
    assemble_scene_macro_context as modular_assemble,
)
from renslate.engine.context.macro import (
    infer_scene_cues as modular_infer_scene_cues,
)
from renslate.engine.context.macro import (
    update_scene_macro_from_flow as modular_update_flow,
)
from renslate.engine.context.prompt_frame import (
    format_rich_dialogue_line as modular_format_rich,
)
from renslate.engine.context.resolver import (
    InterlocutorResolver as ModularResolver,
)
from renslate.engine.context.resolver import (
    ResolvedInterlocutor as ModularResolved,
)
from renslate.engine.context.tokens import (
    _CHARS_PER_TOKEN_CJK,
    _CHARS_PER_TOKEN_EN,
    _CHARS_PER_TOKEN_VI,
)
from renslate.engine.context.tokens import (
    estimate_tokens as modular_estimate_tokens,
)
from renslate.engine.graph import RelationshipGraph


@dataclass(slots=True)
class DummySceneFlowItem:
    kind: str
    name: str
    active_sprites: tuple[str, ...] = ()
    current_label: str = ""


def _make_unit(
    uid: str,
    seq: int,
    source: str,
    *,
    speaker: str | None = None,
    masked: str | None = None,
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        masked_source=masked or source,
        speaker=speaker,
        status=Status.PENDING,
    )


class TestContextSubpackageParity:
    """Verifies that the subpackage modules export identical signatures and parity."""

    def test_forwarding_shim_and_direct_module_identity(self) -> None:
        assert ModularResolver is InterlocutorResolver
        assert ModularResolved is ResolvedInterlocutor
        assert ModularContextBuilder is ContextBuilder
        assert modular_estimate_tokens is estimate_tokens
        assert modular_format_rich is format_rich_dialogue_line
        assert modular_infer_scene_cues is infer_scene_cues
        assert modular_assemble is assemble_scene_macro_context
        assert modular_update_flow is update_scene_macro_from_flow

    def test_token_constants(self) -> None:
        assert _CHARS_PER_TOKEN_CJK == 1.2
        assert _CHARS_PER_TOKEN_VI == 2.8
        assert _CHARS_PER_TOKEN_EN == 3.6


class TestInterlocutorResolverTurnTakingAndMemory:
    """Verifies turn-taking, vocatives, sliding window deque memory, and pronoun resolution."""

    def test_turn_taking_two_speakers(self) -> None:
        resolver = InterlocutorResolver()
        u1 = _make_unit("d:1", 1, "Good morning, Sylvie.", speaker="ren")
        res1 = resolver.resolve(u1, active_speakers=["ren", "sylvie"])
        assert res1.speaker == "ren"
        assert res1.listener == "sylvie"
        assert res1.resolution_method == "vocative"

        u2 = _make_unit("d:2", 2, "Good morning to you too!", speaker="sylvie")
        res2 = resolver.resolve(u2, prev_unit=u1, active_speakers=["ren", "sylvie"])
        assert res2.speaker == "sylvie"
        assert res2.listener == "ren"
        assert res2.resolution_method == "turn_taking"

    def test_narration_pause_memory_window(self) -> None:
        """Memory survives up to history_window (default 12) through narration pauses."""
        resolver = InterlocutorResolver(history_window=12)
        u1 = _make_unit("d:1", 1, "Can you hear me, Ren?", speaker="sylvie")
        resolver.resolve(u1, active_speakers=["ren", "sylvie", "third_person"])

        # 6 narration lines (speaker is None)
        prev = u1
        for i in range(2, 8):
            narr = _make_unit(f"d:{i}", i, f"Leaves rustle in the silent wind ({i}).", speaker=None)
            res_narr = resolver.resolve(narr, prev_unit=prev)
            assert res_narr.is_monologue is True
            prev = narr

        # Ren speaks at seq=8 (within history_window=12 from u1.seq=1)
        u8 = _make_unit("d:8", 8, "I hear you clearly, Sylvie.", speaker="ren")
        res8 = resolver.resolve(
            u8, prev_unit=prev, active_speakers=["ren", "sylvie", "third_person"]
        )
        assert res8.speaker == "ren"
        assert res8.listener == "sylvie"
        assert res8.resolution_method == "vocative"  # Detected vocative 'Sylvie'

        # Ren speaks again at seq=9 without vocative -> uses recent_turn to Sylvie
        u9 = _make_unit("d:9", 9, "Where should we head next?", speaker="ren")
        res9 = resolver.resolve(
            u9, prev_unit=u8, active_speakers=["ren", "sylvie", "third_person"]
        )
        assert res9.speaker == "ren"
        assert res9.listener == "sylvie"
        assert res9.resolution_method == "recent_turn"

    def test_narration_pause_exceeding_window(self) -> None:
        """Memory expires when narration gap exceeds history_window."""
        resolver = InterlocutorResolver(history_window=5)
        u1 = _make_unit("d:1", 1, "Sylvie, wait!", speaker="ren")
        resolver.resolve(u1, active_speakers=["ren", "sylvie", "npc1", "npc2"])

        prev = u1
        for i in range(2, 9):
            narr = _make_unit(
                f"d:{i}", i, f"A long walk across the mountains ({i}).", speaker=None
            )
            resolver.resolve(narr, prev_unit=prev)
            prev = narr

        # seq=9 (diff 8 > history_window 5) with >1 other speaker -> cannot assume recent turn
        u9 = _make_unit("d:9", 9, "Where is everyone?", speaker="ren")
        res9 = resolver.resolve(
            u9, prev_unit=prev, active_speakers=["ren", "sylvie", "npc1", "npc2"]
        )
        assert res9.resolution_method == "default"
        assert res9.listener is None

    def test_monologue_brackets(self) -> None:
        resolver = InterlocutorResolver()
        u_mono = _make_unit("d:1", 1, "(Should I really step inside?)", speaker="ren")
        res = resolver.resolve(u_mono)
        assert res.is_monologue is True
        assert res.listener is None

        u_mono_jp = _make_unit("d:2", 2, "『Có chuyện gì vậy nhỉ?』", speaker="ren")
        res_jp = resolver.resolve(u_mono_jp)
        assert res_jp.is_monologue is True
        assert res_jp.listener is None

    def test_dynamic_symmetric_and_asymmetric_pronoun_resolution(self) -> None:
        resolver = InterlocutorResolver()
        graph = RelationshipGraph()
        # Direct relationship ren -> liana
        graph.add_edge(
            RelationshipEdge(
                source_tag="ren",
                target_tag="liana",
                kind=RelationshipKind.ROMANCE,
                speaker_pronoun="anh",
                listener_pronoun="em",
            )
        )

        # 1. Direct query
        pair_direct = resolver.resolve_pronouns("ren", "liana", graph=graph)
        assert pair_direct == ("anh", "em")

        # 2. Symmetric reciprocal query (liana -> ren is not in graph, should invert)
        pair_reciprocal = resolver.resolve_pronouns("liana", "ren", graph=graph)
        assert pair_reciprocal == ("em", "anh")

        # 3. Fallback to CharacterProfile pronoun_map
        chars = {
            "master": CharacterProfile(
                "master", "Sư Phụ", pronoun_map={"disciple": "ta - con"}
            ),
            "disciple": CharacterProfile("disciple", "Đồ Nhi"),
        }
        # Direct from master profile
        p_master = resolver.resolve_pronouns("master", "disciple", char_map=chars)
        assert p_master == ("ta", "con")

        # Reciprocal from disciple querying master's profile
        p_disciple = resolver.resolve_pronouns("disciple", "master", char_map=chars)
        assert p_disciple == ("con", "ta")


class TestMacroContextAndSceneFlow:
    """Verifies scene cue inference and AST scene flow integration."""

    def test_infer_scene_cues_keywords(self) -> None:
        loc, atm = infer_scene_cues("scene_forest_night_storm")
        assert "Rừng" in loc
        assert "Đêm khuya" in atm or "Bão tố" in atm

        loc_room, atm_tense = infer_scene_cues("label_classroom_tense")
        assert "Lớp học" in loc_room
        assert "Căng thẳng" in atm_tense

    def test_assemble_scene_macro_context(self) -> None:
        macro = assemble_scene_macro_context(
            scene_id="sc_01",
            label="scene_garden_sunset",
            active_characters=["ren", "sylvie"],
            summary="Ren and Sylvie meet in the garden.",
            previous_summary="Sylvie was waiting alone.",
        )
        assert macro.scene_id == "sc_01"
        assert "Vườn hoa" in macro.location
        assert "Hoàng hôn" in macro.atmosphere
        assert macro.active_characters == ("ren", "sylvie")
        assert macro.metadata.get("previous_summary") == "Sylvie was waiting alone."

    def test_update_scene_macro_from_flow(self) -> None:
        base_macro = SceneMacroContext(scene_id="s1", label="start")
        flow_items = [
            DummySceneFlowItem(kind="scene", name="castle_hall"),
            DummySceneFlowItem(kind="show", name="king", active_sprites=("king", "guard")),
        ]
        updated = update_scene_macro_from_flow(base_macro, flow_items)
        assert "Lâu đài" in updated.location or "Đại sảnh" in updated.location
        assert "king" in updated.active_characters
        assert "guard" in updated.active_characters


class TestPromptFrameFormatting:
    """Verifies rich frame tags formatting for dialogues and monologues."""

    def test_format_rich_dialogue_line_monologue(self) -> None:
        u = _make_unit("d:101", 101, "(I have to find a way out...)", speaker="ren")
        r = ResolvedInterlocutor(speaker="ren", is_monologue=True, detected_emotion="hesitant")
        chars = {"ren": CharacterProfile("ren", "Ren")}
        formatted = format_rich_dialogue_line(u, r, chars)
        expected = (
            "d:101 | [Độc thoại/Nội tâm: Ren | Cảm xúc: hesitant] "
            "(I have to find a way out...)"
        )
        assert formatted == expected

    def test_format_rich_dialogue_line_dialogue_with_relationship(self) -> None:
        u = _make_unit("d:102", 102, "Sylvie, stay close.", speaker="ren")
        r = ResolvedInterlocutor(
            speaker="ren", listener="sylvie", is_monologue=False, detected_emotion="emphatic"
        )
        chars = {
            "ren": CharacterProfile("ren", "Ren"),
            "sylvie": CharacterProfile("sylvie", "Sylvie"),
        }
        graph = RelationshipGraph()
        graph.add_edge(
            RelationshipEdge("ren", "sylvie", RelationshipKind.ROMANCE, "anh", "em")
        )
        formatted = format_rich_dialogue_line(u, r, chars, graph=graph)
        expected = (
            "d:102 | [Người nói: Ren -> Người nghe: Sylvie | Quan hệ: ROMANCE "
            "(Ren xưng 'anh', gọi 'em') | Cảm xúc: emphatic] Sylvie, stay close."
        )
        assert formatted == expected


class TestContextBuilderAndKnapsackBudget:
    """Verifies ContextBuilder multi-tier prompt assembly and budget constraints."""

    def test_deterministic_stable_character_prefix(self) -> None:
        builder = ContextBuilder(Config())
        chars = {
            "zack": CharacterProfile("zack", "Zack"),
            "alice": CharacterProfile("alice", "Alice"),
            "bob": CharacterProfile("bob", "Bob"),
        }
        u = _make_unit("d:1", 1, "Hello.", speaker="alice")
        prompt = builder.build_user_prompt([u], characters=chars)
        # Profiles must be sorted alphabetically by tag/name: Alice -> Bob -> Zack
        idx_alice = prompt.index("Alice")
        idx_bob = prompt.index("Bob")
        idx_zack = prompt.index("Zack")
        assert idx_alice < idx_bob < idx_zack

    def test_resolved_listener_retained_under_budget_pressure(self) -> None:
        """FLAW-07 resolution: active speakers AND listeners retained under budget."""
        builder = ContextBuilder(Config(batch=BatchConfig(max_source_tokens=30)))
        # Create many extra character profiles to blow the budget
        chars = {
            f"extra_{i}": CharacterProfile(f"extra_{i}", f"ExtraChar_{i}" * 10) for i in range(25)
        }
        chars["hero"] = CharacterProfile("hero", "Hero", gender="male")
        chars["princess"] = CharacterProfile("princess", "Princess", gender="female")

        # hero is speaker, princess is listener
        u1 = _make_unit("d:1", 1, "Princess, we must depart.", speaker="hero")
        resolver = InterlocutorResolver()

        prompt = builder.build_user_prompt([u1], characters=chars, resolver=resolver)
        # Both Hero and Princess profiles must remain in prompt despite tight budget
        assert "Hero" in prompt
        assert "Princess" in prompt
