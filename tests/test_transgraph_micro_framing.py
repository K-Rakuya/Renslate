"""Comprehensive unit test suite for Milestone 5: TransGraph In-Context Micro-Framing.

Verifies:
1. Active persona injection (only batch speakers/listeners appear).
2. Complete omission of inactive characters from prompt frames.
3. Active reciprocal dyadic address invariants and algebraic vector projection.
4. >= 50% prompt token reduction on multi-character casts.
5. Shifting interlocutors across consecutive batches.
6. Single speaker, narrator, and soliloquy edge cases.
7. Integration within ContextBuilder.build_user_prompt and ContextBundle.
"""

from __future__ import annotations

from typing import Any

from renslate.core.config import Config
from renslate.core.models import (
    CharacterProfile,
    KinshipKind,
    RelationalVector,
    RelationshipEdge,
    RelationshipKind,
    SceneMacroContext,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.context import (
    ContextBuilder,
    ContextBundle,
    build_transgraph_frame,
    estimate_tokens,
)
from renslate.engine.graph import RelationshipGraph


def _make_unit(
    uid: str,
    seq: int,
    source: str,
    *,
    speaker: str | None = None,
    target: str | None = None,
    label: str | None = "scene_01",
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        masked_source=source,
        speaker=speaker,
        label=label,
        status=Status.PENDING,
    )


class DuckGroundingUnit:
    """Duck-typed grounding unit mimicking Ren'Py AST parser output."""

    def __init__(
        self,
        uid: str,
        seq: int,
        speaker: str | None,
        text: str,
        target: str | None = None,
        active_presence: frozenset[str] = frozenset(),
    ) -> None:
        self.uid = uid
        self.seq = seq
        self.speaker = speaker
        self.text = text
        self.source = text
        self.target = target
        self.active_presence = active_presence


class TestActivePersonaInjection:
    """Tests verifying active personas are extracted and injected into the TransGraph frame."""

    def test_active_speaker_and_explicit_listener_injected(self) -> None:
        batch = [
            DuckGroundingUnit("u1", 1, "alex", "Hello Aria.", target="aria"),
        ]
        chars = {
            "alex": CharacterProfile("alex", "Alex", gender="male", role="Hero"),
            "aria": CharacterProfile("aria", "Aria", gender="female", role="Heroine"),
        }
        frame = build_transgraph_frame(batch, chars)
        assert "- alex (male, Hero)" in frame
        assert "- aria (female, Heroine)" in frame
        assert "[ACTIVE INTERACTION SUBGRAPH]" in frame

    def test_vocative_addressed_listener_discovered_from_dialogue(self) -> None:
        # Standard Unit without explicit target, listener discovered via InterlocutorResolver
        batch = [
            _make_unit("u1", 1, "Aria, watch out behind you!", speaker="alex"),
        ]
        chars = {
            "alex": CharacterProfile("alex", "Alex", gender="male", role="Hero"),
            "aria": CharacterProfile("aria", "Aria", gender="female", role="Heroine"),
        }
        frame = build_transgraph_frame(batch, chars)
        assert "alex" in frame
        assert "aria" in frame

    def test_supports_both_character_profile_objects_and_dictionaries(self) -> None:
        batch = [
            DuckGroundingUnit("u1", 1, "c1", "Welcome traveler.", target="c2"),
        ]
        dict_profiles: dict[str, dict[str, Any]] = {
            "c1": {"gender": "male", "role": "Innkeeper"},
            "c2": {"gender": "female", "role": "Traveler"},
        }
        frame = build_transgraph_frame(batch, dict_profiles)  # type: ignore[arg-type]
        assert "- c1 (male, Innkeeper)" in frame
        assert "- c2 (female, Traveler)" in frame

    def test_case_insensitive_matching_and_display_name_canonicalization(self) -> None:
        batch = [
            DuckGroundingUnit("u1", 1, "ALEX", "Greetings Aria.", target="ARIA"),
        ]
        chars = {
            "alex": CharacterProfile("alex", "Alex", gender="male", role="Hero"),
            "aria": CharacterProfile("aria", "Aria", gender="female", role="Heroine"),
        }
        frame = build_transgraph_frame(batch, chars)
        # Should resolve to canonical keys "alex" and "aria" without duplicating
        assert "- alex (male, Hero)" in frame
        assert "- aria (female, Heroine)" in frame
        assert "alex -> aria: (tôi, bạn) | aria -> alex: (bạn, tôi)" in frame
        assert "ALEX" not in frame
        assert "ARIA" not in frame

    def test_special_characters_in_display_names_and_roles(self) -> None:
        batch = [
            DuckGroundingUnit("u1", 1, "leader", "Assemble troops!"),
        ]
        profiles = {
            "leader": {"gender": "male", "role": 'Commander "Iron Fist" <Supreme>'},
        }
        frame = build_transgraph_frame(batch, profiles)  # type: ignore[arg-type]
        assert 'Commander "Iron Fist" <Supreme>' in frame


class TestInactiveCharacterOmission:
    """Tests verifying complete exclusion of inactive characters from prompt frames."""

    def test_complete_omission_of_inactive_roster_in_large_universe(self) -> None:
        batch = [
            DuckGroundingUnit("u1", 1, "hero", "Follow me, Priestess.", target="priestess"),
        ]
        universe: dict[str, CharacterProfile] = {
            f"npc_{i:02d}": CharacterProfile(f"npc_{i:02d}", f"NPC {i}", role="Villager")
            for i in range(30)
        }
        universe["hero"] = CharacterProfile("hero", "Hero", gender="male", role="Protagonist")
        universe["priestess"] = CharacterProfile("priestess", "Priestess", gender="female", role="Healer")

        frame = build_transgraph_frame(batch, universe)
        assert "- hero (male, Protagonist)" in frame
        assert "- priestess (female, Healer)" in frame

        # All 30 inactive NPCs must be 100% absent from the frame
        for i in range(30):
            assert f"npc_{i:02d}" not in frame
            assert f"NPC {i}" not in frame

    def test_sublinear_frame_scaling_on_50_character_universe(self) -> None:
        universe = {f"c{i}": {"gender": "unknown", "role": f"Villager {i}"} for i in range(50)}
        universe["c0"] = {"gender": "male", "role": "Knight"}
        universe["c1"] = {"gender": "female", "role": "Mage"}

        batch = [DuckGroundingUnit("u1", 1, "c0", "Ready?", target="c1")]
        frame = build_transgraph_frame(batch, universe)  # type: ignore[arg-type]

        assert "c0" in frame and "c1" in frame
        assert "c49" not in frame
        # Active frame size should be bounded and minimal (O(active))
        lines = frame.splitlines()
        assert len(lines) <= 8

    def test_shifting_interlocutors_across_consecutive_batches(self) -> None:
        roster = {
            "ren": CharacterProfile("ren", "Ren", gender="male", role="Hero"),
            "liana": CharacterProfile("liana", "Liana", gender="female", role="Priestess"),
            "boris": CharacterProfile("boris", "Boris", gender="male", role="King"),
            "clara": CharacterProfile("clara", "Clara", gender="female", role="Knight"),
        }

        # Turn 1: Ren & Liana
        batch1 = [DuckGroundingUnit("u1", 1, "ren", "Liana, report.", target="liana")]
        frame1 = build_transgraph_frame(batch1, roster)
        assert "ren" in frame1 and "liana" in frame1
        assert "boris" not in frame1 and "clara" not in frame1

        # Turn 2: Boris & Clara
        batch2 = [DuckGroundingUnit("u2", 2, "boris", "Clara, defend the gates.", target="clara")]
        frame2 = build_transgraph_frame(batch2, roster)
        assert "boris" in frame2 and "clara" in frame2
        assert "ren" not in frame2 and "liana" not in frame2

        # Turn 3: Ren & Boris
        batch3 = [DuckGroundingUnit("u3", 3, "ren", "Your Majesty Boris.", target="boris")]
        frame3 = build_transgraph_frame(batch3, roster)
        assert "ren" in frame3 and "boris" in frame3
        assert "liana" not in frame3 and "clara" not in frame3


class TestDyadicReciprocalAddressInvariants:
    """Tests verifying closed reciprocal address rules and algebraic pronoun invariants."""

    def test_reciprocal_pronoun_pair_from_relationship_graph(self) -> None:
        graph = RelationshipGraph()
        graph.add_edge(
            RelationshipEdge(
                source_tag="hero",
                target_tag="heroine",
                kind=RelationshipKind.ROMANCE,
                speaker_pronoun="anh",
                listener_pronoun="em",
            )
        )
        batch = [DuckGroundingUnit("u1", 1, "hero", "I will protect you.", target="heroine")]
        chars = {
            "hero": CharacterProfile("hero", "Hero", gender="male"),
            "heroine": CharacterProfile("heroine", "Heroine", gender="female"),
        }

        frame = build_transgraph_frame(batch, chars, graph=graph)
        assert "hero -> heroine: (anh, em)" in frame
        assert "heroine -> hero: (em, anh)" in frame

    def test_algebraic_relational_vector_projection_invariance(self) -> None:
        graph = RelationshipGraph()
        # Sovereign-Subject relationship
        vec = RelationalVector(
            kinship=KinshipKind.SOVEREIGN_SUBJECT,
            delta_hierarchy=0.9,
            affinity=0.0,
            register=0.85,
        )
        graph.add_edge(
            RelationshipEdge(
                source_tag="king",
                target_tag="knight",
                kind=RelationshipKind.SUPERIOR_SUBORDINATE,
                vector=vec,
            )
        )
        batch = [DuckGroundingUnit("u1", 1, "king", "Hear my decree.", target="knight")]
        chars = {
            "king": CharacterProfile("king", "King", gender="male", role="Sovereign"),
            "knight": CharacterProfile("knight", "Knight", gender="male", role="Vassal"),
        }

        frame = build_transgraph_frame(batch, chars, graph=graph)
        assert "king -> knight: (trẫm, khanh)" in frame
        assert "knight -> king: (khanh, trẫm)" in frame

    def test_mentor_disciple_relational_vector_projection(self) -> None:
        graph = RelationshipGraph()
        vec = RelationalVector(
            kinship=KinshipKind.MENTOR_DISCIPLE,
            delta_hierarchy=0.6,
            affinity=0.5,
            register=0.8,
        )
        graph.add_edge(
            RelationshipEdge(
                source_tag="master",
                target_tag="disciple",
                kind=RelationshipKind.MENTOR_DISCIPLE,
                vector=vec,
            )
        )
        batch = [DuckGroundingUnit("u1", 1, "master", "Practice more.", target="disciple")]
        chars = {
            "master": CharacterProfile("master", "Master", gender="male", role="Teacher"),
            "disciple": CharacterProfile("disciple", "Disciple", gender="male", role="Student"),
        }

        frame = build_transgraph_frame(batch, chars, graph=graph)
        assert "master -> disciple: (thầy, trò)" in frame
        assert "disciple -> master: (trò, thầy)" in frame

    def test_reciprocal_inference_when_only_one_direction_defined_in_pronoun_map(self) -> None:
        # Only A defines pronoun_map to B; B does not define pronoun_map
        chars = {
            "elder": CharacterProfile("elder", "Elder", gender="male", pronoun_map={"junior": "anh - em"}),
            "junior": CharacterProfile("junior", "Junior", gender="male", pronoun_map={}),
        }
        batch = [DuckGroundingUnit("u1", 1, "elder", "Hey junior.", target="junior")]
        frame = build_transgraph_frame(batch, chars)
        assert "elder -> junior: (anh, em)" in frame
        assert "junior -> elder: (em, anh)" in frame

    def test_default_neutral_reciprocal_fallback_when_no_rules_exist(self) -> None:
        chars = {
            "stranger1": CharacterProfile("stranger1", "Stranger 1"),
            "stranger2": CharacterProfile("stranger2", "Stranger 2"),
        }
        batch = [DuckGroundingUnit("u1", 1, "stranger1", "Excuse me.", target="stranger2")]
        frame = build_transgraph_frame(batch, chars)
        assert "stranger1 -> stranger2: (tôi, bạn)" in frame
        assert "stranger2 -> stranger1: (bạn, tôi)" in frame

    def test_multi_party_triad_dyadic_closure(self) -> None:
        chars = {
            "a": CharacterProfile("a", "A", pronoun_map={"b": "tớ - cậu", "c": "anh - em"}),
            "b": CharacterProfile("b", "B", pronoun_map={"a": "cậu - tớ", "c": "anh - em"}),
            "c": CharacterProfile("c", "C", pronoun_map={"a": "em - anh", "b": "em - anh"}),
        }
        batch = [
            DuckGroundingUnit("u1", 1, "a", "Let's go together.", target="b"),
            DuckGroundingUnit("u2", 2, "b", "Wait for C.", target="c"),
        ]
        frame = build_transgraph_frame(batch, chars)
        # All three dyads (a-b, a-c, b-c) must be represented with reciprocal closure
        assert "a -> b: (tớ, cậu)" in frame and "b -> a: (cậu, tớ)" in frame
        assert "a -> c: (anh, em)" in frame and "c -> a: (em, anh)" in frame
        assert "b -> c: (anh, em)" in frame and "c -> b: (em, anh)" in frame


class TestEdgeCasesAndBoundaries:
    """Tests verifying edge cases: monologue, narration, empty batches, and scene transitions."""

    def test_ambient_narration_produces_clean_empty_subgraph(self) -> None:
        batch = [
            _make_unit("u1", 1, "The autumn leaves fell silently across the path.", speaker=None),
        ]
        frame = build_transgraph_frame(batch, {})
        assert "[ACTIVE SCENE: default]" in frame
        assert "Active Personas:\n- None (Ambient monologue/narration)" in frame
        assert "Active Pronoun Invariants:\n- No active interlocutor dyad" in frame

    def test_non_dialogue_speaker_tag_treated_as_narration(self) -> None:
        batch = [
            _make_unit("u1", 1, "System initializing...", speaker="system"),
            _make_unit("u2", 2, "Deep in thought...", speaker="thought"),
        ]
        frame = build_transgraph_frame(batch, {})
        assert "- None (Ambient monologue/narration)" in frame
        assert "- No active interlocutor dyad" in frame

    def test_single_speaker_soliloquy(self) -> None:
        batch = [
            _make_unit("u1", 1, "Where did everyone go?", speaker="hero"),
        ]
        chars = {"hero": CharacterProfile("hero", "Hero", gender="male", role="Wanderer")}
        frame = build_transgraph_frame(batch, chars)
        assert "- hero (male, Wanderer)" in frame
        assert "Active Pronoun Invariants:\n- No active interlocutor dyad" in frame

    def test_explicit_scene_spec_reflected_in_header(self) -> None:
        batch = [DuckGroundingUnit("u1", 1, "hero", "Inside the temple.")]
        frame = build_transgraph_frame(batch, {}, scene_spec="ancient_ruins_hall")
        assert "[ACTIVE SCENE: ancient_ruins_hall]" in frame

    def test_scene_macro_context_header_fallback(self) -> None:
        batch = [DuckGroundingUnit("u1", 1, "hero", "Dark here.")]
        macro = SceneMacroContext(scene_id="s_dungeon", label="dungeon_depths", location="Underground Cellar")
        frame = build_transgraph_frame(batch, {}, scene_macro=macro)
        assert "[ACTIVE SCENE: dungeon_depths]" in frame

    def test_empty_batch_units_handled_gracefully(self) -> None:
        frame = build_transgraph_frame([], {})
        assert "[ACTIVE INTERACTION SUBGRAPH]" in frame
        assert "- None (Ambient monologue/narration)" in frame
        assert "- No active interlocutor dyad" in frame


class TestTokenReductionMetric:
    """Tests verifying >= 50% prompt token reduction on multi-character games."""

    def test_at_least_50_percent_token_reduction_on_25_character_cast(self) -> None:
        large_universe: dict[str, CharacterProfile] = {
            f"char_{i:02d}": CharacterProfile(
                tag=f"char_{i:02d}",
                display_name=f"Villager {i}",
                gender="male" if i % 2 == 0 else "female",
                role=f"Village Merchant {i} with long elaborate backstory",
                speech_style="Humble and respectful",
                pronoun_map={f"char_{j:02d}": "tôi - bạn" for j in range(max(0, i - 3), min(25, i + 3))},
            )
            for i in range(25)
        }
        large_universe["alex"] = CharacterProfile(
            "alex", "Alex", gender="male", role="Hero", pronoun_map={"aria": "anh - em"}
        )
        large_universe["aria"] = CharacterProfile(
            "aria", "Aria", gender="female", role="Heroine", pronoun_map={"alex": "em - anh"}
        )

        batch = [DuckGroundingUnit("u1", 1, "alex", "Hey Aria.", target="aria")]

        # Global character dump representation
        global_dump_lines = []
        for p in large_universe.values():
            global_dump_lines.append(
                f"- {p.display_name} (tag: {p.tag}): {p.gender} | Vai trò: {p.role} | Giọng: {p.speech_style}"
            )
        global_dump_text = "HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:\n" + "\n".join(global_dump_lines)

        micro_frame_text = build_transgraph_frame(batch, large_universe)

        global_tokens = estimate_tokens(global_dump_text)
        micro_tokens = estimate_tokens(micro_frame_text)

        reduction = (len(global_dump_text) - len(micro_frame_text)) / len(global_dump_text)
        token_reduction = (global_tokens - micro_tokens) / global_tokens

        assert reduction >= 0.50, f"Expected >= 50% char reduction, got {reduction:.2%}"
        assert token_reduction >= 0.50, f"Expected >= 50% token reduction, got {token_reduction:.2%}"
        # Empirical reduction typically exceeds 75%
        assert reduction >= 0.70

    def test_token_reduction_on_50_character_cast_exceeds_80_percent(self) -> None:
        huge_universe = {
            f"c{i}": CharacterProfile(f"c{i}", f"Character {i}", role="Minor NPC", gender="unknown")
            for i in range(50)
        }
        huge_universe["c0"] = CharacterProfile("c0", "Leader", role="Protagonist", gender="male")
        huge_universe["c1"] = CharacterProfile("c1", "Follower", role="Ally", gender="female")

        batch = [DuckGroundingUnit("u1", 1, "c0", "Advance!", target="c1")]

        global_dump = "\n".join(f"- {p.display_name}: {p.role} ({p.gender})" for p in huge_universe.values())
        micro_frame = build_transgraph_frame(batch, huge_universe)

        reduction = (len(global_dump) - len(micro_frame)) / len(global_dump)
        assert reduction >= 0.80, f"Expected >= 80% reduction on 50 chars, got {reduction:.2%}"


class TestContextBuilderIntegration:
    """Tests verifying integration of TransGraph within ContextBuilder and ContextBundle."""

    def test_build_user_prompt_with_transgraph_flag_replaces_global_dump(self) -> None:
        cfg = Config()
        builder = ContextBuilder(cfg)

        cast: dict[str, CharacterProfile] = {
            f"extra_{i:02d}": CharacterProfile(f"extra_{i:02d}", f"Extra {i}", role="Citizen")
            for i in range(20)
        }
        cast["ren"] = CharacterProfile("ren", "Ren", gender="male", role="Hero", pronoun_map={"liana": "anh - em"})
        cast["liana"] = CharacterProfile("liana", "Liana", gender="female", role="Priestess", pronoun_map={"ren": "em - anh"})

        units = [
            _make_unit("d:1", 1, "Liana, are you ready?", speaker="ren"),
            _make_unit("d:2", 2, "Yes, Ren.", speaker="liana"),
        ]

        prompt_transgraph = builder.build_user_prompt(units, characters=cast, transgraph=True)
        prompt_legacy = builder.build_user_prompt(units, characters=cast, transgraph=False)

        # TransGraph prompt must contain the micro-framing headers
        assert "[ACTIVE SCENE: default]" in prompt_transgraph
        assert "[ACTIVE INTERACTION SUBGRAPH]" in prompt_transgraph
        assert "Active Personas:" in prompt_transgraph
        assert "ren" in prompt_transgraph and "liana" in prompt_transgraph

        # Inactive characters must NOT appear in TransGraph prompt
        for i in range(20):
            assert f"extra_{i:02d}" not in prompt_transgraph

        # Legacy prompt contains the global dump
        assert "HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:" in prompt_legacy
        assert "extra_00" in prompt_legacy

        # Verify >= 50% token reduction of TransGraph prompt context over legacy prompt context
        reduction = (len(prompt_legacy) - len(prompt_transgraph)) / len(prompt_legacy)
        assert reduction >= 0.50, f"Expected >= 50% reduction in user prompt, got {reduction:.2%}"

    def test_context_bundle_transgraph_flag_drives_build_user_prompt(self) -> None:
        cfg = Config()
        builder = ContextBuilder(cfg)

        cast = {
            "boris": CharacterProfile("boris", "Boris", gender="male", role="King"),
            "clara": CharacterProfile("clara", "Clara", gender="female", role="Knight"),
            "david": CharacterProfile("david", "David", gender="male", role="Advisor"),
        }
        units = [
            _make_unit("d:1", 1, "Clara, defend the city.", speaker="boris"),
        ]

        bundle_tg = ContextBundle(characters=cast, transgraph=True)
        prompt = builder.build_user_prompt(units, bundle=bundle_tg)

        assert "[ACTIVE INTERACTION SUBGRAPH]" in prompt
        assert "boris" in prompt
        # Inactive David must not be in prompt
        assert "david" not in prompt

    def test_context_builder_use_transgraph_constructor_switch(self) -> None:
        cfg = Config()
        builder_tg = ContextBuilder(cfg, use_transgraph=True)

        cast = {
            "hero": CharacterProfile("hero", "Hero", gender="male", role="Warrior"),
            "mage": CharacterProfile("mage", "Mage", gender="female", role="Sorceress"),
            "guard": CharacterProfile("guard", "Guard", gender="male", role="Sentinel"),
        }
        units = [_make_unit("d:1", 1, "Mage, cast the spell!", speaker="hero")]

        prompt = builder_tg.build_user_prompt(units, characters=cast)
        assert "[ACTIVE INTERACTION SUBGRAPH]" in prompt
        assert "hero" in prompt
        assert "guard" not in prompt

    def test_legacy_mode_preserves_prefix_cache_invariance_on_small_cast(self) -> None:
        # Verify backward compatibility: small cast without transgraph preserves exact prefix
        cfg = Config()
        builder = ContextBuilder(cfg)
        chars = {
            "a": CharacterProfile("a", "A", role="Role A"),
            "b": CharacterProfile("b", "B", role="Role B"),
        }
        units = [_make_unit("d:1", 1, "Hello.", speaker="a")]
        prompt = builder.build_user_prompt(units, characters=chars, transgraph=False)
        assert "HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:" in prompt
        assert "A (tag: a)" in prompt
        assert "B (tag: b)" in prompt
