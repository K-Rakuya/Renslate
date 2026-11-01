# ruff: noqa: RUF001, RUF002
"""Empirical adversarial stress-test suite for Milestone 4 Context Architecture.

Targets:
1. InterlocutorResolver sliding window memory across narrative pauses:
   - Turn-taking across narration gaps up to 12 turns, and expiration at 13+ turns.
   - Same-speaker monologue runs and stage directions in nested quotes
     (“‘(Laughs) Hey, Liana!’”, 「(What is happening?)」, «(Clears throat) Dear Arthur...»).
   - Multi-party turnover with 4+ distinct speakers (including AST co-presence pruning).
   - Dynamic symmetric and asymmetric pronoun resolution (resolve_pronouns).
2. ContextBuilder Knapsack Budgeting under extreme token pressure:
   - Active speaker AND resolved listener profiles and pronoun rules retained
     even when available budget drops below standard thresholds (FLAW-07 regression check).
"""

from __future__ import annotations

import pytest

from renslate.core.config import BatchConfig, Config, LLMConfig
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.context import (
    ContextBuilder,
    InterlocutorResolver,
)
from renslate.engine.graph import RelationshipGraph


def _make_unit(
    uid: str,
    seq: int,
    source: str,
    *,
    speaker: str | None = None,
    label: str | None = "scene_default",
    ref: str | None = "script.rpy:100",
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        masked_source=source,
        speaker=speaker,
        label=label,
        ref=ref,
        status=Status.PENDING,
    )


# =============================================================================
# 1. SLIDING WINDOW MEMORY ACROSS NARRATIVE PAUSES
# =============================================================================


class TestSlidingWindowNarrativePauses:
    """Stress-test turn-taking sliding window across narrative pauses."""

    @pytest.mark.parametrize("gap_diff", [1, 2, 3, 5, 8, 11, 12])
    def test_turn_taking_narration_gaps_up_to_12_retained(self, gap_diff: int) -> None:
        """Verify turn-taking memory retains speaker across gaps up to 12 sequence turns."""
        resolver = InterlocutorResolver(history_window=12)
        active_speakers = {"alice", "bob", "charlie"}

        # Alice speaks at seq 1
        u_alice = _make_unit("d:1", 1, "Hello everyone, can you hear me?", speaker="alice")
        resolver.resolve(u_alice, active_speakers=active_speakers)

        # Intervening narration units
        for g in range(1, gap_diff):
            u_narr = _make_unit(f"n:{g}", 1 + g, f"Narrative pause line {g}.", speaker=None)
            resolver.resolve(u_narr, active_speakers=active_speakers)

        # Bob speaks at seq 1 + gap_diff (sequence difference is gap_diff)
        u_bob = _make_unit("d:bob", 1 + gap_diff, "I hear you clearly.", speaker="bob")
        resolved = resolver.resolve(u_bob, active_speakers=active_speakers)

        if gap_diff == 1:
            # Immediate adjacent dialogue turn
            assert resolved.listener == "alice"
            assert resolved.resolution_method == "turn_taking"
        else:
            # Reached across narrative pause via recent_turn sliding window
            assert resolved.listener == "alice"
            assert resolved.resolution_method == "recent_turn"

    @pytest.mark.parametrize("gap_diff", [13, 14, 15, 20, 25])
    def test_turn_taking_narration_gaps_13_plus_expired(self, gap_diff: int) -> None:
        """Verify turn-taking strictly expires at 13+ turns in multi-party scenes."""
        resolver = InterlocutorResolver(history_window=12)
        active_speakers = {"alice", "bob", "charlie"}

        # Alice speaks at seq 1
        u_alice = _make_unit("d:1", 1, "Hello everyone.", speaker="alice")
        resolver.resolve(u_alice, active_speakers=active_speakers)

        # 12+ intervening narration units
        for g in range(1, gap_diff):
            u_narr = _make_unit(f"n:{g}", 1 + g, f"Long narration {g}.", speaker=None)
            resolver.resolve(u_narr, active_speakers=active_speakers)

        # Bob speaks at seq 1 + gap_diff (difference >= 13)
        u_bob = _make_unit("d:bob", 1 + gap_diff, "Is anyone still there?", speaker="bob")
        resolved = resolver.resolve(u_bob, active_speakers=active_speakers)

        # Must strictly expire because diff > 12 and 3 active speakers prevent dyad fallback
        assert resolved.listener is None
        assert resolved.resolution_method == "default"

    def test_dyad_fallback_when_sliding_window_expires_in_two_party_scene(self) -> None:
        """In a 2-person scene, dyad fallback protects resolution even after 20 narrative turns."""
        resolver = InterlocutorResolver(history_window=12)
        active_speakers = {"alice", "bob"}

        u_alice = _make_unit("d:1", 1, "Don't leave me alone.", speaker="alice")
        resolver.resolve(u_alice, active_speakers=active_speakers)

        # 20 narration turns
        for g in range(1, 21):
            u_narr = _make_unit(f"n:{g}", 1 + g, f"Deep silence stretches {g}.", speaker=None)
            resolver.resolve(u_narr, active_speakers=active_speakers)

        u_bob = _make_unit("d:bob", 22, "I am still right here.", speaker="bob")
        resolved = resolver.resolve(u_bob, active_speakers=active_speakers)

        # recent_turn expired, but dyad fallback kicks in
        assert resolved.listener == "alice"
        assert resolved.resolution_method == "dyad"

    def test_custom_history_window_parameter(self) -> None:
        """Verify custom history_window bounds (e.g. window=5)."""
        resolver = InterlocutorResolver(history_window=5)
        active = {"alice", "bob", "charlie"}

        u1 = _make_unit("d:1", 1, "Wait for me.", speaker="alice")
        resolver.resolve(u1, active_speakers=active)

        # Narration for 4 turns: seq 2, 3, 4, 5
        for g in range(1, 5):
            u_pause = _make_unit(f"n:{g}", 1 + g, "Pacing.", speaker=None)
            resolver.resolve(u_pause, active_speakers=active)

        # Bob speaks at seq 6 (diff = 5 <= 5) -> retained
        u_retained = _make_unit("d:ret", 6, "I am waiting.", speaker="bob")
        res_ret = resolver.resolve(u_retained, active_speakers=active)
        assert res_ret.listener == "alice"
        assert res_ret.resolution_method == "recent_turn"

        # Now test expiration at diff = 6 (seq 7 from seq 1) on fresh resolver
        resolver2 = InterlocutorResolver(history_window=5)
        resolver2.resolve(u1, active_speakers=active)
        for g in range(1, 6):
            u_pause2 = _make_unit(f"n:{g}", 1 + g, "Pacing.", speaker=None)
            resolver2.resolve(u_pause2, active_speakers=active)
        u_exp = _make_unit("d:exp", 7, "I am waiting.", speaker="bob")
        res_exp = resolver2.resolve(u_exp, active_speakers=active)
        assert res_exp.listener is None
        assert res_exp.resolution_method == "default"

    def test_scene_boundary_clears_sliding_window_memory(self) -> None:
        """Scene boundary transitions (label change or seq jump > 30) must flush window memory."""
        resolver = InterlocutorResolver()
        active = {"alice", "bob", "charlie"}

        u1 = _make_unit("d:1", 1, "I will see you tomorrow.", speaker="alice", label="chapter_1")
        resolver.resolve(u1, active_speakers=active)

        # Immediate next turn but in different label (scene boundary)
        u2 = _make_unit("d:2", 2, "Where is everyone?", speaker="bob", label="chapter_2")
        res2 = resolver.resolve(u2, active_speakers=active)

        # Must not resolve Alice from chapter_1
        assert res2.listener is None
        assert res2.resolution_method == "default"


# =============================================================================
# 2. NESTED QUOTES AND SAME-SPEAKER MONOLOGUE RUNS
# =============================================================================


class TestNestedQuotesAndMonologues:
    """Stress-test stage directions, nested quotes, and same-speaker monologue runs."""

    def test_nested_double_single_quotes_with_stage_direction_and_vocative(self) -> None:
        """Test “‘(Laughs) Hey, Liana!’” -> peels double/single quotes, extracts vocative."""
        resolver = InterlocutorResolver()
        u = _make_unit("d:1", 1, "“‘(Laughs) Hey, Liana!’”", speaker="arthur")
        res = resolver.resolve(u, known_names={"liana": "Liana"})

        assert not res.is_monologue
        assert res.speaker == "arthur"
        assert res.listener == "liana"
        assert res.resolution_method == "vocative"

    def test_japanese_brackets_with_internal_parentheses_monologue(self) -> None:
        """Test 「(What is happening?)」 -> correctly parsed as internal monologue."""
        resolver = InterlocutorResolver()
        u = _make_unit("d:1", 1, "「(What is happening?)」", speaker="arthur")
        res = resolver.resolve(u)

        assert res.is_monologue
        assert res.speaker == "arthur"
        assert res.listener is None
        assert res.resolution_method == "monologue"

    def test_guillemets_with_stage_direction_and_dear_vocative(self) -> None:
        """Test «(Clears throat) Dear Arthur...» -> peels guillemets, extracts vocative."""
        resolver = InterlocutorResolver()
        u = _make_unit("d:1", 1, "«(Clears throat) Dear Arthur...»", speaker="liana")
        res = resolver.resolve(u, known_names={"arthur": "Arthur"})

        assert not res.is_monologue
        assert res.speaker == "liana"
        assert res.listener == "arthur"
        assert res.resolution_method == "vocative"
        assert res.detected_emotion == "hesitant"

    def test_japanese_full_width_brackets_and_thought_symbols(self) -> None:
        """Test 『...』 and full-width Japanese parentheses 「（ため息）...」."""
        resolver = InterlocutorResolver()

        # Japanese double bracket monologue
        u1 = _make_unit("d:1", 1, "『(Thinking) What should I do?』", speaker="arthur")
        res1 = resolver.resolve(u1)
        assert res1.is_monologue
        assert res1.resolution_method == "monologue"

        # Japanese bracket with full-width parentheses stage direction only
        u2 = _make_unit("d:2", 2, "「（ため息）...」", speaker="arthur")
        res2 = resolver.resolve(u2)
        assert res2.is_monologue
        assert res2.resolution_method == "monologue"

    def test_stage_directions_with_pure_punctuation_are_monologues(self) -> None:
        """Stage directions followed only by punctuation are inner reactions."""
        resolver = InterlocutorResolver()

        variations = [
            "(Sighs) ...",
            "「(Gasps) ...!?」",
            "«(Chuckles) —»",
            "“(Shakes head) ...”",
        ]
        for idx, text in enumerate(variations):
            u = _make_unit(f"d:{idx}", idx + 1, text, speaker="arthur")
            res = resolver.resolve(u)
            assert res.is_monologue, f"Failed for {text}"
            assert res.resolution_method == "monologue"

    def test_same_speaker_monologue_run_preserves_listener_for_dialogue_resumption(self) -> None:
        """Same-speaker monologue run (3 turns) must preserve listener for dialogue resumption."""
        resolver = InterlocutorResolver()
        active = {"arthur", "liana", "merlin"}
        known = {"liana": "Liana", "merlin": "Merlin"}

        # Turn 1: Arthur addresses Liana directly
        u1 = _make_unit("d:1", 1, "Hey Liana, take a look at this ancient map.", speaker="arthur")
        res1 = resolver.resolve(u1, active_speakers=active, known_names=known)
        assert res1.listener == "liana"
        assert res1.resolution_method == "vocative"

        # Turn 2: Arthur inner monologue
        u2 = _make_unit("d:2", 2, "「(What is happening?)」", speaker="arthur")
        res2 = resolver.resolve(u2, active_speakers=active, known_names=known)
        assert res2.is_monologue

        # Turn 3: Arthur another inner monologue / stage direction
        u3 = _make_unit("d:3", 3, "“‘(Sighs)’”", speaker="arthur")
        res3 = resolver.resolve(u3, active_speakers=active, known_names=known)
        assert res3.is_monologue

        # Turn 4: Arthur speaks dialogue again without repeating vocative
        u4 = _make_unit("d:4", 4, "Do you recognize these runes?", speaker="arthur")
        res4 = resolver.resolve(u4, active_speakers=active, known_names=known)

        # Must preserve Liana via same_speaker_continuation
        assert not res4.is_monologue
        assert res4.listener == "liana"
        assert res4.resolution_method == "same_speaker_continuation"

    def test_same_speaker_extended_monologue_chain(self) -> None:
        """A long 6-turn monologue chain still preserves listener on turn 8."""
        resolver = InterlocutorResolver()
        active = {"hero", "princess", "mage"}
        known = {"princess": "Princess"}

        # Hero addresses Princess
        u0 = _make_unit("d:0", 0, "Princess, stay close.", speaker="hero")
        resolver.resolve(u0, active_speakers=active, known_names=known)

        # 6 consecutive monologue turns by hero (seq 1 to 6)
        for i in range(1, 7):
            u_mono = _make_unit(f"d:{i}", i, f"(Hero thinks deeply part {i}...)", speaker="hero")
            res_mono = resolver.resolve(u_mono, active_speakers=active, known_names=known)
            assert res_mono.is_monologue

        # Turn 7: Hero resumes spoken dialogue
        u7 = _make_unit("d:7", 7, "The coast is finally clear.", speaker="hero")
        res7 = resolver.resolve(u7, active_speakers=active, known_names=known)

        assert res7.listener == "princess"
        assert res7.resolution_method == "same_speaker_continuation"


# =============================================================================
# 3. MULTI-PARTY TURNOVER WITH 4+ DISTINCT SPEAKERS
# =============================================================================


class TestMultiPartyTurnover:
    """Stress-test 4+ distinct speakers, dyadic sub-conversations, and AST co-presence."""

    def test_four_party_turnover_comprehensive(self) -> None:
        """Stress-test 4 speakers (Alice, Bob, Charlie, Dana) with AST co-presence."""
        resolver = InterlocutorResolver()
        active = {"alice", "bob", "charlie", "dana"}
        known = {"alice": "Alice", "bob": "Bob", "charlie": "Charlie", "dana": "Dana"}

        script = [
            # 1. Vocative initiation
            ("d:1", 1, "alice", "Bob, look here!", "bob", "vocative", None),
            # 2. Vocative response
            ("d:2", 2, "bob", "What is it, Alice?", "alice", "vocative", None),
            # 3. Interruption by 3rd speaker to 4th
            ("d:3", 3, "charlie", "Dana, did you bring the supplies?", "dana", "vocative", None),
            # 4. Medial vocative answer
            ("d:4", 4, "dana", "Yes, Charlie, right here.", "charlie", "vocative", None),
            # 5. Cross-turn vocative from Bob to Charlie
            ("d:5", 5, "bob", "Charlie, give me a hand.", "charlie", "vocative", None),
            # 6. Adjacency response (Charlie -> Bob)
            ("d:6", 6, "charlie", "Coming.", "bob", "turn_taking", None),
            # 7. Dana speaks to Alice
            ("d:7", 7, "dana", "Alice, look at the sky.", "alice", "vocative", None),
            # 8. Adjacency response (Alice -> Dana)
            ("d:8", 8, "alice", "It's getting dark.", "dana", "turn_taking", None),
            # 9. Rapid turnover without vocatives (Bob -> Alice)
            ("d:9", 9, "bob", "Should we set up camp?", "alice", "turn_taking", None),
            # 10. Rapid turnover (Charlie -> Bob)
            ("d:10", 10, "charlie", "I think we should.", "bob", "turn_taking", None),
            # 11. Rapid turnover (Dana -> Charlie)
            ("d:11", 11, "dana", "I will gather firewood.", "charlie", "turn_taking", None),
            # 12. Rapid turnover (Alice -> Dana)
            ("d:12", 12, "alice", "I will pitch the tent.", "dana", "turn_taking", None),
            # 13-14. Narration pause
            ("d:13", 13, None, "Night fell over the mountain.", None, "monologue", None),
            ("d:14", 14, None, "The fire crackled warmly.", None, "monologue", None),
            # 15. Bob speaks after narrative pause -> addresses recent speaker (Alice)
            ("d:15", 15, "bob", "Good work today.", "alice", "recent_turn", None),
            # 16. Narration: Alice and Bob enter the tent, leaving Charlie and Dana outside
            ("d:16", 16, None, "Alice and Bob went inside the tent.", None, "monologue", None),
            # 17. Charlie speaks outside -> AST co-presence prunes Alice/Bob, resolving Dana!
            (
                "d:17",
                17,
                "charlie",
                "Are you still awake?",
                "dana",
                "recent_turn",
                {"charlie", "dana"},
            ),
        ]

        for uid, seq, spk, text, exp_lis, exp_meth, pres in script:
            u = _make_unit(uid, seq, text, speaker=spk)
            res = resolver.resolve(
                u,
                active_speakers=active,
                known_names=known,
                present_characters=pres,
            )
            assert res.listener == exp_lis, (
                f"Turn {seq} expected listener {exp_lis}, got {res.listener}"
            )
            assert res.resolution_method == exp_meth, (
                f"Turn {seq} expected method {exp_meth}, got {res.resolution_method}"
            )

    def test_rapid_round_robin_turnover_pure_adjacency(self) -> None:
        """Alice -> Bob -> Charlie -> Dana -> Alice pure adjacency chain."""
        resolver = InterlocutorResolver()
        active = {"alice", "bob", "charlie", "dana"}

        u1 = _make_unit("d:1", 1, "Starting.", speaker="alice")
        res1 = resolver.resolve(u1, active_speakers=active)
        assert res1.listener is None

        u2 = _make_unit("d:2", 2, "Second.", speaker="bob")
        res2 = resolver.resolve(u2, active_speakers=active)
        assert res2.listener == "alice"
        assert res2.resolution_method == "turn_taking"

        u3 = _make_unit("d:3", 3, "Third.", speaker="charlie")
        res3 = resolver.resolve(u3, active_speakers=active)
        assert res3.listener == "bob"
        assert res3.resolution_method == "turn_taking"

        u4 = _make_unit("d:4", 4, "Fourth.", speaker="dana")
        res4 = resolver.resolve(u4, active_speakers=active)
        assert res4.listener == "charlie"
        assert res4.resolution_method == "turn_taking"

        u5 = _make_unit("d:5", 5, "Loop back.", speaker="alice")
        res5 = resolver.resolve(u5, active_speakers=active)
        assert res5.listener == "dana"
        assert res5.resolution_method == "turn_taking"


# =============================================================================
# 4. DYNAMIC SYMMETRIC AND ASYMMETRIC PRONOUN RESOLUTION
# =============================================================================


class TestDynamicPronounResolution:
    """Stress-test resolve_pronouns across RelationshipGraph and CharacterProfile."""

    def test_symmetric_pronoun_inversion_graph(self) -> None:
        """Directed edge A -> B ('anh', 'em') inverts for B -> A ('em', 'anh')."""
        resolver = InterlocutorResolver()
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

        # Direct query
        assert resolver.resolve_pronouns("hero", "heroine", graph=graph) == ("anh", "em")
        # Reciprocal symmetric inversion
        assert resolver.resolve_pronouns("heroine", "hero", graph=graph) == ("em", "anh")

    def test_asymmetric_pronoun_resolution_graph(self) -> None:
        """Asymmetric roles (Master vs Disciple) preserve independent direct edges."""
        resolver = InterlocutorResolver()
        graph = RelationshipGraph()
        # Master calls himself 'ta' and disciple 'ngươi'
        graph.add_edge(
            RelationshipEdge(
                source_tag="master",
                target_tag="disciple",
                kind=RelationshipKind.MENTOR_DISCIPLE,
                speaker_pronoun="ta",
                listener_pronoun="ngươi",
            )
        )
        # Disciple calls himself 'con' and master 'sư phụ'
        graph.add_edge(
            RelationshipEdge(
                source_tag="disciple",
                target_tag="master",
                kind=RelationshipKind.MENTOR_DISCIPLE,
                speaker_pronoun="con",
                listener_pronoun="sư phụ",
            )
        )

        assert resolver.resolve_pronouns("master", "disciple", graph=graph) == ("ta", "ngươi")
        assert resolver.resolve_pronouns("disciple", "master", graph=graph) == ("con", "sư phụ")

    def test_symmetric_pronoun_character_profile_map(self) -> None:
        """Profile A with pronoun_map={'bob': 'em - anh'} inverts when querying B -> A."""
        resolver = InterlocutorResolver()
        chars = {
            "alice": CharacterProfile("alice", "Alice", pronoun_map={"bob": "em - anh"}),
            "bob": CharacterProfile("bob", "Bob"),
        }

        # Direct from Alice
        assert resolver.resolve_pronouns("alice", "bob", char_map=chars) == ("em", "anh")
        # Reciprocal from Bob
        assert resolver.resolve_pronouns("bob", "alice", char_map=chars) == ("anh", "em")

    def test_protagonist_pronoun_resolution(self) -> None:
        """Heroine with pronoun_map={'mc': 'em - anh'} resolves against 'protag'."""
        resolver = InterlocutorResolver()
        chars = {
            "heroine": CharacterProfile("heroine", "Heroine", pronoun_map={"mc": "em - anh"}),
            "hero": CharacterProfile("hero", "Hero", role="Protagonist"),
        }

        assert resolver.resolve_pronouns("heroine", "hero", char_map=chars) == ("em", "anh")
        assert resolver.resolve_pronouns("hero", "heroine", char_map=chars) == ("anh", "em")

    def test_case_insensitivity_and_display_name_lookup(self) -> None:
        """Pronoun resolution must handle uppercase tags and full display names."""
        resolver = InterlocutorResolver()
        cp1 = CharacterProfile("liana", "Công chúa Liana", pronoun_map={"arthur": "em - anh"})
        cp2 = CharacterProfile("arthur", "Hiệp sĩ Arthur", pronoun_map={"liana": "anh - em"})

        char_map = {
            "liana": cp1,
            "LIANA": cp1,
            "arthur": cp2,
            "ARTHUR": cp2,
            "Công chúa Liana": cp1,
            "Hiệp sĩ Arthur": cp2,
        }

        # Uppercase tags
        assert resolver.resolve_pronouns("LIANA", "ARTHUR", char_map=char_map) == ("em", "anh")
        # Vietnamese Display Names
        assert resolver.resolve_pronouns(
            "Công chúa Liana", "Hiệp sĩ Arthur", char_map=char_map
        ) == ("em", "anh")
        # Unknown character returns None safely
        assert resolver.resolve_pronouns("liana", "stranger", char_map=char_map) is None


# =============================================================================
# 5. CONTEXT BUILDER KNAPSACK BUDGETING (FLAW-07 REGRESSION CHECK)
# =============================================================================


class TestKnapsackBudgetingUnderExtremePressure:
    """FLAW-07 Regression: Active speakers AND resolved listeners retained under budget."""

    def test_active_speaker_and_resolved_listener_retained_under_extreme_token_pressure(
        self,
    ) -> None:
        """Assert both active speaker AND listener are kept when budget sheds other profiles."""
        chars = {
            f"extra_{i}": CharacterProfile(
                f"extra_{i}", f"ExtraCharacter_{i}", pronoun_map={"all": "tôi - bạn"}
            )
            for i in range(30)
        }
        chars["hero"] = CharacterProfile(
            "hero", "Hero", gender="male", role="Hero", pronoun_map={"princess": "anh - em"}
        )
        chars["princess"] = CharacterProfile(
            "princess",
            "Princess",
            gender="female",
            role="Princess",
            pronoun_map={"hero": "em - anh"},
        )

        u = _make_unit("d:1", 1, "Princess, wait for me!", speaker="hero")

        # Configure small context window (1300 tokens) causing 30-char roster to blow budget
        cfg = Config(llm=LLMConfig(n_ctx=1300), batch=BatchConfig(max_source_tokens=30))
        builder = ContextBuilder(cfg)
        resolver = InterlocutorResolver()

        prompt = builder.build_user_prompt([u], characters=chars, resolver=resolver)

        # 1. Both Hero and Princess must be retained in the profile block
        assert "Hero (tag: hero)" in prompt
        assert "Princess (tag: princess)" in prompt

        # 2. Pronoun rules for both must be retained
        assert "với princess (xưng 'anh', gọi 'em')" in prompt
        assert "với hero (xưng 'em', gọi 'anh')" in prompt

        # 3. None of the 30 extra characters should be in prompt (knapsack shed)
        assert "ExtraCharacter_0" not in prompt
        assert "ExtraCharacter_15" not in prompt
        assert "ExtraCharacter_29" not in prompt

        # 4. Rich dialogue frame must carry the resolved listener
        assert "Người nghe: Princess" in prompt

    def test_multi_speaker_and_listeners_retained_under_budget(self) -> None:
        """A batch with 3 distinct characters retains all three while shedding 25 NPCs."""
        chars = {
            f"npc_{i}": CharacterProfile(
                f"npc_{i}", f"Villager_{i}", pronoun_map={"all": "tôi - bạn"}
            )
            for i in range(25)
        }
        chars["hero"] = CharacterProfile(
            "hero",
            "Hero",
            gender="male",
            pronoun_map={"princess": "anh - em", "villain": "ta - ngươi"},
        )
        chars["princess"] = CharacterProfile(
            "princess", "Princess", gender="female", pronoun_map={"hero": "em - anh"}
        )
        chars["villain"] = CharacterProfile(
            "villain", "Villain", gender="male", pronoun_map={"hero": "ngươi - ta"}
        )

        u1 = _make_unit("d:1", 1, "Princess, stay back!", speaker="hero")
        u2 = _make_unit("d:2", 2, "Villain, show yourself!", speaker="hero")
        u3 = _make_unit("d:3", 3, "Fools, you will perish.", speaker="villain")

        cfg = Config(llm=LLMConfig(n_ctx=1400), batch=BatchConfig(max_source_tokens=60))
        builder = ContextBuilder(cfg)
        resolver = InterlocutorResolver()

        prompt = builder.build_user_prompt([u1, u2, u3], characters=chars, resolver=resolver)

        # Hero, Princess, and Villain must all be present
        assert "Hero" in prompt
        assert "Princess" in prompt
        assert "Villain" in prompt

        # Unrelated NPCs shed
        assert "Villager_0" not in prompt
        assert "Villager_12" not in prompt

    def test_stateful_resolver_listener_not_in_batch_retained(self) -> None:
        """When batch has only Bob, resolver keeps Alice (from prior turn) in prompt!"""
        chars = {f"npc_{i}": CharacterProfile(f"npc_{i}", f"Villager_{i}") for i in range(25)}
        chars["alice"] = CharacterProfile(
            "alice", "Alice", gender="female", pronoun_map={"bob": "em - anh"}
        )
        chars["bob"] = CharacterProfile(
            "bob", "Bob", gender="male", pronoun_map={"alice": "anh - em"}
        )

        resolver = InterlocutorResolver()
        # Alice spoke in previous turn
        u_prev = _make_unit("d:0", 0, "Hey Bob!", speaker="alice")
        resolver.resolve(u_prev, active_speakers={"alice", "bob"})

        # Current batch ONLY contains Bob (Alice is absent from batch units)
        u_curr = _make_unit("d:1", 1, "I hear you.", speaker="bob")

        cfg = Config(llm=LLMConfig(n_ctx=1350), batch=BatchConfig(max_source_tokens=30))
        builder = ContextBuilder(cfg)

        prompt = builder.build_user_prompt([u_curr], characters=chars, resolver=resolver)

        # Bob is active speaker -> retained
        assert "Bob" in prompt
        # Alice was NOT in batch, but is resolved listener -> MUST BE RETAINED!
        assert "Alice" in prompt
        # NPCs shed
        assert "Villager_0" not in prompt

    def test_rich_dialogue_line_preserves_pronoun_metadata_under_exhausted_budget(self) -> None:
        """When budget is low, rich dialogue line preserves rules even if profile block drops."""
        chars = {
            "hero": CharacterProfile("hero", "Hero", pronoun_map={"princess": "anh - em"}),
            "princess": CharacterProfile("princess", "Princess", pronoun_map={"hero": "em - anh"}),
        }
        u = _make_unit("d:1", 1, "Princess, hurry!", speaker="hero")

        cfg = Config(llm=LLMConfig(n_ctx=1000), batch=BatchConfig(max_source_tokens=30))
        builder = ContextBuilder(cfg)
        resolver = InterlocutorResolver()

        prompt = builder.build_user_prompt([u], characters=chars, resolver=resolver)

        # Profile block may drop due to <80 tokens, but rich dialogue line has relational metadata
        assert "d:1 |" in prompt
        assert "Người nghe: Princess" in prompt
        assert "Hero xưng 'anh', gọi 'em'" in prompt
