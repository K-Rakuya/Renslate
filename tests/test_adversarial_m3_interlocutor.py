"""Adversarial stress harness & empirical verification for Milestone 3 (InterlocutorResolver).

Probing:
1. Rapid turn-taking churn & 3+ speaker free-for-alls
2. Kinship title address resolution and dynamic alias poisoning
3. Leading vs trailing title vocatives ("Father", "Master", "Sensei", "Sư phụ")
4. Stage directions in parentheses vs true monologues
5. Quoted monologues and Japanese corner bracket delimiters
6. Punctuation boundaries (ellipsis '...', '…', tilde '~', em-dash, full-width)
7. Vocative collisions with common English words ("May", "Will", "Hope", "Rose")
8. Latency micro-benchmark across 50,000+ / 100,000 calls (< 1.0ms per 25 units)
"""

from __future__ import annotations

import time

from renslate.core.models import Status, Unit, UnitKind
from renslate.engine.context import InterlocutorResolver


def make_unit(uid: str, source: str, speaker: str | None = None, seq: int = 1) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        status=Status.PENDING,
        speaker=speaker,
    )


# ==============================================================================
# 1. Kinship Title Address Resolution & Dynamic Alias Poisoning
# ==============================================================================


class TestAdversarialKinshipResolution:
    """Stress tests probing kinship/title resolution in multi-party scenes."""

    def test_kinship_pronoun_inversion_and_alias_poisoning(self) -> None:
        """BUG 1: In multi-party scenes, replying to an elder with 'him/her' in sentence

        must NOT invert the listener to the 3rd party or poison the alias table.
        Scenario:
        1. Lord Vane speaks: 'Luke, where is your sister?'
        2. Evelyn answers Vane directly: 'Do not be harsh on him, Father.'
        Expected: listener is 'vane', and _aliases['father'] is 'vane'.
        """
        resolver = InterlocutorResolver()
        active = {"vane", "evelyn", "luke"}
        names = {"vane": "Lord Vane", "evelyn": "Evelyn", "luke": "Luke"}

        u1 = make_unit("d:1", "Luke, where is your sister?", speaker="vane", seq=1)
        u2 = make_unit("d:2", "Do not be harsh on him, Father.", speaker="evelyn", seq=2)

        r2 = resolver.resolve(u2, u1, active, names)

        assert r2.listener == "vane", (
            f"Expected listener 'vane', got {r2.listener!r}. "
            "Excluding prev_spk because of 'him' erroneously assigned Evelyn's response to Luke!"
        )
        assert resolver._aliases.get("father") == "vane", (
            f"Expected _aliases['father'] to be 'vane', got {resolver._aliases.get('father')!r}."
        )

    def test_leading_kinship_vocative_resolution(self) -> None:
        """BUG 2: Leading title vocatives ('Father, ...', 'Master, ...', 'Sensei, ...')

        must resolve to the addressed character even at the start of a sentence.
        """
        resolver = InterlocutorResolver()
        active = {"vane", "evelyn", "luke"}
        names = {"vane": "Lord Vane", "evelyn": "Evelyn", "luke": "Luke"}

        # Evelyn addresses Father at the start of dialogue
        u1 = make_unit("d:1", "Father, please listen to my explanation.", speaker="evelyn")
        r1 = resolver.resolve(u1, None, active, names)

        err_msg = (
            f"Expected leading 'Father' to resolve to 'vane', got {r1.listener!r} "
            f"(method: {r1.resolution_method})."
        )
        assert r1.listener == "vane", err_msg
        assert r1.resolution_method == "vocative"

    def test_leading_kinship_vocative_not_overridden_by_prev_turn(self) -> None:
        """BUG 2b: Leading title vocative must override turn-taking adjacency from prev unit."""
        resolver = InterlocutorResolver()
        active = {"vane", "evelyn", "luke"}
        names = {"vane": "Lord Vane", "evelyn": "Evelyn", "luke": "Luke"}

        # Luke speaks first
        u_luke = make_unit("d:0", "I cannot agree with this decision.", speaker="luke")
        # Evelyn speaks next, but addresses Father, NOT Luke
        u_evelyn = make_unit("d:1", "Father, please reconsider Luke's request.", speaker="evelyn")

        r = resolver.resolve(u_evelyn, u_luke, active, names)

        err_msg = (
            f"Expected listener 'vane' via vocative, but turn-taking assigned it to 'luke': "
            f"{r.listener!r}"
        )
        assert r.listener == "vane", err_msg
        assert r.resolution_method == "vocative"


# ==============================================================================
# 2. Monologue Classification vs Stage Directions & Quotes
# ==============================================================================


class TestAdversarialMonologueAndQuotes:
    """Stress tests probing parentheses, stage directions, and quote enclosures."""

    def test_stage_directions_in_parentheses_not_classified_as_monologue(self) -> None:
        """BUG 3: Parenthesized stage directions / SFX at the start of dialogue

        must NOT classify the entire line as an inner monologue.
        """
        resolver = InterlocutorResolver()
        active = {"ren", "liana"}
        names = {"ren": "Ren", "liana": "Liana"}

        u1 = make_unit("d:1", "(Laughs) Hey Liana, look at this!", speaker="ren")
        r1 = resolver.resolve(u1, None, active, names)

        assert not r1.is_monologue, "Stage direction (Laughs) was misclassified as inner monologue!"
        assert r1.listener == "liana", f"Expected listener 'liana', got {r1.listener!r}"
        assert r1.resolution_method == "vocative"

        u2 = make_unit("d:2", "(Sigh) What should we do, Ren?", speaker="liana")
        r2 = resolver.resolve(u2, u1, active, names)

        assert not r2.is_monologue, "Stage direction (Sigh) was misclassified as inner monologue!"
        assert r2.listener == "ren", f"Expected listener 'ren', got {r2.listener!r}"

    def test_outer_quoted_inner_monologue_detected(self) -> None:
        """BUG 4: Inner monologue enclosed in outer quotes must still be recognized as monologue."""
        resolver = InterlocutorResolver()
        active = {"ren", "liana"}
        names = {"ren": "Ren", "liana": "Liana"}

        # Previous dialogue was from Liana
        u_prev = make_unit("d:1", "Are you ready, Ren?", speaker="liana")

        # Ren has an inner thought wrapped in quotation marks
        u_thought = make_unit("d:2", "\"(What should I do now...?)\"", speaker="ren")
        r = resolver.resolve(u_thought, u_prev, active, names)

        msg = (
            "Quoted thought was not recognized as monologue and falsely bound to Liana "
            "via turn-taking!"
        )
        assert r.is_monologue, msg
        assert r.listener is None

    def test_japanese_corner_brackets_handling(self) -> None:
        """BUG 5: Japanese quotation marks (「...」, 『...』) should be handled cleanly."""
        resolver = InterlocutorResolver()
        active = {"ren", "liana"}
        names = {"ren": "Ren", "liana": "Liana"}

        u_dial = make_unit("d:1", "「Liana, wait for me!」", speaker="ren")
        r_dial = resolver.resolve(u_dial, None, active, names)
        assert r_dial.listener == "liana", f"Expected 'liana', got {r_dial.listener!r}"

        u_mono = make_unit("d:2", "「(What is happening?)」", speaker="ren")
        r_mono = resolver.resolve(u_mono, None, active, names)
        assert r_mono.is_monologue, "Japanese bracketed thought failed to resolve as monologue!"


# ==============================================================================
# 3. Punctuation Delimiters & Vocative Collisions
# ==============================================================================


class TestAdversarialPunctuationAndVocatives:
    """Stress tests probing non-standard punctuation delimiters and name collisions."""

    def test_ellipsis_and_tilde_vocative_delimiters(self) -> None:
        """BUG 6: Dramatic ellipsis ('...', '…') and affectionate tilde ('~')

        are standard visual novel dialogue punctuation and must delimit vocatives.
        """
        resolver = InterlocutorResolver()
        active = {"ren", "liana"}
        names = {"ren": "Ren", "liana": "Liana"}

        # Ellipsis leading
        u1 = make_unit("d:1", "Liana... are you okay?", speaker="ren")
        r1 = resolver.resolve(u1, None, active, names)
        assert r1.listener == "liana", f"Ellipsis '...' failed to delimit vocative: {r1.listener!r}"

        # Unicode ellipsis
        u2 = make_unit("d:2", "Liana… please wake up!", speaker="ren")
        r2 = resolver.resolve(u2, None, active, names)
        msg = f"Unicode ellipsis '…' failed to delimit vocative: {r2.listener!r}"
        assert r2.listener == "liana", msg

        # Tilde vocative
        u3 = make_unit("d:3", "Liana~ where are you?", speaker="ren")
        r3 = resolver.resolve(u3, None, active, names)
        assert r3.listener == "liana", f"Tilde '~' failed to delimit vocative: {r3.listener!r}"

    def test_common_english_words_collision_guards(self) -> None:
        """Verifies that normal English words ('May', 'Will', 'Hope', 'Rose')

        do not falsely trigger vocative resolution when used non-vocatively.
        """
        resolver = InterlocutorResolver()
        active = {"ren", "liana", "may", "will", "hope", "rose"}
        names = {
            "ren": "Ren",
            "liana": "Liana",
            "may": "May",
            "will": "Will",
            "hope": "Hope",
            "rose": "Rose",
        }

        prev = make_unit("d:0", "What do you think, Ren?", speaker="liana")

        non_vocative_cases = [
            "May I ask you a question?",
            "I will be right there.",
            "There is no hope left for them.",
            "She picked a red rose from the garden.",
            "Come what may, we must survive.",
            "Against his will, he had to comply.",
            "He wept with all hope gone.",
        ]

        for text in non_vocative_cases:
            u = make_unit("d:test", text, speaker="ren")
            r = resolver.resolve(u, prev, active, names)
            # Should fall back to turn-taking with Liana, NOT bind to May, Will, Hope, or Rose
            assert r.listener == "liana", (
                f"Text {text!r} falsely bound to {r.listener!r} instead of Liana!"
            )
            assert r.resolution_method == "turn_taking"


# ==============================================================================
# 4. Rapid Turn Churn & 3+ Speaker Free-for-All
# ==============================================================================


class TestAdversarialTurnChurnAndFreeForAll:
    """Stress tests rapid alternation across 4+ speakers with interleaved monologues."""

    def test_rapid_alternation_4_speakers_with_monologue_interruption(self) -> None:
        """Validates listener continuity across a chaotic 4-speaker battle sequence."""
        resolver = InterlocutorResolver()
        active = {"ren", "liana", "scarlet", "boss"}
        names = {"ren": "Ren", "liana": "Liana", "scarlet": "Scarlet", "boss": "Demon Lord"}

        # 1. Ren attacks Boss
        u1 = make_unit("d:1", "Demon Lord, your reign ends here!", speaker="ren")
        r1 = resolver.resolve(u1, None, active, names)
        assert r1.listener == "boss"
        assert r1.resolution_method == "vocative"

        # 2. Boss laughs and attacks Ren
        u2 = make_unit("d:2", "Insolent pest! Die!", speaker="boss")
        r2 = resolver.resolve(u2, u1, active, names)
        assert r2.listener == "ren"
        assert r2.resolution_method == "turn_taking"

        # 3. Liana casts a shield for Ren
        u3 = make_unit("d:3", "Ren, look out!", speaker="liana")
        r3 = resolver.resolve(u3, u2, active, names)
        assert r3.listener == "ren"
        assert r3.resolution_method == "vocative"

        # 4. Ren inner thought during the clash
        u4 = make_unit(
            "d:4", "(That shield barely held... his power is terrifying.)", speaker="ren"
        )
        r4 = resolver.resolve(u4, u3, active, names)
        assert r4.is_monologue

        # 5. Scarlet leaps in from the side
        u5 = make_unit("d:5", "Don't just stand there, Ren! Strike now!", speaker="scarlet")
        r5 = resolver.resolve(u5, u4, active, names)
        assert r5.listener == "ren"
        assert r5.resolution_method == "vocative"

        # 6. Boss speaks to everyone
        u6 = make_unit("d:6", "Pathetic worms, all of you!", speaker="boss")
        r6 = resolver.resolve(u6, u5, active, names)
        assert r6.listener == "scarlet"  # Turn-taking from previous dialogue speaker


# ==============================================================================
# 5. Latency Micro-Benchmark (100,000 Iterations)
# ==============================================================================


class TestAdversarialLatencyBenchmark:
    """Confirms O(1) execution and < 1.0ms per 25-unit batch budget across 100,000 calls."""

    def test_micro_benchmark_100k_calls(self) -> None:
        resolver = InterlocutorResolver()
        active = {"ren", "liana", "scarlet", "vane", "evelyn"}
        names = {
            "ren": "Ren",
            "liana": "Liana",
            "scarlet": "Scarlet",
            "vane": "Lord Vane",
            "evelyn": "Evelyn",
        }

        samples = [
            ("d:1", "Liana, are you ready?", "ren"),
            ("d:2", "Yes, Ren, let's go!", "liana"),
            ("d:3", "Wait for me too!", "scarlet"),
            ("d:4", "(I have a bad feeling about this...)", "scarlet"),
            ("d:5", "Be careful, everyone.", "vane"),
        ]
        units = [
            make_unit(uid, text, speaker=spk, seq=i)
            for i, (uid, text, spk) in enumerate(samples)
        ]

        total_calls = 100_000
        start = time.perf_counter()
        for i in range(total_calls):
            prev = units[(i - 1) % len(units)]
            curr = units[i % len(units)]
            resolver.resolve(curr, prev, active, names)
        elapsed = time.perf_counter() - start

        ms_per_25 = (elapsed / total_calls) * 25 * 1000.0
        assert ms_per_25 < 1.0, f"Exceeded latency budget: {ms_per_25:.4f} ms per 25 units"
