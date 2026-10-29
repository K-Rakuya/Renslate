"""Listener resolution must survive narration gaps, same-speaker runs and two-person scenes."""

from __future__ import annotations

from renslate.core.config import Config
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.context import ContextBuilder, InterlocutorResolver
from renslate.engine.graph import RelationshipGraph


def _u(seq: int, text: str, speaker: str | None, label: str = "scene1") -> Unit:
    return Unit(
        f"d:{seq}", UnitKind.DIALOGUE, seq, text, Status.PENDING, speaker=speaker, label=label
    )


def test_narration_between_turns_keeps_listener() -> None:
    r = InterlocutorResolver()
    first = _u(1, "I made tea for you.", "mei")
    narr = _u(2, "Rain taps against the window.", None)
    reply = _u(3, "Thank you, I needed that.", "ren")

    r.resolve(first, active_speakers={"mei", "ren", "kai"})
    r.resolve(narr, prev_unit=first, active_speakers={"mei", "ren", "kai"})
    res = r.resolve(reply, prev_unit=narr, active_speakers={"mei", "ren", "kai"})

    assert res.listener == "mei"
    assert res.resolution_method == "recent_turn"


def test_same_speaker_run_keeps_original_addressee() -> None:
    r = InterlocutorResolver()
    a1 = _u(1, "Where were you last night?", "ren")
    b1 = _u(2, "Out.", "mei")
    b2 = _u(3, "Does it matter?", "mei")
    active = {"ren", "mei", "kai"}

    r.resolve(a1, active_speakers=active)
    r.resolve(b1, prev_unit=a1, active_speakers=active)
    res = r.resolve(b2, prev_unit=b1, active_speakers=active)

    assert res.listener == "ren"
    assert res.resolution_method == "recent_turn"


def test_recent_turn_expires_outside_window() -> None:
    r = InterlocutorResolver(history_window=3)
    active = {"mei", "ren", "kai"}
    r.resolve(_u(1, "Hello there.", "mei"), active_speakers=active)
    gap = _u(19, "Hours pass in silence.", None)
    r.resolve(gap, active_speakers=active)
    res = r.resolve(_u(20, "Anyone home?", "ren"), prev_unit=gap, active_speakers=active)
    assert res.listener is None
    assert res.resolution_method == "default"


def test_history_window_zero_disables_recent_turn() -> None:
    r = InterlocutorResolver(history_window=0)
    a1 = _u(1, "Hello there.", "mei")
    narr = _u(2, "A pause.", None)
    r.resolve(a1, active_speakers={"mei", "ren", "kai"})
    r.resolve(narr, prev_unit=a1, active_speakers={"mei", "ren", "kai"})
    res = r.resolve(_u(3, "Hmm.", "ren"), prev_unit=narr, active_speakers={"mei", "ren", "kai"})
    assert res.listener is None


def test_dyad_when_exactly_one_other_participant() -> None:
    r = InterlocutorResolver()
    res = r.resolve(_u(1, "We should go.", "ren"), active_speakers={"ren", "mei"})
    assert res.listener == "mei"
    assert res.resolution_method == "dyad"

    r2 = InterlocutorResolver()
    res2 = r2.resolve(_u(1, "We should go.", "ren"), active_speakers={"ren", "mei", "kai"})
    assert res2.listener is None


def test_dyad_ignores_narration_tags() -> None:
    r = InterlocutorResolver()
    res = r.resolve(
        _u(1, "We should go.", "ren"), active_speakers={"ren", "narrator", "th", "mei"}
    )
    assert res.listener == "mei"


def test_scene_boundary_clears_recent_turns() -> None:
    r = InterlocutorResolver()
    a = _u(1, "Good morning.", "mei", label="scene1")
    r.resolve(a, active_speakers={"mei", "ren", "kai"})
    new_scene = _u(2, "Is this seat taken?", "ren", label="scene2")
    res = r.resolve(new_scene, prev_unit=a, active_speakers={"mei", "ren", "kai"})
    assert res.listener is None


def test_repeated_resolution_does_not_evict_other_speaker() -> None:
    """The pipeline resolves each unit twice (prompt + QA); memory must stay stable."""
    r = InterlocutorResolver()
    active = {"mei", "ren", "kai"}
    r.resolve(_u(1, "Hi.", "mei"), active_speakers=active)
    prev: Unit | None = None
    for seq in range(2, 12):
        unit = _u(seq, "Well.", "ren")
        r.resolve(unit, prev_unit=prev, active_speakers=active)
        res = r.resolve(unit, prev_unit=prev, active_speakers=active)
        prev = unit
    assert res.listener == "mei"


def test_prompt_carries_pair_pronouns_after_narration_gap() -> None:
    chars = {
        "mei": CharacterProfile(tag="mei", display_name="Mei", gender="female"),
        "ren": CharacterProfile(tag="ren", display_name="Ren", gender="male"),
        "kai": CharacterProfile(tag="kai", display_name="Kai", gender="male"),
    }
    graph = RelationshipGraph()
    graph.add_edge(
        RelationshipEdge(
            "ren", "mei", RelationshipKind.ROMANCE, speaker_pronoun="anh", listener_pronoun="em"
        )
    )
    batch = [
        _u(1, "I made tea for you.", "mei"),
        _u(2, "Rain taps against the window.", None),
        _u(3, "Thank you, I needed that.", "ren"),
        _u(4, "Kai waves from the door.", "kai"),
    ]
    prompt = ContextBuilder(Config()).build_user_prompt(
        batch, characters=chars, graph=graph, resolver=InterlocutorResolver()
    )
    ren_line = next(line for line in prompt.splitlines() if line.startswith("d:3 |"))
    assert "Người nghe: Mei" in ren_line
    assert "xưng 'anh', gọi 'em'" in ren_line
