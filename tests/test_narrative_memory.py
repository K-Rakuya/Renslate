"""Tests for AST stage-grounded multi-turn narrative memory."""

from __future__ import annotations

from renslate.engine.context.narrative_memory import (
    GroundedDialogueTurn,
    NarrativeMemory,
)


def test_grounded_dialogue_turn_slots_and_properties() -> None:
    turn = GroundedDialogueTurn(
        speaker="alice",
        listener="bob",
        source_text="Good morning.",
        target_text="Chào buổi sáng.",
        scene_id="scene_courtyard",
        active_sprites=("alice smile", "bob neutral"),
    )
    assert turn.speaker == "alice"
    assert turn.listener == "bob"
    assert turn.source_text == "Good morning."
    assert turn.target_text == "Chào buổi sáng."
    assert turn.source == "Good morning."
    assert turn.target == "Chào buổi sáng."
    assert turn.scene_id == "scene_courtyard"
    assert turn.active_sprites == ("alice smile", "bob neutral")


def test_grounded_dialogue_turn_keyword_aliases() -> None:
    turn = GroundedDialogueTurn(
        speaker="speaker_a",
        listener="speaker_b",
        source="Where are you going?",
        target="Cậu đang đi đâu đấy?",
    )
    assert turn.source_text == "Where are you going?"
    assert turn.target_text == "Cậu đang đi đâu đấy?"
    assert turn.source == "Where are you going?"
    assert turn.target == "Cậu đang đi đâu đấy?"
    assert turn.scene_id == ""
    assert turn.active_sprites == ()


def test_narrative_memory_capacity_ring_buffer() -> None:
    memory = NarrativeMemory(capacity=3)
    assert len(memory) == 0

    for i in range(5):
        memory.record(
            GroundedDialogueTurn(
                speaker="a",
                listener="b",
                source_text=f"Line {i}",
                target_text=f"Câu {i}",
                scene_id="scene_main",
            )
        )

    assert len(memory) == 3
    turns = memory.turns
    assert [t.source_text for t in turns] == ["Line 2", "Line 3", "Line 4"]


def test_narrative_memory_default_capacity_is_six() -> None:
    memory = NarrativeMemory()
    for i in range(10):
        memory.record(
            GroundedDialogueTurn(
                speaker="a",
                listener="b",
                source_text=f"Turn {i}",
                target_text=f"Luot {i}",
                scene_id="scene_study",
            )
        )
    assert len(memory) == 6
    assert memory.turns[0].source_text == "Turn 4"
    assert memory.turns[-1].source_text == "Turn 9"


def test_narrative_memory_format_history_with_sprites() -> None:
    memory = NarrativeMemory(capacity=4)
    memory.record(
        GroundedDialogueTurn(
            speaker="alice",
            listener="bob",
            source_text="Don't look at me like that!",
            target_text="Đừng nhìn em như thế chứ!",
            scene_id="scene_bedroom",
            active_sprites=("alice blush", "bob neutral"),
        )
    )

    history = memory.format_history(current_scene="scene_bedroom")
    expected = (
        "- alice -> bob [alice (blush), bob (neutral)]:"
        ' "Don\'t look at me like that!" => "Đừng nhìn em như thế chứ!"'
    )
    assert history == expected
    assert "alice -> bob" in history
    assert "alice (blush)" in history
    assert "bob (neutral)" in history


def test_narrative_memory_format_history_without_sprites_or_listener() -> None:
    memory = NarrativeMemory(capacity=4)
    memory.record(
        GroundedDialogueTurn(
            speaker="alice",
            listener=None,
            source_text="What a strange dream...",
            target_text="Một giấc mơ thật kỳ lạ...",
            scene_id="scene_bedroom",
        )
    )
    memory.record(
        GroundedDialogueTurn(
            speaker=None,
            listener=None,
            source_text="The morning bell rang loudly.",
            target_text="Chuông buổi sáng vang lên inh ỏi.",
            scene_id="scene_bedroom",
        )
    )

    history = memory.format_history(current_scene="scene_bedroom")
    lines = history.splitlines()
    assert len(lines) == 2
    assert lines[0] == '- alice: "What a strange dream..." => "Một giấc mơ thật kỳ lạ..."'
    assert (
        lines[1]
        == '- narrator: "The morning bell rang loudly." => "Chuông buổi sáng vang lên inh ỏi."'
    )


def test_narrative_memory_flushes_on_scene_mismatch() -> None:
    memory = NarrativeMemory(capacity=4)
    memory.record(
        GroundedDialogueTurn(
            speaker="alice",
            listener="bob",
            source_text="Farewell.",
            target_text="Tạm biệt anh.",
            scene_id="scene_bedroom",
        )
    )

    # Calling format_history with a different active scene flushes memory
    assert memory.format_history(current_scene="scene_street") == ""
    assert len(memory) == 0


def test_narrative_memory_record_flushes_on_scene_transition() -> None:
    memory = NarrativeMemory(capacity=4)
    memory.record(
        GroundedDialogueTurn(
            speaker="alice",
            listener="bob",
            source_text="Inside the room.",
            target_text="Bên trong căn phòng.",
            scene_id="scene_interior",
        )
    )
    assert len(memory) == 1

    memory.record(
        GroundedDialogueTurn(
            speaker="bob",
            listener="alice",
            source_text="Now we are outside.",
            target_text="Giờ chúng ta đã ở bên ngoài.",
            scene_id="scene_exterior",
        )
    )
    assert len(memory) == 1
    assert memory.turns[0].source_text == "Now we are outside."


def test_narrative_memory_clear_and_empty() -> None:
    memory = NarrativeMemory(capacity=4)
    assert memory.format_history("scene_a") == ""

    memory.record(
        GroundedDialogueTurn(
            speaker="alice",
            listener="bob",
            source_text="Hello.",
            target_text="Xin chào.",
            scene_id="scene_a",
        )
    )
    assert len(memory) == 1
    memory.clear()
    assert len(memory) == 0
    assert memory.format_history("scene_a") == ""
