"""Tests for renslate.context."""

from __future__ import annotations

from renslate.config import Config
from renslate.context import ContextBuilder, LookbackItem, estimate_tokens
from renslate.models import CharacterProfile, GlossaryEntry, Status, Unit, UnitKind


def test_estimate_tokens() -> None:
    en_tok = estimate_tokens("Hello world this is a test sentence.")
    assert en_tok > 0
    vi_tok = estimate_tokens("Xin chào thế giới đây là câu kiểm tra.", is_vietnamese=True)
    assert vi_tok > 0


def test_system_prompt_composition() -> None:
    cfg = Config()
    builder = ContextBuilder(cfg, custom_style_guide="Luôn xưng 'tớ' và 'cậu'")
    sys_prompt = builder.build_system_prompt()
    assert "chuyên gia bản địa hóa" in sys_prompt
    assert "Luôn xưng 'tớ' và 'cậu'" in sys_prompt
    assert "tự nhiên, trung tính" in sys_prompt


def test_user_prompt_with_all_priorities() -> None:
    cfg = Config()
    builder = ContextBuilder(cfg)

    unit = Unit(
        uid="d:intro",
        kind=UnitKind.DIALOGUE,
        seq=1,
        source="Hello Eileen!",
        status=Status.PENDING,
        speaker="e",
        masked_source="Hello ⟦V1⟧!",
    )

    glossary = [GlossaryEntry("Eileen", "Eileen", notes="Tên riêng")]
    chars = {
        "e": CharacterProfile(
            tag="e",
            display_name="Eileen",
            gender="female",
            role="Heroine",
            speech_style="vui tươi",
        )
    }
    lookback = [LookbackItem("e", "Start game", "Bắt đầu")]

    user_prompt = builder.build_user_prompt(
        batch_units=[unit],
        glossary_matches=glossary,
        characters=chars,
        lookback=lookback,
    )

    assert "d:intro" in user_prompt
    assert "GLOSSARY" in user_prompt
    assert "Eileen -> Eileen" in user_prompt
    assert "HỒ SƠ NHÂN VẬT" in user_prompt
    assert "Heroine" in user_prompt
    assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" in user_prompt


def test_empty_glossary_candidate_is_injected_as_candidate() -> None:
    cfg = Config()
    builder = ContextBuilder(cfg)
    unit = Unit("d:test", UnitKind.DIALOGUE, 1, "Hello", Status.PENDING)
    # Candidate with empty translation should not appear in mandatory GLOSSARY,
    # but should be actively surfaced under DANH TỪ RIÊNG & THUẬT NGỮ CẦN ĐỒNG NHẤT.
    glossary = [GlossaryEntry("CandidateTerm", "", kind="candidate")]
    prompt = builder.build_user_prompt(batch_units=[unit], glossary_matches=glossary)
    assert "THUẬT NGỮ BẮT BUỘC (GLOSSARY)" not in prompt
    assert "DANH TỪ RIÊNG & THUẬT NGỮ CẦN ĐỒNG NHẤT" in prompt
    assert "CandidateTerm" in prompt


def test_relational_pronouns_with_pseudo_speakers_and_multi_characters() -> None:
    cfg = Config()
    builder = ContextBuilder(cfg)
    units = [
        Unit("d:1", UnitKind.DIALOGUE, 1, "Hello", Status.PENDING, speaker="mc"),
        Unit("d:2", UnitKind.DIALOGUE, 2, "Thinking...", Status.PENDING, speaker="t"),
        Unit("d:3", UnitKind.DIALOGUE, 3, "Hi Human", Status.PENDING, speaker="c"),
    ]
    chars = {
        "mc": CharacterProfile("mc", "Protagonist", pronoun_map={"c": "anh - em"}),
        "c": CharacterProfile("c", "Charlotte", pronoun_map={"mc": "em - anh"}),
        "t": CharacterProfile("t", "Thinking"),
    }
    prompt = builder.build_user_prompt(batch_units=units, characters=chars)

    assert "QUAN HỆ XƯNG HÔ ĐỐI THOẠI TRONG PHÂN ĐOẠN:" in prompt
    assert "Protagonist (mc)" in prompt and "Charlotte (c)" in prompt
    assert "TUYỆT ĐỐI CẤM dịch 'you' thành 'em'" in prompt
    assert "TUYỆT ĐỐI CẤM dịch 'you' thành 'anh'" in prompt
    assert "[Người nói: Charlotte (c)] Hi Human" in prompt
    assert "[Người nói: Protagonist (mc)] Hello" in prompt
    assert "[Độc thoại/Nội tâm: t] Thinking..." in prompt


def test_relational_pronouns_resolved_from_lookback_when_single_speaker() -> None:
    cfg = Config()
    builder = ContextBuilder(cfg)
    units = [
        Unit("d:1", UnitKind.DIALOGUE, 2, "Why are you here?", Status.PENDING, speaker="c"),
    ]
    chars = {
        "mc": CharacterProfile("mc", "Protagonist", pronoun_map={"c": "anh - em"}),
        "c": CharacterProfile("c", "Charlotte", pronoun_map={"mc": "em - anh"}),
    }
    lookback = [LookbackItem("mc", "Hello", "Xin chào")]
    prompt = builder.build_user_prompt(batch_units=units, characters=chars, lookback=lookback)

    assert "QUAN HỆ XƯNG HÔ ĐỐI THOẠI TRONG PHÂN ĐOẠN:" in prompt
    assert "Protagonist (mc)" in prompt and "Charlotte (c)" in prompt
    assert "TUYỆT ĐỐI CẤM dịch 'you' thành 'em'" in prompt


def test_strict_non_inversion_rules_in_system_prompt() -> None:
    cfg = Config()
    builder = ContextBuilder(cfg)
    sys_prompt = builder.build_system_prompt()

    assert "TUYỆT ĐỐI CẤM ĐẢO NGƯỢC" in sys_prompt
    assert "ĐẠI TỪ NGÔI THỨ NHẤT (I, me, my, we): BẮT BUỘC dịch là X" in sys_prompt
    assert "ĐẠI TỪ NGÔI THỨ HAI (you, your, yourself): BẮT BUỘC dịch là Y" in sys_prompt


def test_infer_scene_cues() -> None:
    from renslate.engine.context import infer_scene_cues

    loc, atmos = infer_scene_cues("forest_night_mystery")
    assert "Rừng" in loc
    assert "Đêm khuya" in atmos

    loc2, atmos2 = infer_scene_cues("classroom_sunset")
    assert "Lớp học" in loc2
    assert "Hoàng hôn" in atmos2

    loc3, atmos3 = infer_scene_cues("arbitrary_label")
    assert loc3 == ""
    assert atmos3 == ""


def test_format_rich_dialogue_line_variants() -> None:
    from renslate.core.models import RelationshipEdge, RelationshipKind
    from renslate.engine.context import (
        ResolvedInterlocutor,
        format_rich_dialogue_line,
    )
    from renslate.engine.graph import RelationshipGraph

    chars = {
        "ren": CharacterProfile("ren", "Ren"),
        "liana": CharacterProfile("liana", "Liana"),
    }
    graph = RelationshipGraph()
    graph.add_edge(RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em"))

    u_mono = Unit("d:1", UnitKind.DIALOGUE, 1, "(Where am I...?)", Status.PENDING, speaker="ren")
    r_mono = ResolvedInterlocutor(speaker="ren", is_monologue=True, detected_emotion="hesitant")
    line_mono = format_rich_dialogue_line(u_mono, r_mono, chars, graph=graph)
    assert line_mono == "d:1 | [Độc thoại/Nội tâm: Ren | Cảm xúc: hesitant] (Where am I...?)"

    u_dial = Unit("d:2", UnitKind.DIALOGUE, 2, "I love you!", Status.PENDING, speaker="ren")
    r_dial = ResolvedInterlocutor(
        speaker="ren", listener="liana", is_monologue=False, detected_emotion="emphatic"
    )
    line_dial = format_rich_dialogue_line(u_dial, r_dial, chars, graph=graph)
    expected_line = (
        "d:2 | [Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE "
        "(Ren xưng 'anh', gọi 'em') | Cảm xúc: emphatic] I love you!"
    )
    assert line_dial == expected_line


def test_context_builder_rich_dialogue_prompt_with_graph() -> None:
    from renslate.core.models import RelationshipEdge, RelationshipKind, SceneSummary
    from renslate.engine.graph import RelationshipGraph

    cfg = Config()
    builder = ContextBuilder(cfg)
    chars = {
        "ren": CharacterProfile("ren", "Ren"),
        "liana": CharacterProfile("liana", "Liana"),
    }
    graph = RelationshipGraph()
    graph.add_edge(RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em"))

    units = [
        Unit(
            "d:1",
            UnitKind.DIALOGUE,
            1,
            "Liana, please wait for me!",
            Status.PENDING,
            speaker="ren",
        ),
        Unit("d:2", UnitKind.DIALOGUE, 2, "I'm right here.", Status.PENDING, speaker="liana"),
    ]
    scene = SceneSummary("s1", label="forest_night", summary="Night in the forest.")
    prompt = builder.build_user_prompt(units, characters=chars, scene_summary=scene, graph=graph)

    assert "SCENE MACRO CONTEXT" in prompt
    assert "forest_night" in prompt
    expected_f1 = (
        "d:1 | [Người nói: Ren -> Người nghe: Liana | Quan hệ: ROMANCE "
        "(Ren xưng 'anh', gọi 'em') | Cảm xúc: emphatic] Liana, please wait for me!"
    )
    assert expected_f1 in prompt
    assert "[Người nói: Liana -> Người nghe: Ren] I'm right here." in prompt


def test_unit_file_derivation_from_ref() -> None:
    u = Unit("d:1", UnitKind.DIALOGUE, 1, "Hello", Status.PENDING, ref="game/ch1.rpy:42")
    assert u.file == "game/ch1.rpy"

    u_explicit = Unit(
        "d:2",
        UnitKind.DIALOGUE,
        2,
        "World",
        Status.PENDING,
        ref="game/ch1.rpy:42",
        file="custom/path.rpy",
    )
    assert u_explicit.file == "custom/path.rpy"

    u_no_colon = Unit("d:3", UnitKind.DIALOGUE, 3, "Test", Status.PENDING, ref="game/script.rpy")
    assert u_no_colon.file == "game/script.rpy"


def test_interlocutor_isolation_on_scene_label_change() -> None:
    from renslate.engine.context import InterlocutorResolver

    resolver = InterlocutorResolver()
    u1 = Unit(
        "d:1",
        UnitKind.DIALOGUE,
        1,
        "Goodbye for now.",
        Status.PENDING,
        speaker="ren",
        label="scene_forest",
    )
    res1 = resolver.resolve(u1)
    assert res1.speaker == "ren"
    assert res1.listener is None

    # u2 is in a new scene label
    u2 = Unit(
        "d:2",
        UnitKind.DIALOGUE,
        2,
        "Welcome to the academy.",
        Status.PENDING,
        speaker="teacher",
        label="scene_classroom",
    )
    res2 = resolver.resolve(u2, prev_unit=u1)
    # Speaker must NOT leak across scene boundary into listener
    assert res2.speaker == "teacher"
    assert res2.listener is None
    assert res2.resolution_method != "turn_taking"


def test_interlocutor_isolation_on_script_file_change() -> None:
    from renslate.engine.context import InterlocutorResolver

    resolver = InterlocutorResolver()
    u1 = Unit(
        "d:1",
        UnitKind.DIALOGUE,
        1,
        "End of chapter 1.",
        Status.PENDING,
        speaker="ren",
        ref="game/ch1.rpy:100",
    )
    res1 = resolver.resolve(u1)
    assert res1.speaker == "ren"

    # u2 is in chapter 2 file
    u2 = Unit(
        "d:2",
        UnitKind.DIALOGUE,
        2,
        "Beginning of chapter 2.",
        Status.PENDING,
        speaker="narrator_char",
        ref="game/ch2.rpy:5",
    )
    res2 = resolver.resolve(u2, prev_unit=u1)
    assert res2.speaker == "narrator_char"
    assert res2.listener is None


def test_interlocutor_turn_taking_resumes_within_new_scene() -> None:
    from renslate.engine.context import InterlocutorResolver

    resolver = InterlocutorResolver()
    # Scene 1 dialogue
    u1 = Unit(
        "d:1",
        UnitKind.DIALOGUE,
        1,
        "Line in scene 1",
        Status.PENDING,
        speaker="ren",
        label="scene_1",
    )
    res1 = resolver.resolve(u1)
    assert res1.listener is None

    # Scene 2 dialogue line 1 (transition: listener isolated)
    u2 = Unit(
        "d:2",
        UnitKind.DIALOGUE,
        2,
        "Line 1 in scene 2",
        Status.PENDING,
        speaker="liana",
        label="scene_2",
    )
    res2 = resolver.resolve(u2, prev_unit=u1)
    assert res2.speaker == "liana"
    assert res2.listener is None

    # Scene 2 dialogue line 2 (turn-taking resumes within scene 2)
    u3 = Unit(
        "d:3",
        UnitKind.DIALOGUE,
        3,
        "Line 2 in scene 2",
        Status.PENDING,
        speaker="ren",
        label="scene_2",
    )
    res3 = resolver.resolve(u3, prev_unit=u2)
    assert res3.speaker == "ren"
    assert res3.listener == "liana"
    assert res3.resolution_method == "turn_taking"


def test_interlocutor_isolation_with_narrator_at_transition() -> None:
    from renslate.engine.context import InterlocutorResolver

    resolver = InterlocutorResolver()
    u1 = Unit(
        "d:1",
        UnitKind.DIALOGUE,
        1,
        "Stay safe.",
        Status.PENDING,
        speaker="ren",
        label="scene_1",
    )
    resolver.resolve(u1)

    # Transition unit is narrator (no speaker)
    u2 = Unit(
        "d:2",
        UnitKind.DIALOGUE,
        2,
        "The next morning arrived quietly.",
        Status.PENDING,
        speaker=None,
        label="scene_2",
    )
    res2 = resolver.resolve(u2, prev_unit=u1)
    assert res2.is_monologue is True

    # First spoken line in scene 2
    u3 = Unit(
        "d:3",
        UnitKind.DIALOGUE,
        3,
        "Good morning everyone.",
        Status.PENDING,
        speaker="liana",
        label="scene_2",
    )
    res3 = resolver.resolve(u3, prev_unit=u2)
    assert res3.speaker == "liana"
    assert res3.listener is None


def test_kinship_alias_cleared_across_scene_boundary() -> None:
    from renslate.engine.context import InterlocutorResolver

    resolver = InterlocutorResolver()
    # In scene 1, "father" is bound to Lord Vane
    u1 = Unit(
        "d:1",
        UnitKind.DIALOGUE,
        1,
        "I understand, father.",
        Status.PENDING,
        speaker="daughter",
        label="scene_1",
    )
    # Simulate known alias
    resolver._aliases["father"] = "Lord Vane"
    assert "father" in resolver._aliases

    # Moving to scene 2 clears aliases
    u2 = Unit(
        "d:2",
        UnitKind.DIALOGUE,
        2,
        "A new quest awaits.",
        Status.PENDING,
        speaker="hero",
        label="scene_2",
    )
    resolver.resolve(u2, prev_unit=u1)
    assert "father" not in resolver._aliases


def test_build_user_prompt_scene_boundary_rich_formatting() -> None:
    from renslate.engine.graph import RelationshipGraph

    cfg = Config()
    builder = ContextBuilder(cfg)
    chars = {
        "ren": CharacterProfile("ren", "Ren"),
        "liana": CharacterProfile("liana", "Liana"),
        "guard": CharacterProfile("guard", "Guard"),
    }
    graph = RelationshipGraph()

    units = [
        Unit(
            "d:1",
            UnitKind.DIALOGUE,
            1,
            "Let's leave the forest.",
            Status.PENDING,
            speaker="ren",
            label="scene_forest",
        ),
        Unit(
            "d:2",
            UnitKind.DIALOGUE,
            2,
            "Halt! Who goes there?",
            Status.PENDING,
            speaker="guard",
            label="scene_castle_gate",
        ),
        Unit(
            "d:3",
            UnitKind.DIALOGUE,
            3,
            "We are travelers.",
            Status.PENDING,
            speaker="liana",
            label="scene_castle_gate",
        ),
    ]

    prompt = builder.build_user_prompt(units, characters=chars, graph=graph)

    # d:1 in scene_forest: speaker ren, no listener
    assert "d:1 | [Người nói: ren] Let's leave the forest." in prompt

    # d:2 in scene_castle_gate: Guard speaking at scene transition -> MUST NOT have Ren as listener!
    assert "d:2 | [Người nói: guard | Cảm xúc: inquiring] Halt! Who goes there?" in prompt
    assert "Người nghe" not in prompt.split("d:2 |")[1].split("d:3 |")[0]

    # d:3 in scene_castle_gate: Liana speaking to Guard -> turn taking in same scene!
    assert "d:3 | [Người nói: Liana -> Người nghe: Guard] We are travelers." in prompt

