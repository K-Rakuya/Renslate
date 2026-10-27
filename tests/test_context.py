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

