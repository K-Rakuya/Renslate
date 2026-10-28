"""Tests for Stable Prefix Prompt KV Cache architecture."""

from __future__ import annotations

from renslate.core.config import Config
from renslate.core.models import (
    CharacterProfile,
    GlossaryEntry,
    RelationshipEdge,
    RelationshipKind,
    SceneMacroContext,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.context import ContextBuilder
from renslate.engine.graph import RelationshipGraph


def test_stable_prefix_ordering_and_layer_hierarchy() -> None:
    cfg = Config()
    builder = ContextBuilder(cfg)

    chars = {
        "ren": CharacterProfile("ren", "Ren", gender="male", role="Hero"),
        "liana": CharacterProfile("liana", "Liana", gender="female", role="Priestess"),
    }
    glossary = [GlossaryEntry("Sect", "Tông môn")]
    macro = SceneMacroContext(scene_id="s1", label="forest_night", summary="Night in the forest.")
    graph = RelationshipGraph()
    graph.add_edge(RelationshipEdge("ren", "liana", RelationshipKind.ROMANCE, "anh", "em"))

    units = [
        Unit("d:1", UnitKind.DIALOGUE, 1, "Wait for me!", Status.PENDING, speaker="ren"),
        Unit("d:2", UnitKind.DIALOGUE, 2, "I am here.", Status.PENDING, speaker="liana"),
    ]

    sys_prompt = builder.build_system_prompt()
    user_prompt = builder.build_user_prompt(
        units,
        glossary_matches=glossary,
        characters=chars,
        scene_macro=macro,
        graph=graph,
    )

    # Layer 1: Invariant System Prompt with Universal Pronoun Anti-Inversion Rule
    assert "QUY TẮC CẤM ĐẢO NGƯỢC" in sys_prompt or "TUYỆT ĐỐI CẤM ĐẢO NGƯỢC" in sys_prompt

    # Layer 2: Stable Game-Level Character Profiles
    idx_chars = user_prompt.index("HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:")

    # Layer 3.1: Scene Macro
    macro_hdr = "BỐI CẢNH PHÂN CẢNH & TÓM TẮT CẢNH HIỆN TẠI (SCENE MACRO CONTEXT):"
    idx_scene = user_prompt.index(macro_hdr)

    # Layer 3.2: Talking Pairs
    idx_rel = user_prompt.index("QUAN HỆ XƯNG HÔ ĐỐI THOẠI TRONG PHÂN ĐOẠN:")

    # Layer 3.3: Glossary
    idx_glossary = user_prompt.index("THUẬT NGỮ BẮT BUỘC (GLOSSARY):")

    # Layer 3.6: Dialogue Units
    idx_dialogue = user_prompt.index("CÁC DÒNG CẦN DỊCH")

    # Verify strict hierarchy: Layer 2 < Layer 3.1 < Layer 3.2 < Layer 3.3 < Layer 3.6
    assert idx_chars < idx_scene < idx_rel < idx_glossary < idx_dialogue


def test_prefix_cache_invariance_across_different_batches() -> None:
    """Verify that the prompt prefix through Layer 2 is identical across batches."""
    cfg = Config()
    builder = ContextBuilder(cfg)

    # Visual novel cast of characters
    chars = {
        "ren": CharacterProfile(
            "ren",
            "Ren",
            gender="male",
            role="Hero",
            pronoun_map={"liana": "anh - em", "elder": "con - sư phụ"},
        ),
        "liana": CharacterProfile(
            "liana",
            "Liana",
            gender="female",
            role="Priestess",
            pronoun_map={"ren": "em - anh"},
        ),
        "elder": CharacterProfile(
            "elder",
            "Elder",
            gender="male",
            role="Mentor",
            pronoun_map={"ren": "ta - con"},
        ),
    }

    # Batch 1: Ren and Liana talking, with Sect glossary
    batch1_units = [
        Unit("d:1", UnitKind.DIALOGUE, 1, "Hello Liana", Status.PENDING, speaker="ren"),
        Unit("d:2", UnitKind.DIALOGUE, 2, "Hello Ren", Status.PENDING, speaker="liana"),
    ]
    batch1_glossary = [GlossaryEntry("Sect", "Tông môn")]
    prompt1 = builder.build_user_prompt(
        batch1_units,
        characters=chars,
        glossary_matches=batch1_glossary,
    )

    # Batch 2: Ren and Elder talking, with Sword glossary
    batch2_units = [
        Unit(
            "d:3",
            UnitKind.DIALOGUE,
            3,
            "Master Elder, please teach me",
            Status.PENDING,
            speaker="ren",
        ),
        Unit("d:4", UnitKind.DIALOGUE, 4, "Listen carefully", Status.PENDING, speaker="elder"),
    ]
    batch2_glossary = [GlossaryEntry("Sword", "Kiếm pháp")]
    prompt2 = builder.build_user_prompt(
        batch2_units,
        characters=chars,
        glossary_matches=batch2_glossary,
    )

    # Both batches must have Layer 2 at token 0 of user prompt
    assert prompt1.startswith("HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:")
    assert prompt2.startswith("HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:")

    # Extract Layer 2 block from both prompts
    prefix1 = prompt1.split("\n\nQUAN HỆ XƯNG HÔ")[0]
    prefix2 = prompt2.split("\n\nQUAN HỆ XƯNG HÔ")[0]

    # Verify that the entire Layer 2 prefix is 100% byte-for-byte identical!
    assert prefix1 == prefix2


def test_deterministic_character_profiles_sorting() -> None:
    """Verify profiles are sorted deterministically regardless of dictionary insertion order."""
    cfg = Config()
    builder = ContextBuilder(cfg)

    # Dict order 1: Z, A, M
    chars1 = {
        "zack": CharacterProfile("zack", "Zack"),
        "alice": CharacterProfile("alice", "Alice"),
        "mary": CharacterProfile("mary", "Mary"),
    }
    # Dict order 2: M, Z, A
    chars2 = {
        "mary": CharacterProfile("mary", "Mary"),
        "zack": CharacterProfile("zack", "Zack"),
        "alice": CharacterProfile("alice", "Alice"),
    }

    units = [Unit("d:1", UnitKind.DIALOGUE, 1, "Hello", Status.PENDING, speaker="alice")]

    prompt1 = builder.build_user_prompt(units, characters=chars1)
    prompt2 = builder.build_user_prompt(units, characters=chars2)

    # Both prompts should produce Alice, Mary, Zack in alphabetical order
    p1_profiles = prompt1.split("\n\nCÁC DÒNG CẦN DỊCH")[0]
    p2_profiles = prompt2.split("\n\nCÁC DÒNG CẦN DỊCH")[0]
    assert p1_profiles == p2_profiles
    assert p1_profiles.index("Alice") < p1_profiles.index("Mary") < p1_profiles.index("Zack")
