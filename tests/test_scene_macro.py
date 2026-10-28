"""Tests for Hierarchical Scene Memory Model (SceneMacroContext) & Context Integration [Adv-05]."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from renslate.core.config import Config, LLMConfig
from renslate.core.db import Database
from renslate.core.models import (
    CharacterProfile,
    GlossaryEntry,
    SceneMacroContext,
    SceneSummary,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.analyze import extract_scene_macro_contexts, segment_scenes
from renslate.engine.context import ContextBuilder, ContextBundle, LookbackItem


class TestSceneMacroContextModel:
    """Unit tests for SceneMacroContext dataclass definition and methods."""

    def test_instantiation_and_defaults(self) -> None:
        ctx = SceneMacroContext(scene_id="s1", label="intro")
        assert ctx.scene_id == "s1"
        assert ctx.label == "intro"
        assert ctx.file == ""
        assert ctx.summary == ""
        assert ctx.location == ""
        assert ctx.atmosphere == ""
        assert ctx.active_characters == ()
        assert ctx.temporal_anchor == ""
        assert ctx.first_seq == 0
        assert ctx.last_seq == 0
        assert ctx.metadata == {}

    def test_immutability_frozen_slots(self) -> None:
        ctx = SceneMacroContext(scene_id="s1", label="intro", location="Castle")
        with pytest.raises(FrozenInstanceError):
            ctx.location = "Dungeon"  # type: ignore[misc]

    def test_active_characters_normalization(self) -> None:
        # Pass a list of strings
        ctx = SceneMacroContext(
            scene_id="s1",
            label="intro",
            active_characters=["Alice", "Bob"],  # type: ignore[arg-type]
        )
        assert isinstance(ctx.active_characters, tuple)
        assert ctx.active_characters == ("Alice", "Bob")

    def test_is_empty(self) -> None:
        empty_ctx = SceneMacroContext(scene_id="s1", label="intro")
        assert empty_ctx.is_empty() is True

        ctx_with_loc = SceneMacroContext(scene_id="s1", label="intro", location="Forest")
        assert ctx_with_loc.is_empty() is False

        ctx_with_chars = SceneMacroContext(
            scene_id="s1", label="intro", active_characters=("Hero",)
        )
        assert ctx_with_chars.is_empty() is False

    def test_with_updates(self) -> None:
        ctx = SceneMacroContext(
            scene_id="s1",
            label="intro",
            location="Castle",
            atmosphere="Tense",
        )
        updated = ctx.with_updates(location="Courtyard", summary="Meeting at dawn")
        assert updated.scene_id == "s1"
        assert updated.label == "intro"
        assert updated.location == "Courtyard"
        assert updated.atmosphere == "Tense"
        assert updated.summary == "Meeting at dawn"
        # Original remains unmodified
        assert ctx.location == "Castle"

    def test_format_prompt_header(self) -> None:
        ctx = SceneMacroContext(
            scene_id="s1",
            label="scene_castle_night",
            location="Lâu đài cổ",
            atmosphere="U ám / Căng thẳng",
            temporal_anchor="Nửa đêm",
            active_characters=("Kỵ sĩ", "Phù thủy"),
            summary="Hai người chạm trán trước cổng thành.",
        )
        header = ctx.format_prompt_header()
        assert "BỐI CẢNH PHÂN CẢNH & TÓM TẮT CẢNH HIỆN TẠI (SCENE MACRO CONTEXT):" in header
        assert "- Phân cảnh (Scene Label): scene_castle_night" in header
        assert "- Địa điểm (Location): Lâu đài cổ" in header
        assert "- Không khí / Cảm xúc (Atmosphere): U ám / Căng thẳng" in header
        assert "- Thời điểm (Time): Nửa đêm" in header
        assert "- Nhân vật hiện diện (Characters): Kỵ sĩ, Phù thủy" in header
        assert "- Tóm tắt tình huống (Summary): Hai người chạm trán trước cổng thành." in header

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        ctx = SceneMacroContext(
            scene_id="scene_001",
            label="ch1_start",
            file="game/ch1.rpy",
            summary="Opening chapter",
            location="Library",
            atmosphere="Quiet",
            active_characters=("Scholar", "Apprentice"),
            temporal_anchor="Morning",
            first_seq=10,
            last_seq=55,
            metadata={"priority": 1, "weather": "rain"},
        )
        data = ctx.to_dict()
        assert data["active_characters"] == ["Scholar", "Apprentice"]
        assert data["metadata"] == {"priority": 1, "weather": "rain"}

        reconstructed = SceneMacroContext.from_dict(data)
        assert reconstructed == ctx

    def test_from_dict_edge_cases(self) -> None:
        # Invalid container type
        with pytest.raises(TypeError, match="Expected dict or Mapping"):
            SceneMacroContext.from_dict("invalid")  # type: ignore[arg-type]

        # Missing scene_id
        with pytest.raises(ValueError, match="requires a non-empty 'scene_id'"):
            SceneMacroContext.from_dict({"label": "foo"})

        # Comma-separated string for active_characters
        ctx = SceneMacroContext.from_dict(
            {
                "scene_id": "s1",
                "label": "foo",
                "active_characters": "hero, villain, bystander",
            }
        )
        assert ctx.active_characters == ("hero", "villain", "bystander")


class TestSceneMacroExtraction:
    """Tests for extract_scene_macro_contexts in renslate.engine.analyze."""

    def test_extract_scene_macro_contexts_full_lifecycle(self, tmp_path: Path) -> None:
        db = Database.open(tmp_path / "test.db", create=True)
        try:
            db.save_unit(
                Unit(
                    "d:1",
                    UnitKind.DIALOGUE,
                    1,
                    "Halt!",
                    Status.PENDING,
                    speaker="guard",
                    label="scene_castle_night_tense",
                    ref="game/castle.rpy:10",
                )
            )
            db.save_unit(
                Unit(
                    "d:2",
                    UnitKind.DIALOGUE,
                    2,
                    "Let me pass.",
                    Status.PENDING,
                    speaker="hero",
                    label="scene_castle_night_tense",
                    ref="game/castle.rpy:12",
                )
            )
            # Second scene after sequence jump and label change
            db.save_unit(
                Unit(
                    "d:3",
                    UnitKind.DIALOGUE,
                    100,
                    "Welcome class.",
                    Status.PENDING,
                    speaker="sensei",
                    label="scene_school_morning",
                    ref="game/school.rpy:5",
                )
            )

            macros = extract_scene_macro_contexts(db)
            assert len(macros) == 2

            m1 = macros[0]
            assert m1.label == "scene_castle_night_tense"
            assert "Lâu đài" in m1.location
            assert "Đêm khuya" in m1.atmosphere or "Căng thẳng" in m1.atmosphere
            assert "guard" in m1.active_characters
            assert "hero" in m1.active_characters
            assert m1.file == "game/castle.rpy"
            assert "Đêm khuya" in m1.temporal_anchor
            assert m1.first_seq == 1
            assert m1.last_seq == 2
            assert m1.metadata.get("dialogue_count") == 2

            m2 = macros[1]
            assert m2.label == "scene_school_morning"
            assert "Trường học" in m2.location
            assert "sensei" in m2.active_characters
            assert m2.file == "game/school.rpy"
            assert "Buổi sáng" in m2.temporal_anchor
            assert m2.first_seq == 100
            assert m2.last_seq == 100
        finally:
            db.close()

    def test_extract_empty_or_no_dialogues(self, tmp_path: Path) -> None:
        db = Database.open(tmp_path / "test.db", create=True)
        try:
            assert extract_scene_macro_contexts(db) == []
            db.save_unit(Unit("s:1", UnitKind.STRING, 1, "Play game", Status.PENDING))
            assert extract_scene_macro_contexts(db) == []
        finally:
            db.close()

    def test_extract_with_passed_units_sequence(self, tmp_path: Path) -> None:
        db = Database.open(tmp_path / "test.db", create=True)
        try:
            units = [
                Unit("d:1", UnitKind.DIALOGUE, 1, "Hello", Status.PENDING, label="label_dusk"),
                Unit("d:2", UnitKind.DIALOGUE, 2, "World", Status.PENDING, label="label_dusk"),
            ]
            macros = extract_scene_macro_contexts(db, units=units)
            assert len(macros) == 1
            assert macros[0].label == "label_dusk"
            assert "Hoàng hôn" in macros[0].temporal_anchor
        finally:
            db.close()

    def test_segment_scenes_backward_compat(self, tmp_path: Path) -> None:
        db = Database.open(tmp_path / "test.db", create=True)
        try:
            db.save_unit(Unit("d:1", UnitKind.DIALOGUE, 1, "One", Status.PENDING, label="sc1"))
            db.save_unit(Unit("d:2", UnitKind.DIALOGUE, 2, "Two", Status.PENDING, label="sc1"))
            db.save_unit(Unit("d:3", UnitKind.DIALOGUE, 3, "Three", Status.PENDING, label="sc2"))

            scenes = segment_scenes(db)
            assert len(scenes) == 2
            assert isinstance(scenes[0], SceneSummary)
            assert scenes[0].label == "sc1"
            assert scenes[0].first_seq == 1
            assert scenes[0].last_seq == 2
            assert scenes[0].summary == "Cảnh 'sc1' (từ dòng 1 đến 2)"
        finally:
            db.close()


class TestContextIntegration:
    """Tests for ContextBundle and ContextBuilder scene macro integration."""

    def test_context_bundle_initialization_and_immutability(self) -> None:
        gl = [GlossaryEntry(term="mana", translation="năng lượng")]
        chars = {"hero": CharacterProfile(tag="hero", display_name="Hero")}
        macro = SceneMacroContext(scene_id="s1", label="chap1", location="Forest")
        lb = [LookbackItem(speaker="hero", source="Hold on", target="Chờ đã")]

        bundle = ContextBundle(
            glossary_matches=gl,
            characters=chars,
            scene_macro=macro,
            lookback=lb,
        )
        assert isinstance(bundle.glossary_matches, tuple)
        assert isinstance(bundle.lookback, tuple)
        assert bundle.scene_macro == macro
        assert bundle.characters == chars

    def test_build_user_prompt_with_scene_macro(self) -> None:
        cb = ContextBuilder(Config())
        units = [Unit("d:1", UnitKind.DIALOGUE, 1, "Look out!", Status.PENDING, speaker="hero")]
        macro = SceneMacroContext(
            scene_id="s_dungeon",
            label="scene_dungeon_dark",
            location="Ngục tối",
            atmosphere="U ám",
            temporal_anchor="Nửa đêm",
            active_characters=("hero", "goblin"),
            summary="Entering dangerous territory.",
        )

        prompt = cb.build_user_prompt(units, scene_macro=macro)
        assert "BỐI CẢNH PHÂN CẢNH & TÓM TẮT CẢNH HIỆN TẠI (SCENE MACRO CONTEXT):" in prompt
        assert "- Phân cảnh (Scene Label): scene_dungeon_dark" in prompt
        assert "- Địa điểm (Location): Ngục tối" in prompt
        assert "- Không khí / Cảm xúc (Atmosphere): U ám" in prompt
        assert "- Thời điểm (Time): Nửa đêm" in prompt
        assert "- Nhân vật hiện diện (Characters): hero, goblin" in prompt
        assert "- Tóm tắt tình huống (Summary): Entering dangerous territory." in prompt

    def test_build_user_prompt_with_bundle(self) -> None:
        cb = ContextBuilder(Config())
        units = [Unit("d:1", UnitKind.DIALOGUE, 1, "I see light.", Status.PENDING, speaker="hero")]
        macro = SceneMacroContext(scene_id="s_dungeon", label="scene_dawn", location="Đồng cỏ")
        bundle = ContextBundle(
            glossary_matches=[GlossaryEntry(term="light", translation="ánh sáng")],
            scene_macro=macro,
        )

        prompt = cb.build_user_prompt(units, bundle=bundle)
        assert "THUẬT NGỮ BẮT BUỘC (GLOSSARY):" in prompt
        assert "- light -> ánh sáng" in prompt
        assert "BỐI CẢNH PHÂN CẢNH & TÓM TẮT CẢNH HIỆN TẠI (SCENE MACRO CONTEXT):" in prompt
        assert "- Phân cảnh (Scene Label): scene_dawn" in prompt

    def test_build_user_prompt_with_legacy_scene_summary(self) -> None:
        cb = ContextBuilder(Config())
        units = [Unit("d:1", UnitKind.DIALOGUE, 1, "Good morning", Status.PENDING, speaker="hero")]
        legacy = SceneSummary(
            scene_id="s_class",
            label="scene_classroom_morning",
            summary="Class is starting.",
        )

        prompt = cb.build_user_prompt(units, scene_summary=legacy)
        assert "BỐI CẢNH PHÂN CẢNH & TÓM TẮT CẢNH HIỆN TẠI (SCENE MACRO CONTEXT):" in prompt
        assert "- Phân cảnh (Scene Label): scene_classroom_morning" in prompt
        assert "- Địa điểm (Location): Lớp học" in prompt
        assert "- Không khí / Cảm xúc (Atmosphere): Buổi sáng" in prompt
        assert "- Tóm tắt tình huống (Summary): Class is starting." in prompt

    def test_scene_macro_budget_overflow_excluded(self) -> None:
        cfg = Config(llm=LLMConfig(n_ctx=512))
        cb = ContextBuilder(cfg)
        units = [
            Unit(
                f"d:{i}",
                UnitKind.DIALOGUE,
                i,
                f"Line content {i}",
                Status.PENDING,
                speaker="hero",
            )
            for i in range(15)
        ]
        huge_macro = SceneMacroContext(
            scene_id="huge",
            label="scene_large",
            summary="A massive narrative description that consumes a vast token budget... " * 150,
        )
        prompt = cb.build_user_prompt(units, scene_macro=huge_macro)
        assert "SCENE MACRO CONTEXT" not in prompt
        for u in units:
            assert u.uid in prompt
