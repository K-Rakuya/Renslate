"""Unit tests for Character Persona Grounding (1.1, 1.2) and Same-Speaker Listener Continuation (2.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from renslate.core.config import Config
from renslate.core.models import CharacterProfile, Status, Unit, UnitKind
from renslate.core.project import Project
from renslate.engine.analyze import (
    bootstrap_characters_with_llm,
    build_persona_user_prompt,
    collect_character_narration_context,
    extract_stylometrics,
    find_protagonist_tag,
)
from renslate.engine.context import ContextBuilder, InterlocutorResolver, format_rich_dialogue_line


def _u(seq: int, text: str, speaker: str | None = None, label: str = "scene1") -> Unit:
    return Unit(
        f"d:{seq}", UnitKind.DIALOGUE, seq, text, Status.PENDING, speaker=speaker, label=label
    )


def test_collect_character_narration_context() -> None:
    chars = {
        "charlotte": CharacterProfile(tag="charlotte", display_name="Charlotte"),
        "ren": CharacterProfile(tag="ren", display_name="Ren"),
    }
    units = [
        _u(1, "The castle was quiet.", None),
        _u(2, "Charlotte held her staff tightly, staring at the gate.", None),
        _u(3, "She breathed slowly.", None),
        _u(4, "Ren was nowhere to be seen.", None),
        _u(5, "I am ready.", "charlotte"),
    ]
    ctx = collect_character_narration_context(units, ["charlotte", "ren"], chars, max_lines=2)
    assert len(ctx["charlotte"]) == 1
    assert "Charlotte held her staff" in ctx["charlotte"][0]
    assert len(ctx["ren"]) == 1
    assert "Ren was nowhere" in ctx["ren"][0]


def test_find_protagonist_tag() -> None:
    # 1. By tag
    chars1 = {"mc": CharacterProfile("mc", "Hero"), "alice": CharacterProfile("alice", "Alice")}
    assert find_protagonist_tag(chars1, []) == "mc"

    # 2. By role
    chars2 = {
        "ren": CharacterProfile("ren", "Ren", role="Nhân vật chính (Protagonist)"),
        "alice": CharacterProfile("alice", "Alice"),
    }
    assert find_protagonist_tag(chars2, []) == "ren"


def test_build_persona_user_prompt_includes_mc_and_narration() -> None:
    chars = {"mei": CharacterProfile("mei", "Mei")}
    samples = {"mei": ["Hello there."]}
    stys = {"mei": extract_stylometrics(samples["mei"])}
    narr = {"mei": ["Mei looked down at her hands, smiling softly."]}

    prompt = build_persona_user_prompt(
        ["mei"],
        chars,
        samples,
        stys,
        narration_by_speaker=narr,
        mc_tag="ren",
    )
    assert "[LƯU Ý NHÂN VẬT CHÍNH - MC]: Nhân vật chính của game có tag là 'ren'" in prompt
    assert "Mei looked down at her hands" in prompt


def test_bootstrap_characters_bidirectional_mc_pronoun_mapping(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)

    class MockLLM:
        def complete_json(
            self, system: str, user: str, *args: Any, **kwargs: Any
        ) -> tuple[dict[str, Any], int, int]:
            return (
                {
                    "mei": {
                        "gender": "female",
                        "role": "Bạn thuở nhỏ",
                        "pronoun_map": {"mc": "em - anh"},
                    }
                },
                50,
                30,
            )

    with project.open_db() as db:
        # Protagonist has tag "ren"
        db.save_character(CharacterProfile("ren", "Ren", role="Nhân vật chính (Protagonist)"))
        db.save_unit(_u(1, "Ren, wait for me!", "mei"))
        chars = bootstrap_characters_with_llm(project, db, MockLLM(), max_chars=5)

    assert "mei" in chars
    p = chars["mei"]
    # Both "mc" and "ren" are mapped
    assert p.pronoun_map.get("mc") == "em - anh"
    assert p.pronoun_map.get("ren") == "em - anh"


def test_format_rich_dialogue_line_mc_fallback() -> None:
    char_map = {
        "mei": CharacterProfile("mei", "Mei", pronoun_map={"mc": "em - anh"}),
        "ren": CharacterProfile("ren", "Ren", role="Nhân vật chính (Protagonist)"),
    }
    unit = _u(1, "Why are you here?", "mei")
    resolver = InterlocutorResolver()
    # Resolve listener as "ren"
    resolved = resolver.resolve(unit, active_speakers={"mei", "ren"})
    line = format_rich_dialogue_line(unit, resolved, char_map)
    assert "Người nói: Mei -> Người nghe: Ren" in line
    assert "Mei xưng 'em', gọi 'anh'" in line


def test_same_speaker_continuation_keeps_listener() -> None:
    r = InterlocutorResolver()
    # 3 active characters in scene (not a simple dyad)
    active = {"ren", "mei", "kai"}

    # Ren addresses Mei explicitly via vocative in line 1
    u1 = _u(1, "Mei, look at the sky.", "ren")
    res1 = r.resolve(u1, active_speakers=active)
    assert res1.listener == "mei"
    assert res1.resolution_method == "vocative"

    # Line 2: Ren speaks again, no vocative in sentence
    u2 = _u(2, "The stars are unusually bright tonight.", "ren")
    res2 = r.resolve(u2, prev_unit=u1, active_speakers=active)
    assert res2.listener == "mei"
    assert res2.resolution_method == "same_speaker_continuation"

    # Line 3: Ren speaks a 3rd sentence consecutively
    u3 = _u(3, "I think tomorrow will be clear.", "ren")
    res3 = r.resolve(u3, prev_unit=u2, active_speakers=active)
    assert res3.listener == "mei"
    assert res3.resolution_method == "same_speaker_continuation"


def test_speaker_scoped_aliases_no_cross_speaker_leakage() -> None:
    """Ensure dynamic vocative/kinship aliases are scoped by speaker to prevent leakage."""
    r = InterlocutorResolver()
    active = {"ren", "eileen", "vane"}
    names = {"ren": "Ren", "eileen": "Eileen", "vane": "Lord Vane"}

    # Ren addresses Lord Vane as 'Father'
    u1 = _u(1, "Father, the harvest is ready.", "ren")
    res1 = r.resolve(u1, None, active_speakers=active, known_names=names)
    assert res1.listener == "vane"
    assert res1.resolution_method == "vocative"

    # Now Eileen speaks. Eileen addresses 'Father' but Eileen is NOT Ren.
    # Eileen's speaker alias should not inherit Ren's unless explicitly global fallback.
    # But specifically, if Eileen speaks to Ren, Ren's kinship aliases do not contaminate Eileen's turn.
    u2 = _u(2, "Ren, did you speak to him?", "eileen")
    res2 = r.resolve(u2, prev_unit=u1, active_speakers=active, known_names=names)
    assert res2.listener == "ren"
    assert res2.resolution_method == "vocative"

    # Turn taking back to Ren: Ren speaks without vocative; should target Eileen (turn taking)
    u3 = _u(3, "Yes, I just did.", "ren")
    res3 = r.resolve(u3, prev_unit=u2, active_speakers=active, known_names=names)
    assert res3.listener == "eileen"
    assert res3.resolution_method == "turn_taking"


def test_scene_presence_filtering_blocks_absent_characters() -> None:
    """Ensure characters not present in the active scene macro cannot be selected as listeners."""
    r = InterlocutorResolver()
    # Scene has only Ren and Eileen present
    present = {"ren", "eileen"}
    # Active speakers in broader project or historical turns
    active = {"ren", "eileen", "kai"}

    # Ren speaks to Kai in an earlier scene
    u1 = _u(1, "Kai, stand guard.", "ren")
    r.resolve(u1, active_speakers=active)

    # Next scene starts, only Ren and Eileen present
    u2 = _u(2, "The coast is clear now.", "ren")
    # Even though Kai was recent turn, Kai is NOT in present_characters!
    res2 = r.resolve(u2, active_speakers=active, present_characters=present)
    assert res2.listener == "eileen"
    assert res2.listener != "kai"


def test_large_seq_gap_triggers_scene_boundary() -> None:
    """A seq gap > 30 units triggers scene boundary and resets turn-taking state."""
    r = InterlocutorResolver()
    u1 = _u(10, "Hello Eileen.", "ren")
    r.resolve(u1, active_speakers={"ren", "eileen"})

    # Sequence jumps from 10 to 45 (> 30 gap)
    u2 = _u(45, "Where are we now?", "ren")
    # Because gap > 30, it resets effective_prev and interlocutor memory
    res2 = r.resolve(u2, prev_unit=u1, active_speakers={"ren", "eileen"})
    assert res2.resolution_method in ("dyad", "default")


def test_jit_lazy_scene_summarization_in_pipeline(tmp_path: Path) -> None:
    """Verify TranslationPipeline._prepare_batch triggers on-demand scene summarization."""
    from unittest.mock import patch

    from renslate.core.models import SceneSummary
    from renslate.engine.translate import TranslationPipeline

    root = tmp_path / "jit_test"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)

    with project.open_db() as db:
        db.save_character(CharacterProfile("ren", "Ren", "male"))
        db.save_character(CharacterProfile("eileen", "Eileen", "female"))

        # Define a scene from seq 1 to 5, without an existing summary
        sc = SceneSummary(
            scene_id="sc_tavern_1",
            label="tavern_meeting",
            first_seq=1,
            last_seq=5,
            summary="",
        )
        db.save_scene(sc)

        u1 = _u(1, "Welcome to the tavern, Ren.", "eileen", label="tavern_meeting")
        u2 = _u(2, "Thank you, Eileen.", "ren", label="tavern_meeting")
        u3 = _u(3, "Have a drink.", "eileen", label="tavern_meeting")
        db.save_unit(u1)
        db.save_unit(u2)
        db.save_unit(u3)

        cfg = Config()
        pipeline = TranslationPipeline(project, db, cfg)
        assert pipeline._scene_ctx.get("sc_tavern_1") is None

        mock_scene_summary = {
            "summary": "Ren và Eileen gặp nhau tại quán rượu để bàn kế hoạch.",
            "location": "Quán rượu (Tavern)",
            "atmosphere": "Ấm cúng, thư giãn",
            "time": "Buổi tối",
        }

        with patch.object(
            pipeline.llm,
            "complete_json",
            return_value=(mock_scene_summary, 100, 50),
        ):
            # _prepare_batch for [u1, u2] should trigger JIT summarization
            prep = pipeline._prepare_batch([u1, u2])

        # Assert scene_ctx was lazily created and loaded
        loaded_ctx = pipeline._scene_ctx.get("sc_tavern_1")
        assert loaded_ctx is not None
        assert loaded_ctx.summary == "Ren và Eileen gặp nhau tại quán rượu để bàn kế hoạch."
        assert loaded_ctx.location == "Quán rượu (Tavern)"
        assert loaded_ctx.atmosphere == "Ấm cúng, thư giãn"

        # Assert scene_macro is attached to prepared batch
        assert prep.scene_macro is loaded_ctx

        # Assert build_user_prompt produces prompt with scene macro header
        prompt = pipeline.context.build_user_prompt(batch_units=prep.batch, scene_macro=prep.scene_macro)
        assert "BỐI CẢNH PHÂN CẢNH & TÓM TẮT CẢNH HIỆN TẠI" in prompt
        assert "Quán rượu (Tavern)" in prompt


def test_batch_dialogue_units_strictly_ordered_by_seq() -> None:
    """Ensure partition_batches and build_user_prompt sort units chronologically to prevent dialogue chaos."""
    from renslate.engine.translate import partition_batches

    # Intentionally scrambled dialogue units
    u_9 = _u(9, "Final verdict.", "ren")
    u_2 = _u(2, "Wait, what happened?", "eileen")
    u_5 = _u(5, "I investigated the scene.", "ren")
    u_1 = _u(1, "The body was found here.", "kael")

    shuffled = [u_9, u_2, u_5, u_1]
    batches = partition_batches(shuffled, max_lines=10)
    assert len(batches) == 1
    batch = batches[0]

    # Verify batch is strictly ordered by seq
    assert [u.seq for u in batch] == [1, 2, 5, 9]

    # Verify build_user_prompt also maintains strict seq order
    cfg = Config()
    cb = ContextBuilder(cfg)
    prompt = cb.build_user_prompt(batch_units=shuffled)
    lines_section = prompt.split("CÁC DÒNG CẦN DỊCH (bắt buộc dịch đầy đủ đúng id):")[1]

    idx_1 = lines_section.index("d:1")
    idx_2 = lines_section.index("d:2")
    idx_5 = lines_section.index("d:5")
    idx_9 = lines_section.index("d:9")
    assert idx_1 < idx_2 < idx_5 < idx_9


def test_prompt_system_and_user_completely_free_of_glossary() -> None:
    """Verify that glossary mechanism and prompt text are completely eliminated."""
    from renslate.engine.context import DEFAULT_SYSTEM_PROMPT, QUOTED_RULES

    # Invariant: system prompt rules must not contain any glossary rule
    assert "GLOSSARY" not in QUOTED_RULES
    assert "THUẬT NGỮ" not in QUOTED_RULES
    assert "GLOSSARY" not in DEFAULT_SYSTEM_PROMPT

    # Invariant: user prompt must not render glossary sections
    cfg = Config()
    cb = ContextBuilder(cfg)
    u1 = _u(1, "Welcome to the Grand Sect, elder.", "ren")

    prompt = cb.build_user_prompt([u1])
    assert "GLOSSARY" not in prompt
    assert "THUẬT NGỮ BẮT BUỘC" not in prompt
    assert "DANH TỪ RIÊNG & THUẬT NGỮ CẦN ĐỒNG NHẤT" not in prompt


