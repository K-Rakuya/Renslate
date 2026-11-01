"""LLM scene summaries feeding the translation prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from renslate.core.config import Config
from renslate.core.models import (
    CharacterProfile,
    SceneMacroContext,
    SceneSummary,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.context import ContextBuilder
from renslate.engine.scenes import (
    build_scene_prompt,
    load_scene_contexts,
    parse_scene_result,
    summarize_scenes_with_llm,
)

GOOD = {
    "summary": "Ren và Mei trú mưa trong quán cà phê, Mei lo lắng về kỳ thi.",
    "location": "Quán cà phê",
    "atmosphere": "Ấm áp, hơi căng thẳng",
    "time": "Chiều muộn",
}


class FakeLLM:
    def __init__(self, answer: Any = GOOD) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def complete_json(self, system: str, user: str, **_kw: Any) -> tuple[Any, int, int]:
        self.prompts.append(user)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer, 5, 5


def _units(label: str, start: int, n: int) -> list[Unit]:
    return [
        Unit(
            f"d:{start + i}", UnitKind.DIALOGUE, start + i, f"Line {start + i} {label}.",
            Status.PENDING, speaker="ren" if i % 2 == 0 else "mei", label=label,
        )
        for i in range(n)
    ]


@pytest.fixture
def project(tmp_path: Path) -> Project:
    root = tmp_path / "g"
    (root / "game").mkdir(parents=True)
    proj = Project.create(root)
    with proj.open_db() as db:
        db.save_characters_bulk(
            [
                CharacterProfile(tag="ren", display_name="Ren"),
                CharacterProfile(tag="mei", display_name="Mei"),
            ]
        )
        db.save_units_bulk(_units("cafe", 1, 10) + _units("tiny", 11, 3) + _units("gym", 14, 10))
        db.save_scene(SceneSummary("s1", "cafe", 1, 10, "Cảnh 'cafe'"))
        db.save_scene(SceneSummary("s2", "tiny", 11, 13, "Cảnh 'tiny'"))
        db.save_scene(SceneSummary("s3", "gym", 14, 23, "Cảnh 'gym'"))
    return proj


def test_parse_scene_result_validates() -> None:
    assert parse_scene_result(GOOD)["location"] == "Quán cà phê"  # type: ignore[index]
    assert parse_scene_result({"summary": "ngắn"}) is None
    assert parse_scene_result("nope") is None
    cleaned = parse_scene_result({"summary": "Ren nói {i}chào{/i} với [name] rất lâu."})
    assert cleaned is not None and "{" not in cleaned["summary"] and "[" not in cleaned["summary"]


def test_prompt_samples_long_scene_but_keeps_ends() -> None:
    lines = [("Ren", f"Câu số {i} " + "x" * 80) for i in range(200)]
    prompt = build_scene_prompt(SceneSummary("s", "l", 1, 200, ""), lines, ["Trước đó ổn."],
                                budget_tokens=300)
    assert "Câu số 0 " in prompt and "Câu số 199 " in prompt
    assert "trích" in prompt and "Trước đó ổn." in prompt
    assert prompt.count("[Ren]") < 200


def test_summarize_skips_short_scenes_and_persists(project: Project) -> None:
    llm = FakeLLM()
    with project.open_db() as db:
        stats = summarize_scenes_with_llm(project, db, llm)
        assert stats["summarized"] == 2 and stats["skipped"] == 1
        scenes = {s.scene_id: s for s in db.get_scenes()}
        assert scenes["s1"].summary == GOOD["summary"]
        assert scenes["s2"].summary.startswith("Cảnh")
        # second scene prompt carries the previous scene's summary
        assert GOOD["summary"] in llm.prompts[1]
        # resumable: nothing left to do, no new calls
        n = len(llm.prompts)
        again = summarize_scenes_with_llm(project, db, llm)
        assert again["summarized"] == 0 and len(llm.prompts) == n


def test_summarize_force_and_seq_range_and_cap(project: Project) -> None:
    llm = FakeLLM()
    with project.open_db() as db:
        stats = summarize_scenes_with_llm(project, db, llm, seq_range=(14, 20))
        assert stats["summarized"] == 1 and len(llm.prompts) == 1
        stats = summarize_scenes_with_llm(project, db, llm, max_scenes=0)
        assert stats["summarized"] == 0
        summarize_scenes_with_llm(project, db, llm, force=True)
        assert len(llm.prompts) == 1 + 2


def test_llm_failure_and_bad_output_are_ignored(project: Project) -> None:
    with project.open_db() as db:
        assert summarize_scenes_with_llm(project, db, FakeLLM(RuntimeError("down")))[
            "summarized"
        ] == 0
        assert summarize_scenes_with_llm(project, db, FakeLLM({"summary": ""}))["summarized"] == 0
        assert db.get_scenes()[0].summary.startswith("Cảnh")


def test_load_scene_contexts_chains_previous_and_detects_moved_scene(project: Project) -> None:
    with project.open_db() as db:
        summarize_scenes_with_llm(project, db, FakeLLM())
        ctx = load_scene_contexts(db, db.get_scenes())
        assert set(ctx) == {"s1", "s3"}
        assert ctx["s1"].location == "Quán cà phê" and ctx["s1"].temporal_anchor == "Chiều muộn"
        assert "previous_summary" not in ctx["s1"].metadata
        # s2 has no LLM summary, so the chain restarts instead of linking s1 to s3
        assert "previous_summary" not in ctx["s3"].metadata
        # the scene was re-segmented to a different start: stale summary is not reused
        db.save_scene(SceneSummary("s1", "cafe", 2, 10, "Cảnh 'cafe'"))
        assert "s1" not in load_scene_contexts(db, db.get_scenes())


def test_header_includes_previous_summary() -> None:
    macro = SceneMacroContext(
        scene_id="a", label="x", summary="Hiện tại.", metadata={"previous_summary": "Trước đó."}
    )
    header = macro.format_prompt_header()
    assert "Trước đó." in header and "Hiện tại." in header


def test_user_prompt_uses_rich_scene_macro() -> None:
    units = _units("cafe", 1, 2)
    macro = SceneMacroContext(
        scene_id="s1", label="cafe", summary=GOOD["summary"], location="Quán cà phê"
    )
    prompt = ContextBuilder(Config()).build_user_prompt(
        batch_units=units, glossary_matches=[], characters={}, scene_macro=macro
    )
    assert GOOD["summary"] in prompt
