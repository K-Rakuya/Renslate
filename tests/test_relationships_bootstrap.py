"""LLM relationship/address-form bootstrapping for frequently interacting pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from renslate.core.db import Database
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.graph import RelationshipGraph
from renslate.engine.relationships import (
    bootstrap_relationships_with_llm,
    build_relationship_prompt,
    collect_pair_evidence,
    parse_pair_result,
)


class FakeLLM:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def complete_json(self, system: str, user: str, **_kw: Any) -> tuple[Any, int, int]:
        self.prompts.append(user)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer, 10, 10


GOOD = {
    "kind": "friendship_close",
    "a_to_b": {"self": "tớ", "other": "cậu"},
    "b_to_a": {"self": "mình", "other": "cậu"},
    "confidence": 0.95,
    "evidence": "Hai người đùa giỡn thân mật.",
}


def _make_project(tmp_path: Path) -> Project:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    return Project.create(root)


def _seed(db: Database) -> None:
    db.save_characters_bulk(
        [
            CharacterProfile(tag="ren", display_name="Ren", gender="male", role="Nhân vật chính"),
            CharacterProfile(tag="mei", display_name="Mei", gender="female"),
            CharacterProfile(tag="kai", display_name="Kai", gender="male"),
        ]
    )
    units: list[Unit] = []
    seq = 0
    # ren <-> mei: 8 alternating turns with a narration line in the middle
    for i in range(8):
        seq += 1
        speaker = "ren" if i % 2 == 0 else "mei"
        units.append(
            Unit(
                f"d:{seq}", UnitKind.DIALOGUE, seq, f"Line number {i} of their chat.",
                Status.PENDING, speaker=speaker, label="cafe",
            )
        )
        if i == 3:
            seq += 1
            units.append(
                Unit(f"d:{seq}", UnitKind.DIALOGUE, seq, "Rain keeps falling.", Status.PENDING,
                     speaker=None, label="cafe")
            )
    # kai appears with ren only twice (below default min_turns) in another scene
    for i in range(2):
        seq += 1
        units.append(
            Unit(
                f"d:{seq}", UnitKind.DIALOGUE, seq, f"Kai exchange {i} here.", Status.PENDING,
                speaker="kai" if i == 0 else "ren", label="gym",
            )
        )
    db.save_units_bulk(units)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    proj = _make_project(tmp_path)
    with proj.open_db() as db:
        _seed(db)
    return proj


def test_collect_pair_evidence_counts_resolved_listeners(project: Project) -> None:
    with project.open_db() as db:
        ev = collect_pair_evidence(db.all_units(), db.get_characters())
    assert ("mei", "ren") in ev
    assert ev[("mei", "ren")].turns == 8
    assert all(spk in ("ren", "mei") for spk, _ in ev[("mei", "ren")].lines)
    # narration never counts as a turn
    assert all("Rain" not in text for _, text in ev[("mei", "ren")].lines)


def test_collect_pair_evidence_skips_non_human_and_collective(tmp_path: Path) -> None:
    proj = _make_project(tmp_path)
    with proj.open_db() as db:
        db.save_characters_bulk(
            [
                CharacterProfile(tag="ren", display_name="Ren"),
                CharacterProfile(tag="system", display_name="System"),
                CharacterProfile(tag="crowd", display_name="Crowd"),
            ]
        )
        units = [
            Unit(f"d:{i}", UnitKind.DIALOGUE, i, f"Message {i} for you.", Status.PENDING,
                 speaker=("ren", "system", "crowd")[i % 3], label="s")
            for i in range(1, 13)
        ]
        db.save_units_bulk(units)
        assert collect_pair_evidence(db.all_units(), db.get_characters()) == {}


def test_prompt_contains_pair_profiles_and_sampled_lines(project: Project) -> None:
    with project.open_db() as db:
        chars = db.get_characters()
        ev = collect_pair_evidence(db.all_units(), chars)[("mei", "ren")]
    prompt = build_relationship_prompt([ev], chars, lines_per_pair=4)
    assert 'Cặp "mei|ren"' in prompt
    assert "Mei" in prompt and "Ren" in prompt
    assert "giới tính female" in prompt
    assert prompt.count('  [') == 4  # exactly lines_per_pair samples
    assert "Line number 0" in prompt  # story start represented
    assert "Line number 7" in prompt  # story end represented


def test_bootstrap_writes_edges_profiles_json_and_meta(project: Project) -> None:
    llm = FakeLLM({"mei|ren": GOOD})
    with project.open_db() as db:
        stats = bootstrap_relationships_with_llm(project, db, llm)
        assert stats == {"pairs": 1, "resolved": 1, "edges": 2}

        e1 = db.get_relationship("mei", "ren")
        e2 = db.get_relationship("ren", "mei")
        assert e1 is not None and e2 is not None
        assert e1.kind is RelationshipKind.FRIENDSHIP_CLOSE
        assert (e1.speaker_pronoun, e1.listener_pronoun) == ("tớ", "cậu")
        assert (e2.speaker_pronoun, e2.listener_pronoun) == ("mình", "cậu")
        assert e1.source_type == "llm"
        assert e1.confidence == pytest.approx(0.9)  # clamped
        assert "thân mật" in e1.notes

        chars = db.get_characters()
        assert chars["mei"].pronoun_map["ren"] == "tớ - cậu"
        assert chars["ren"].pronoun_map["mei"] == "mình - cậu"
        assert db.get_meta("relationships_bootstrapped")
        assert db.get_meta("rel_llm_done:mei|ren")

    graph = RelationshipGraph.from_json_file(project.relationships_path)
    assert graph.get_edge("mei", "ren") is not None
    assert json.loads(project.characters_path.read_text(encoding="utf-8"))["mei"]["pronoun_map"]


def test_bootstrap_is_resumable_and_force_redoes(project: Project) -> None:
    llm = FakeLLM({"mei|ren": GOOD})
    with project.open_db() as db:
        bootstrap_relationships_with_llm(project, db, llm)
        assert len(llm.prompts) == 1
        again = bootstrap_relationships_with_llm(project, db, llm)
        assert again["pairs"] == 0
        assert len(llm.prompts) == 1
        forced = bootstrap_relationships_with_llm(project, db, llm, force=True)
        assert forced["pairs"] == 1
        assert len(llm.prompts) == 2


def test_manual_edges_are_never_overwritten(project: Project) -> None:
    with project.open_db() as db:
        db.save_relationship(
            RelationshipEdge(
                "mei", "ren", RelationshipKind.ROMANCE, "em", "anh", 1.0, "manual", "tay"
            )
        )
        stats = bootstrap_relationships_with_llm(project, db, FakeLLM({"mei|ren": GOOD}))
        assert stats["edges"] == 1  # only ren -> mei written
        kept = db.get_relationship("mei", "ren")
        assert kept is not None
        assert (kept.kind, kept.speaker_pronoun, kept.listener_pronoun, kept.source_type) == (
            RelationshipKind.ROMANCE, "em", "anh", "manual",
        )
        written = db.get_relationship("ren", "mei")
        assert written is not None and written.source_type == "llm"


def test_pair_fully_manual_is_skipped_without_llm_call(project: Project) -> None:
    with project.open_db() as db:
        for a, b in (("mei", "ren"), ("ren", "mei")):
            db.save_relationship(
                RelationshipEdge(a, b, RelationshipKind.ROMANCE, "x", "y", 1.0, "manual")
            )
        llm = FakeLLM({"mei|ren": GOOD})
        stats = bootstrap_relationships_with_llm(project, db, llm)
        assert stats["pairs"] == 0
        assert llm.prompts == []


def test_hand_typed_pronoun_map_without_edge_is_preserved(project: Project) -> None:
    with project.open_db() as db:
        mei = db.get_characters()["mei"]
        db.save_character(
            CharacterProfile(
                tag="mei", display_name=mei.display_name, gender=mei.gender,
                pronoun_map={"ren": "em - anh"},
            )
        )
        bootstrap_relationships_with_llm(project, db, FakeLLM({"mei|ren": GOOD}))
        assert db.get_characters()["mei"].pronoun_map["ren"] == "em - anh"
        # the graph edge is still written (prompt prefers graph pairs)
        assert db.get_relationship("mei", "ren") is not None


@pytest.mark.parametrize(
    "answer",
    [
        {"mei|ren": {"kind": "BOSS_FIGHT", "a_to_b": {"self": "tớ", "other": "cậu"}}},
        {"mei|ren": {"kind": "ROMANCE", "a_to_b": {"self": "anh", "other": "anh"}}},
        {"mei|ren": "friends"},
        {"other|pair": GOOD},
        {},
    ],
)
def test_invalid_answers_write_nothing_and_do_not_mark_done(
    project: Project, answer: dict[str, Any]
) -> None:
    with project.open_db() as db:
        stats = bootstrap_relationships_with_llm(project, db, FakeLLM(answer))
        assert stats["edges"] == 0
        assert db.count_relationships() == 0
        assert not db.get_meta("rel_llm_done:mei|ren")


def test_llm_failure_is_tolerated(project: Project) -> None:
    with project.open_db() as db:
        stats = bootstrap_relationships_with_llm(project, db, FakeLLM(RuntimeError("down")))
        assert stats == {"pairs": 1, "resolved": 0, "edges": 0}
        assert not db.get_meta("rel_llm_done:mei|ren")


def test_min_turns_and_max_pairs(project: Project) -> None:
    with project.open_db() as db:
        llm = FakeLLM({})
        assert bootstrap_relationships_with_llm(project, db, llm, min_turns=100)["pairs"] == 0
        assert bootstrap_relationships_with_llm(project, db, llm, max_pairs=0)["pairs"] == 0
        # kai|ren has only 1 turn so it is excluded by default min_turns
        assert bootstrap_relationships_with_llm(project, db, llm, force=True)["pairs"] == 1


def test_graph_passed_in_is_updated(project: Project) -> None:
    graph = RelationshipGraph()
    with project.open_db() as db:
        bootstrap_relationships_with_llm(
            project, db, FakeLLM({"mei|ren": GOOD}), graph=graph
        )
    assert graph.get_edge("mei", "ren") is not None


def test_parse_pair_result_variants() -> None:
    parsed = parse_pair_result(
        {"kind": "FAMILY_ELDER", "a_to_b": "con - mẹ", "b_to_a": "mẹ - con", "confidence": "x"}
    )
    assert parsed is not None
    assert parsed.a_to_b == ("con", "mẹ")
    assert parsed.confidence == pytest.approx(0.8)

    one_way = parse_pair_result({"kind": "SERVANT_MASTER", "a_to_b": {"self": "tôi", "other": "cô"}})
    assert one_way is not None
    assert one_way.b_to_a == ("", "")

    assert parse_pair_result({"kind": "ROMANCE"}) is None
    assert parse_pair_result(None) is None
    long_form = {"kind": "ROMANCE", "a_to_b": {"self": "x" * 40, "other": "em"}}
    assert parse_pair_result(long_form) is None


def test_translation_prompt_uses_bootstrapped_pair_rule(project: Project) -> None:
    """End to end: edges learned by the bootstrap reach the next translation prompt."""
    from unittest.mock import patch

    from renslate.engine.translate import TranslationPipeline

    with project.open_db() as db:
        bootstrap_relationships_with_llm(project, db, FakeLLM({"mei|ren": GOOD}))

    prompts: list[str] = []

    def fake_complete(system: str, user: str, **kw: Any) -> tuple[dict[str, str], int, int]:
        prompts.append(user)
        schema = kw.get("schema") or {}
        return ({uid: "Tạm được." for uid in schema.get("properties", {})}, 10, 10)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=fake_complete),
    ):
        TranslationPipeline(project, db, project.config).run(limit=6)

    assert prompts, "pipeline never called the LLM"
    first = prompts[0]
    assert "xưng 'tớ', gọi 'cậu'" in first  # mei -> ren
    assert "xưng 'mình', gọi 'cậu'" in first  # ren -> mei
