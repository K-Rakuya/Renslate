"""TM reuse must not freeze speaker/listener-dependent pronouns."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from renslate.core.models import Origin, Status, TMEntry, Unit, UnitKind
from renslate.core.project import Project
from renslate.engine.tm import TranslationMemory, is_context_sensitive_line
from renslate.engine.translate import TranslationPipeline


def _unit(source: str, *, speaker: str | None, kind: UnitKind = UnitKind.DIALOGUE) -> Unit:
    return Unit("d:x", kind, 1, source, Status.PENDING, speaker=speaker)


@pytest.mark.parametrize(
    ("source", "speaker", "kind", "expected"),
    [
        ("Thank you, I owe you one.", "alice", UnitKind.DIALOGUE, True),
        ("Are you alright, sister?", "bob", UnitKind.DIALOGUE, True),
        ("{i}You{/i} again?", "bob", UnitKind.DIALOGUE, True),
        ("Yes.", "alice", UnitKind.DIALOGUE, False),
        ("The wind howls.", "alice", UnitKind.DIALOGUE, False),
        # Narration / thoughts have no addressee.
        ("I walked home alone.", None, UnitKind.DIALOGUE, False),
        ("I wonder what you think.", "narrator", UnitKind.DIALOGUE, False),
        ("I should go.", "th", UnitKind.DIALOGUE, False),
        # UI / strings are always context-free.
        ("Are you sure you want to quit?", None, UnitKind.UI, False),
        ("Your name", None, UnitKind.STRING, False),
        # Variables and tags must not count as pronouns.
        ("Hello [you_name]!", "alice", UnitKind.DIALOGUE, False),
    ],
)
def test_is_context_sensitive_line(
    source: str, speaker: str | None, kind: UnitKind, expected: bool
) -> None:
    assert is_context_sensitive_line(_unit(source, speaker=speaker, kind=kind)) is expected


def _project(tmp_path: Path) -> Project:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    return Project.create(root)


def test_pipeline_retranslates_context_sensitive_machine_tm_hit(tmp_path: Path) -> None:
    project = _project(tmp_path)
    with project.open_db() as db:
        db.save_tm(TMEntry("Thank you, I owe you.", "Cảm ơn anh.", origin=Origin.LLM))
        db.save_unit(
            Unit(
                "d:1", UnitKind.DIALOGUE, 1, "Thank you, I owe you.", Status.PENDING,
                speaker="alice",
            )
        )

    with (
        project.open_db() as db,
        patch(
            "renslate.llm.LLMClient.complete_json",
            return_value=({"d:1": "Cảm ơn ngài, tôi nợ ngài."}, 20, 10),
        ) as llm,
    ):
        stats = TranslationPipeline(project, db, project.config).run()
        unit = db.get_unit("d:1")

    assert llm.call_count == 1
    assert stats.tm_reused == 0
    assert stats.tm_context_skipped == 1
    assert stats.llm_translated == 1
    assert unit is not None
    assert unit.target == "Cảm ơn ngài, tôi nợ ngài."
    assert unit.origin is Origin.LLM


def test_pipeline_still_reuses_context_free_and_manual_entries(tmp_path: Path) -> None:
    project = _project(tmp_path)
    with project.open_db() as db:
        db.save_tm(TMEntry("The wind howls.", "Gió rít.", origin=Origin.LLM))
        db.save_tm(TMEntry("Thank you, I owe you.", "Cảm ơn bạn.", origin=Origin.MANUAL))
        db.save_unit(
            Unit("d:1", UnitKind.DIALOGUE, 1, "The wind howls.", Status.PENDING, speaker="alice")
        )
        db.save_unit(
            Unit(
                "d:2", UnitKind.DIALOGUE, 2, "Thank you, I owe you.", Status.PENDING,
                speaker="alice",
            )
        )

    def _no_llm(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("LLM must not be called")

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=_no_llm),
    ):
        stats = TranslationPipeline(project, db, project.config).run()
        u2 = db.get_unit("d:2")

    assert stats.tm_reused == 2
    assert stats.tm_context_skipped == 0
    assert u2 is not None
    assert u2.target == "Cảm ơn bạn."


def test_record_never_overwrites_locked_entry_and_counts_uses(tmp_path: Path) -> None:
    project = _project(tmp_path)
    with project.open_db() as db:
        db.save_tm(TMEntry("You are late.", "Cậu trễ rồi.", origin=Origin.MANUAL))
        tm = TranslationMemory(db)
        tm.record("You are late.", "Anh đến muộn.", origin=Origin.LLM)
        entry = tm.lookup_exact_entry("You are late.")
        assert entry is not None
        assert entry.target == "Cậu trễ rồi."
        assert entry.origin is Origin.MANUAL

        tm.record("Fine day.", "Ngày đẹp.", origin=Origin.LLM)
        tm.record("Fine day.", "Ngày đẹp trời.", origin=Origin.LLM)
        again = tm.lookup_exact_entry("Fine day.")
        assert again is not None
        assert again.uses == 2
        assert again.target == "Ngày đẹp trời."
