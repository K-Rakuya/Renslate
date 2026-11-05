"""Unit test verifying that TranslationPipeline broadcasts LiveDialogueItem to unit_cb."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from renslate.core.models import Status, Unit, UnitKind
from renslate.core.project import Project
from renslate.engine.translate import TranslationPipeline
from renslate.ui.studio import LiveDialogueItem


def test_translation_pipeline_unit_callback_broadcasting(tmp_path: Path) -> None:
    (tmp_path / "game").mkdir()
    project = Project.create(tmp_path)
    with project.open_db() as db:
        db.save_unit(
            Unit(
                uid="d:test_001",
                kind=UnitKind.DIALOGUE,
                speaker="hero",
                source="Hello world!",
                seq=1,
            )
        )

    received_items: list[LiveDialogueItem] = []

    def on_unit(item: LiveDialogueItem) -> None:
        received_items.append(item)

    with project.open_db() as db:
        pipeline = TranslationPipeline(project, db, unit_cb=on_unit)
        pipeline.llm = MagicMock()
        pipeline.llm.complete_json.return_value = ({"d:test_001": "Xin chào thế giới!"}, 10, 10)

        stats = pipeline.run(limit=1)

    assert stats.llm_translated == 1
    assert len(received_items) == 1
    item = received_items[0]
    assert item.uid == "d:test_001"
    assert item.source == "Hello world!"
    assert item.target == "Xin chào thế giới!"
    assert item.speaker == "hero"
    assert item.status == Status.TRANSLATED.value
