"""Tests for TransGraph dyad-aware dynamic TM exemplar retrieval and prompt injection."""

from __future__ import annotations

from pathlib import Path

from renslate.core.db import Database
from renslate.core.models import Origin, Status, Unit, UnitKind
from renslate.engine.context.builder import ContextBuilder, ContextBundle
from renslate.engine.context.exemplars import (
    ExemplarItem,
    TMDyadExemplarRetriever,
    format_exemplars_section,
)


def test_exemplar_item_dataclass_properties() -> None:
    item = ExemplarItem(source="Hello", target="Xin chào")
    assert item.source == "Hello"
    assert item.target == "Xin chào"
    assert item.similarity == 1.0

    custom = ExemplarItem(source="Goodbye", target="Tạm biệt", similarity=0.85)
    assert custom.similarity == 0.85
    assert hasattr(custom, "__slots__")


def test_tm_dyad_exemplar_retriever_pulls_similar_verified_pairs(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with Database.open(db_path, create=True) as db:
        db.add_tm("What are you doing here?", "Cậu làm gì ở đây thế?", origin=Origin.USER)
        db.add_tm("I don't understand.", "Tôi không hiểu.", origin=Origin.USER)

        retriever = TMDyadExemplarRetriever(db)
        exemplars = retriever.retrieve_for_dyad(speaker="c", listener="mc", sources=["What are you doing?"])
        assert len(exemplars) >= 1
        assert exemplars[0].target == "Cậu làm gì ở đây thế?"


def test_format_exemplars_section_empty_and_populated() -> None:
    assert format_exemplars_section([]) == ""
    assert TMDyadExemplarRetriever.format_exemplars_section([]) == ""

    items = [
        ExemplarItem(source="Wait!", target="Đợi đã!"),
        ExemplarItem(source="Let's go.", target="Đi thôi."),
    ]
    formatted = TMDyadExemplarRetriever.format_exemplars_section(items)
    assert "### CÂU MẪU ĐÃ ĐƯỢC XÁC NHẬN TRONG DỰ ÁN (PROJECT-VERIFIED EXEMPLARS):" in formatted
    assert "- Wait! => Đợi đã!" in formatted
    assert "- Let's go. => Đi thôi." in formatted


def test_context_builder_injects_tm_exemplars() -> None:
    builder = ContextBuilder()
    prompt = builder.build_system_prompt(exemplars=[ExemplarItem(source="Wait!", target="Đợi đã!")])
    assert "CÂU MẪU ĐÃ ĐƯỢC XÁC NHẬN" in prompt
    assert "Wait! => Đợi đã!" in prompt


def test_exemplar_retriever_filters_unverified_single_use_llm_drafts(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with Database.open(db_path, create=True) as db:
        db.add_tm("What are you doing?", "Mày đang làm gì?", origin=Origin.LLM, uses=1)

        retriever = TMDyadExemplarRetriever(db)
        exemplars = retriever.retrieve_for_dyad(speaker="c", listener="mc", sources=["What are you doing?"])
        assert exemplars == []


def test_exemplar_retriever_accepts_high_reuse_count(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with Database.open(db_path, create=True) as db:
        db.add_tm("What are you doing?", "Cậu đang làm gì đấy?", origin=Origin.LLM, uses=3)

        retriever = TMDyadExemplarRetriever(db)
        exemplars = retriever.retrieve_for_dyad(speaker="c", listener="mc", sources=["What are you doing?"])
        assert len(exemplars) == 1
        assert exemplars[0].target == "Cậu đang làm gì đấy?"


def test_exemplar_retriever_respects_max_results(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with Database.open(db_path, create=True) as db:
        db.add_tm("Where are you going today?", "Hôm nay cậu đi đâu?", origin=Origin.USER)
        db.add_tm("Where are you going now?", "Bây giờ cậu đi đâu?", origin=Origin.USER)
        db.add_tm("Where are you going tomorrow?", "Ngày mai cậu đi đâu?", origin=Origin.USER)

        retriever = TMDyadExemplarRetriever(db)
        exemplars = retriever.retrieve_for_dyad(
            speaker="c",
            listener="mc",
            sources=["Where are you going?"],
            max_results=2,
        )
        assert len(exemplars) == 2


def test_exemplar_retriever_empty_sources_and_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with Database.open(db_path, create=True) as db:
        retriever = TMDyadExemplarRetriever(db)
        assert retriever.retrieve_for_dyad(speaker="c", listener="mc", sources=[]) == []
        assert retriever.retrieve_for_dyad(speaker="c", listener="mc", sources=["Hello"]) == []


def test_exemplar_retriever_speaker_history_fallback(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with Database.open(db_path, create=True) as db:
        unit = Unit(
            uid="u1",
            kind=UnitKind.DIALOGUE,
            seq=1,
            speaker="heroine",
            source="Don't worry about me.",
            target="Đừng lo cho em.",
            status=Status.REVIEWED,
            origin=Origin.MANUAL,
        )
        db.save_unit(unit)

        retriever = TMDyadExemplarRetriever(db)
        exemplars = retriever.retrieve_for_dyad(
            speaker="heroine",
            listener="mc",
            sources=["Don't worry about it."],
        )
        assert len(exemplars) >= 1
        assert exemplars[0].target == "Đừng lo cho em."


def test_context_bundle_with_exemplars() -> None:
    bundle = ContextBundle(
        exemplars=[ExemplarItem(source="Look out!", target="Cẩn thận!")]
    )
    assert len(bundle.exemplars) == 1
    assert bundle.exemplars[0].source == "Look out!"
