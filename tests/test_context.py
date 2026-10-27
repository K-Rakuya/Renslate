"""Tests for vitrans.context."""

from __future__ import annotations

from vitrans.config import Config
from vitrans.context import ContextBuilder, LookbackItem, estimate_tokens
from vitrans.models import CharacterProfile, GlossaryEntry, Status, Unit, UnitKind


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
