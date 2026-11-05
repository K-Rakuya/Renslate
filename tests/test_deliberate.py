"""Tests for confidence-gated deliberative reflection engine."""

from __future__ import annotations

from typing import Any

from renslate.core.models import Unit, UnitKind
from renslate.engine.deliberate import DeliberativeGate, DeliberativeReflector


class MockLLM:
    """Mock LLM client returning predetermined dictionary or tuple."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.call_count = 0
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


def test_deliberative_gate_selects_only_anomalous_or_complex_units() -> None:
    gate = DeliberativeGate()

    # Simple clear line -> pass through, NO reflection
    u_simple = Unit(uid="1", source="I agree.", kind=UnitKind.DIALOGUE)
    assert not gate.should_reflect(u_simple, "Tôi đồng ý.", calque_score=0.0, qa_issues=[])

    # High calque density -> trigger reflection
    u_calque = Unit(uid="2", source="I was surprised by your arrival.", kind=UnitKind.DIALOGUE)
    assert gate.should_reflect(
        u_calque,
        "Tôi đã bị làm cho ngạc nhiên bởi sự đến của bạn.",
        calque_score=0.5,
        qa_issues=[],
    )

    # Boundary calque density (0.35)
    assert gate.should_reflect(u_simple, "Bản dịch thô.", calque_score=0.35, qa_issues=[])
    assert not gate.should_reflect(u_simple, "Bản dịch ổn.", calque_score=0.34, qa_issues=[])

    # QA issues flag reflection
    assert gate.should_reflect(u_simple, "Tôi đi đây.", calque_score=0.1, qa_issues=["pronoun_mismatch"])

    # Complex rhetorical question / exclamation directed at interlocutor
    u_rhetorical = Unit(uid="3", source="Why on earth would you say that?!", kind=UnitKind.DIALOGUE)
    assert gate.should_reflect(u_rhetorical, "Sao bạn lại nói thế?!", calque_score=0.1, qa_issues=[])

    u_tag_q = Unit(uid="4", source="You don't really believe that, do you?", kind=UnitKind.DIALOGUE)
    assert gate.should_reflect(u_tag_q, "Cậu không thực sự tin điều đó, phải không?", calque_score=0.1, qa_issues=[])

    # Simple question and simple exclamation should NOT trigger reflection
    u_simple_q = Unit(uid="5", source="What time is it?", kind=UnitKind.DIALOGUE)
    assert not gate.should_reflect(u_simple_q, "Mấy giờ rồi?", calque_score=0.1, qa_issues=[])

    u_simple_excl = Unit(uid="6", source="Good morning!", kind=UnitKind.DIALOGUE)
    assert not gate.should_reflect(u_simple_excl, "Chào buổi sáng!", calque_score=0.0, qa_issues=[])


def test_deliberative_gate_filter_reflection_batch_caps_ratio() -> None:
    gate = DeliberativeGate()
    units = [
        Unit(uid=f"u_{i}", source=f"Line {i}", kind=UnitKind.DIALOGUE)
        for i in range(10)
    ]
    drafts = {u.uid: f"Bản dịch {u.uid}" for u in units}

    # 4 units flagged: u_1 with qa issue, u_2, u_3, u_4 with high calque
    calque_scores = {
        "u_0": 0.1,
        "u_1": 0.2,
        "u_2": 0.6,
        "u_3": 0.5,
        "u_4": 0.4,
        "u_5": 0.0,
    }
    qa_issues: dict[str, list[Any]] = {
        "u_1": ["pronoun_warning"],
    }

    # With max_ratio=0.25 on 10 units, capped at int(10 * 0.25) = 2 units
    filtered = gate.filter_reflection_batch(
        units=units,
        drafts=drafts,
        calque_scores=calque_scores,
        qa_issues=qa_issues,
        max_ratio=0.25,
    )
    assert len(filtered) == 2
    # Prioritizes QA issue (u_1) and highest calque (u_2)
    filtered_uids = {u.uid for u in filtered}
    assert "u_1" in filtered_uids
    assert "u_2" in filtered_uids

    # Empty units list returns empty list
    assert gate.filter_reflection_batch([], {}, {}, {}) == []

    # Units with no triggers return empty list
    clean_scores = {u.uid: 0.05 for u in units}
    assert gate.filter_reflection_batch(units, drafts, clean_scores, {}) == []


def test_deliberative_reflector_preserves_unflagged_drafts() -> None:
    reflector = DeliberativeReflector(llm=None)
    drafts = {"1": "Tôi đồng ý.", "2": "Câu dịch lỗi"}
    result = reflector.refine_batch(
        units=[Unit(uid="1", source="I agree.", kind=UnitKind.DIALOGUE)],
        drafts=drafts,
        qualified_uids=set(),
        transgraph_frame="",
    )
    assert result == drafts


def test_deliberative_reflector_builds_prompt_with_frame_and_reasons() -> None:
    reflector = DeliberativeReflector(llm=None)
    units = [
        Unit(uid="u1", source="What are you doing?!", kind=UnitKind.DIALOGUE, speaker="Aria"),
    ]
    drafts = {"u1": "Bạn đang làm gì vậy?!"}
    reasons = {"u1": ["rhetorical_complexity", "high_calque"]}
    frame = "Aria -> MC (ROMANCE, intimate)"

    prompt = reflector.build_reflection_prompt(units, drafts, reasons, transgraph_frame=frame)
    assert "Aria -> MC (ROMANCE, intimate)" in prompt
    assert "u1" in prompt
    assert "What are you doing?!" in prompt
    assert "Bạn đang làm gì vậy?!" in prompt
    assert "placeholder" in prompt.lower()


def test_deliberative_reflector_refines_batch_with_llm_verification() -> None:
    llm = MockLLM({
        "u1": "Cậu đang làm cái trò gì thế hả?!",
        "u2": "Bản dịch thiếu placeholder",  # Invalid: misses placeholder
    })
    reflector = DeliberativeReflector(llm=llm)

    u1 = Unit(uid="u1", source="What are you doing?!", kind=UnitKind.DIALOGUE, speaker="Aria")
    u2 = Unit(uid="u2", source="Look at [player]!", kind=UnitKind.DIALOGUE, speaker="Aria")
    units = [u1, u2]

    drafts = {
        "u1": "Bạn đang làm gì vậy?!",
        "u2": "Nhìn [player] kìa!",
        "u3": "Dòng không được chọn phản tư",
    }

    result = reflector.refine_batch(
        units=units,
        drafts=drafts,
        qualified_uids={"u1", "u2"},
        transgraph_frame="",
    )

    assert llm.call_count == 1
    # u1 was valid and refined
    assert result["u1"] == "Cậu đang làm cái trò gì thế hả?!"
    # u2 failed validation (missing [player] placeholder) so draft was kept
    assert result["u2"] == "Nhìn [player] kìa!"
    # u3 was not in qualified_uids so kept
    assert result["u3"] == "Dòng không được chọn phản tư"


def test_deliberative_reflector_handles_tuple_response_from_llm() -> None:
    # LLMClient.complete_json typically returns (dict, prompt_tokens, completion_tokens)
    llm = MockLLM(({"u1": "Em hiểu rồi."}, 50, 20))
    reflector = DeliberativeReflector(llm=llm)

    u1 = Unit(uid="u1", source="I understand.", kind=UnitKind.DIALOGUE)
    result = reflector.refine_batch(
        units=[u1],
        drafts={"u1": "Tôi hiểu."},
        qualified_uids={"u1"},
    )
    assert result["u1"] == "Em hiểu rồi."


def test_deliberative_reflector_handles_llm_exception_gracefully() -> None:
    class FailingLLM:
        def complete_json(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("LLM connection timeout")

    reflector = DeliberativeReflector(llm=FailingLLM())
    u1 = Unit(uid="u1", source="Hello", kind=UnitKind.DIALOGUE)
    drafts = {"u1": "Xin chào"}
    result = reflector.refine_batch(
        units=[u1],
        drafts=drafts,
        qualified_uids={"u1"},
    )
    assert result == drafts


def test_deliberative_reflector_handles_prefixed_uid_keys() -> None:
    llm = MockLLM({"123": "Chào buổi tối."})
    reflector = DeliberativeReflector(llm=llm)

    u1 = Unit(uid="d:123", source="Good evening.", kind=UnitKind.DIALOGUE)
    result = reflector.refine_batch(
        units=[u1],
        drafts={"d:123": "Chào."},
        qualified_uids={"d:123"},
    )
    assert result["d:123"] == "Chào buổi tối."


def test_deliberative_gate_vietnamese_rhetorical_in_draft() -> None:
    gate = DeliberativeGate()
    u = Unit(uid="u1", source="You already knew this.", kind=UnitKind.DIALOGUE, speaker="Aria")
    assert gate.should_reflect(u, "Chẳng phải cậu đã biết chuyện này rồi sao?", calque_score=0.1, qa_issues=[])


def test_deliberative_gate_narration_does_not_trigger_rhetorical() -> None:
    gate = DeliberativeGate()
    u_narration = Unit(uid="u1", source="Why on earth...?", kind=UnitKind.STRING, speaker=None)
    assert not gate.should_reflect(u_narration, "Tại sao chứ...?", calque_score=0.1, qa_issues=[])

