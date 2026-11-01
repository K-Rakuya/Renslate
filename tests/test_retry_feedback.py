from __future__ import annotations

from collections.abc import Sequence

from renslate.core.models import Status, Unit, UnitKind
from renslate.engine.repair import RetryLadder, UnitFailure, format_failure_feedback


def unit(i: int) -> Unit:
    return Unit(f"d:{i}", UnitKind.DIALOGUE, i, f"Line {i}", Status.PENDING)


def failure(i: int, code: str = "qa_error", candidate: str = "") -> UnitFailure:
    return UnitFailure(unit(i), code, f"detail {i}", candidate)


def test_feedback_lists_each_failure_with_code_detail_and_rejected_output() -> None:
    text = format_failure_feedback([failure(1, "raw_markup", "sai thẻ"), failure(2)])
    assert "d:1: raw_markup | detail 1" in text
    assert 'bản dịch bị loại: "sai thẻ"' in text
    assert "d:2: qa_error" in text
    assert format_failure_feedback([]) == ""


def test_feedback_clips_long_candidates() -> None:
    text = format_failure_feedback([failure(1, candidate="x " * 400)])
    assert text.count("x") < 200 and text.endswith('…"')


def run_ladder(
    n_failed: int, passing_after: str
) -> tuple[list[tuple[list[str], str]], tuple[list[UnitFailure], list[tuple[str, int]]]]:
    batch = [unit(i) for i in range(n_failed + 2)]
    failures = [failure(i, candidate=f"bad{i}") for i in range(n_failed)]
    calls: list[tuple[list[str], str]] = []

    def attempt(units: Sequence[Unit], feedback: str) -> list[UnitFailure]:
        calls.append(([u.uid for u in units], feedback))
        if passing_after == "cluster" and len(units) > 1:
            return []
        return [failure(int(u.uid[2:]), candidate="again") for u in units]

    stages: list[tuple[str, int]] = []
    result = RetryLadder().run(batch, failures, attempt, lambda s, c: stages.append((s, c)))
    return calls, (result, stages)


def test_many_failures_go_through_clusters_then_stop_when_fixed() -> None:
    calls, (result, stages) = run_ladder(8, "cluster")
    assert result == []
    assert stages == [("cluster", 8)]
    assert [len(c[0]) for c in calls] == [4, 4]
    assert all('bản dịch bị loại: "bad' in fb for _, fb in calls)


def test_stubborn_failures_escalate_to_single_with_latest_feedback() -> None:
    calls, (result, stages) = run_ladder(5, "never")
    assert [s for s, _ in stages] == ["cluster", "single"]
    assert [len(c[0]) for c in calls[:3]] == [2, 2, 1]
    singles = calls[3:]
    assert len(singles) == 5 and all(len(c[0]) == 1 for c in singles)
    assert all('"again"' in fb for _, fb in singles)
    assert len(result) == 5


def test_few_failures_skip_cluster_stage() -> None:
    calls, (result, stages) = run_ladder(2, "never")
    assert [s for s, _ in stages] == ["single"]
    assert len(calls) == 2


def test_single_unit_batch_is_not_retried() -> None:
    called: list[int] = []

    def _dummy(u: Sequence[Unit], f: str) -> list[UnitFailure]:
        called.append(1)
        return []

    out = RetryLadder().run([unit(1)], [failure(1)], _dummy, None)
    assert called == [] and len(out) == 1
