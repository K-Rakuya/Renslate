from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from renslate.core.models import Unit

__all__ = [
    "RetryLadder",
    "UnitFailure",
    "format_failure_feedback",
]

_MAX_CANDIDATE_CHARS = 240
_MAX_DETAIL_CHARS = 200


@dataclass(frozen=True, slots=True)
class UnitFailure:
    unit: Unit
    code: str
    detail: str = ""
    candidate: str = ""


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def format_failure_feedback(failures: Sequence[UnitFailure]) -> str:
    if not failures:
        return ""
    lines = ["LỖI Ở LẦN DỊCH TRƯỚC (bắt buộc sửa đúng các lỗi này, giữ nguyên mọi thẻ và biến):"]
    for failure in failures:
        parts = [f"- {failure.unit.uid}: {failure.code}"]
        if failure.detail:
            parts.append(_clip(failure.detail, _MAX_DETAIL_CHARS))
        if failure.candidate:
            parts.append(f'bản dịch bị loại: "{_clip(failure.candidate, _MAX_CANDIDATE_CHARS)}"')
        lines.append(" | ".join(parts))
    return "\n".join(lines)


AttemptFn = Callable[[Sequence[Unit], str], list[UnitFailure]]
LogFn = Callable[[str, int], None]


@dataclass(frozen=True, slots=True)
class RetryLadder:
    cluster_trigger: int = 3
    cluster_min: int = 2
    cluster_max: int = 6

    def cluster_size(self, failing: int) -> int:
        return max(self.cluster_min, min(self.cluster_max, failing // 2))

    def run(
        self,
        batch: Sequence[Unit],
        failures: Sequence[UnitFailure],
        attempt: AttemptFn,
        on_stage: LogFn | None = None,
    ) -> list[UnitFailure]:
        if not failures or len(batch) <= 1:
            return list(failures)

        remaining = list(failures)
        if len(remaining) > self.cluster_trigger:
            if on_stage is not None:
                on_stage("cluster", len(remaining))
            size = self.cluster_size(len(remaining))
            feedback_by_uid = {f.unit.uid: f for f in remaining}
            next_remaining: list[UnitFailure] = []
            for start in range(0, len(remaining), size):
                chunk = [f.unit for f in remaining[start : start + size]]
                feedback = format_failure_feedback([feedback_by_uid[u.uid] for u in chunk])
                next_remaining.extend(attempt(chunk, feedback))
            remaining = next_remaining

        if remaining and on_stage is not None:
            on_stage("single", len(remaining))
        final: list[UnitFailure] = []
        latest: Mapping[str, UnitFailure] = {f.unit.uid: f for f in remaining}
        for failure in remaining:
            feedback = format_failure_feedback([latest[failure.unit.uid]])
            final.extend(attempt([failure.unit], feedback))
        return final
