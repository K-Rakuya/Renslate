"""Confidence-gated deliberative reflection engine for high-touch literary dialogue."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from renslate.core.models import Unit, UnitKind
from renslate.renpy.mask import reconcile_translation

__all__ = ["DeliberativeGate", "DeliberativeReflector"]

_RHETORICAL_PUNCT_RE = re.compile(r"(?:\?{2,}|!{2,}|\?!|!\?|\.{2,}\?|\.{2,}!|…\?|…!)")
_RHETORICAL_TAG_Q_RE = re.compile(
    r",\s*(?:right|huh|eh|okay|ok)\s*\?"
    r"|,\s*(?:is|isn't|are|aren't|do|don't|did|didn't|does|doesn't|can|can't|could|couldn't|will|won't|would|wouldn't|should|shouldn't|have|haven't|has|hasn't)\s+(?:it|you|we|they|he|she|i)\s*\?",
    re.IGNORECASE,
)
_RHETORICAL_SUBTEXT_RE = re.compile(
    r"\b(?:"
    r"how could you|how dare you|who do you think|what makes you think|"
    r"why on earth|what on earth|how on earth|where on earth|who on earth|"
    r"what the (?:hell|heck)|why the (?:hell|heck)|how the (?:hell|heck)|who the (?:hell|heck)|"
    r"are you kidding|are you serious|are you insane|are you out of your mind|"
    r"don't tell me|as if\b|"
    r"give me a break|you've got to be kidding"
    r")\b",
    re.IGNORECASE,
)
_VI_RHETORICAL_RE = re.compile(
    r"\b(?:"
    r"chẳng phải|không phải|sao lại|lẽ nào|thế quái nào|"
    r"đùa à|đùa đấy à|làm cái trò gì"
    r")\b",
    re.IGNORECASE,
)


def _has_rhetorical_complexity(unit: Unit, draft_text: str) -> bool:
    if unit.kind != UnitKind.DIALOGUE and not unit.speaker:
        return False
    source = unit.source.strip()
    if not source:
        return False
    if _RHETORICAL_PUNCT_RE.search(source):
        return True
    if _RHETORICAL_TAG_Q_RE.search(source):
        return True
    if _RHETORICAL_SUBTEXT_RE.search(source):
        return True
    return bool(draft_text and _VI_RHETORICAL_RE.search(draft_text) and ("?" in draft_text or "!" in draft_text))


class DeliberativeGate:
    """Evaluates completed draft translations to selectively identify units needing reflection."""

    def __init__(self, calque_threshold: float = 0.35) -> None:
        self.calque_threshold = calque_threshold

    def should_reflect(
        self,
        unit: Unit,
        draft_text: str,
        calque_score: float = 0.0,
        qa_issues: Sequence[Any] | None = None,
    ) -> bool:
        if calque_score >= self.calque_threshold:
            return True
        if qa_issues:
            for issue in qa_issues:
                code = getattr(issue, "code", str(issue)).lower()
                if "ratio" in code or "length" in code:
                    continue
                return True
        return bool(_has_rhetorical_complexity(unit, draft_text))

    def filter_reflection_batch(
        self,
        units: Sequence[Unit],
        drafts: dict[str, str],
        calque_scores: dict[str, float],
        qa_issues: dict[str, list[Any]],
        max_ratio: float = 0.25,
    ) -> list[Unit]:
        if not units:
            return []

        candidates: list[tuple[float, int, Unit]] = []
        for idx, unit in enumerate(units):
            d_text = drafts.get(unit.uid, "")
            c_score = calque_scores.get(unit.uid, 0.0)
            issues = qa_issues.get(unit.uid, [])
            if self.should_reflect(unit, d_text, c_score, issues):
                priority = (10.0 if issues else 0.0) + c_score
                candidates.append((priority, idx, unit))

        if not candidates:
            return []

        max_count = max(1, int(len(units) * max_ratio))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[:max_count]
        selected.sort(key=lambda item: item[1])
        return [item[2] for item in selected]


class DeliberativeReflector:
    """Refines flagged draft translations through focused System-2 reflection."""

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm

    def build_reflection_prompt(
        self,
        units: Sequence[Unit],
        drafts: dict[str, str],
        reasons: dict[str, list[str]],
        transgraph_frame: str = "",
    ) -> str:
        prompt_parts: list[str] = [
            "Bạn là biên dịch viên văn học cao cấp chuyên về Visual Novel.",
            "Hãy tinh chỉnh các câu thoại bản nháp sau đây để xóa bỏ hoàn toàn văn phong dịch gượng gạo, "
            "chuyển hóa thành lời thoại tiếng Việt tự nhiên, giữ nguyên 100% placeholder.",
            "",
            "### QUY TẮC BẮT BUỘC:",
            "1. Giữ nguyên 100% tất cả placeholder (⟦V...⟧, [name], {i},...) đúng cú pháp.",
            "2. Tuân thủ tuyệt đối mối quan hệ nhân vật và đại từ xưng hô đã thiết lập.",
            "3. Trả về định dạng JSON thuần túy: {\"uid\": \"câu dịch tiếng Việt tự nhiên đã tinh chỉnh\"}.",
        ]

        if transgraph_frame.strip():
            prompt_parts.extend([
                "",
                "### KHUNG TƯƠNG TÁC QUAN HỆ NHÂN VẬT (TRANSGRAPH FRAME):",
                transgraph_frame.strip(),
            ])

        prompt_parts.extend([
            "",
            "### CÁC DÒNG CẦN TINH CHỈNH:",
        ])

        for u in units:
            draft = drafts.get(u.uid, "")
            u_reasons = reasons.get(u.uid, [])
            reason_desc = ", ".join(u_reasons) if u_reasons else "Cần tinh chỉnh văn phong tự nhiên"
            speaker_str = f" [{u.speaker}]" if u.speaker else ""
            prompt_parts.append(
                f"- UID: {u.uid}{speaker_str}\n  Gốc: {u.source}\n  Bản nháp: {draft}\n  Lý do: {reason_desc}"
            )

        return "\n".join(prompt_parts)

    def refine_batch(
        self,
        units: Sequence[Unit],
        drafts: dict[str, str],
        qualified_uids: set[str],
        transgraph_frame: str = "",
    ) -> dict[str, str]:
        if not qualified_uids or self.llm is None:
            return dict(drafts)

        reflect_units = [u for u in units if u.uid in qualified_uids]
        if not reflect_units:
            return dict(drafts)

        reasons: dict[str, list[str]] = {u.uid: ["deliberative_refinement"] for u in reflect_units}
        user_prompt = self.build_reflection_prompt(
            reflect_units, drafts, reasons, transgraph_frame=transgraph_frame
        )
        system_prompt = (
            "You are a professional literary editor refining Vietnamese dialogue. "
            "Return a valid JSON object mapping UID to refined Vietnamese translation."
        )

        try:
            resp = self.llm.complete_json(system_prompt, user_prompt)
            if isinstance(resp, tuple):
                result_map = resp[0]
            elif isinstance(resp, dict):
                result_map = resp
            else:
                result_map = {}
        except Exception:
            return dict(drafts)

        if not isinstance(result_map, dict):
            return dict(drafts)

        refined = dict(drafts)
        for u in reflect_units:
            cand = result_map.get(u.uid)
            if cand is None and ":" in u.uid:
                cand = result_map.get(u.uid.split(":", 1)[1])
            if not isinstance(cand, str) or not cand.strip():
                continue

            cleaned, issues = reconcile_translation(u.source, cand)
            if not issues and cleaned.strip():
                refined[u.uid] = cleaned

        return refined
