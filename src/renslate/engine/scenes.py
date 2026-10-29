"""LLM-written scene summaries that feed the translation prompt.

``segment_scenes`` only knows where a scene starts and ends; its "summary" is just
``"Cảnh 'label' (từ dòng a đến b)"`` and location/mood come from keywords in the Ren'Py
label. A translator model cannot resolve "he", "it", "that" or choose a register without
knowing what is going on. This module reads each scene's dialogue once, asks the local LLM
for a short factual summary plus location/atmosphere/time, and stores it so every later
batch of that scene (and the one after it) is translated with that context.

Work is lazy and resumable: scenes are summarised just before they are translated
(``seq_range``), each result is committed on its own, and a scene already summarised is
skipped unless ``force`` is set.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from renslate.core.db import Database
from renslate.core.models import (
    CharacterProfile,
    SceneMacroContext,
    SceneSummary,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.context import estimate_tokens

_LOG = logging.getLogger("renslate.scenes")

__all__ = [
    "SCENE_SYSTEM_PROMPT",
    "build_scene_prompt",
    "load_scene_contexts",
    "parse_scene_result",
    "summarize_scenes_with_llm",
]

_CTX_PREFIX = "scene_ctx:"
_TAG_OR_VAR = re.compile(r"\{[^{}]*\}|\[[^\[\]]*\]|⟦[^⟦⟧]*⟧")
_WS = re.compile(r"\s+")
_NON_SPEECH = frozenset({"", "t", "th", "thought", "thoughts", "thinking", "none", "all", "ui"})
_MAX_SUMMARY = 420
_MAX_FIELD = 90

SCENE_SYSTEM_PROMPT = (
    "Bạn là biên tập viên visual novel. Đọc các dòng thoại của MỘT phân cảnh và viết bối cảnh "
    "ngắn gọn để người dịch hiểu chuyện gì đang diễn ra.\n"
    "Trả về DUY NHẤT một đối tượng JSON gồm:\n"
    '- "summary": 2-3 câu tiếng Việt: ai đang làm gì, mâu thuẫn hoặc tâm trạng chính, '
    "điều quan trọng mỗi người đã biết. Chỉ dùng thông tin có trong đoạn thoại, không bịa.\n"
    '- "location": địa điểm ngắn gọn bằng tiếng Việt, để chuỗi rỗng nếu không rõ.\n'
    '- "atmosphere": cảm xúc/không khí chủ đạo, vài từ.\n'
    '- "time": thời điểm trong ngày/thời gian nếu có manh mối, nếu không thì chuỗi rỗng.\n'
    "Không giải thích ngoài JSON."
)


def _clean(text: str, limit: int) -> str:
    cleaned = _WS.sub(" ", _TAG_OR_VAR.sub(" ", text)).strip()
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def _fit_excerpt(
    lines: Sequence[tuple[str, str]], budget_tokens: int
) -> list[tuple[str, str]]:
    """Keep the scene's lines whole when they fit; otherwise sample evenly (first/last kept)."""
    costs = [estimate_tokens(f"[{s}] {t}") + 2 for s, t in lines]
    total = sum(costs)
    if total <= budget_tokens or len(lines) <= 6:
        return list(lines)
    keep = max(6, int(len(lines) * budget_tokens / total))
    step = (len(lines) - 1) / (keep - 1)
    idx = sorted({round(i * step) for i in range(keep)})
    return [lines[i] for i in idx]


def build_scene_prompt(
    scene: SceneSummary,
    lines: Sequence[tuple[str, str]],
    previous_summaries: Sequence[str] = (),
    *,
    budget_tokens: int = 1400,
) -> str:
    """User prompt for one scene: previous context, then its (possibly sampled) lines."""
    parts: list[str] = []
    if previous_summaries:
        parts.append("Diễn biến ngay trước đó:\n" + "\n".join(f"- {s}" for s in previous_summaries))
    chosen = _fit_excerpt(lines, budget_tokens)
    note = "" if len(chosen) == len(lines) else f", trích {len(chosen)} dòng tiêu biểu"
    body = "\n".join(f"[{spk}] {text}" for spk, text in chosen)
    parts.append(f"Phân cảnh '{scene.label}' ({len(lines)} dòng thoại{note}):\n{body}")
    return "\n\n".join(parts)


def parse_scene_result(data: Any) -> dict[str, str] | None:
    """Validate the model's answer; return cleaned fields or None when unusable."""
    if not isinstance(data, Mapping):
        return None
    summary = _clean(str(data.get("summary") or ""), _MAX_SUMMARY)
    if len(summary) < 10:
        return None
    return {
        "summary": summary,
        "location": _clean(str(data.get("location") or ""), _MAX_FIELD),
        "atmosphere": _clean(str(data.get("atmosphere") or ""), _MAX_FIELD),
        "time": _clean(str(data.get("time") or ""), _MAX_FIELD),
    }


def _scene_lines(
    units: Sequence[Unit],
    seqs: Sequence[int],
    scene: SceneSummary,
    characters: Mapping[str, CharacterProfile],
) -> list[tuple[str, str]]:
    lo = bisect_left(seqs, scene.first_seq)
    hi = bisect_right(seqs, scene.last_seq)
    out: list[tuple[str, str]] = []
    for u in units[lo:hi]:
        if u.kind is not UnitKind.DIALOGUE:
            continue
        text = _clean(u.source, 200)
        if not text:
            continue
        tag = (u.speaker or "").strip()
        if tag.lower() in _NON_SPEECH and tag:
            speaker = "Suy nghĩ"
        elif not tag:
            speaker = "Dẫn truyện"
        else:
            prof = characters.get(tag)
            speaker = prof.display_name if prof and prof.display_name else tag
        out.append((speaker, text))
    return out


def _load_ctx(db: Database, scene: SceneSummary) -> dict[str, Any] | None:
    raw = db.get_meta(_CTX_PREFIX + scene.scene_id)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    # Scene ids are positional; ignore a stored summary if the scene moved or changed.
    if data.get("first_seq") != scene.first_seq or data.get("label") != scene.label:
        return None
    return data


def summarize_scenes_with_llm(
    project: Project,
    db: Database,
    llm: Any,
    *,
    seq_range: tuple[int, int] | None = None,
    min_lines: int = 8,
    max_scenes: int | None = None,
    force: bool = False,
    budget_tokens: int = 1400,
) -> dict[str, int]:
    """Summarise scenes that lack an LLM summary. Returns counters.

    ``seq_range`` limits work to scenes overlapping ``(first_seq, last_seq)`` (the units about
    to be translated). Scenes shorter than ``min_lines`` dialogue lines are skipped: they
    carry little context and are not worth a model call.
    """
    stats = {"scenes": 0, "summarized": 0, "skipped": 0}
    scenes = sorted(db.get_scenes(), key=lambda s: s.first_seq)
    if not scenes:
        return stats

    units = sorted(db.all_units(), key=lambda u: u.seq)
    seqs = [u.seq for u in units]
    characters = db.get_characters()
    done_summaries: dict[str, str] = {}
    for sc in scenes:
        ctx = _load_ctx(db, sc)
        if ctx and ctx.get("summary"):
            done_summaries[sc.scene_id] = str(ctx["summary"])

    summarized_now = 0
    for pos, scene in enumerate(scenes):
        if seq_range is not None and (
            scene.last_seq < seq_range[0] or scene.first_seq > seq_range[1]
        ):
            continue
        stats["scenes"] += 1
        if scene.scene_id in done_summaries and not force:
            continue
        if max_scenes is not None and summarized_now >= max_scenes:
            break

        lines = _scene_lines(units, seqs, scene, characters)
        if sum(1 for spk, _ in lines if spk != "Dẫn truyện") < min_lines:
            stats["skipped"] += 1
            continue

        previous = [
            done_summaries[s.scene_id]
            for s in scenes[max(0, pos - 2) : pos]
            if s.scene_id in done_summaries
        ]
        prompt = build_scene_prompt(scene, lines, previous, budget_tokens=budget_tokens)
        try:
            result, _, _ = llm.complete_json(SCENE_SYSTEM_PROMPT, prompt, max_tokens=512)
        except Exception as exc:
            _LOG.warning("Tóm tắt phân cảnh %s thất bại: %s", scene.scene_id, exc)
            continue
        parsed = parse_scene_result(result)
        if parsed is None:
            _LOG.info("Phân cảnh %s: phản hồi LLM không hợp lệ, bỏ qua.", scene.scene_id)
            continue

        payload = {
            **parsed,
            "label": scene.label,
            "first_seq": scene.first_seq,
            "last_seq": scene.last_seq,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with db.transaction():
            db.save_scene(dataclasses.replace(scene, summary=parsed["summary"]))
            db.set_meta(_CTX_PREFIX + scene.scene_id, json.dumps(payload, ensure_ascii=False))
        done_summaries[scene.scene_id] = parsed["summary"]
        stats["summarized"] += 1
        summarized_now += 1
    return stats


def load_scene_contexts(
    db: Database, scenes: Sequence[SceneSummary]
) -> dict[str, SceneMacroContext]:
    """Rebuild the rich per-scene context (summary, location, mood, previous scene) in memory.

    Pure in-memory result so the pipeline's prefetch thread never touches SQLite.
    """
    ordered = sorted(scenes, key=lambda s: s.first_seq)
    loaded: dict[str, dict[str, Any]] = {}
    for sc in ordered:
        ctx = _load_ctx(db, sc)
        if ctx and ctx.get("summary"):
            loaded[sc.scene_id] = ctx

    result: dict[str, SceneMacroContext] = {}
    previous_summary = ""
    for sc in ordered:
        ctx = loaded.get(sc.scene_id)
        if ctx is not None:
            metadata: dict[str, Any] = {}
            if previous_summary:
                metadata["previous_summary"] = previous_summary
            result[sc.scene_id] = SceneMacroContext(
                scene_id=sc.scene_id,
                label=sc.label,
                summary=str(ctx["summary"]),
                location=str(ctx.get("location") or ""),
                atmosphere=str(ctx.get("atmosphere") or ""),
                temporal_anchor=str(ctx.get("time") or ""),
                first_seq=sc.first_seq,
                last_seq=sc.last_seq,
                metadata=metadata,
            )
            previous_summary = str(ctx["summary"])
        else:
            previous_summary = ""
    return result
