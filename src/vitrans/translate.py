"""Core translation pipeline orchestrator for vitrans."""

from __future__ import annotations

import logging
import uuid
from collections import deque
from collections.abc import Callable, MutableSequence, Sequence
from dataclasses import dataclass, field
from typing import Any

from vitrans.builtin_ui import get_builtin_ui_translation
from vitrans.config import Config
from vitrans.context import ContextBuilder, LookbackItem, estimate_tokens
from vitrans.db import Database, utc_timestamp
from vitrans.errors import LLMError
from vitrans.glossary import Glossary
from vitrans.llm import LLMClient
from vitrans.mask import auto_repair_placeholders, mask, unmask, validate_raw
from vitrans.models import BatchLog, Origin, Status, Unit
from vitrans.project import Project
from vitrans.qa import _CJK_PATTERN, QAChecker, clean_translation, repair_glossary_leakage
from vitrans.tm import TranslationMemory

_LOG = logging.getLogger("vitrans.translate")

ProgressCallback = Callable[[int, int, str], None]


@dataclass(slots=True)
class TranslationStats:
    total_pending: int = 0
    seeded: int = 0
    tm_reused: int = 0
    llm_translated: int = 0
    failed: int = 0
    batches_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    errors: list[str] = field(default_factory=list)


def partition_batches(
    units: Sequence[Unit], max_lines: int = 25, max_tokens: int = 700
) -> list[list[Unit]]:
    """Partition units into batches by scene/label continuity and token capacity."""
    batches: list[list[Unit]] = []
    current_batch: list[Unit] = []
    current_tokens = 0
    current_label: str | None = None

    for unit in units:
        unit_tokens = estimate_tokens(unit.masked_source or unit.source)
        label_changed = current_label is not None and unit.label != current_label
        kind_changed = bool(current_batch and current_batch[-1].kind != unit.kind)

        if current_batch and (
            len(current_batch) >= max_lines
            or current_tokens + unit_tokens > max_tokens
            or kind_changed
            or (label_changed and len(current_batch) >= 10)
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(unit)
        current_tokens += unit_tokens
        current_label = unit.label

    if current_batch:
        batches.append(current_batch)

    return batches


class TranslationPipeline:
    """Manages end-to-end translation using built-in seeds, TM, and LLM batches."""

    def __init__(self, project: Project, db: Database, config: Config) -> None:
        self.project = project
        self.db = db
        self.config = config
        self.tm = TranslationMemory(db)
        self.glossary = Glossary.load_db(db)
        # Also check project root glossary.json if present
        gl_file = project.work_dir / "glossary.json"
        if gl_file.is_file():
            file_glossary = Glossary.load_file(gl_file)
            for e in file_glossary.entries:
                self.glossary.add(e)

        self.characters = db.get_characters()
        self.qa = QAChecker(max_length_ratio=config.qa.max_length_ratio)
        self.llm = LLMClient(config.llm)

        # Custom style guide if present
        style_guide = ""
        style_path = project.work_dir / "style.md"
        if style_path.is_file():
            style_guide = style_path.read_text(encoding="utf-8")
        self.context = ContextBuilder(config, custom_style_guide=style_guide)

    def run(
        self,
        *,
        limit: int | None = None,
        include_failed: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> TranslationStats:
        """Execute the translation pipeline over pending units."""
        stats = TranslationStats()
        pending = self.db.pending_units(limit=limit, include_failed=include_failed)
        stats.total_pending = len(pending)

        if not pending:
            return stats

        need_llm: list[Unit] = []
        lookback_capacity = self.config.batch.lookback * 4
        lookback_buffer: deque[LookbackItem] = deque(maxlen=lookback_capacity)

        if pending:
            first_seq = pending[0].seq
            recent_units = self.db.recent_translated_units(
                limit=lookback_capacity, before_seq=first_seq
            )
            for ru in recent_units:
                if ru.target:
                    lookback_buffer.append(LookbackItem(ru.speaker, ru.source, ru.target))

        # ── Step 1: Pre-process with Built-in Seeds and Exact TM ────────
        with self.db.transaction():
            for u in pending:
                # 1. Built-in UI seed
                seed = get_builtin_ui_translation(u.source)
                if seed is not None:
                    self.db.update_translation(
                        uid=u.uid,
                        target=seed,
                        origin=Origin.SEED,
                        status=Status.TRANSLATED,
                    )
                    self.tm.record(u.source, seed, origin=Origin.SEED)
                    stats.seeded += 1
                    lookback_buffer.append(LookbackItem(u.speaker, u.source, seed))
                    continue

                # 2. TM exact match
                tm_match = self.tm.lookup_exact(u.source) or self.tm.lookup_exact(u.source.lower())
                if tm_match is not None:
                    self.db.update_translation(
                        uid=u.uid,
                        target=tm_match,
                        origin=Origin.TM,
                        status=Status.TRANSLATED,
                    )
                    stats.tm_reused += 1
                    lookback_buffer.append(LookbackItem(u.speaker, u.source, tm_match))
                    continue

                need_llm.append(u)

        if not need_llm:
            return stats

        # ── Step 2: Partition into batches for LLM ──────────────────────
        batches = partition_batches(
            need_llm,
            max_lines=self.config.batch.max_lines,
            max_tokens=self.config.batch.max_source_tokens,
        )
        stats.batches_count = len(batches)

        processed_units = 0
        total_to_process = len(need_llm)

        for batch_idx, batch in enumerate(batches, start=1):
            if progress_cb:
                progress_cb(
                    processed_units,
                    total_to_process,
                    f"Đang dịch lô {batch_idx}/{len(batches)} ({len(batch)} dòng)...",
                )

            failing_units = self._translate_batch(batch, lookback_buffer, stats)
            if failing_units and len(batch) > 1:
                # If multiple lines failed, retry in smaller sub-batches first
                if len(failing_units) > 3:
                    _LOG.warning(
                        "Lô %d có %d dòng thất bại QA/unmask, thử lại theo cụm nhỏ...",
                        batch_idx,
                        len(failing_units),
                    )
                    sub_size = max(2, min(6, len(failing_units) // 2))
                    still_failing: list[Unit] = []
                    for k in range(0, len(failing_units), sub_size):
                        sub_chunk = failing_units[k : k + sub_size]
                        retry_fails = self._translate_batch(sub_chunk, lookback_buffer, stats)
                        still_failing.extend(retry_fails)
                    failing_units = still_failing

                if failing_units:
                    _LOG.warning(
                        "Lô %d có %d dòng thất bại sau thử lại cụm, chuyển sang dịch đơn lẻ...",
                        batch_idx,
                        len(failing_units),
                    )
                    for single_unit in failing_units:
                        self._translate_batch([single_unit], lookback_buffer, stats)

            processed_units += len(batch)

        return stats

    def _translate_batch(
        self,
        batch: Sequence[Unit],
        lookback: MutableSequence[LookbackItem],
        stats: TranslationStats,
    ) -> list[Unit]:
        """Translate a single batch of units, validate output, and commit to DB.

        Returns a list of units that failed validation/translation and require individual retry.
        """
        batch_id = f"b_{uuid.uuid4().hex[:12]}"
        texts = [u.masked_source or u.source for u in batch]
        glossary_matches = self.glossary.match_batch(texts)

        # Collect fuzzy matches for prompt hints
        fuzzy_hints = {}
        for u in batch:
            matches = self.tm.lookup_fuzzy(
                u.source, threshold=self.config.tm.fuzzy_threshold, max_results=1
            )
            if matches:
                fuzzy_hints[u.uid] = matches

        user_prompt = self.context.build_user_prompt(
            batch_units=batch,
            glossary_matches=glossary_matches,
            characters=self.characters,
            lookback=list(lookback),
            fuzzy_matches=fuzzy_hints,
        )
        system_prompt = self.context.build_system_prompt()

        has_cjk_source = any(_CJK_PATTERN.search(u.source) for u in batch)
        prop_schema: dict[str, Any] = {"type": "string"}
        if not has_cjk_source:
            prop_schema["pattern"] = r"^[^\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff]+$"

        batch_schema = {
            "type": "object",
            "properties": {u.uid: dict(prop_schema) for u in batch},
            "required": [u.uid for u in batch],
            "additionalProperties": False,
        }

        try:
            budget = ContextBuilder.calculate_output_budget(len(batch))
            result_map, p_tokens, c_tokens = self.llm.complete_json(
                system_prompt, user_prompt, max_tokens=budget, schema=batch_schema
            )
            stats.tokens_in += p_tokens
            stats.tokens_out += c_tokens
        except LLMError as error:
            stats.errors.append(f"Lô {batch_id}: {error}")
            # Record failed batch
            self.db.save_batch(
                BatchLog(
                    batch_id=batch_id,
                    unit_uids=tuple(u.uid for u in batch),
                    tokens_in=0,
                    tokens_out=0,
                    attempts=1,
                    error=str(error),
                    created_at=utc_timestamp(),
                )
            )
            if len(batch) == 1:
                self.db.mark_failed(batch[0].uid, f"LLM error: {error}")
                stats.failed += 1
                return []
            return list(batch)

        # Process and validate returned translations
        unit_updates: list[tuple[Unit, str, tuple[str, ...], Status]] = []
        failed_units: list[Unit] = []

        # Flexible key resolver: maps canonical uids, stripped prefix uids, and trimmed strings
        res_by_key: dict[str, str] = {}
        for k, v in result_map.items():
            k_clean = str(k).strip()
            res_by_key[k_clean] = v
            if ":" in k_clean:
                _, _, rest = k_clean.partition(":")
                res_by_key[rest.strip()] = v

        for u in batch:
            raw_target = (
                res_by_key.get(u.uid)
                or (res_by_key.get(u.uid[2:]) if u.uid.startswith(("s:", "d:", "u:")) else None)
            )
            if raw_target is None:
                for prefix in ("s:", "d:", "u:"):
                    if f"{prefix}{u.uid}" in res_by_key:
                        raw_target = res_by_key[f"{prefix}{u.uid}"]
                        break

            if raw_target is None:
                _LOG.info("Dòng %s thiếu bản dịch trong phản hồi của LLM.", u.uid)
                failed_units.append(u)
                continue

            # Auto-clean
            cleaned = clean_translation(raw_target, speaker=u.speaker, source=u.source)

            # Unmask if source had placeholders
            masked = mask(u.source)
            final_target = cleaned
            if masked.tokens:
                cleaned = auto_repair_placeholders(masked, cleaned)
                if "⟦" in cleaned:
                    try:
                        final_target = unmask(masked, cleaned)
                    except Exception as unmask_err:
                        _LOG.info(
                            "Dòng %s không thể unmask (%s): '%s' -> '%s'",
                            u.uid,
                            unmask_err,
                            u.source,
                            cleaned,
                        )
                        failed_units.append(u)
                        continue
                else:
                    # If target stripped placeholders but source had a single enclosing
                    # tag pair (e.g. {i}...{i}), attempt auto-wrapping before raw validation.
                    if (
                        len(masked.tokens) == 1
                        and masked.tokens[0].close_raw is not None
                        and "{" not in cleaned
                        and "[" not in cleaned
                    ):
                        candidate = f"{masked.tokens[0].raw}{cleaned}{masked.tokens[0].close_raw}"
                        if not validate_raw(u.source, candidate):
                            cleaned = candidate

                    raw_issues = validate_raw(u.source, cleaned)
                    if any(
                        i.code
                        in (
                            "bad_markup",
                            "missing_placeholder",
                            "unknown_placeholder",
                            "duplicate_placeholder",
                            "tag_order",
                        )
                        for i in raw_issues
                    ):
                        _LOG.info(
                            "Dòng %s lỗi raw markup/placeholder (%s): '%s'",
                            u.uid,
                            [i.code for i in raw_issues],
                            cleaned,
                        )
                        failed_units.append(u)
                        continue
                    final_target = cleaned
            else:
                if "{" in cleaned or "[" in cleaned:
                    raw_issues = validate_raw(u.source, cleaned)
                    if any(i.code in ("bad_markup", "tag_order") for i in raw_issues):
                        _LOG.info(
                            "Dòng %s lỗi cú pháp markup/thứ tự thẻ (%s): '%s'",
                            u.uid,
                            [i.code for i in raw_issues],
                            cleaned,
                        )
                        failed_units.append(u)
                        continue
                final_target = cleaned

            # Unit-specific glossary matches
            unit_glossary = self.glossary.match_text(u.masked_source or u.source)

            # Auto-repair any leaked English glossary terms before QA
            final_target = repair_glossary_leakage(final_target, unit_glossary)

            # Run QA checks
            masked_for_qa = masked if ("⟦" in final_target) else None
            qa_issues = self.qa.check_unit(
                u.uid,
                u.source,
                masked_for_qa,
                final_target,
                glossary_matches=unit_glossary,
                speaker=u.speaker,
            )
            max_ratio = self.config.qa.max_length_ratio
            # Only genuine runaway repetition loops are fatal errors.
            # Normal length ratio warnings (ratio > 1.8) are recorded by QA as
            # warnings and saved with Status.QA_FLAGGED for review.
            is_runaway_loop = (
                len(u.source) > 0
                and len(cleaned) > 100
                and (len(cleaned) / max(len(u.source), 1) > max(max_ratio * 1.5, 3.0))
            )
            has_error_issue = any(i.severity == "error" for i in qa_issues) or is_runaway_loop
            if has_error_issue:
                reason = (
                    "runaway_loop"
                    if is_runaway_loop
                    else ", ".join(i.code for i in qa_issues if i.severity == "error")
                )
                _LOG.info(
                    "Dòng %s không đạt validation (%s): '%s' -> '%s'",
                    u.uid,
                    reason,
                    u.source,
                    final_target,
                )
                failed_units.append(u)
                continue

            flags = tuple(sorted(i.code for i in qa_issues))
            status = Status.QA_FLAGGED if flags else Status.TRANSLATED
            unit_updates.append((u, final_target, flags, status))

        # Immediately commit all valid units
        if unit_updates:
            with self.db.transaction():
                for u, target, flags, status in unit_updates:
                    self.db.update_translation(
                        uid=u.uid,
                        target=target,
                        origin=Origin.LLM,
                        status=status,
                        qa_flags=flags,
                    )
                    self.tm.record(u.source, target, origin=Origin.LLM)
                    lookback.append(LookbackItem(u.speaker, u.source, target))
                    stats.llm_translated += 1

                self.db.save_batch(
                    BatchLog(
                        batch_id=batch_id,
                        unit_uids=tuple(u.uid for u in batch),
                        tokens_in=p_tokens,
                        tokens_out=c_tokens,
                        attempts=1,
                        error=None,
                        created_at=utc_timestamp(),
                    )
                )

        if not failed_units:
            return []

        if len(batch) == 1:
            # Single unit failed
            fail_reason = (
                "missing_result"
                if (not result_map or batch[0].uid not in result_map)
                else "qa_or_unmask_error"
            )
            self.db.mark_failed(batch[0].uid, fail_reason)
            stats.failed += 1
            return []

        return failed_units
