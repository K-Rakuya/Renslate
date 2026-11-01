"""Core translation pipeline orchestrator for renslate."""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from collections import deque
from collections.abc import Callable, MutableSequence, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from renslate.core.config import Config
from renslate.core.db import Database, utc_timestamp
from renslate.core.errors import LLMError
from renslate.core.models import (
    LOCKED_ORIGINS,
    BatchLog,
    GlossaryEntry,
    Origin,
    SceneMacroContext,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.context import (
    ContextBuilder,
    InterlocutorResolver,
    LookbackItem,
    estimate_tokens,
)
from renslate.engine.glossary import Glossary
from renslate.engine.graph import RelationshipGraph
from renslate.engine.hallucination import (
    HallucinationGuard,
    persist_hallucination_audit,
)
from renslate.engine.knowledge import refresh_character_genders
from renslate.engine.llm import LLMClient, generate_batch_gbnf
from renslate.engine.qa import _CJK_PATTERN, QAChecker, clean_translation
from renslate.engine.repair import RetryLadder, UnitFailure
from renslate.engine.scenes import load_scene_contexts
from renslate.engine.tm import FuzzyMatch, TranslationMemory, is_context_sensitive_line
from renslate.renpy.builtin_ui import get_builtin_ui_translation
from renslate.renpy.mask import (
    auto_repair_placeholders,
    enforce_tag_invariants,
    mask,
    unmask,
    validate_raw,
)

_LOG = logging.getLogger("renslate.translate")

ProgressCallback = Callable[[int, int, str], None]

_OUTER_TAGS_PREFIX_RE = re.compile(r"^(?:\{[^\}]+\})+")
_OUTER_TAGS_SUFFIX_RE = re.compile(r"(?:\{[^\}]+\})+$")


def _realign_outer_tags(source: str, translation: str) -> str:
    pref_match = _OUTER_TAGS_PREFIX_RE.match(source)
    suff_match = _OUTER_TAGS_SUFFIX_RE.search(source)
    if not pref_match or not suff_match:
        return translation
    prefix = pref_match.group(0)
    suffix = suff_match.group(0)
    inner = _OUTER_TAGS_PREFIX_RE.sub("", translation)
    inner = _OUTER_TAGS_SUFFIX_RE.sub("", inner).strip()

    src_inner = source[len(prefix) : len(source) - len(suffix)]
    if src_inner.startswith(" ") and not inner.startswith(" "):
        inner = " " + inner
    if src_inner.endswith(" ") and not inner.endswith(" "):
        inner = inner + " "
    return f"{prefix}{inner}{suffix}"


@dataclass(slots=True)
class TranslationStats:
    total_pending: int = 0
    seeded: int = 0
    tm_reused: int = 0
    tm_context_skipped: int = 0
    llm_translated: int = 0
    failed: int = 0
    batches_count: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _PreparedBatch:
    batch: Sequence[Unit]
    batch_id: str
    glossary_matches: Sequence[GlossaryEntry]
    fuzzy_hints: dict[str, list[FuzzyMatch]]
    active_scene: Any
    system_prompt: str
    batch_schema: dict[str, Any]
    batch_grammar: str
    budget: int
    single_key: str | None
    scene_macro: Any = None


@dataclass(slots=True)
class DialogueTurn:
    """A contiguous run of dialogue units spoken by the same character or narration."""

    speaker: str | None
    normalized_speaker: str | None
    units: list[Unit]
    tokens: int
    is_narration: bool


@dataclass(slots=True)
class TurnCluster:
    """A cohesive conversational interaction block (e.g. exchange turn pair)."""

    units: list[Unit]
    tokens: int
    speakers: set[str]
    kind: UnitKind
    label: str | None
    file: str | None


def _extract_unit_file(unit: Unit | None) -> str | None:
    """Safely extract source script file path from Unit.file or Unit.ref."""
    if unit is None:
        return None
    f = getattr(unit, "file", None)
    if isinstance(f, str) and f:
        return f
    ref = getattr(unit, "ref", None)
    if isinstance(ref, str) and ref:
        return ref.rsplit(":", 1)[0] if ":" in ref else ref
    return None


def _normalize_speaker(speaker: str | None) -> tuple[str | None, bool]:
    """Classify speaker into normalized tag and narration boolean."""
    if not speaker:
        return None, True
    clean = speaker.strip()
    lower = clean.lower()
    _NON_DIALOGUE = {
        "t",
        "th",
        "thought",
        "thinking",
        "narrator",
        "narration",
        "none",
        "all",
        "system",
        "ui",
        "menu",
    }
    if lower in _NON_DIALOGUE:
        return lower, True
    return clean, False


def _cluster_dialogue_slice(
    units: Sequence[Unit], max_lines: int, max_tokens: int
) -> list[TurnCluster]:
    """Group dialogue units into cohesive turn exchange clusters."""
    if not units:
        return []

    first = units[0]
    kind = first.kind
    label = first.label
    file = _extract_unit_file(first)

    if kind != UnitKind.DIALOGUE:
        return [
            TurnCluster(
                units=[u],
                tokens=estimate_tokens(u.masked_source or u.source),
                speakers=set(),
                kind=kind,
                label=label,
                file=file,
            )
            for u in units
        ]

    # 1. Group consecutive lines by the same speaker into DialogueTurn
    turns: list[DialogueTurn] = []
    for u in units:
        spk, is_narr = _normalize_speaker(u.speaker)
        u_tok = estimate_tokens(u.masked_source or u.source)
        if turns and turns[-1].normalized_speaker == spk and turns[-1].is_narration == is_narr:
            turns[-1].units.append(u)
            turns[-1].tokens += u_tok
        else:
            turns.append(
                DialogueTurn(
                    speaker=u.speaker,
                    normalized_speaker=spk,
                    units=[u],
                    tokens=u_tok,
                    is_narration=is_narr,
                )
            )

    # 2. Cluster turns into conversational beats (exchange pairs between speaker pairs)
    target_cluster_lines = max(1, min(max(1, max_lines // 2), 8))
    target_cluster_tokens = max(1, min(max(1, max_tokens // 2), 350))

    clusters: list[TurnCluster] = []
    curr_turns: list[DialogueTurn] = []
    curr_speakers: set[str] = set()
    curr_lines = 0
    curr_toks = 0

    def flush_cluster() -> None:
        nonlocal curr_turns, curr_speakers, curr_lines, curr_toks
        if not curr_turns:
            return
        cluster_units: list[Unit] = []
        for t in curr_turns:
            cluster_units.extend(t.units)
        clusters.append(
            TurnCluster(
                units=cluster_units,
                tokens=curr_toks,
                speakers=set(curr_speakers),
                kind=kind,
                label=label,
                file=file,
            )
        )
        curr_turns = []
        curr_speakers = set()
        curr_lines = 0
        curr_toks = 0

    for turn in turns:
        t_lines = len(turn.units)
        t_toks = turn.tokens

        if turn.is_narration:
            if curr_turns and (
                curr_lines >= target_cluster_lines or curr_toks >= target_cluster_tokens
            ):
                flush_cluster()
            curr_turns.append(turn)
            curr_lines += t_lines
            curr_toks += t_toks
            continue

        spk_tag = turn.normalized_speaker or ""
        if not curr_turns:
            curr_turns.append(turn)
            curr_speakers.add(spk_tag)
            curr_lines += t_lines
            curr_toks += t_toks
        else:
            is_existing_speaker = spk_tag in curr_speakers
            pair_complete = len(curr_speakers) >= 2
            size_reached = (curr_lines >= target_cluster_lines) or (
                curr_toks >= target_cluster_tokens
            )

            if is_existing_speaker:
                if size_reached and pair_complete:
                    flush_cluster()
                    curr_turns.append(turn)
                    curr_speakers.add(spk_tag)
                    curr_lines += t_lines
                    curr_toks += t_toks
                else:
                    curr_turns.append(turn)
                    curr_lines += t_lines
                    curr_toks += t_toks
            else:
                if pair_complete or (size_reached and curr_speakers):
                    flush_cluster()
                    curr_turns.append(turn)
                    curr_speakers.add(spk_tag)
                    curr_lines += t_lines
                    curr_toks += t_toks
                else:
                    curr_turns.append(turn)
                    curr_speakers.add(spk_tag)
                    curr_lines += t_lines
                    curr_toks += t_toks

    flush_cluster()

    # 3. Oversized cluster splitting safety guard
    sanitized_clusters: list[TurnCluster] = []
    for c in clusters:
        if len(c.units) <= max_lines and c.tokens <= max_tokens:
            sanitized_clusters.append(c)
        else:
            sub_u: list[Unit] = []
            sub_tok = 0
            for u in c.units:
                u_tok = estimate_tokens(u.masked_source or u.source)
                if sub_u and (len(sub_u) >= max_lines or sub_tok + u_tok > max_tokens):
                    sanitized_clusters.append(
                        TurnCluster(
                            units=sub_u,
                            tokens=sub_tok,
                            speakers=c.speakers,
                            kind=c.kind,
                            label=c.label,
                            file=c.file,
                        )
                    )
                    sub_u = []
                    sub_tok = 0
                sub_u.append(u)
                sub_tok += u_tok
            if sub_u:
                sanitized_clusters.append(
                    TurnCluster(
                        units=sub_u,
                        tokens=sub_tok,
                        speakers=c.speakers,
                        kind=c.kind,
                        label=c.label,
                        file=c.file,
                    )
                )

    return sanitized_clusters


def partition_batches(
    units: Sequence[Unit],
    max_lines: int = 25,
    max_tokens: int = 700,
    *,
    isolate_scenes: bool = True,
    allow_cross_scene_packing: bool = False,
) -> list[list[Unit]]:
    """Partition units into batches respecting scene boundaries and dialogue turns.

    Uses knapsack-inspired contiguous atomic bin-packing to keep conversational turn pairs
    (e.g., character exchanges) cohesive within the same batch.
    """
    if not units:
        return []

    # Strictly sort by sequence order (seq, uid) to ensure unbroken chronological dialogue context
    units = sorted(units, key=lambda u: (u.seq, u.uid))

    if len(units) == 1:
        return [[units[0]]]

    # Step 1: Slice units into scene segments by (kind, label, file)
    scene_slices: list[list[Unit]] = []
    curr_slice: list[Unit] = []
    prev_u: Unit | None = None

    for u in units:
        if prev_u is not None:
            kind_changed = u.kind != prev_u.kind
            label_changed = (u.label or "").strip() != (prev_u.label or "").strip()
            file_changed = (_extract_unit_file(u) or "").strip() != (
                _extract_unit_file(prev_u) or ""
            ).strip()

            if kind_changed or (isolate_scenes and (label_changed or file_changed)):
                if curr_slice:
                    scene_slices.append(curr_slice)
                curr_slice = []

        curr_slice.append(u)
        prev_u = u

    if curr_slice:
        scene_slices.append(curr_slice)

    # Step 2: Cluster dialogue turns and pack clusters using contiguous knapsack bin packing
    batches: list[list[Unit]] = []
    current_batch: list[Unit] = []
    current_tokens = 0

    for s_slice in scene_slices:
        clusters = _cluster_dialogue_slice(s_slice, max_lines=max_lines, max_tokens=max_tokens)

        if not allow_cross_scene_packing and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        for cluster in clusters:
            c_units = cluster.units
            c_tokens = cluster.tokens
            c_lines = len(c_units)

            if current_batch and (
                len(current_batch) + c_lines > max_lines or current_tokens + c_tokens > max_tokens
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.extend(c_units)
            current_tokens += c_tokens

    if current_batch:
        batches.append(current_batch)

    return [sorted(b, key=lambda u: (u.seq, u.uid)) for b in batches]


class TranslationPipeline:
    """Manages end-to-end translation using built-in seeds, TM, and LLM batches."""

    def __init__(
        self,
        project: Project,
        db: Database,
        config: Config,
        graph: RelationshipGraph | None = None,
        resolver: InterlocutorResolver | None = None,
        hallucination_guard: HallucinationGuard | None = None,
    ) -> None:
        self.project = project
        self.db = db
        self.config = config
        self.tm = TranslationMemory(db)
        # Glossary mechanism disabled during translation to prevent vocabulary pollution
        self.glossary = Glossary([])

        self.characters = db.get_characters()
        self.scenes = db.get_scenes()
        self._scene_ctx = load_scene_contexts(db, self.scenes)
        self.qa = QAChecker(max_length_ratio=config.qa.max_length_ratio)
        self.llm = LLMClient(config.llm)
        self._retry_ladder = RetryLadder()

        # Initialize RelationshipGraph (injected, synced from json, or loaded from db)
        if graph is not None:
            self.graph = graph
        else:
            rel_path = getattr(project, "relationships_path", None)
            if rel_path is None and hasattr(project, "work_dir"):
                rel_path = project.work_dir / "relationships.json"
            if rel_path and rel_path.is_file():
                try:
                    self.graph = RelationshipGraph.sync(db, rel_path)
                except Exception:
                    try:
                        self.graph = RelationshipGraph.from_db(db)
                    except Exception:
                        self.graph = RelationshipGraph()
            else:
                try:
                    self.graph = RelationshipGraph.from_db(db)
                except Exception:
                    self.graph = RelationshipGraph()

        # Initialize InterlocutorResolver
        self.resolver = resolver if resolver is not None else InterlocutorResolver()
        self._current_scene_label: str | None = None
        self._current_prepared: _PreparedBatch | None = None

        # Initialize Continuous LLM Hallucination Guard
        self.guard = hallucination_guard
        if self.guard is None and getattr(config, "hallucination_check_enabled", False):
            self.guard = HallucinationGuard(config=config.llm, llm_client=self.llm)
        self.hallucination_guard = self.guard

        # Custom style guide if present
        style_guide = ""
        style_path = project.work_dir / "style.md"
        if style_path.is_file():
            style_guide = style_path.read_text(encoding="utf-8")
        self.context = ContextBuilder(config, custom_style_guide=style_guide)

    def refresh_knowledge(self) -> None:
        """Re-read data other tools changed and fix character facts from script evidence.

        The pipeline snapshots characters/scenes once in ``__init__``; a long-lived instance
        (UI, resumed run) would otherwise translate with stale profiles. Runs on the main
        thread before any prefetch worker starts.
        """
        try:
            refresh_character_genders(self.db)
        except sqlite3.Error as error:
            _LOG.warning("Không cập nhật được giới tính nhân vật từ kịch bản: %s", error)
        self.characters = self.db.get_characters()
        self.scenes = self.db.get_scenes()
        self._scene_ctx = load_scene_contexts(self.db, self.scenes)

    def run(
        self,
        *,
        limit: int | None = None,
        include_failed: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> TranslationStats:
        """Execute the translation pipeline over pending units."""
        stats = TranslationStats()
        self.refresh_knowledge()
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

        with self.db.transaction():
            for u in pending:
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

                tm_entry = self.tm.lookup_exact_entry(u.source) or self.tm.lookup_exact_entry(
                    u.source.lower()
                )
                if tm_entry is not None:
                    # Human-curated entries are always reused. Machine entries are reused only
                    # for context-free lines: dialogue carrying I/you/kinship words must be
                    # re-translated so the pronouns follow the *current* speaker/listener pair.
                    if tm_entry.origin in LOCKED_ORIGINS or not is_context_sensitive_line(u):
                        tm_match = tm_entry.target
                        self.db.update_translation(
                            uid=u.uid,
                            target=tm_match,
                            origin=Origin.TM,
                            status=Status.TRANSLATED,
                        )
                        stats.tm_reused += 1
                        lookback_buffer.append(LookbackItem(u.speaker, u.source, tm_match))
                        continue
                    stats.tm_context_skipped += 1

                need_llm.append(u)

        if not need_llm:
            return stats

        need_llm.sort(key=lambda u: (u.seq, u.uid))
        batches = partition_batches(
            need_llm,
            max_lines=self.config.batch.max_lines,
            max_tokens=self.config.batch.max_source_tokens,
        )
        stats.batches_count = len(batches)

        processed_units = 0
        total_to_process = len(need_llm)

        if len(batches) <= 1:
            for batch_idx, batch in enumerate(batches, start=1):
                if progress_cb:
                    progress_cb(
                        processed_units,
                        total_to_process,
                        f"Đang dịch lô {batch_idx}/{len(batches)} ({len(batch)} dòng)...",
                    )

                self._settle_batch(batch, batch_idx, lookback_buffer, stats)

                processed_units += len(batch)

            return stats

        with ThreadPoolExecutor(max_workers=1) as executor:
            next_prep_future = executor.submit(self._prepare_batch, batches[0])

            for batch_idx, batch in enumerate(batches, start=1):
                if progress_cb:
                    progress_cb(
                        processed_units,
                        total_to_process,
                        f"Đang dịch lô {batch_idx}/{len(batches)} ({len(batch)} dòng)...",
                    )

                prepared = next_prep_future.result()
                if batch_idx < len(batches):
                    next_prep_future = executor.submit(self._prepare_batch, batches[batch_idx])

                self._current_prepared = prepared
                self._settle_batch(batch, batch_idx, lookback_buffer, stats)

                processed_units += len(batch)

        return stats

    def _prepare_batch(self, batch: Sequence[Unit]) -> _PreparedBatch:
        # Strictly sort units by sequence order (seq, uid)
        batch = sorted(batch, key=lambda u: (u.seq, u.uid))
        batch_id = f"b_{uuid.uuid4().hex[:12]}"
        glossary_matches: Sequence[GlossaryEntry] = ()

        # Collect fuzzy matches for prompt hints
        fuzzy_hints: dict[str, list[FuzzyMatch]] = {}
        for u in batch:
            matches = self.tm.lookup_fuzzy(
                u.source, threshold=self.config.tm.fuzzy_threshold, max_results=1
            )
            if matches:
                fuzzy_hints[u.uid] = matches

        # Find active scene summary covering this batch
        active_scene = None
        if self.scenes and batch:
            b_first = batch[0].seq
            b_last = batch[-1].seq
            for sc in self.scenes:
                if sc.first_seq <= b_last and sc.last_seq >= b_first:
                    active_scene = sc
                    break
                if sc.label and sc.label == batch[0].label:
                    active_scene = sc
                    break

        # JIT Scene Summarization: ensure active scene has an LLM narrative summary
        scene_macro = None
        if active_scene is not None:
            scene_macro = self._scene_ctx.get(active_scene.scene_id)
            if (scene_macro is None or not scene_macro.summary) and not active_scene.summary:
                from renslate.engine.scenes import summarize_single_scene_with_llm

                try:
                    jit_macro = summarize_single_scene_with_llm(
                        self.project,
                        self.db,
                        self.llm,
                        active_scene,
                        min_lines=2,
                        characters=self.characters,
                        scenes=self.scenes,
                    )
                    if jit_macro is not None:
                        self._scene_ctx[active_scene.scene_id] = jit_macro
                        scene_macro = jit_macro
                except Exception as exc:
                    _LOG.debug(
                        "JIT scene summarization skipped for %s: %s", active_scene.scene_id, exc
                    )
            elif scene_macro is None and active_scene.summary:
                loc = getattr(active_scene, "location", "")
                atmos = getattr(active_scene, "atmosphere", "")
                scene_macro = SceneMacroContext(
                    scene_id=active_scene.scene_id,
                    label=active_scene.label,
                    first_seq=active_scene.first_seq,
                    last_seq=active_scene.last_seq,
                    summary=active_scene.summary,
                    location=loc,
                    atmosphere=atmos,
                )
                self._scene_ctx[active_scene.scene_id] = scene_macro

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
        batch_grammar = generate_batch_gbnf([u.uid for u in batch], allow_cjk=has_cjk_source)
        batch_source_tokens = sum(estimate_tokens(u.masked_source or u.source) for u in batch)
        budget = ContextBuilder.calculate_output_budget(
            len(batch),
            source_tokens=batch_source_tokens,
            n_ctx=self.config.llm.n_ctx,
        )
        single_key = batch[0].uid if len(batch) == 1 else None

        return _PreparedBatch(
            batch=batch,
            batch_id=batch_id,
            glossary_matches=glossary_matches,
            fuzzy_hints=fuzzy_hints,
            active_scene=active_scene,
            system_prompt=system_prompt,
            batch_schema=batch_schema,
            batch_grammar=batch_grammar,
            budget=budget,
            single_key=single_key,
            scene_macro=scene_macro,
        )

    def _settle_batch(
        self,
        batch: Sequence[Unit],
        batch_idx: int,
        lookback: MutableSequence[LookbackItem],
        stats: TranslationStats,
    ) -> None:
        failures = self._translate_batch(batch, lookback, stats)

        def attempt(units: Sequence[Unit], feedback: str) -> list[UnitFailure]:
            return self._translate_batch(units, lookback, stats, retry_feedback=feedback)

        def announce(stage: str, count: int) -> None:
            _LOG.warning(
                "Lô %d có %d dòng thất bại, thử lại theo bậc '%s'", batch_idx, count, stage
            )

        self._retry_ladder.run(batch, failures, attempt, announce)

    def _translate_batch(
        self,
        batch: Sequence[Unit],
        lookback: MutableSequence[LookbackItem],
        stats: TranslationStats,
        retry_feedback: str = "",
    ) -> list[UnitFailure]:
        prepared = getattr(self, "_current_prepared", None)
        self._current_prepared = None
        if prepared is None or prepared.batch is not batch:
            prepared = self._prepare_batch(batch)
        return self._execute_prepared_batch(prepared, lookback, stats, retry_feedback)

    def _execute_prepared_batch(
        self,
        prep: _PreparedBatch,
        lookback: MutableSequence[LookbackItem],
        stats: TranslationStats,
        retry_feedback: str = "",
    ) -> list[UnitFailure]:
        batch = prep.batch
        batch_id = prep.batch_id
        batch_label = batch[0].label if batch else None
        if batch_label != self._current_scene_label:
            self.resolver.reset()
            self._current_scene_label = batch_label

        user_prompt = self.context.build_user_prompt(
            batch_units=batch,
            glossary_matches=prep.glossary_matches,
            characters=self.characters,
            scene_summary=prep.active_scene,
            scene_macro=prep.scene_macro,
            lookback=list(lookback),
            fuzzy_matches=prep.fuzzy_hints,
            graph=self.graph,
            resolver=self.resolver,
            retry_feedback=retry_feedback,
        )

        try:
            result_map, p_tokens, c_tokens = self.llm.complete_json(
                prep.system_prompt,
                user_prompt,
                max_tokens=prep.budget,
                grammar=prep.batch_grammar,
                schema=prep.batch_schema,
                fallback_single_key=prep.single_key,
            )
            stats.tokens_in += p_tokens
            stats.tokens_out += c_tokens
        except LLMError as error:
            stats.errors.append(f"Lô {batch_id}: {error}")
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
            return [UnitFailure(u, "llm_error", str(error)) for u in batch]

        unit_updates: list[tuple[Unit, str, tuple[str, ...], Status]] = []
        failed_units: list[UnitFailure] = []

        res_by_key: dict[str, str] = {}
        for k, v in result_map.items():
            k_clean = str(k).strip()
            res_by_key[k_clean] = v
            if ":" in k_clean:
                _, _, rest = k_clean.partition(":")
                res_by_key[rest.strip()] = v

        active_speakers = {
            item.speaker.strip()
            for item in batch
            if item.speaker
            and item.speaker.strip().lower()
            not in (
                "t",
                "th",
                "thought",
                "thinking",
                "narrator",
                "narration",
                "none",
                "all",
                "system",
                "ui",
                "menu",
            )
        }
        known_names: dict[str, str] = {}
        for p in self.characters.values():
            if p.tag and p.display_name:
                known_names[p.tag] = p.display_name
            if p.display_name:
                known_names[p.display_name] = p.display_name

        prev_u: Unit | None = None
        for u in batch:
            raw_target = res_by_key.get(u.uid) or (
                res_by_key.get(u.uid[2:]) if u.uid.startswith(("s:", "d:", "u:")) else None
            )
            if raw_target is None:
                for prefix in ("s:", "d:", "u:"):
                    if f"{prefix}{u.uid}" in res_by_key:
                        raw_target = res_by_key[f"{prefix}{u.uid}"]
                        break

            if raw_target is None:
                _LOG.info("Dòng %s thiếu bản dịch trong phản hồi của LLM.", u.uid)
                failed_units.append(
                    UnitFailure(u, "missing_result", "Dòng này bị thiếu trong phản hồi JSON.")
                )
                prev_u = u
                continue

            cleaned = clean_translation(raw_target, speaker=u.speaker, source=u.source)
            cleaned = enforce_tag_invariants(u.source, cleaned)

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
                        failed_units.append(
                            UnitFailure(u, "unmask_error", str(unmask_err), cleaned)
                        )
                        prev_u = u
                        continue
                else:
                    if masked.tokens and "{" not in cleaned and "[" not in cleaned:
                        candidate = _realign_outer_tags(u.source, cleaned)
                        if not any(
                            i.code
                            in (
                                "bad_markup",
                                "missing_placeholder",
                                "unknown_placeholder",
                                "duplicate_placeholder",
                                "tag_order",
                            )
                            for i in validate_raw(u.source, candidate)
                        ):
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
                        realigned = _realign_outer_tags(u.source, cleaned)
                        if realigned != cleaned and not any(
                            i.code
                            in (
                                "bad_markup",
                                "missing_placeholder",
                                "unknown_placeholder",
                                "duplicate_placeholder",
                                "tag_order",
                            )
                            for i in validate_raw(u.source, realigned)
                        ):
                            cleaned = realigned
                        else:
                            _LOG.info(
                                "Dòng %s lỗi raw markup/placeholder (%s): '%s'",
                                u.uid,
                                [i.code for i in raw_issues],
                                cleaned,
                            )
                            failed_units.append(
                                UnitFailure(
                                    u, "raw_markup", ", ".join(i.code for i in raw_issues), cleaned
                                )
                            )
                            prev_u = u
                            continue
                    final_target = cleaned
            else:
                if "{" in cleaned or "[" in cleaned:
                    if not masked.tokens and ("{" in cleaned and "}" in cleaned):
                        candidate = _OUTER_TAGS_PREFIX_RE.sub("", cleaned)
                        candidate = _OUTER_TAGS_SUFFIX_RE.sub("", candidate).strip()
                        if not any(
                            i.code in ("bad_markup", "tag_order")
                            for i in validate_raw(u.source, candidate)
                        ):
                            cleaned = candidate

                    raw_issues = validate_raw(u.source, cleaned)
                    if any(i.code in ("bad_markup", "tag_order") for i in raw_issues):
                        _LOG.info(
                            "Dòng %s lỗi cú pháp markup/thứ tự thẻ (%s): '%s'",
                            u.uid,
                            [i.code for i in raw_issues],
                            cleaned,
                        )
                        failed_units.append(
                            UnitFailure(
                                u, "raw_markup", ", ".join(i.code for i in raw_issues), cleaned
                            )
                        )
                        prev_u = u
                        continue
                final_target = cleaned
            final_target = enforce_tag_invariants(u.source, final_target)

            masked_for_qa = masked if ("⟦" in final_target) else None
            pronoun_rules = None
            if u.speaker and self.characters:
                spk_clean = u.speaker.strip()
                spk_prof = self.characters.get(spk_clean) or self.characters.get(spk_clean.lower())
                target_tag: str | None = None

                # 1. First attempt to resolve listener via InterlocutorResolver
                resolved = self.resolver.resolve(
                    unit=u,
                    prev_unit=prev_u,
                    active_speakers=active_speakers,
                    known_names=known_names,
                )
                if resolved and resolved.listener:
                    target_tag = resolved.listener

                # 2. Query RelationshipGraph for bilateral edge
                if target_tag and self.graph:
                    target_prof = self.characters.get(target_tag) or self.characters.get(
                        target_tag.lower()
                    )
                    edge = self.graph.get_edge(spk_clean, target_tag) or self.graph.get_edge(
                        spk_clean.lower(), target_tag.lower()
                    )
                    if not edge and spk_prof:
                        edge = self.graph.get_edge(spk_prof.tag, target_tag) or (
                            self.graph.get_edge(spk_prof.tag, target_prof.tag)
                            if target_prof
                            else None
                        )
                    if edge and edge.speaker_pronoun and edge.listener_pronoun:
                        pronoun_rules = (edge.speaker_pronoun, edge.listener_pronoun)

                # 3. Fallback to character profile pronoun map if not resolved from graph
                if pronoun_rules is None and spk_prof and spk_prof.pronoun_map:
                    if not target_tag:
                        other_speakers = [
                            item.speaker
                            for item in batch
                            if item.speaker
                            and item.speaker.lower() not in ("t", "narrator", "thought")
                            and item.speaker.lower() != spk_clean.lower()
                        ]
                        if not other_speakers and lookback:
                            other_speakers = [
                                item.speaker
                                for item in reversed(lookback)
                                if item.speaker
                                and item.speaker.lower() not in ("t", "narrator", "thought")
                                and item.speaker.lower() != spk_clean.lower()
                            ]
                        if other_speakers:
                            target_spk = other_speakers[0].strip()
                            target_prof = self.characters.get(target_spk) or self.characters.get(
                                target_spk.lower()
                            )
                            target_tag = target_prof.tag if target_prof else target_spk

                    if target_tag:
                        target_prof = self.characters.get(target_tag) or self.characters.get(
                            target_tag.lower()
                        )
                        rule_text = spk_prof.pronoun_map.get(target_tag) or (
                            spk_prof.pronoun_map.get(target_prof.display_name)
                            if target_prof
                            else None
                        )
                        if rule_text and " - " in rule_text and not rule_text.startswith("xưng"):
                            p_parts = [p.strip() for p in rule_text.split(" - ", 1)]
                            if len(p_parts) == 2:
                                pronoun_rules = (p_parts[0], p_parts[1])

            qa_issues = self.qa.check_unit(
                u.uid,
                u.source,
                masked_for_qa,
                final_target,
                glossary_matches=(),
                speaker=u.speaker,
                pronoun_rules=pronoun_rules,
            )
            max_ratio = self.config.qa.max_length_ratio
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
                failed_units.append(
                    UnitFailure(
                        u,
                        "qa_error",
                        "; ".join(i.message for i in qa_issues if i.severity == "error") or reason,
                        final_target,
                    )
                )
                prev_u = u
                continue

            flags = tuple(sorted(i.code for i in qa_issues))
            status = Status.QA_FLAGGED if flags else Status.TRANSLATED
            unit_updates.append((u, final_target, flags, status))
            prev_u = u

        if self.guard is not None and unit_updates:
            from dataclasses import replace

            candidate_units = [replace(u, target=tgt) for u, tgt, _, _ in unit_updates]
            guard_results = self.guard.scan_batch(candidate_units, context_window=user_prompt)
            persist_hallucination_audit(self.db, guard_results)

            hal_by_id = {r.unit_id: r for r in guard_results if r.has_hallucination}
            if hal_by_id:
                filtered_updates: list[tuple[Unit, str, tuple[str, ...], Status]] = []
                for u, tgt, flags, status in unit_updates:
                    r = hal_by_id.get(u.uid)
                    if r is not None and r.confidence >= 0.70:
                        _LOG.info(
                            "Dòng %s bị HallucinationGuard gắn cờ %s (độ tin cậy %.2f): %s",
                            u.uid,
                            r.category,
                            r.confidence,
                            r.explanation,
                        )
                        detail = r.explanation
                        if r.suggested_correction:
                            detail = f"{detail} (Gợi ý: {r.suggested_correction})"
                        failed_units.append(
                            UnitFailure(
                                u,
                                r.category or "hallucination",
                                detail,
                                tgt,
                            )
                        )
                    else:
                        if r is not None:
                            flags = tuple(
                                sorted(set(flags) | {f"qa_hallucination_{r.category.lower()}"})
                            )
                            status = Status.QA_FLAGGED
                        filtered_updates.append((u, tgt, flags, status))
                unit_updates = filtered_updates

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
            self.db.mark_failed(batch[0].uid, failed_units[0].code)
            stats.failed += 1
            return []

        return failed_units
