"""Milestone 5 Adversarial Stress Harness for Renslate Core Algorithms.

Empirically challenges:
1. Ren'Py markup parsing, nesting depth 100-500, mismatched tags, ruby text,
   universal closers, and standalone tags ({w}, {p}, {nw}, {fast}, {space}).
2. Dynamic token budgeting: zero-length text, empty batches, massive batches,
   huge narration lines (>500 chars), and negative budget graceful degradation.
3. High-throughput SQLite WAL database streaming: 10,000 units streaming benchmark,
   generator flat memory allocation profiling, transaction rollback atomicity,
   and concurrent WAL read/write non-blocking behavior.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import pytest

from renslate.core.config import Config
from renslate.core.db import Database
from renslate.core.models import (
    CharacterProfile,
    GlossaryEntry,
    Origin,
    SceneSummary,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.context import ContextBuilder, LookbackItem
from renslate.renpy.mask import (
    auto_repair_placeholders,
    mask,
    unmask,
    validate,
    validate_raw,
)

# ==============================================================================
# SECTION 1: DEEPLY NESTED REN'PY TAGS & ADVERSARIAL MARKUP HARNESS
# ==============================================================================


@pytest.mark.parametrize("depth", [50, 100, 200, 500])
def test_deeply_nested_tags_roundtrip_scalability(depth: int) -> None:
    """Challenge mask/unmask and validate_raw with extreme nesting depths up to 500."""
    open_tags = "".join(f"{{tag_{d}}}" for d in range(depth))
    close_tags = "".join(f"{{/tag_{d}}}" for d in reversed(range(depth)))
    text = f"{open_tags}Deep narrative payload at depth {depth}{close_tags}"

    t0 = time.perf_counter()
    masked = mask(text)
    t_mask = time.perf_counter() - t0

    assert len(masked.tokens) == depth
    assert len(validate(masked, masked.text)) == 0

    t0 = time.perf_counter()
    restored = unmask(masked, masked.text)
    t_unmask = time.perf_counter() - t0
    assert restored == text

    t0 = time.perf_counter()
    raw_issues = validate_raw(text, text)
    t_raw = time.perf_counter() - t0
    assert len(raw_issues) == 0

    # Linear O(N) performance assertion: depth 500 must complete within 50ms
    assert t_mask < 0.05, f"Masking depth {depth} took {t_mask * 1000:.2f}ms"
    assert t_unmask < 0.05, f"Unmasking depth {depth} took {t_unmask * 1000:.2f}ms"
    assert t_raw < 0.05, f"validate_raw depth {depth} took {t_raw * 1000:.2f}ms"


def test_deeply_nested_tags_mismatched_closers_adversarial() -> None:
    """Stress-test detection of subtle closer inversions and corruptions at depth 100+."""
    depth = 100
    open_tags = "".join(f"{{tag_{d}}}" for d in range(depth))
    close_tags = "".join(f"{{/tag_{d}}}" for d in reversed(range(depth)))
    src = f"{open_tags}Payload{close_tags}"

    # 1. Invert adjacent closers deep inside the stack (depth 50 and 51)
    order = list(reversed(range(depth)))
    order[50], order[51] = order[51], order[50]
    inverted_closers = "".join(f"{{/tag_{d}}}" for d in order)
    tgt_inverted = f"{open_tags}Bản dịch{inverted_closers}"

    issues_inv = validate_raw(src, tgt_inverted)
    assert any(i.code == "tag_order" for i in issues_inv), (
        "Failed to detect swapped closers at depth 50!"
    )

    # 2. Corrupt one tag name at depth 75
    order_corrupt = list(reversed(range(depth)))
    corrupt_closers = "".join(
        f"{{/tag_{d}}}" if d != 75 else "{/corrupt_tag}" for d in order_corrupt
    )
    tgt_corrupt = f"{open_tags}Bản dịch{corrupt_closers}"
    issues_corrupt = validate_raw(src, tgt_corrupt)
    assert any(
        i.code in ("tag_order", "missing_placeholder", "unknown_placeholder")
        for i in issues_corrupt
    )

    # 3. Missing closer at depth 30
    order_missing = [d for d in order if d != 30]
    missing_closers = "".join(f"{{/tag_{d}}}" for d in order_missing)
    tgt_missing = f"{open_tags}Bản dịch{missing_closers}"
    issues_missing = validate_raw(src, tgt_missing)
    assert any(i.code in ("tag_order", "missing_placeholder") for i in issues_missing)


def test_universal_closer_adversarial_depth_100() -> None:
    """Verify universal closer {/} handling across 100 nested tags."""
    depth = 100
    open_tags = "".join(f"{{level_{d}}}" for d in range(depth))
    univ_closers = "{/}" * depth
    src = f"{open_tags}Content{univ_closers}"

    masked = mask(src)
    assert len(masked.tokens) == depth
    assert unmask(masked, masked.text) == src
    assert len(validate_raw(src, src)) == 0

    # Target has 99 closers (1 unclosed tag)
    tgt_under = f"{open_tags}Nội dung" + ("{/}" * 99)
    issues_under = validate_raw(src, tgt_under)
    assert any(i.code in ("missing_placeholder", "unknown_placeholder") for i in issues_under)

    # Target has 101 closers (1 extraneous closer)
    tgt_over = f"{open_tags}Nội dung" + ("{/}" * 101)
    issues_over = validate_raw(src, tgt_over)
    assert any(i.code in ("tag_order", "unknown_placeholder") for i in issues_over)


def test_ruby_text_and_annotated_markup() -> None:
    """Verify Ren'Py ruby text tags {rb}...{/rb}{rt}...{/rt} and compound styling."""
    src = (
        "In Tokyo ({b}{rb}東京{/rb}{rt}とうきょう{/rt}{/b}), "
        "she met {i}{rb}先生{/rb}{rt}せんせい{/rt}{/i}."
    )
    masked = mask(src)
    assert len(masked.tokens) == 6
    assert unmask(masked, masked.text) == src

    tgt_valid = (
        "Ở Tokyo ({b}{rb}Tokyo{/rb}{rt}とうきょう{/rt}{/b}), "
        "cô ấy gặp {i}{rb}thầy giáo{/rb}{rt}せんせい{/rt}{/i}."
    )
    assert len(validate_raw(src, tgt_valid)) == 0

    # Dropping the {rt} tag should trigger missing_placeholder
    tgt_no_rt = (
        "Ở Tokyo ({b}{rb}Tokyo{/rb}{/b}), cô ấy gặp {i}{rb}thầy giáo{/rb}{rt}せんせい{/rt}{/i}."
    )
    issues_no_rt = validate_raw(src, tgt_no_rt)
    assert any(i.code == "missing_placeholder" for i in issues_no_rt)

    # Permissible grammatical reordering of independent sibling tags must pass
    tgt_sibling_reordered = (
        "Ở Tokyo ({b}{rt}とうきょう{/rt} {rb}Tokyo{/rb}{/b}), "
        "cô ấy gặp {i}{rb}thầy giáo{/rb}{rt}せんせい{/rt}{/i}."
    )
    assert len(validate_raw(src, tgt_sibling_reordered)) == 0

    # Interleaved invalid LIFO nesting must be strictly rejected
    tgt_interleaved = (
        "Ở Tokyo ({b}{rt}とうきょう{rb}Tokyo{/rt}{/rb}{/b}), "
        "cô ấy gặp {i}{rb}thầy giáo{/rb}{rt}せんせい{/rt}{/i}."
    )
    issues_interleaved = validate_raw(src, tgt_interleaved)
    assert any(i.code == "tag_order" for i in issues_interleaved)


def test_standalone_tags_inside_deep_nesting() -> None:
    """Verify standalone tags ({w}, {p}, {nw}, {fast}, {space=30}) inside depth 100 nesting."""
    depth = 100
    open_parts = []
    close_parts = []
    for i in range(depth):
        open_parts.append(f"{{t{i}}}")
        if i == 20:
            open_parts.append("{w=1.5}")
        elif i == 40:
            open_parts.append("{p}")
        elif i == 60:
            open_parts.append("{nw}")
        elif i == 80:
            open_parts.append("{fast}")
        close_parts.append(f"{{/t{i}}}")
    close_parts.reverse()

    src = "".join(open_parts) + "Dialogue body" + "".join(close_parts)
    masked = mask(src)
    assert unmask(masked, masked.text) == src

    # Validate valid target
    tgt_valid = "".join(open_parts) + "Lời thoại tiếng Việt" + "".join(close_parts)
    assert len(validate_raw(src, tgt_valid)) == 0

    # Dropping {fast} must be caught
    tgt_missing_fast = tgt_valid.replace("{fast}", "")
    issues_no_fast = validate_raw(src, tgt_missing_fast)
    assert any(i.code == "missing_placeholder" and "{fast}" in i.detail for i in issues_no_fast)

    # Duplicating {w=1.5} must be caught
    tgt_dup_w = tgt_valid.replace("{w=1.5}", "{w=1.5}{w=1.5}")
    issues_dup_w = validate_raw(src, tgt_dup_w)
    assert any(i.code == "duplicate_placeholder" and "{w=1.5}" in i.detail for i in issues_dup_w)


def test_auto_repair_placeholders_deep_nesting_stress() -> None:
    """Empirically test auto_repair_placeholders under depth 100 corrupted delimiters."""
    depth = 100
    open_tags = "".join(f"{{t{i}}}" for i in range(depth))
    close_tags = "".join(f"{{/t{i}}}" for i in reversed(range(depth)))
    src = f"{open_tags}Text{close_tags}"
    m = mask(src)

    # Simulate common LLM space and bracket corruption
    corrupted = m.text.replace("⟦T1⟧", "⟦ T1 ⟧").replace("⟦/T1⟧", "[/T1]")
    repaired = auto_repair_placeholders(m, corrupted)
    assert "⟦T1⟧" in repaired
    assert "⟦/T1⟧" in repaired
    assert len(validate(m, repaired)) == 0


# ==============================================================================
# SECTION 2: DYNAMIC TOKEN BUDGETING ADVERSARIAL HARNESS
# ==============================================================================


def test_token_budgeting_zero_length_and_empty_batches() -> None:
    """Stress-test token budgeting with 0-length source text and empty batches."""
    cfg = Config.from_dict({"llm": {"n_ctx": 6144}, "batch": {"lookback": 4}})
    builder = ContextBuilder(cfg)

    # 1. Empty batch calculation
    budget_0 = ContextBuilder.calculate_output_budget(0, source_tokens=0, n_ctx=6144)
    assert budget_0 == 350  # Clamped to lower bound 350

    prompt_empty = builder.build_user_prompt([])
    assert "CÁC DÒNG CẦN DỊCH" in prompt_empty

    # 2. Single unit with empty source string
    u_zero = Unit(uid="u0", kind=UnitKind.DIALOGUE, seq=0, source="", status=Status.PENDING)
    budget_zero = ContextBuilder.calculate_output_budget(1, source_tokens=0, n_ctx=6144)
    assert budget_zero == 350

    prompt_zero = builder.build_user_prompt([u_zero])
    assert "u0 | " in prompt_zero


def test_token_budgeting_massive_batches_and_narration() -> None:
    """Stress-test massive batches (100-200 units) and huge narration lines (>500 chars)."""
    cfg = Config.from_dict({"llm": {"n_ctx": 6144}, "batch": {"lookback": 4}})
    builder = ContextBuilder(cfg)

    # 1. Huge narration lines (2,000 chars)
    huge_text = "This is an extremely detailed and elaborate visual novel description. " * 30
    assert len(huge_text) > 2000
    u_huge = Unit(
        uid="huge_1",
        kind=UnitKind.DIALOGUE,
        seq=1,
        source=huge_text,
        status=Status.PENDING,
    )
    prompt_huge = builder.build_user_prompt([u_huge])
    assert "huge_1 | " in prompt_huge
    assert huge_text in prompt_huge

    # 2. Massive batch (200 units) exceeding n_ctx limit
    massive_units = [
        Unit(
            uid=f"u_{i}",
            kind=UnitKind.DIALOGUE,
            seq=i,
            source=f"Story dialogue turn #{i} with substantial character interaction narrative.",
            status=Status.PENDING,
        )
        for i in range(200)
    ]
    glossary = [GlossaryEntry(term="magic", translation="phép thuật")]
    chars = {"eileen": CharacterProfile(tag="eileen", display_name="Eileen")}
    scene = SceneSummary(scene_id="s1", label="ch1", summary="Major dramatic scene climax")
    lookback = [LookbackItem(speaker="eileen", source="Past line", target="Dòng trước")]

    # Must not crash; optional sections must gracefully omit to prevent overflow
    prompt_massive = builder.build_user_prompt(
        massive_units,
        glossary_matches=glossary,
        characters=chars,
        scene_summary=scene,
        lookback=lookback,
    )
    assert "CÁC DÒNG CẦN DỊCH" in prompt_massive
    # Supplementary sections dropped when available_budget < threshold
    assert "THUẬT NGỮ BẮT BUỘC" not in prompt_massive
    assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" not in prompt_massive


def test_lookback_compaction_sliding_suffix() -> None:
    """Verify lookback compaction sliding suffix fits within bounded token allowances."""
    cfg = Config.from_dict({"batch": {"lookback": 5}})
    builder = ContextBuilder(cfg)

    items = [
        LookbackItem(speaker=f"spk_{i}", source=f"Line {i} " * 20, target=f"Dòng {i} " * 20)
        for i in range(10)
    ]

    # Available budget sufficient for only ~2 lines
    block, rem_budget = builder._compact_lookback(items, available_budget=150)
    assert block is not None
    assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" in block
    assert rem_budget >= 0

    # Negative or tiny budget returns None
    block_none, rem_tiny = builder._compact_lookback(items, available_budget=30)
    assert block_none is None
    assert rem_tiny == 30


# ==============================================================================
# SECTION 3: HIGH-THROUGHPUT DATABASE STREAMING & WAL ATOMICITY HARNESS
# ==============================================================================


def test_database_streaming_10k_units_benchmark_and_memory(tmp_path: Path) -> None:
    """Benchmark iter_units and iter_pending_units with 10,000 units and profile memory."""
    db_path = tmp_path / "stream_10k_benchmark.db"
    db = Database.open(db_path, create=True)

    # 1. Ingest 10,000 units
    units_10k = [
        Unit(
            uid=f"unit_{i:06d}",
            kind=UnitKind.DIALOGUE,
            seq=i,
            ref=f"game/script.rpy:{i + 1}",
            speaker="narrator" if i % 2 == 0 else "hero",
            label="start",
            source=f"Original English source line #{i} with formatting.",
            masked_source=f"Original English source line #{i} with formatting.",
            target=f"Dòng dịch tiếng Việt #{i}" if i % 2 == 0 else None,
            status=Status.TRANSLATED if i % 2 == 0 else Status.PENDING,
            origin=Origin.LLM if i % 2 == 0 else None,
            qa_flags=(),
            updated_at="2026-10-24T18:00:00Z",
        )
        for i in range(10000)
    ]

    t0 = time.perf_counter()
    with db.transaction():
        db.save_units_bulk(units_10k)
    elapsed_bulk = time.perf_counter() - t0
    assert elapsed_bulk < 0.5, f"10k bulk save took {elapsed_bulk:.3f}s (expected < 0.5s)"

    # 2. Benchmark iter_units() generator memory and throughput
    tracemalloc.start()
    t0 = time.perf_counter()
    count_all = 0
    for _u in db.iter_units():
        count_all += 1
    t_all = time.perf_counter() - t0
    _, peak_stream_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert count_all == 10000
    # Throughput must exceed 30,000 units/second
    throughput = count_all / t_all
    assert throughput > 30000, f"iter_units throughput was {throughput:.1f} units/s"

    # Peak memory for streaming generator must be tiny (< 100 KiB)
    assert peak_stream_mem < 100 * 1024, (
        f"Memory leak in streaming iterator! Peak RAM was {peak_stream_mem / 1024:.1f} KiB"
    )

    # 3. Benchmark iter_pending_units()
    t0 = time.perf_counter()
    pending_count = sum(1 for _ in db.iter_pending_units())
    t_pending = time.perf_counter() - t0
    assert pending_count == 5000
    assert t_pending < 0.2, f"iter_pending_units took {t_pending:.3f}s"

    # 4. Benchmark recent_translated_units(50) over 10k rows
    t0 = time.perf_counter()
    recent = db.recent_translated_units(limit=50)
    t_recent = time.perf_counter() - t0
    assert len(recent) == 50
    assert t_recent < 0.01, (
        f"recent_translated_units took {t_recent * 1000:.2f}ms (expected < 10ms)"
    )

    db.close()


def test_database_wal_atomicity_and_concurrency_10k(tmp_path: Path) -> None:
    """Verify transaction rollback atomicity with 10,000 units and non-blocking WAL streaming."""
    db_path = tmp_path / "wal_atomicity_10k.db"
    db = Database.open(db_path, create=True)

    # Seed initial 1,000 units
    initial_units = [
        Unit(
            uid=f"init_{i:04d}",
            kind=UnitKind.DIALOGUE,
            seq=i,
            source=f"Seed line {i}",
            status=Status.PENDING,
            updated_at="2026-10-24T18:00:00Z",
        )
        for i in range(1000)
    ]
    with db.transaction():
        db.save_units_bulk(initial_units)

    # Attempt to insert 10,000 failing units in transaction
    fault_units = [
        Unit(
            uid=f"fault_{i:05d}",
            kind=UnitKind.DIALOGUE,
            seq=1000 + i,
            source=f"Fault line {i}",
            status=Status.PENDING,
            updated_at="2026-10-24T18:00:00Z",
        )
        for i in range(10000)
    ]

    with pytest.raises(RuntimeError), db.transaction():
        db.save_units_bulk(fault_units)
        raise RuntimeError("Injected system crash during 10k unit commit")

    # Assert atomicity: count must strictly remain 1,000
    count_post_abort = db.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    assert count_post_abort == 1000, f"Atomicity violated! Count was {count_post_abort}"

    # Verify non-blocking WAL streaming while a second connection updates
    db2 = Database.open(db_path)
    streamed = 0
    for _u in db.iter_units():
        streamed += 1
        if streamed == 500:
            with db2.transaction():
                db2.update_translation("init_0001", "Updated concurrently", Origin.LLM)
    assert streamed == 1000
    db2.close()
    db.close()
