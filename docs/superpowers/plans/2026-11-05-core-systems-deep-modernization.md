# Core Systems Deep Modernization Implementation Plan (2026+)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul Renslate's remaining core subsystems (Markup Protection, Translation Memory, Hallucination Defense, and RPYC Bytecode Processing) into world-class, deep architectural modules conforming to November 2026 engineering standards and the 8 Renvi Quality Pillars in `AGENTS.md`.

**Architecture:** Hexagonal Ports & Adapters architecture operating with pure Python 3.10+ standard library. Employs a Deterministic Pushdown Automaton (DPDA) for context-free markup preservation, SQLite FTS5 character trigram indexing with bit-parallel edit distance for sub-millisecond translation memory, a 4-tier hallucination taxonomy with structured negative feedback ladders, and hardened Ren'Py 8.x AST bytecode decompilation with robust path hashing.

**Tech Stack:** Pure Python 3.10+ Standard Library (`dependencies = []`), SQLite 3 with FTS5 and WAL mode, `urllib.request` persistent connection pooling, `dataclasses` with slots, `pytest`, `ruff`.

**Spec:** [AGENTS.md](file:///home/rakuya/Documents/Renvi/AGENTS.md) and [ARCHITECTURE_DECISIONS.md](file:///home/rakuya/Documents/Renvi/docs/ARCHITECTURE_DECISIONS.md).

## Global Constraints

- **Pure Python Standard Library:** 100% standard library compliance (`dependencies = []` in `pyproject.toml`). No external ML libraries or compilation dependencies.
- **Ren'Py General-Purpose Paradigm:** Zero hardcoded game titles, character IDs, or ad-hoc patches. Works identically on any Visual Novel `.rpy` and `.rpyc`.
- **Zero Benchmark Leakage:** Benchmark paths and game titles must never appear in code, tests, docs, or git commits. Use neutral names (`CommercialVN-Alpha`, `CommercialVN-Beta`) and dynamic environment variables (`RENVI_TEST_GAME_RPY`, `RENVI_TEST_GAME_RPYC`).
- **Clean Code & Self-Documenting:** No commentary apologies, workaround flags, or junk comments. Concise docstrings only for public protocols.
- **Single-Worker SQLite Concurrency:** ACID WAL transactions with composite indexes, resilient against crash and thread re-entry.
- **Mathematical Invariants:** Bijective token preservation $\tau(\mathbf{T}) \leftrightarrow \mathbf{T}'$ and algebraic reciprocal pronouns $\Pi(\mathbf{R}_{(B, A)}, g_B, g_A) \equiv \sigma(\Pi(\mathbf{R}_{(A, B)}, g_A, g_B))$.

## Review Focus

1. **Tag Nesting and Clause Reordering:** English SVO moving to Vietnamese topic-comment structures must maintain strictly balanced Ren'Py tags without orphan tags or unclosed braces.
2. **Sub-Millisecond Retrieval at Scale:** Translation Memory fuzzy search across 150,000+ units must return in < 1ms with near-zero Python heap footprint.
3. **Infinite Token Repetition Traps:** 4-bit and 8-bit quantized local GGUF models looping on repetitive tokens must be immediately detected and cleanly interrupted.
4. **Adversarial Directory Slicing:** Game paths containing subdirectories with the substring `game` (e.g. `/path/to/mygame/game/script.rpy`) must compute identical canonical say IDs to standard paths.
5. **Corrupted Bytecode Graceful Degradation:** Corrupt `.rpyc` zlib blocks must return safe `None` results rather than unhandled `zlib.error` exceptions.

---

### Task 1: Deterministic Pushdown Automaton (DPDA) Parser & Invariant Markup Engine

**Files:**
- Modify: `src/renslate/renpy/mask.py`
- Test: `tests/test_mask.py`

**Interfaces:**
- Consumes: Raw dialogue text with Ren'Py interpolations (`[var]`, `%(var)s`), formatting tags (`{b}`, `{color=...}`, `{size=...}`, `{font=...}`, `{a=...}`, `{cps=...}`), self-closing tags (`{image=...}`, `{/}`), and escape brackets (`[[`, `{{`, `%%`).
- Produces: 
  * `parse_markup_ast(text: str) -> MarkupAST`: Returns an immutable AST representation of the text.
  * `mask(source: str) -> MaskedText`: DPDA-driven token protection replacing tags with synthetic bijective placeholders (`⟦V1⟧`, `⟦T1⟧...⟦/T1⟧`).
  * `reconcile_translation(source: str, target: str) -> tuple[str, list[PlaceholderIssue]]`: Validates stack balance and bijection, reconstructing clean canonical Ren'Py text.

- [ ] **Step 1: Write the failing tests for DPDA tag reconciliation and clause permutation**

```python
def test_dpda_clause_reordering_with_nested_formatting() -> None:
    source = "I saw {b}{i}[player]{/i}{/b} yesterday, {color=#ff0000}didn't you?{/color}"
    # Target reorders the clauses (Vietnamese topic-prominent order)
    target = "{color=#ff0000}Không phải sao?{/color} Tôi đã thấy {b}{i}[player]{/i}{/b} hôm qua."
    cleaned, issues = reconcile_translation(source, target)
    assert not issues, f"Expected no issues, got {issues}"
    assert cleaned == target
    assert "{b}{i}[player]{/i}{/b}" in cleaned
    assert "{color=#ff0000}Không phải sao?{/color}" in cleaned

def test_dpda_rejects_unbalanced_nested_tags() -> None:
    source = "{b}{i}Hello{/i}{/b}"
    bad_target = "{b}{i}Xin chào{/b}{/i}"  # Interleaved tags
    cleaned, issues = reconcile_translation(source, bad_target)
    assert any(i.code in ("unbalanced_tag", "tag_mismatch", "unmask_error") for i in issues)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_mask.py -k "test_dpda_" -v`
Expected: FAIL (missing methods / DPDA stack validation)

- [ ] **Step 3: Implement Deterministic Pushdown Automaton in `src/renslate/renpy/mask.py`**

Refactor `mask.py` into a formal DPDA:
1. Define formal stack states: `TEXT`, `INTERPOLATION`, `TAG_NAME`, `TAG_PARAM`, `ESCAPE`.
2. Implement pushdown stack transition table validating proper push/pop order on nested `{...}` and `[...]`.
3. Support self-closing `{/}` and standard `{/tag}` tags with strict FIFO stack unwinding.
4. Ensure `reconcile_translation` handles token permutation gracefully while rejecting improper interleaving or unbalanced depths.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_mask.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/renslate/renpy/mask.py tests/test_mask.py
git commit -m "feat(mask): implement deterministic pushdown automaton for context-free markup reconciliation"
```

---

### Task 2: SQLite FTS5 Trigram Translation Memory & Bit-Parallel Levenshtein

**Files:**
- Modify: `src/renslate/core/db/schema.py`
- Modify: `src/renslate/engine/tm.py`
- Test: `tests/test_tm.py`

**Interfaces:**
- Consumes: `tm` table in SQLite database.
- Produces:
  * Schema migration v8: `tm_fts` virtual table with `tokenize='trigram'` and synchronization triggers (`tm_ai`, `tm_ad`, `tm_au`).
  * `myers_bit_parallel_levenshtein(s1: str, s2: str, max_distance: int) -> int`: Sub-microsecond edit distance for short strings using 64-bit integer bitmasks.
  * `TranslationMemory.lookup_fuzzy(source: str, threshold: float = 0.88, max_results: int = 1) -> list[FuzzyMatch]`: Sub-millisecond candidate filtering via SQLite FTS5 followed by bit-parallel ranking.

- [ ] **Step 1: Write the failing tests for FTS5 trigram indexing and sub-millisecond retrieval**

```python
def test_tm_fts5_migration_and_fuzzy_retrieval(tmp_path: Path) -> None:
    db_path = tmp_path / "test_tm.db"
    with Database.open(db_path, create=True) as db:
        db.add_tm("Open application preferences", "Mở tùy chọn ứng dụng", origin=Origin.USER)
        db.add_tm("Save current game state", "Lưu trạng thái trò chơi hiện tại", origin=Origin.USER)
        tm = TranslationMemory(db)
        # Query with high similarity typo
        results = tm.lookup_fuzzy("Open application preference", threshold=0.85)
        assert len(results) == 1
        assert results[0].target == "Mở tùy chọn ứng dụng"
        assert results[0].score >= 0.85

def test_myers_bit_parallel_levenshtein_accuracy() -> None:
    from renslate.engine.tm import myers_bit_parallel_distance
    assert myers_bit_parallel_distance("kitten", "sitting", 10) == 3
    assert myers_bit_parallel_distance("flaw", "lawn", 10) == 2
    assert myers_bit_parallel_distance("exact", "exact", 10) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tm.py -k "test_tm_fts5_migration_or_myers" -v`
Expected: FAIL

- [ ] **Step 3: Implement Migration v8 and Myers Bit-Parallel Distance in `schema.py` and `tm.py`**

1. In `src/renslate/core/db/schema.py`:
   - Add Migration 8 creating `tm_fts USING fts5(src_norm, content='tm', content_rowid='rowid', tokenize='trigram')`.
   - Add insert, delete, and update triggers keeping `tm_fts` synchronized with `tm`.
2. In `src/renslate/engine/tm.py`:
   - Implement `myers_bit_parallel_distance(a: str, b: str, max_dist: int) -> int` in pure Python.
   - Refactor `TranslationMemory.lookup_fuzzy` to execute `SELECT rowid, src_norm, target FROM tm_fts WHERE tm_fts MATCH ? LIMIT 25` against SQLite FTS5 when available, eliminating the in-memory Python `_ngram_index` heap overhead.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_tm.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/renslate/core/db/schema.py src/renslate/engine/tm.py tests/test_tm.py
git commit -m "feat(tm): implement sqlite fts5 trigram virtual table and myers bit-parallel edit distance"
```

---

### Task 3: Multi-Tier Hallucination Taxonomy & Structured Feedback Ladder

**Files:**
- Modify: `src/renslate/engine/hallucination/scanner.py`
- Modify: `src/renslate/engine/hallucination/taxonomy.py`
- Modify: `src/renslate/engine/qa.py`
- Modify: `src/renslate/engine/repair.py`
- Test: `tests/test_qa.py`
- Test: `tests/test_adversarial_m3_qa.py`

**Interfaces:**
- Consumes: Source dialogue units, raw LLM translations, and active relational frame $\Pi(\mathbf{R})$.
- Produces:
  * `HallucinationScanner.scan(unit: Unit, translation: str, frame: RelationalFrame | None) -> list[QAAnomaly]`: Scans for token repetition traps, semantic deletion, pronoun inversion, and LLM boilerplate.
  * `build_feedback_correction_frame(unit: Unit, anomalies: list[QAAnomaly]) -> str`: Builds targeted, negative-feedback prompt instructions for deterministic LLM regeneration.

- [ ] **Step 1: Write the failing tests for repetition trap detection and feedback prompt generation**

```python
def test_repetition_trap_scanner_catches_cyclic_quant_loops() -> None:
    source = "Wait for me outside."
    # Cyclic loop typical of low-temp 4-bit quant LLMs
    bad_output = "Đợi tôi ở ngoài. Đợi tôi ở ngoài. Đợi tôi ở ngoài. Đợi tôi ở ngoài."
    scanner = HallucinationScanner()
    anomalies = scanner.scan_repetition(source, bad_output)
    assert any(a.anomaly_type == "REPETITION_LOOP" for a in anomalies)

def test_feedback_frame_injects_specific_repetition_constraint() -> None:
    unit = Unit(uid="u1", source="Wait for me outside.", kind=UnitKind.DIALOGUE)
    anomaly = QAAnomaly(anomaly_type="REPETITION_LOOP", detail="Repeated 4-gram 'Đợi tôi ở ngoài'")
    frame = build_feedback_correction_frame(unit, [anomaly])
    assert "KHÔNG ĐƯỢC LẶP LẠI" in frame
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_qa.py -k "test_repetition_trap" -v`
Expected: FAIL

- [ ] **Step 3: Implement 4-tier scanners and feedback frame generator**

1. In `src/renslate/engine/hallucination/scanner.py`:
   - Implement `scan_repetition`: Sliding window compression ratio ($L_{\text{compressed}} / L_{\text{raw}}$) and n-gram set entropy to identify cyclic loops with length > 20 characters.
   - Implement `scan_boilerplate`: Detect conversational prefix/suffix hallucination (`Certainly`, `Here is the translation`, `Đây là câu dịch:`, markdown fences).
   - Implement `scan_truncation`: Flag translations that drop > 60% of source content or end with trailing unclosed quotes/ellipses.
2. In `src/renslate/engine/repair.py`:
   - Implement `build_feedback_correction_frame`: Generate structured negative feedback directives instructing the model on the exact error to avoid during sub-batch regeneration.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_qa.py tests/test_adversarial_m3_qa.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/renslate/engine/hallucination/ src/renslate/engine/qa.py src/renslate/engine/repair.py tests/test_qa.py tests/test_adversarial_m3_qa.py
git commit -m "feat(qa): implement multi-tier hallucination scanner and structured negative feedback ladder"
```

---

### Task 4: Ren'Py Bytecode (RPYC) Resilient Decompiler & Path Normalization

**Files:**
- Modify: `src/renslate/renpy/rpyc/archive.py`
- Modify: `src/renslate/renpy/rpyc/extractor.py`
- Modify: `src/renslate/renpy/rpyc/decompiler.py`
- Test: `tests/test_empirical_challenger_m2.py`
- Test: `tests/test_rpyc.py`

**Interfaces:**
- Consumes: Raw `.rpyc` files, streams, or directories containing game scripts.
- Produces:
  * `compute_canonical_say_identifier(label, filename, linenumber, text) -> str`: Normalized relative path hashing that cleanly trims root paths without substring collisions on directory names ending in `game`.
  * `read_rpyc_data(source, slot) -> bytes | None`: Safely handles corrupt or partial zlib compressed streams by returning `None` instead of raising unhandled `zlib.error`.
  * Removal of obsolete `@pytest.mark.xfail` markers in `tests/test_empirical_challenger_m2.py`.

- [ ] **Step 1: Write/Update the test asserting full pass on adversarial RPYC conditions**

```python
def test_adversarial_path_with_game_suffix_directory_passes_cleanly() -> None:
    expected = compute_canonical_say_identifier("start", "game/script.rpy", 10, "Hello")
    adversarial_path = "/home/user/renpy_game/game/script.rpy"
    actual = compute_canonical_say_identifier("start", adversarial_path, 10, "Hello")
    assert actual == expected

def test_adversarial_read_rpyc_data_corrupt_zlib_stream_returns_none() -> None:
    corrupt_rpyc = bytearray(create_rpyc({1: b"valid payload"}))
    corrupt_rpyc[40:50] = b"\xff" * 10
    bio = io.BytesIO(corrupt_rpyc)
    result = read_rpyc_data(bio, 1)
    assert result is None
```

- [ ] **Step 2: Run test to verify behavior without xfail**

Remove `@pytest.mark.xfail` from `tests/test_empirical_challenger_m2.py` and run:
Run: `.venv/bin/pytest tests/test_empirical_challenger_m2.py -v`
Expected: Verify both tests pass without `XPASS` warnings.

- [ ] **Step 3: Harden path normalization and zlib decompression in `extractor.py` and `archive.py`**

1. In `src/renslate/renpy/rpyc/extractor.py`:
   - Replace naive `path.index("game/")` with reverse segment matching (`parts = Path(filename).parts; if "game" in parts: idx = len(parts) - 1 - parts[::-1].index("game")`) to strictly bind to the innermost `game/` folder.
2. In `src/renslate/renpy/rpyc/archive.py`:
   - Wrap `zlib.decompress` in `read_rpyc_data` within `try...except (zlib.error, struct.error): return None` ensuring symmetry with `RpycFile.read_slot`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_empirical_challenger_m2.py tests/test_rpyc.py -v`
Expected: 100% PASS with 0 XPASS.

- [ ] **Step 5: Commit**

```bash
git add src/renslate/renpy/rpyc/ tests/test_empirical_challenger_m2.py
git commit -m "fix(rpyc): harden canonical path normalization against game directory collisions and safe corrupt zlib decompression"
```

---

### Task 5: End-to-End Hexagonal Callsite Audit & Legacy Purge

**Files:**
- Modify: `src/renslate/cli.py`
- Modify: `src/renslate/engine/translate.py`
- Modify: `src/renslate/ui/wizard.py`
- Test: Full test suite (`pytest`)

**Interfaces:**
- Consumes: All modernized modules across domain, engine, renpy, and adapters.
- Produces:
  * 100% clean call paths verifying no runtime code silently falls back to legacy or ad-hoc heuristics.
  * Pristine code style: 0 ruff warnings, 0 type errors, 100% test suite passing.

- [ ] **Step 1: Write an architectural callsite audit test verifying runtime imports**

```python
def test_zero_stealth_fallbacks_in_cli_and_translate() -> None:
    import inspect
    from renslate.engine.translate import TranslationPipeline
    from renslate.cli import app
    source_pipeline = inspect.getsource(TranslationPipeline)
    assert "pronoun_map" not in source_pipeline
    assert "glossary" not in source_pipeline
```

- [ ] **Step 2: Run test to verify it fails if any legacy heuristic remains**

Run: `.venv/bin/pytest tests/test_cli.py -k "test_zero_stealth_fallbacks" -v`

- [ ] **Step 3: Audit all callsites across CLI, engine, and UI**

1. Verify that `TranslationPipeline` uses `reconcile_translation` from `renslate.renpy.mask` and `HallucinationScanner` from `renslate.engine.hallucination`.
2. Verify that `TranslationMemory` utilizes FTS5 Trigram indexing.
3. Clean up any obsolete docstrings or commented-out logic.

- [ ] **Step 4: Run full test suite and linters to verify 100% green status**

Run: `.venv/bin/pytest -q`
Expected: 2,094+ passed, 0 failed, 0 xpass.
Run: `.venv/bin/ruff check src/ tests/ scripts/`
Expected: All checks passed!

- [ ] **Step 5: Commit**

```bash
git add src/renslate/ tests/
git commit -m "refactor(architecture): complete hexagonal callsite audit and eliminate legacy fallbacks"
```
