# High-Fidelity Literary Translation Quality Overhaul (Revision 3 - Leveraging 2026 Subagent Architectures)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Radically maximize Renslate's visual novel translation quality into natural, publication-grade Vietnamese by integrating and leveraging the cutting-edge 2026 architectures established in the recent subagent runs: Ren'Py Structural AST Grounding, 4D Relational Vector Algebra $\Pi(\mathbf{R})$, TransGraph Micro-Framing, Confidence-Gated Deliberative Reflection (latency-optimized), Reference-Free Syntactic Calque Scoring (zero regex replacement), and Document-Level Multi-Turn Dialogue Memory.

**Architecture:** Hexagonal domain-driven architecture in 100% pure Python 3.10+ standard library. Directly bridges the AST Grounding Engine (`grounding.py`), 4D Relational Vector Projections (`relational.py`), and TransGraph Micro-Frames (`builder.py`) into the runtime translation pipeline. Replaces flat single-pass literal generation with a high-throughput, context-grounded cognitive workflow: multi-turn dialogue memory and dyad-grounded TM exemplars steer the first-pass draft, while an uncertainty/calque confidence gate selectively routes only complex or high-ambiguity units (~10-15% of lines) to a fast deliberative reflection pass, preserving maximum inference speed and zero hardcoding.

**Tech Stack:** Pure Python 3.10+ Standard Library (`dependencies = []`), SQLite 3 WAL mode with composite indexes, `urllib.request` persistent keep-alive connection pooling, `dataclasses` with slots, `pytest`, `ruff`.

**Spec:** [AGENTS.md](file:///home/rakuya/Documents/Renvi/AGENTS.md) (8 Renvi Quality Pillars), AST Grounding Engine (`src/renslate/renpy/grounding.py`), 4D Relational Vector Domain (`src/renslate/domain/relational.py`), and WMT@ACL 2025/2026 (*Lost in Mimicry: Verified Training-Free Post-Editing for LLM Translationese*).

## Global Constraints

- **Pure Python Standard Library:** 100% Python standard library (`dependencies = []` in `pyproject.toml`).
- **Deep Integration with 2026 Subagent Modules:** Must directly reuse and extend `grounding.py`, `relational.py`, `stage.py`, `mob_roles.py`, and `build_transgraph_frame()`.
- **Zero Hardcoding & Zero Speculation:** No hardcoded character names, no game-specific aliases, and no speculative idiolect profiles. 100% derived from AST syntax, spatial presence, relational vectors, and verified project TM entries.
- **Zero Regex String Replacements on Target Text:** Absolutely NO naive regex find-and-replace for Vietnamese grammatical words (`bị`, `được`, `của`). Anti-translationese is achieved via prompt contrastive directives and LLM deliberative re-writing.
- **Latency & Throughput Protection:** Deliberative reflection must be confidence-gated (triggered only on ~10-15% of units), ensuring pipeline throughput remains high on local single-GPU LLMs.
- **Zero Benchmark Leakage:** Benchmark paths and game titles must never appear in code, tests, docs, or git history.
- **Mathematical Reciprocal Invariant:** Vietnamese pronoun pairing must strictly satisfy algebraic symmetry: $\Pi(\mathbf{R}_{(B, A)}, g_B, g_A) \equiv \sigma(\Pi(\mathbf{R}_{(A, B)}, g_A, g_B))$.
- **Clean Code & Self-Documenting:** No commentary apologies, workaround flags, or junk comments. Concise docstrings only for public protocols.

## Review Focus

1. **AST Stage Grounding Continuity:** Dialogue units loaded for translation must carry their physical stage context (`active_sprites`, `scene_id`, and speaker expressions from `StageTracker`) into the multi-turn dialogue memory.
2. **Confidence Gate Selectivity:** The deliberative reflection pass must trigger strictly on complex, ambiguous, or calque-heavy dialogue lines, never blindly doubling latency across 100% of units.
3. **Grammar Integrity without Regex Calque Replacement:** Anti-translationese must never blindly strip legitimate grammatical uses of `bị` (e.g. `bị thương`), `được` (e.g. `được tặng quà`), or `của` (e.g. `lựa chọn của tôi`).
4. **Dyad-Aware TM Exemplar Grounding:** In-context exemplars must be dynamically retrieved from the project's own verified SQLite TM matching the active TransGraph dyad $(A, B)$ rather than static or hallucinated examples.
5. **JSON Output Stability:** Deliberative reflection and batch outputs must return strictly valid JSON key-value pairs conforming to unit IDs without markdown commentary or key drift.

---

### Task 1: AST Stage-Grounded Multi-Turn Dialogue Memory

**Files:**
- Create: `src/renslate/engine/context/narrative_memory.py`
- Modify: `src/renslate/engine/context/builder.py`
- Modify: `src/renslate/engine/translate.py`
- Test: `tests/test_narrative_memory.py`

**Interfaces:**
- Consumes: AST-grounded dialogue units (`GroundedDialogue` from `renslate.renpy.grounding`), `StageTracker` active sprites, speaker/listener dyad, and target translations.
- Produces:
  * `GroundedDialogueTurn`: Stores `speaker`, `listener`, `source_text`, `target_text`, `scene_id`, and `active_sprites` (with expressions like `blush`, `angry`, `smile`).
  * `NarrativeMemory`: Ring buffer tracking the last $K$ turns (default $K=6$) within the active scene. Automatically flushes on scene boundaries (`_is_scene_boundary`).
  * `format_multi_turn_history(current_scene_id: str) -> str`: Formats prior conversational turns into a compact dialogue history section within the prompt to eliminate translation amnesia.

- [ ] **Step 1: Write failing tests for AST stage-grounded dialogue memory**

```python
def test_narrative_memory_tracks_ast_grounded_sprites_and_dialogue() -> None:
    from renslate.engine.context.narrative_memory import NarrativeMemory, GroundedDialogueTurn
    mem = NarrativeMemory(capacity=4)
    mem.record(GroundedDialogueTurn(
        speaker="eileen",
        listener="mc",
        source="Don't look at me like that!",
        target="Đừng nhìn em như thế chứ!",
        scene_id="bg_bedroom",
        active_sprites=("eileen blush", "mc neutral"),
    ))
    
    history = mem.format_history(current_scene="bg_bedroom")
    assert "eileen -> mc: Don't look at me like that! => Đừng nhìn em như thế chứ!" in history
    assert "eileen (blush)" in history

def test_narrative_memory_clears_on_scene_boundary() -> None:
    from renslate.engine.context.narrative_memory import NarrativeMemory, GroundedDialogueTurn
    mem = NarrativeMemory(capacity=4)
    mem.record(GroundedDialogueTurn(
        speaker="eileen",
        listener="mc",
        source="Bye.",
        target="Tạm biệt anh.",
        scene_id="bg_bedroom",
    ))
    # Boundary: scene changed to bg_street
    assert mem.format_history(current_scene="bg_street") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_narrative_memory.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `NarrativeMemory` in `src/renslate/engine/context/narrative_memory.py`**

1. Create `src/renslate/engine/context/narrative_memory.py`:
   - Implement `GroundedDialogueTurn` dataclass with slots.
   - Implement `NarrativeMemory` ring buffer using `collections.deque(maxlen=capacity)`.
   - Implement `format_history` outputting compact prior turns with speaker emotional cues for prompt injection.
2. In `src/renslate/engine/context/builder.py`:
   - Inject `### BỐI CẢNH ĐỐI THOẠI LIỀN TRƯỚC (RECENT CONVERSATIONAL TURNS)` into the batch prompt when dialogue history is available.
3. In `src/renslate/engine/translate.py`:
   - Feed completed translation turns into `NarrativeMemory`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_narrative_memory.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/renslate/engine/context/narrative_memory.py src/renslate/engine/context/builder.py src/renslate/engine/translate.py tests/test_narrative_memory.py
git commit -m "feat(context): implement ast stage-grounded multi-turn dialogue memory"
```

---

### Task 2: Reference-Free Syntactic Calque Density Scoring & Contrastive Prompting

**Files:**
- Create: `src/renslate/engine/qa/calque_density.py`
- Modify: `src/renslate/engine/context/builder.py`
- Modify: `src/renslate/engine/qa.py`
- Test: `tests/test_calque_density.py`

**Interfaces:**
- Consumes: Source text and candidate translation text.
- Produces:
  * `compute_calque_density(source: str, target: str) -> float`: Calculates normalized syntactic calque score ($0.0 \le D_{\text{calque}} \le 1.0$) based on structural English mimicry:
    - Suspicious passives without adversity/benefit context (`bị/được ... bởi ...`).
    - Word-order rigid mirroring (subclause fronting absence).
    - Unnatural phrase starter calques (`Chà,`, `Bạn thấy đấy,`, `Nói cách khác,`).
  * ZERO regex string mutation: Never performs regex replacements on target text; only returns the numeric density score for confidence gating.
  * Contrastive Structural Prompt Directives in `ContextBuilder.QUOTED_RULES`: Directives instructing the model to prefer native Vietnamese topic-comment structures and active constructions.

- [ ] **Step 1: Write failing tests for reference-free calque density scoring**

```python
def test_calque_density_flags_stiff_formulaic_sentences() -> None:
    from renslate.engine.qa.calque_density import compute_calque_density
    
    source = "Well, I was told by him that you failed."
    stiff_target = "Chà, tôi đã bị nói bởi anh ấy rằng bạn đã thất bại."
    score_stiff = compute_calque_density(source, stiff_target)
    
    natural_target = "Anh ấy có bảo với tôi là cậu trượt rồi đấy."
    score_natural = compute_calque_density(source, natural_target)
    
    assert score_stiff > 0.4
    assert score_natural < 0.15
    assert score_stiff > score_natural

def test_calque_density_does_not_penalize_legitimate_passives() -> None:
    from renslate.engine.qa.calque_density import compute_calque_density
    source = "He was injured in the battle."
    target = "Anh ấy bị thương trong trận chiến."
    assert compute_calque_density(source, target) < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_calque_density.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `compute_calque_density` in `src/renslate/engine/qa/calque_density.py`**

1. Create `src/renslate/engine/qa/calque_density.py`:
   - Implement token-level linguistic features:
     * Flag formulaic English calque starters (`Chà,`, `Bạn thấy đấy,`, `Như bạn biết,`).
     * Flag passive markers `bị/được` immediately preceding agent prepositions `bởi` (classic English passive calque `bị/được ... bởi ...`).
     * Calculate ratio against word count.
   - Return clean scalar float score without altering the text.
2. In `src/renslate/engine/context/builder.py`:
   - Enhance `QUOTED_RULES` Rule 3 with concise contrastive structural guidelines:
     *"Chuyển đổi linh hoạt câu bị động tiếng Anh sang câu chủ động hoặc câu biểu đạt trải nghiệm tự nhiên trong tiếng Việt; tránh dùng 'bị/được ... bởi ...'; lược bỏ từ 'của' khi ngữ cảnh đã rõ quyền sở hữu."*

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_calque_density.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/renslate/engine/qa/calque_density.py src/renslate/engine/context/builder.py tests/test_calque_density.py
git commit -m "feat(qa): implement reference-free syntactic calque density scoring and contrastive structural prompting"
```

---

### Task 3: Confidence-Gated Deliberative Reflection Engine (TransGraph-Conditioned)

**Files:**
- Create: `src/renslate/engine/deliberate.py`
- Modify: `src/renslate/engine/translate.py`
- Test: `tests/test_deliberate.py`

**Interfaces:**
- Consumes: Drafted batch translations, active `build_transgraph_frame()` interaction subgraphs, `compute_calque_density` scores, and `QAChecker` validation issues.
- Produces:
  * `DeliberativeGate`: Evaluates each unit in a completed batch. Only flags a unit for reflection if:
    a) `calque_density >= 0.35`, OR
    b) `QAChecker` reports a pronoun or length anomaly against 4D Relational Vector $\Pi(\mathbf{R})$, OR
    c) Line has high rhetorical complexity (questions addressed to interlocutor with complex clauses or sarcasm markers).
  * `DeliberativeReflector.reflect_batch(units, drafts, reasons, transgraph_frame) -> dict[str, str]`: Executes a single, compact, fast LLM call refining ONLY the flagged subset of units (~10-15% of total volume).
  * Replaces drafted text with refined text only when the reflection pass produces valid, high-quality Vietnamese text satisfying markup invariants.

- [ ] **Step 1: Write failing tests for confidence gating selectivity and reflection execution**

```python
def test_deliberative_gate_selects_only_anomalous_or_complex_units() -> None:
    from renslate.core.models import Unit, UnitKind
    from renslate.engine.deliberate import DeliberativeGate
    
    gate = DeliberativeGate()
    
    # Simple clear line -> pass through, NO reflection
    u_simple = Unit(uid="1", source="I agree.", kind=UnitKind.DIALOGUE)
    assert not gate.should_reflect(u_simple, "Tôi đồng ý.", calque_score=0.0, qa_issues=[])
    
    # High calque density -> trigger reflection
    u_calque = Unit(uid="2", source="I was surprised by your arrival.", kind=UnitKind.DIALOGUE)
    assert gate.should_reflect(u_calque, "Tôi đã bị làm cho ngạc nhiên bởi sự đến của bạn.", calque_score=0.5, qa_issues=[])

def test_deliberative_reflector_preserves_unflagged_drafts() -> None:
    from renslate.core.models import Unit, UnitKind
    from renslate.engine.deliberate import DeliberativeReflector
    
    reflector = DeliberativeReflector(llm=None)
    drafts = {"1": "Tôi đồng ý.", "2": "Câu dịch lỗi"}
    result = reflector.refine_batch(
        units=[Unit(uid="1", source="I agree.", kind=UnitKind.DIALOGUE)],
        drafts=drafts,
        qualified_uids=set(),
        transgraph_frame="",
    )
    assert result == drafts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_deliberate.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `DeliberativeGate` and `DeliberativeReflector` in `src/renslate/engine/deliberate.py`**

1. Create `src/renslate/engine/deliberate.py`:
   - Implement `DeliberativeGate`:
     * Threshold: `calque_score >= 0.35` OR `bool(qa_issues)` OR `has_rhetorical_subtext(source)`.
     * Strict budget cap: Maximum 25% of batch units can be reflected in a single batch, protecting latency.
   - Implement `DeliberativeReflector`:
     * Construct a focused reflection prompt conditioned on active TransGraph frame: *"Hãy tinh chỉnh các câu thoại sau đây để xóa bỏ hoàn toàn văn phong dịch gượng gạo, chuyển hóa thành lời thoại tiếng Việt tự nhiên, giữ nguyên 100% placeholder."*
     * Call LLM using JSON mode.
     * Validate output through `reconcile_translation`.
2. In `src/renslate/engine/translate.py`:
   - Wire `DeliberativeGate` into `TranslationPipeline._translate_batch` after Pass 1 generation.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_deliberate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/renslate/engine/deliberate.py src/renslate/engine/translate.py tests/test_deliberate.py
git commit -m "feat(deliberate): implement transgraph-conditioned confidence-gated reflection engine"
```

---

### Task 4: TransGraph Dyad-Aware Dynamic TM Exemplar Grounding

**Files:**
- Create: `src/renslate/engine/context/exemplars.py`
- Modify: `src/renslate/engine/context/builder.py`
- Test: `tests/test_exemplars.py`

**Interfaces:**
- Consumes: Project SQLite database `tm` table (Translation Memory), active interaction dyad $(A, B)$ from `build_transgraph_frame()`, and current batch units.
- Produces:
  * `TMDyadExemplarRetriever`: Queries database for up to 2 high-confidence, verified translation pairs ($0.65 \le \text{similarity} \le 0.95$) relevant to the active speaker-listener pair in the current batch.
  * Formats dynamic few-shot exemplars inside the prompt:
    ```text
    ### CÂU MẪU ĐÃ ĐƯỢC XÁC NHẬN TRONG DỰ ÁN (PROJECT-VERIFIED EXEMPLARS):
    - "Source..." => "Target..."
    ```
  * Zero Speculation & Zero Hardcoding: Pulls strictly from actual historical translations in the project's own database. If the TM is empty, gracefully omits the section with zero overhead.

- [ ] **Step 1: Write failing tests for dyad-aware TM exemplar retrieval and prompt injection**

```python
def test_tm_dyad_exemplar_retriever_pulls_similar_verified_pairs(tmp_path: Path) -> None:
    from renslate.core.db import Database
    from renslate.core.models import Origin
    from renslate.engine.context.exemplars import TMDyadExemplarRetriever
    
    db_path = tmp_path / "test.db"
    with Database.open(db_path, create=True) as db:
        db.add_tm("What are you doing here?", "Cậu làm gì ở đây thế?", origin=Origin.USER)
        db.add_tm("I don't understand.", "Tôi không hiểu.", origin=Origin.USER)
        
        retriever = TMDyadExemplarRetriever(db)
        exemplars = retriever.retrieve_for_dyad(speaker="c", listener="mc", sources=["What are you doing?"])
        assert len(exemplars) >= 1
        assert exemplars[0].target == "Cậu làm gì ở đây thế?"

def test_context_builder_injects_tm_exemplars() -> None:
    from renslate.engine.context.builder import ContextBuilder
    from renslate.engine.context.exemplars import ExemplarItem
    
    builder = ContextBuilder()
    prompt = builder.build_system_prompt(exemplars=[ExemplarItem(source="Wait!", target="Đợi đã!")])
    assert "CÂU MẪU ĐÃ ĐƯỢC XÁC NHẬN" in prompt
    assert "Wait! => Đợi đã!" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_exemplars.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `TMDyadExemplarRetriever` in `src/renslate/engine/context/exemplars.py`**

1. Create `src/renslate/engine/context/exemplars.py`:
   - Implement `TMDyadExemplarRetriever`:
     * Match current batch source tokens against `tm` table entries.
     * Filter for high quality: `origin IN ('user', 'review')` or high reuse count.
     * Return maximum 2 compact exemplars.
2. In `src/renslate/engine/context/builder.py`:
   - Format retrieved exemplars into the system prompt.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_exemplars.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/renslate/engine/context/exemplars.py src/renslate/engine/context/builder.py tests/test_exemplars.py
git commit -m "feat(context): implement transgraph dyad-aware dynamic tm exemplar retrieval"
```

---

### Task 5: End-to-End Pipeline Integration & Benchmark Verification

**Files:**
- Modify: `tests/e2e/test_discourse_e2e_suite.py`
- Modify: `src/renslate/engine/translate.py`
- Test: Full test suite (`pytest`)

**Interfaces:**
- Consumes: All modernized translation quality components integrated with 2026 AST Grounding and TransGraph modules.
- Produces:
  * Full integration test verifying end-to-end translation execution:
    AST Grounding $\rightarrow$ Multi-Turn Memory $\rightarrow$ TransGraph Micro-Framing $\rightarrow$ Calque Density Scoring $\rightarrow$ Confidence-Gated Deliberation $\rightarrow$ Dyad TM Exemplars.
  * 100% test suite passing, 0 ruff errors.

- [ ] **Step 1: Write comprehensive e2e integration test for modernized quality pipeline**

```python
def test_modernized_literary_quality_pipeline_e2e(tmp_path: Path) -> None:
    from renslate.core.db import Database
    from renslate.core.project import Project
    from renslate.engine.translate import TranslationPipeline
    from renslate.core.models import Unit, UnitKind
    
    db_path = tmp_path / "project.db"
    with Database.open(db_path, create=True) as db:
        u1 = Unit(uid="1", source="Who is there?", kind=UnitKind.DIALOGUE, speaker="eileen")
        u2 = Unit(uid="2", source="It is me. Don't be afraid.", kind=UnitKind.DIALOGUE, speaker="mc")
        db.add_unit(u1)
        db.add_unit(u2)
        
        pipeline = TranslationPipeline(Project(tmp_path, db=db))
        assert pipeline.narrative_memory is not None
        assert pipeline.deliberative_reflector is not None
```

- [ ] **Step 2: Run full test suite to verify zero regressions**

Run: `.venv/bin/pytest -q`
Expected: 2,100+ passed, 0 failed.

- [ ] **Step 3: Run ruff linter**

Run: `.venv/bin/ruff check src/ tests/ scripts/`
Expected: All checks passed!

- [ ] **Step 4: Commit**

```bash
git add tests/ src/
git commit -m "test(e2e): complete end-to-end integration and benchmark verification of literary quality pipeline"
```
