# Translation Architecture Overhaul Checkpoint (2026+)

**Status:** ACTIVE
**Active Branch:** `feature/hexagonal-discourse-engine`
**Last Updated:** 2026-11-05T10:46:00+07:00
**Quality Rules Enforced:** [AGENTS.md](file:///home/rakuya/Documents/Renvi/AGENTS.md) (8 Pillars)

---

## Architecture Blueprint

```
                       [ REN'PY AST GROUNDING ENGINE ]
                   (Scene Boundaries, StageTracker, Dyads)
                                     │
                                     ▼
      ┌─────────────────────────────────────────────────────────────┐
      │ Task 1: AST Stage-Grounded Multi-Turn Dialogue Memory       │
      │ ── Module: src/renslate/engine/context/narrative_memory.py  │
      │ ── Test: tests/test_narrative_memory.py                     │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
                                     ▼
      ┌─────────────────────────────────────────────────────────────┐
      │ Task 4: TransGraph Dyad-Aware Dynamic TM Exemplars          │
      │ ── Module: src/renslate/engine/context/exemplars.py         │
      │ ── Test: tests/test_exemplars.py                            │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
                                     ▼
                 [ PASS 1: TRANSGRAPH-CONDITIONED DRAFT ]
             (Dịch thô nhanh có chỉ thị cấu trúc đối chiếu)
                                     │
                                     ▼
      ┌─────────────────────────────────────────────────────────────┐
      │ Task 2: Reference-Free Syntactic Calque Density Scoring     │
      │ ── Module: src/renslate/engine/qa/calque_density.py         │
      │ ── Test: tests/test_calque_density.py                       │
      │ ── TUYỆT ĐỐI KHÔNG DÙNG REGEX THAY THẾ CHUỖI TIẾNG VIỆT     │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
       D_calque < 0.35 & QA OK             D_calque >= 0.35 hoặc cờ QA
       (~85-90% số câu thoại)                 (~10-15% số câu khó nhất)
                     │                               │
                     ▼                               ▼
            [ GIỮ NGUYÊN BẢN DỊCH ]     ┌─────────────────────────────┐
             (Tốc độ tối đa, 1-Pass)     │ Task 3: Confidence-Gated    │
                     │                  │ Deliberative Reflection     │
                     │                  │ (LLM viết lại tự nhiên)     │
                     │                  │ src/renslate/engine/        │
                     │                  │ deliberate.py               │
                     │                  └──────────────┬──────────────┘
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
                                      ▼
                        [ BẢN DỊCH THUẦN VIỆT ĐẠT CHUẨN ]
```

---

## State Matrix

| Task | Module | Scope & Files | State | Commit |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1** | Grounded Narrative Memory | `src/renslate/engine/context/narrative_memory.py`, `tests/test_narrative_memory.py` | COMPLETED | Verified (9 tests passing) |
| **Task 2** | Calque Density Scoring | `src/renslate/engine/qa/calque_density.py`, `tests/test_calque_density.py` | COMPLETED | Verified (7 tests passing, 0 regex mutation) |
| **Task 4** | TransGraph TM Exemplars | `src/renslate/engine/context/exemplars.py`, `tests/test_exemplars.py` | COMPLETED | Verified (10 tests passing, 0 speculation) |
| **Task 3** | Deliberative Reflection | `src/renslate/engine/deliberate.py`, `tests/test_deliberate.py` | COMPLETED | Verified (10 tests passing, 25% budget cap) |
| **Task 5** | Pipeline Integration & E2E | `src/renslate/engine/translate.py`, `src/renslate/engine/context/builder.py`, `tests/e2e/test_discourse_e2e_suite.py` | COMPLETED | Verified (77 E2E tests passing, 0 regression) |

---

## Recovery Protocol
If the agent context window reaches limit or resets:
1. Read this checkpoint file (`docs/CHECKPOINT_TRANSLATION_ARCHITECTURE.md`).
2. Run `.venv/bin/pytest -q` and `git status` to verify current green state.
3. Resume the first non-completed task in the State Matrix according to the implementation plan in `docs/superpowers/plans/2026-11-05-literary-translation-quality-overhaul.md`.
