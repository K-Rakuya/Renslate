# Modernized Hexagonal CLI Architecture Overhaul Plan (2026+)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the entire Command-Line Interface (CLI) of Renslate, decomposing the monolithic `src/renslate/cli.py` (909 lines) into a modular, testable Hexagonal Primary Driving Adapter (`src/renslate/cli/`), exposing 2026 modern literary features (2-Pass Deliberation, AST Narrative Memory, Dyad TM Exemplars, TransGraph), eradicating legacy dead code and duplicate aliases, while preserving 100% backward compatibility across all existing commands and tests.

**Architecture:** Hexagonal Primary Driving Adapter pattern (Ports & Adapters). Decouple CLI argument parsing, presentation formatting, and application use-case execution. The CLI acts strictly as an input translator and output presenter, delegating domain and workflow logic to application services with zero inline SQLite queries.

**Tech Stack:** Pure Python 3.10+ Standard Library (`argparse`, `dataclasses`, `pathlib`, `sys`, `logging`), zero third-party runtime dependencies.

**Spec:** [AGENTS.md](file:///home/rakuya/Documents/Renvi/AGENTS.md) (8 Quality Pillars), [ARCHITECTURE_DECISIONS.md](file:///home/rakuya/Documents/Renvi/docs/ARCHITECTURE_DECISIONS.md) (ADR-001, ADR-021, ADR-022, ADR-032).

---

## Global Constraints

- **Pure Python 3.10+ Standard Library:** `dependencies = []` in `pyproject.toml`, zero external CLI libraries (no Click, no Typer, no Rich).
- **100% Backward Compatibility:** All existing CLI invocations (`renslate init`, `extract`, `translate`, `qa`, `apply`, `run`, `status`, `dashboard`, `decompile`, `font`, `export`, `import`, `preview`, `clean-db`) and their arguments must retain identical behavior and exit codes.
- **Zero Inline SQL Queries in CLI:** CLI handlers must never invoke `database.execute("SELECT ...")` or bypass domain boundaries.
- **Zero Legacy Glossary:** Absolute zero mentions or fallbacks to glossary in CLI flags, help text, or handlers.
- **Clean Code & World-Class Ergonomics:** Group commands into clear lifecycle phases with `argparse.BooleanOptionalAction`, concise ANSI styling with automatic non-TTY `--plain` fallback, and full typing.
- **Zero Regression:** All 2,132 existing unit, adversarial, and e2e tests must continue to pass (0 failure).

---

## Review Focus

1. **Terminal TTY vs Non-TTY / Pipe Detection:** Running `renslate status | grep ...` or redirecting stdout to file must automatically disable ANSI escape codes without crashing or mangling output.
2. **Missing Project Context Resolution:** Running commands outside a valid Ren'Py project root must print standard Vietnamese error messages to `stderr` and exit with code 1 instead of raising unhandled Python tracebacks.
3. **Boolean Optional Action Consistency:** Modern `--deliberate` / `--no-deliberate`, `--guard` / `--no-guard`, and `--bootstrap` / `--no-bootstrap` flags must correctly parse without conflicting defaults or double-negation ambiguities.
4. **Bytecode vs Script Uniformity:** `renslate extract` and `renslate decompile` must seamlessly process both raw `.rpy` scripts and compiled `.rpyc` bytecode without requiring separate manual configuration.
5. **Atomic Error Handling & Exit Codes:** Validation failures in QA or missing files in `import` must return appropriate non-zero exit codes (1 or 2) conforming to Unix CLI conventions.

---

## Structural Decomposition (Before vs After)

### Before (Monolithic 909-line file):
```
src/renslate/
└── cli.py  (909 lines: Parser + 18 Handlers + Direct DB SQL + ANSI codes + TUI launcher)
```

### After (Hexagonal Primary Adapter Subpackage):
```
src/renslate/
├── cli/
│   ├── __init__.py        # Public API export: main(argv), __version__
│   ├── parser.py          # Hierarchical, ergonomic argparse configuration
│   ├── formatter.py       # Standardized console output, issue printing, ANSI helpers
│   └── handlers/
│       ├── __init__.py    # Registry of command handlers
│       ├── project.py     # init, generate, extract, apply
│       ├── translation.py # translate, run, preview
│       ├── discourse.py   # analyze, bootstrap
│       ├── qa.py          # qa, clean_db
│       ├── data.py        # export, import
│       └── tools.py       # decompile, font, dashboard, wizard
└── cli.py                 # 15-line forwarding shim for backward compatibility
```

---

## Command Hierarchy & Modernized Feature Matrix

| Lifecycle Phase | Subcommand | Core Capabilities & Modern 2026 Flags | Backward Alias |
| :--- | :--- | :--- | :--- |
| **Workflow** | `run` | Automated pipeline with live TUI, `--deliberate`, `--guard`, `--concise-tags`, `--retry-failed` | `auto` |
| | `init` | Project creation, target language, Ren'Py SDK configuration | - |
| | `extract` | AST extraction from `.rpy` and `.rpyc` bytecode into SQLite | - |
| | `translate` | Batch translation with 2-Pass Deliberation, `--calque-threshold`, `--exemplars`, `--narrative-memory` | - |
| | `apply` | Atomic, fingerprint-verified write to game script with backup rotation | - |
| **Discourse** | `analyze` | AST structural analysis, speaker detection, co-occurrence matrix | - |
| | `bootstrap` | LLM 4D relational vector $\Pi(\mathbf{R})$ distillation & JIT scene summaries | - |
| **Quality** | `qa` | Multi-phase validation: tags, invariants, calque density scoring, continuous hallucination guard | - |
| | `preview` | Bilingual chronological or recent translation viewer | - |
| | `status` | Project summary metrics, slot counts, status matrix | - |
| **Exchange** | `export` | TSV dataset export with status and QA flag filters | - |
| | `import` | Human review TSV ingestion with dual-mode reconciliation | - |
| | `clean-db` | Database sanitization, deduplication, and label normalization | - |
| **Tools** | `decompile` | Pure Python Ren'Py 6/7/8 bytecode decompiler (.rpyc $\rightarrow$ .rpy) | - |
| | `font` | Automated Vietnamese font discovery and replacement mapping | - |
| | `dashboard` | Standalone real-time ANSI terminal TUI dashboard | - |
| | `wizard` | Interactive multi-step project onboarding setup | `interactive` |

---

## Tasks

### Task 1: CLI Console Output Formatter & ANSI Abstraction

**Files:**
- Create: `src/renslate/cli/formatter.py`
- Test: `tests/test_cli_formatter.py`

**Interfaces:**
- Consumes: Standard terminal streams (`sys.stdout`, `sys.stderr`), `Issue` dataclasses, `ExtractReport`, `ApplyReport`, `TranslationStats`.
- Produces:
  * `ConsoleFormatter`: Centralized, clean terminal output with automatic ANSI stripping when `not isatty()` or `--plain`.
  * `print_issues(issues, max_shown=20)`
  * `print_extract_report(report)`
  * `print_apply_report(report)`
  * `print_translation_stats(stats)`

- [ ] **Step 1: Write failing unit test for `ConsoleFormatter`**

```python
def test_console_formatter_plain_mode(capsys: pytest.CaptureFixture[str]) -> None:
    from renslate.cli.formatter import ConsoleFormatter
    fmt = ConsoleFormatter(plain=True)
    fmt.info("Hello world")
    assert capsys.readouterr().out == "Hello world\n"
```

- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `src/renslate/cli/formatter.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit `feat(cli): implement centralized console formatter`**

---

### Task 2: Domain-Driven Command Handlers Decomposition

**Files:**
- Create: `src/renslate/cli/handlers/__init__.py`
- Create: `src/renslate/cli/handlers/project.py` (init, generate, extract, apply)
- Create: `src/renslate/cli/handlers/translation.py` (translate, run, preview)
- Create: `src/renslate/cli/handlers/discourse.py` (analyze, bootstrap)
- Create: `src/renslate/cli/handlers/qa.py` (qa, clean_db)
- Create: `src/renslate/cli/handlers/data.py` (export, import)
- Create: `src/renslate/cli/handlers/tools.py` (decompile, font, dashboard, wizard)
- Test: `tests/test_cli_handlers.py`

**Interfaces:**
- Each handler signature: `(args: argparse.Namespace) -> int`.
- Strictly delegates to Application Services (`TranslationPipeline`, `run_unattended_pipeline`, `run_analysis`, `QAChecker`, etc.).
- Zero direct raw SQL queries in handlers.

- [ ] **Step 1: Write failing unit tests for decoupled command handlers**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement decoupled handler modules under `src/renslate/cli/handlers/`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit `refactor(cli): decompose command handlers into domain-aligned modules`**

---

### Task 3: Ergonomic Argument Parser with 2026 Quality Flags & Grouping

**Files:**
- Create: `src/renslate/cli/parser.py`
- Test: `tests/test_cli_parser.py`

**Interfaces:**
- `build_parser() -> argparse.ArgumentParser`
- Support modern 2026 quality flags:
  * `--deliberate` / `--no-deliberate` (default: True, gated by confidence)
  * `--calque-threshold` (default: 0.35)
  * `--exemplars` (default: 2)
  * `--narrative-memory` (default: 6)
  * `--transgraph` / `--no-transgraph` (default: True)
- Group help outputs logically: *Workflow Commands*, *Discourse Engine*, *Quality & Audit*, *Data Exchange*, *Tools*.
- Maintain exact flag names and aliases for all existing tests (`-p`, `--lang`, `--limit`, `--retry-failed`, etc.).

- [ ] **Step 1: Write failing unit tests for parser configuration and new 2026 flags**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `src/renslate/cli/parser.py`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit `feat(cli): implement ergonomic parser with 2026 literary quality controls`**

---

### Task 4: Package Assembly, Backward Forwarding Shim & Entrypoint Integration

**Files:**
- Create: `src/renslate/cli/__init__.py`
- Modify: `src/renslate/cli.py` (transform into a lightweight backward-compatibility shim)
- Test: `tests/test_cli.py`, `tests/test_cli_commands.py`

**Interfaces:**
- `src/renslate/cli/__init__.py` exposes `main(argv=None) -> int` and `__version__`.
- `src/renslate/cli.py` re-exports `main`, `_build_parser` to ensure 100% backward compatibility for existing scripts, tests, and CI/CD pipelines.

- [ ] **Step 1: Run existing CLI test suites against the new modular package**
- [ ] **Step 2: Implement `src/renslate/cli/__init__.py` and backward-compatible `src/renslate/cli.py` shim**
- [ ] **Step 3: Run full CLI test suite (`tests/test_cli.py`, `tests/test_cli_commands.py`, `tests/test_adversarial_runner_ui.py`)**
- [ ] **Step 4: Verify 100% pass rate**
- [ ] **Step 5: Commit `feat(cli): complete modular CLI packaging with backward forwarding shim`**

---

### Task 5: Full Regression Testing & Real-World VN Verification

**Files:**
- Modify: `tests/test_cli_commands.py` (add tests for new 2026 literary quality CLI flags)
- Test: Full project test suite (`pytest`)

- [ ] **Step 1: Add new test scenarios verifying CLI invocations with `--deliberate`, `--calque-threshold`, `--exemplars`**
- [ ] **Step 2: Run full test suite across the entire repository (`.venv/bin/pytest -q`)**
  * Assert: 2,140+ tests passed, 0 failed.
- [ ] **Step 3: Run linter verification (`.venv/bin/ruff check src/ tests/ scripts/`)**
  * Assert: All checks passed!
- [ ] **Step 4: Commit and push changes cleanly**
