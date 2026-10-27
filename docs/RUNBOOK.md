# Renslate Operational Runbook & Production Guide

This runbook provides step-by-step procedures for operating, automating, and troubleshooting Renslate translation pipelines in production and unattended overnight runs.

---

## 1. Prerequisites & Environment Setup

### 1.1 Local LLM Backend (llama.cpp)
Renslate is optimized for local `llama.cpp` server (`llama-server`).

**Recommended Hardware & Model Configurations:**
- **Tier 1 (High Quality MoE - ADR-009):** `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` (Hybrid RTX 4060 8GB + Ryzen 7 8845H CPU, Flash Attention, 8-bit KV Cache, Zero-SSD-wear).
- **Tier 2 (Consumer GPU / 8-12 GB VRAM):** Qwen 2.5 14B Instruct (Q4_K_M) or Llama 3.1 8B Instruct (Q6_K).
- **Tier 3 (CPU only / Low VRAM):** Qwen 2.5 7B Instruct (Q4_K_M).

**Start the Native MoE llama-server (Zero SSD Wear, GPU Maximize):**
```bash
./scripts/run_qwen_server.sh
```

Or start a custom `llama-server` manually:
```bash
./llama-server \
  -m /path/to/model.gguf \
  --ctx-size 12288 \
  --fit on --fit-target 768 \
  -fa on -ctk q8_0 -ctv q8_0 \
  -t 8 -tb 8 --cpu-range 0-7 \
  --port 8080 --host 127.0.0.1
```

Verify backend health:
```bash
curl -s http://127.0.0.1:8080/health
```

---

## 2. Standard End-to-End Pipeline

```
┌──────────┐     ┌───────────┐     ┌───────────┐     ┌──────────┐     ┌────────┐
│ init/gen │ ──> │  extract  │ ──> │  analyze  │ ──> │translate │ ──> │ apply  │
└──────────┘     └───────────┘     └───────────┘     └──────────┘     └────────┘
                                                           │
                                                           ▼
                                                     ┌──────────┐
                                                     │ export/qa│
                                                     └──────────┘
```

### Step 1: Initialize Project
Inside your Ren'Py project directory (or specifying `-p /path/to/game`):
```bash
renslate init --target-lang vietnamese
```

### Step 2: Generate Translation Templates (Optional if using Ren'Py SDK)
If you haven't generated translation files yet:
```bash
renslate generate --sdk /path/to/renpy-sdk
```

### Step 3: Extract Translation Units into SQLite
```bash
renslate extract
```

### Step 4: Semantic Analysis & Glossary Mining
```bash
renslate analyze
```
This extracts character profiles (`Character("...", color="...")`), segments visual novel scenes, and mines candidate proper nouns into `.renslate/glossary.json`.

### Step 5: Execute Translation
```bash
renslate translate
```

### Step 5.5: Preview Translated Lines (CLI)
Quickly preview bilingual translated lines directly in your terminal:
```bash
# Preview first 10 translated lines
renslate preview

# Preview latest 20 translated lines
renslate preview -n 20 --latest
```

### Step 6: Quality Assurance Check
```bash
renslate qa
```

### Step 7: Export / Import for Human Review (Optional)
```bash
# Export all lines or QA-flagged lines for human proofreading in Excel/LibreOffice
renslate export -o review.tsv --qa-only

# After editing in natural Vietnamese, re-import:
renslate import -i review.tsv
```

### Step 8: Apply Translations to Ren'Py Game
```bash
# First dry-run to verify slots and fingerprints
renslate apply --dry-run

# Apply and configure Vietnamese unicode font support in Ren'Py
renslate apply --setup-font
```

---

## 3. Autonomous Overnight Automation & Interactive Wizard

Renslate provides native autonomous execution capabilities designed for continuous, unattended overnight runs on visual novels containing tens of thousands of lines.

### 3.1 Native Autonomous Runner (`renslate run` / `renslate auto`)

The autonomous runner is integrated natively into the `renslate` CLI. Both `renslate run` and its alias `renslate auto` execute the end-to-end pipeline with automatic bootstrapping, live dashboard visualization, and resilient error recovery.

```bash
# Run autonomous translation with default chunk size (100 lines per batch cycle)
renslate run -p "/path/to/game"

# Run with auto-apply enabled (automatically updates .rpy files and sets fonts on completion)
renslate run -p "/path/to/game" --auto-apply

# Retry previously failed units and QA-flagged units
renslate run -p "/path/to/game" --retry-failed --retry-flagged

# Run in headless CI or plain logging mode (disables ANSI escape sequences)
renslate run -p "/path/to/game" --plain
```

#### CLI Options & Flags
| Option / Flag | Type | Default | Description |
|---|---|---|---|
| `-p`, `--project` | Path | `.` (Current dir) | Path to the Ren'Py project root containing `game/`. |
| `--chunk-size` | Integer | `100` | Number of translatable units processed per batch transaction cycle. |
| `--retry-failed` | Flag | `False` | Reset and re-evaluate units marked as `Status.FAILED`. |
| `--retry-flagged` | Flag | `False` | Reset and re-evaluate units marked as `Status.QA_FLAGGED`. |
| `--auto-apply` | Flag | `False` | Automatically execute `apply_translations(project, db)` upon 100% completion. |
| `--plain`, `--no-dashboard` | Flag | `False` | Disable ANSI VT100 screen updates; stream clean line logs (ideal for CI/systemd). |
| `-i`, `--interactive` | Flag | `False` | Launch the interactive setup wizard before starting the pipeline. |

#### Real-Time ANSI/VT100 Dashboard Interpretation
When running in an interactive terminal (TTY), Renslate displays a multi-panel dashboard (`LiveDashboard`):

1. **Startup Banner (80-column bordered card):**
   - **Project Path & Language**: Displays the canonical project directory and target language (`VIETNAMESE`).
   - **LLM Endpoint & Active Model**: Probes `llama-server` and identifies active GGUF model name (e.g. `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf`).
   - **Unit Inventory**: Displays total lines, already translated lines (green), pending lines (yellow), and failed lines (red).
   - **Baseline Progress**: Current completion percentage with fractional Unicode block meter.

2. **Real-Time Per-Batch Status Line:**
   - **Batch Chunk Progress**: `Lô #N (+100 dòng trong 14.2s)` with batch processing latency.
   - **Instant Throughput**: Current batch speed (e.g. `7.04 dòng/s` or `422 dòng/phút`).
   - **Global Progress Bar**: Fractional sub-character rendering `[████████▍░░░░] 62.5%` using 1/8th-cell glyphs (`[" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]`).
   - **Completion & ETA**: Ratio of translated/total units, remaining line count, and dynamically calculated time-to-completion (ETA).
   - **Lookback State**: Shows the active dialogue lookback buffer count ensuring narrative continuity.

3. **Session Completion Summary Card:**
   - **Status Badge**: `Hoàn tất phiên dịch` (green) or `Tạm dừng bởi người dùng (Ctrl+C)` (yellow).
   - **Elapsed Time**: Total duration formatted as `HHh MMm SSs`.
   - **Throughput Averages**: Average translation throughput in lines/second and lines/minute.
   - **Persistence Confirmation**: Verification of atomic SQLite WAL transaction commits.

#### Graceful Interrupt Mechanics (`Ctrl+C` / SIGINT)
Long translation runs can be interrupted at any time without data loss:
- **First `Ctrl+C` (Graceful Pause)**:
  - Trapped by `_sig_handler`.
  - Sets `_STOP_REQUESTED = True`.
  - The runner allows the current in-flight LLM batch to finish, writes all translated units into the SQLite WAL database within an ACID transaction, and renders the session summary card.
  - Exits cleanly with status code `0` (or `130`). No partial strings or corrupted JSON are stored.
- **Second `Ctrl+C` (Immediate Exit)**:
  - If the LLM socket is stalled or unresponsive, pressing `Ctrl+C` a second time triggers an immediate forced exit via `os._exit(130)`.

---

### 3.2 Interactive Setup Wizard (`renslate wizard` / `renslate interactive`)

For first-time users or new projects, Renslate provides a guided terminal wizard:

```bash
# Launch interactive wizard
renslate wizard

# Or via alias
renslate interactive

# Or launch wizard directly from run
renslate run -i
```

#### Guided Workflow Steps:
1. **Project Directory Verification**:
   - Prompts for project path.
   - Checks for the existence of `game/` and valid Ren'Py scripts (`.rpy`).
   - Automatically initializes `.renslate/config.json` if missing.
2. **Local LLM Server Discovery**:
   - Probes `http://127.0.0.1:8080/health` (or configured `base_url`).
   - If the server is offline, prompts the user to start `llama-server` and offers auto-retry polling.
3. **Model Parameter & Architecture Inspection**:
   - Queries `/props` or `/v1/models` to detect active model name and context limits (`n_ctx`).
   - Displays active model details in green bold text.
4. **Execution Parameters Configuration**:
   - Prompts for batch chunk size (default: 100).
   - Prompts for auto-apply on completion (`y/N`).
   - Prompts for retrying previously failed or QA-flagged units.
5. **Immediate Launch**:
   - On confirmation, seamlessly hands execution over to `renslate.ui.runner.run_unattended_pipeline`.
   - If executed in a non-interactive pipe or headless script, the wizard validates the project structure and exits gracefully without raising unhandled exceptions.

---

### 3.3 Script Delegation Wrapper (`scripts/unattended_runner.py`)

For backward compatibility with existing crontabs, CI pipelines, and operator workflows, `scripts/unattended_runner.py` is maintained as an ergonomic thin delegation wrapper (~80 lines):

```bash
# Legacy syntax remains 100% functional
python scripts/unattended_runner.py translate -p "/path/to/game" --chunk-size 100 --auto-apply
```

#### Architectural Characteristics:
- **CLI Forwarding**: Parses command-line arguments and forwards execution directly to `renslate.cli.main(["run", *raw_args])`.
- **Public API Re-Exports**: Retains public exports of all 7 core helper functions to ensure uninterrupted test compatibility:
  - `check_server_health(base_url, timeout_s=3.0) -> bool`
  - `detect_server_model(base_url, default_name, timeout_s=3.0) -> str`
  - `wait_for_server(base_url, check_interval=4.0) -> bool`
  - `format_duration(seconds) -> str`
  - `render_progress_bar(pct, width=20) -> str`
  - `query_db_status(db) -> dict[str, int]`
  - `run_unattended_pipeline(...) -> int`
- **Signal Coordination**: Shares signal state (`_STOP_REQUESTED`, `_INTERRUPT_COUNT`, `_sig_handler`) bidirectionally with `renslate.ui.runner` to ensure consistent interrupt behavior across wrapper invocations and direct CLI calls.


---

## 4. Troubleshooting & FAQ

### Issue: `ApplyError: tệp đã thay đổi kể từ lần extract`
- **Cause:** Source `.rpy` translation files were modified externally after `renslate extract` was run, causing a fingerprint mismatch.
- **Resolution:**
  1. Re-run `renslate extract`. It will update slot line numbers and fingerprints without overwriting previously translated units in the SQLite store.
  2. Re-run `renslate apply`.

### Issue: Translation Server Error 400 (`context length exceeded`)
- **Cause:** Large scene or long dialogue batch exceeding `n_ctx`.
- **Resolution:**
  1. Adjust `renslate.toml`: set `batch.max_lines = 20` and `batch.max_source_tokens = 1000`.
  2. Or run CLI with `--batch-size 15`.

### Issue: Missing Placeholders on TSV Import
- **Cause:** A reviewer accidentally modified or deleted a Ren'Py variable (e.g. removed `[player]` or `{b}`).
- **Resolution:** Renslate error report lists the exact row number and missing element. Correct the text in TSV to preserve the original Ren'Py markup and re-run `renslate import -i review.tsv`.
