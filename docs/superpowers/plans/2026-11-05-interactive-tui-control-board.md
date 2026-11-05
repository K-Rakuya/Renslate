# Interactive TUI Control Board & Real-Time Translation Studio (2026+)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Research and build an interactive, visually stunning, zero-dependency Terminal User Interface (TUI) Control Board and Real-Time Translation Studio for Renslate (`renslate`). Enables anyone—from casual translators to power engineers—to toggle feature switches (`[● BẬT] / [○ TẮT]`), adjust numeric parameters with arrow keys, stream full bilingual translations in real-time with zero truncation, and immediately edit or reflect on translated lines in-place, using 100% Pure Python standard library.

**Architecture:** Model-View-Controller TUI Component Pattern over ANSI Terminal Escape Sequences (VT100/POSIX + Windows Virtual Terminal), integrating seamlessly into Hexagonal Primary Driving Adapters (`src/renslate/ui/control_board.py`, `src/renslate/ui/studio.py`, and `src/renslate/cli/`).

**Tech Stack:** Pure Python 3.10+ Standard Library (`sys`, `os`, `termios`/`tty` on POSIX, `msvcrt` on Windows, `dataclasses`, `unicodedata`), zero third-party dependencies (`dependencies = []`).

**Spec:** [AGENTS.md](file:///home/rakuya/Documents/Renvi/AGENTS.md) (8 Quality Pillars), [ARCHITECTURE_DECISIONS.md](file:///home/rakuya/Documents/Renvi/docs/ARCHITECTURE_DECISIONS.md).

---

## 1. Research: Modern 2026 CLI & TUI Paradigms

### The Problem with Traditional CLIs & Previews
1. **Flag Overload & Cognitive Friction:** Passing 15 flags (`--deliberate --calque-threshold 0.35 --exemplars 2 --guard --concise-tags --batch-size 8`) requires memorizing arcane syntax.
2. **Truncated Previews Obstruct Review:** Standard CLI previews truncate long visual novel dialogues with ellipses (`"Tôi nghĩ rằng..."`), making it impossible to evaluate narrative cadence, pronoun consistency, or grammatical flow.
3. **Disconnected Review Loops:** Translators must wait for the whole run to finish, export a TSV, open Excel, edit lines, and re-import. If a character address form is wrong on line 10, the model continues generating 5,000 subsequent lines with that same mistake.
4. **Heavy TUI Frameworks Violate Stdlib Principles:** External libraries (Rich, Textual, Inquirer) break offline and stripped Ren'Py runtimes.

### The 2026 World-Class Solution: Dual-Mode TUI Architecture

```
                    [ RENSLATE TUI EXPERIENCE 2026 ]
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│ VIEW 1: INTERACTIVE CONTROL     │   │ VIEW 2: REAL-TIME FULL STUDIO   │
│ BOARD (Bảng Điều Khiển Cờ)      │   │ (Trình Soi & Sửa Live Không Cắt)│
│                                 │   │                                 │
│ - Toggles: [● BẬT] / [○ TẮT]    │──►│ - Live Streaming từng câu thoại │
│ - Sliders: ◄ 0.35 ►, ◄ 8 ►      │(Go│ - XEM FULL 100%, tự động wrap   │
│ - Live Contextual Info Card     │   │ - [E] Sửa trực tiếp vào DB + TM │
│ - Start/Stop/Config Presets     │   │ - [R] Phản tư viết lại System-2 │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 2. Visual Wireframe Designs

### Screen 1: Interactive Control Board (Cấu hình & Bật/Tắt Cờ)
```text
╭──────────────────────────────────────────────────────────────────────────────╮
│  RENSLATE VISUAL NOVEL LOCALIZATION ENGINE  ──  BẢNG ĐIỀU KHIỂN CẤU HÌNH     │
│  Dự án: [CommercialVN-Alpha]  │  Tiến độ: 1,420 / 3,850 dòng (36.8%)         │
╰──────────────────────────────────────────────────────────────────────────────╯

 CÁC TÙY CHỌN & CỜ ĐIỀU KHIỂN:               THÔNG TIN & KHUYẾN NGHỊ:
─────────────────────────────────────────   ────────────────────────────────────
 ► [● BẬT] 2-Pass Deliberative Reflector     [2-PASS DELIBERATIVE REFLECTION]
   [◄ 0.35 ►] Ngưỡng Calque kích hoạt       - Cơ chế tư duy chậm (System-2) tự
   [◄  2   ►] Dyad TM Exemplars (câu mẫu)     động phát hiện câu dịch thô cứng,
   [◄  6   ►] Ring Buffer Lịch sử Cảnh        bị động gượng gạo và viết lại
   [● BẬT] TransGraph Micro-Framing 4D        thành tiếng Việt văn học tự nhiên.
                                            - Bảo toàn 100% placeholder Ren'Py.
   [● BẬT] Continuous Hallucination Guard   - Khống chế trần ngân sách <= 25%
   [◄  8   ►] Kích thước lô (Batch lines)     số câu để giữ tốc độ tối đa!
   [◄ 0.30 ►] Temperature sinh mẫu LLM      ────────────────────────────────────
   [○ TẮT] Rút gọn nhãn thoại (--concise)   Khuyên dùng: BẬT (Visual Novel)
                                            Tác động tốc độ: ~90% 1-Pass speed
   [○ TẮT] Tự động ghi vào game sau dịch
   [● BẬT] Thiết lập phông chữ tiếng Việt

────────────────────────────────────────────────────────────────────────────────
 [ENTER] Bắt đầu Dịch     [S] Lưu Cấu hình     [R] Khôi phục Mặc định     [Q] Thoát
 (Dùng phím ↑/↓ để chọn dòng, Space để bật/tắt, ←/→ để tăng giảm giá trị)
```

---

### Screen 2: Real-Time Full Translation Studio (Soi Full & Sửa Đổi Trực Tiếp)
```text
╭──────────────────────────────────────────────────────────────────────────────╮
│  RENSLATE LIVE TRANSLATION STUDIO  ──  THEO DÕI & SỬA ĐỔI THỜI GIAN THỰC     │
│  Lô #12/48  •  Tốc độ: 4.2 dòng/s  •  Đã dịch: 1,420/3,850 dòng (36.8%)       │
╰──────────────────────────────────────────────────────────────────────────────╯
 [TAB] Đổi View    [F] Lọc: Tất cả (A)    [↑/↓ / PgUp/PgDn] Cuộn    [E] Sửa dòng
────────────────────────────────────────────────────────────────────────────────

 ┌─ [Seq: 142] d:scene1_0042  ──  hero ──► sage  [LLM • D_calque: 0.12] ────────┐
 │ ENG: Master, the eastern gate has fallen! What should we do now?             │
 │ VIE: Thưa sư phụ, cổng đông đã thất thủ rồi! Chúng ta phải làm gì bây giờ?   │
 └──────────────────────────────────────────────────────────────────────────────┘

 ┌─ [Seq: 143] d:scene1_0043  ──  sage ──► hero  [REFLECTED • System-2] ────────┐
 │ ENG: Calm down. Haste will only lead us into the enemy's trap.               │
 │ VIE: Hãy bình tĩnh lại. Hấp tấp chỉ khiến chúng ta rơi vào bẫy của kẻ địch.  │
 └──────────────────────────────────────────────────────────────────────────────┘

 ►┌─ [Seq: 144] d:scene1_0044  ──  hero ──► sage  [QA_FLAGGED • pronoun_mismatch] ┐
  │ ENG: But my sister is still trapped inside the sacred chamber!              │
  │ VIE: Nhưng em gái tôi vẫn còn đang bị mắc kẹt bên trong gian phòng thánh!   │
  └─────────────────────────────────────────────────────────────────────────────┘
  (Đang chọn dòng này ── Bấm [E] để sửa tại chỗ ── Bấm [R] để LLM viết lại)

────────────────────────────────────────────────────────────────────────────────
 [E] Sửa bản dịch    [R] Phản tư viết lại    [SPACE] Tạm dừng/Tiếp tục    [Q] Thoát
```

---

### In-Place Editing Modal (Khi người dùng bấm phím `E`):
```text
╭──────────────────────────────────────────────────────────────────────────────╮
│  CHỈNH SỬA BẢN DỊCH TẠI CHỖ: d:scene1_0044                                   │
│  Gốc: "But my sister is still trapped inside the sacred chamber!"            │
│  Hiện tại: "Nhưng em gái tôi vẫn còn đang bị mắc kẹt bên trong gian phòng..."│
│                                                                              │
│  Nhập bản dịch mới:                                                          │
│  ► Nhưng muội muội con vẫn còn đang mắc kẹt trong mật thất!_                 │
│                                                                              │
│  [ENTER] Lưu vào Database + TM    [ESC] Hủy bỏ                               │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 3. Architecture & Data Structures

### Module Organization
```
src/renslate/ui/
├── control_board.py      # Interactive TUI Flag Control Board
├── studio.py             # Real-time Full Translation Studio & In-Place Editor
├── theme.py              # ANSI colors, rounded borders, visible length calculation
├── prompts.py            # Cross-platform raw keypress reader (termios & msvcrt)
└── dashboard.py          # Real-time progress metrics collector
```

### Core Data Models (`src/renslate/ui/studio.py`)

```python
@dataclass
class LiveDialogueItem:
    uid: str
    seq: int
    speaker: str
    listener: str | None
    source: str
    target: str
    status: str
    origin: str
    calque_score: float = 0.0
    qa_issues: tuple[str, ...] = ()
    updated_at: str = ""

@dataclass
class StudioState:
    units: list[LiveDialogueItem] = field(default_factory=list)
    selected_idx: int = 0
    scroll_offset: int = 0
    is_paused: bool = False
    filter_mode: str = "all"  # "all", "flagged", "reflected"
    is_editing: bool = False
    edit_buffer: str = ""
```

---

## 4. Implementation Tasks

### Task 1: ControlField State Machine & Navigation Logic

**Files:**
- Create: `src/renslate/ui/control_board.py`
- Test: `tests/test_tui_control_board.py`

- [ ] **Step 1: Write failing unit tests for `ControlField` value manipulation**
  * Test toggle boolean (`value = not value`).
  * Test increment / decrement bounded numeric values with `step`.
  * Test cycle choice lists.
- [ ] **Step 2: Implement state models and navigation handlers in `control_board.py`**
- [ ] **Step 3: Run pytest to verify passes**
- [ ] **Step 4: Commit `feat(ui): implement control field state machine for tui board`**

---

### Task 2: Real-Time Full Dialogue Studio & Word-Wrapping Renderer

**Files:**
- Create: `src/renslate/ui/studio.py`
- Test: `tests/test_tui_studio.py`

- [ ] **Step 1: Write failing unit test for full text wrapping and box rendering**
  * Verify long source and target lines wrap gracefully across multiple visual lines without truncation (`...`).
  * Verify character names (`hero ──► sage`), origins (`[REFLECTED • System-2]`), and calque scores are formatted cleanly.
  * Verify scroll and cursor clamping.
- [ ] **Step 2: Implement `LiveDialogueItem` and `StudioRenderer` in `src/renslate/ui/studio.py`**
- [ ] **Step 3: Run pytest to verify passes**
- [ ] **Step 4: Commit `feat(ui): implement real-time full dialogue studio renderer`**

---

### Task 3: In-Place Editing & Database Transactional Propagation

**Files:**
- Modify: `src/renslate/ui/studio.py`
- Test: `tests/test_tui_studio.py`

- [ ] **Step 1: Write failing unit test for in-place edit workflow**
  * Press 'E' on a selected unit $\rightarrow$ enter new Vietnamese string $\rightarrow$ commits to SQLite `units` table with `Origin.USER`.
  * Verifies new translation is saved into `tm` table for subsequent reuse.
  * Verifies QA Checker validates placeholder invariants.
- [ ] **Step 2: Implement in-place edit input modal and transactional write handler in `studio.py`**
- [ ] **Step 3: Run pytest to verify passes**
- [ ] **Step 4: Commit `feat(ui): implement in-place translation editing with sqlite and tm sync`**

---

### Task 4: Interactive Event Loop & Dual-View Switching

**Files:**
- Modify: `src/renslate/ui/control_board.py`
- Modify: `src/renslate/ui/studio.py`
- Test: `tests/test_tui_control_board.py`

- [ ] **Step 1: Write tests for event loop handling TAB (switch View 1 $\leftrightarrow$ View 2), SPACE (pause/resume), Arrow keys (scroll), and E (edit)**
- [ ] **Step 2: Connect real-time callback from `TranslationPipeline` / runner to append new dialogue items to `StudioState`**
- [ ] **Step 3: Run pytest to verify passes**
- [ ] **Step 4: Commit `feat(ui): integrate dual-view event loop for control board and live studio`**

---

### Task 5: CLI Integration & End-to-End Verification

**Files:**
- Modify: `src/renslate/cli/handlers/translation.py`
- Modify: `src/renslate/cli/parser.py`
- Test: Full test suite (`pytest`)

- [ ] **Step 1: Wire `renslate` (no args), `renslate run`, and `renslate live` to launch the Interactive Studio**
- [ ] **Step 2: Run full test suite across the entire repository (`.venv/bin/pytest -q`)**
  * Assert: 2,132+ tests passing, 0 failure.
- [ ] **Step 3: Run ruff linter (`.venv/bin/ruff check src/ tests/ scripts/`)**
  * Assert: All checks passed!
- [ ] **Step 4: Commit and push changes**
