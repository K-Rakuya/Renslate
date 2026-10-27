"""Comprehensive unit tests for renslate.ui widgets, formatting, and autonomous runner."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from unittest.mock import patch

import renslate.ui.runner as runner_mod
from renslate.core.db import Database
from renslate.core.models import Origin, Status, Unit, UnitKind
from renslate.ui.dashboard import (
    DashboardState,
    LiveDashboard,
    render_banner,
    render_batch_card,
    render_summary_card,
)
from renslate.ui.progress import (
    FRAC_BLOCKS,
    format_duration,
    format_speed,
    render_progress_bar,
)
from renslate.ui.runner import (
    _sig_handler,
    check_server_health,
    detect_server_model,
    query_db_status,
)
from renslate.ui.tables import format_table
from renslate.ui.theme import (
    char_width,
    strip_ansi,
    supports_color,
    visible_len,
)
from renslate.ui.wizard import InteractiveWizard, run_interactive_wizard, run_wizard


class TestThemeAndWidth:
    """Tests for theme.py ANSI styling and character width calculation."""

    def test_strip_ansi_removes_codes(self) -> None:
        colored = "\033[1m\033[32mHello World\033[0m"
        assert strip_ansi(colored) == "Hello World"
        assert strip_ansi("Plain text") == "Plain text"

    def test_char_width_ascii_and_wide(self) -> None:
        assert char_width("a") == 1
        assert char_width("Z") == 1
        assert char_width(" ") == 1
        # East Asian wide character
        assert char_width("中") == 2
        assert char_width("文") == 2
        # Vietnamese precomposed
        assert char_width("ế") == 1
        assert char_width("ữ") == 1

    def test_visible_len_calculation(self) -> None:
        text = "\033[31mRed\033[0m"
        assert visible_len(text) == 3
        wide_text = "\033[34m中文\033[0m"
        assert visible_len(wide_text) == 4

    def test_supports_color_no_color_env(self) -> None:
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            assert supports_color() is False

    def test_supports_color_dumb_term(self) -> None:
        with patch.dict(os.environ, {"TERM": "dumb"}, clear=True):
            assert supports_color() is False


class TestProgressMeters:
    """Tests for progress.py formatting and smooth fractional block rendering."""

    def test_format_duration_edge_cases(self) -> None:
        assert format_duration(-5.0) == "--:--:--"
        assert format_duration(0.0) == "--:--:--"
        assert format_duration(float("nan")) == "--:--:--"
        assert format_duration(float("inf")) == "--:--:--"
        assert format_duration(59.0) == "00m 59s"
        assert format_duration(61.0) == "01m 01s"
        assert format_duration(3600.0) == "01h 00m 00s"
        assert format_duration(3665.0) == "01h 01m 05s"

    def test_format_speed_scales(self) -> None:
        assert format_speed(0.0) == "0.0 u/s"
        assert format_speed(-1.0) == "0.0 u/s"
        # Below 0.2 u/s formats as units/minute
        assert "u/min" in format_speed(0.1)
        assert "u/s" in format_speed(15.5)

    def test_render_progress_bar_clamping(self) -> None:
        b_neg = render_progress_bar(-20.0, width=10)
        assert "0.0%" in b_neg
        assert "░" * 10 in b_neg

        b_over = render_progress_bar(120.0, width=10)
        assert "100.0%" in b_over
        assert "█" * 10 in b_over

    def test_render_progress_bar_fractional_blocks(self) -> None:
        # 25% of width 10 is 2.5 cells: 2 full blocks, 1 half block (4/8ths -> "▌"), 7 empty
        b_25 = render_progress_bar(25.0, width=10)
        assert "25.0%" in b_25
        assert "██" in b_25
        assert FRAC_BLOCKS[4] in b_25


class TestUnicodeTables:
    """Tests for tables.py box-drawing tables with borders and alignment."""

    def test_format_table_empty(self) -> None:
        assert format_table([], []) == ""

    def test_format_table_alignment_and_truncation(self) -> None:
        headers = ["ID", "Name", "Score"]
        rows = [
            ["1", "Alice", "95.5"],
            ["2", "Very Long Name Exceeding Limit", "88.0"],
        ]
        out = format_table(
            headers,
            rows,
            max_widths=[5, 10, 6],
            alignments=["left", "center", "right"],
        )
        assert "┌" in out
        assert "ID" in out
        assert "Alice" in out
        assert "…" in out  # Truncated long name
        assert "┘" in out


class TestDashboardAndCards:
    """Tests for dashboard.py live dashboard and summary cards."""

    def test_live_dashboard_render_plain(self) -> None:
        st = DashboardState(
            model_name="Qwen-Test",
            server_status="Online",
            total_units=200,
            completed_units=100,
            pending_units=90,
            failed_units=10,
            speed_u_s=12.5,
            elapsed_s=80.0,
            eta_s=7.2,
            current_batch=2,
            batch_latency_s=1.25,
            lookback_size=5,
        )
        dash = LiveDashboard(st)
        rendered = dash.render(width=75, plain=True)
        assert "Renslate Autonomous Engine" in rendered
        assert "Qwen-Test" in rendered
        assert "100/200" in rendered
        assert "50.0%" in rendered
        assert "Failed: 10" in rendered

    def test_render_banner_output(self) -> None:
        items = [("Model", "Qwen"), ("Language", "Vietnamese")]
        banner = render_banner("Renslate Test Runner", width=60, items=items, plain=True)
        assert "Renslate Test Runner" in banner
        assert "Model" in banner
        assert "Qwen" in banner

    def test_render_summary_card_output(self) -> None:
        st = DashboardState(
            model_name="Qwen",
            total_units=100,
            completed_units=98,
            pending_units=0,
            failed_units=2,
            speed_u_s=10.0,
            elapsed_s=10.0,
        )
        summary = render_summary_card(st, width=60, plain=True)
        assert "TỔNG KẾT PHIÊN DỊCH RENSLATE" in summary
        assert "98" in summary
        assert "2" in summary

    def test_render_batch_card_output(self) -> None:
        msg = render_batch_card(1, 25, 2.5, 10.0, failed=1, plain=True)
        assert "Lô #1" in msg
        assert "+25 dòng" in msg
        assert "1 lỗi" in msg


class TestWizardAndRunner:
    """Tests for wizard.py and runner.py server discovery and process controls."""

    def test_wizard_non_interactive_execution(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        with patch("sys.stdin.isatty", return_value=False):
            code = run_interactive_wizard(tmp_path)
            assert code == 0
            assert run_wizard is run_interactive_wizard

    def test_wizard_interactive_prompt_and_confirm(self) -> None:
        wiz = InteractiveWizard()
        with patch("builtins.input", side_effect=["custom_val", "y"]):
            assert wiz.prompt("Enter value", default="default") == "custom_val"
            assert wiz.confirm("Confirm action") is True

    def test_query_db_status_aggregation(self, tmp_path: Path) -> None:
        db = Database.open(tmp_path / "test_status.db", create=True)
        try:
            u1 = Unit(
                uid="s:1",
                kind=UnitKind.DIALOGUE,
                seq=1,
                source="Hello",
                status=Status.TRANSLATED,
                speaker="narrator",
                masked_source="Hello",
                target="Xin chào",
                origin=Origin.LLM,
            )
            u2 = Unit(
                uid="s:2",
                kind=UnitKind.DIALOGUE,
                seq=2,
                source="World",
                status=Status.PENDING,
                speaker="narrator",
                masked_source="World",
                target=None,
                origin=None,
            )
            db.save_units_bulk([u1, u2])
            counts = query_db_status(db)
            assert counts.get("translated") == 1
            assert counts.get("pending") == 1
        finally:
            db.close()

    def test_server_health_probe_endpoints(self) -> None:
        # Offline endpoint should return False cleanly
        assert check_server_health("http://127.0.0.1:59998", timeout_s=0.01) is False
        res = detect_server_model("http://127.0.0.1:59998", "fallback.gguf", timeout_s=0.01)
        assert res == "fallback.gguf"

    def test_sig_handler_sets_stop_flag(self) -> None:
        import sys
        from typing import Any

        runner_mod._STOP_REQUESTED = False
        runner_mod._INTERRUPT_COUNT = 0
        unattended: Any = sys.modules.get("scripts.unattended_runner")
        if unattended is not None:
            unattended._STOP_REQUESTED = False
            unattended._INTERRUPT_COUNT = 0

        _sig_handler(signal.SIGINT, None)
        assert runner_mod._STOP_REQUESTED is True

        runner_mod._STOP_REQUESTED = False
        runner_mod._INTERRUPT_COUNT = 0
        if unattended is not None:
            unattended._STOP_REQUESTED = False
            unattended._INTERRUPT_COUNT = 0
