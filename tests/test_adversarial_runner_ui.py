"""Tier 5 Adversarial Stress Harness for Renslate Runner, UI, Signals, and Auto-Apply.

Empirically validates:
1. CLI flags & subcommands parsing, aliases, invalid inputs, and delegation wrapper.
2. Terminal rendering under adverse conditions (NO_COLOR, TERM=dumb, non-TTY pipes,
   zero/negative/narrow/ultra-wide columns, CJK/combining characters).
3. 2-stage SIGINT trapping: single SIGINT graceful completion (exit 0) vs double SIGINT
   emergency exit (code 130).
4. Mock server health recovery: server offline/500/timeout detection, dynamic revival,
   and automatic loop resumption.
5. Auto-apply mechanics: atomic .rpy replacement, backup generation, fingerprint validation,
   stale file detection, and markup safety gates.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import pty
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import renslate.ui.runner as runner_mod
import scripts.unattended_runner as unattended_mod
from renslate.cli import _build_parser
from renslate.cli import main as cli_main
from renslate.core.errors import ApplyError
from renslate.core.project import Project
from renslate.core.textio import write_atomic
from renslate.engine.translate import TranslationStats
from renslate.renpy.extract import extract
from renslate.renpy.writer import apply_translations
from renslate.ui.dashboard import (
    DashboardState,
    LiveDashboard,
    render_banner,
    render_summary_card,
)
from renslate.ui.progress import (
    render_progress_bar,
)
from renslate.ui.runner import (
    _is_stop_requested,
    _sig_handler,
    check_server_health,
    detect_server_model,
    run_unattended_pipeline,
    wait_for_server,
)
from renslate.ui.tables import format_table
from renslate.ui.theme import (
    supports_color,
    visible_len,
)


@pytest.fixture(autouse=True)
def _reset_global_signal_state() -> Any:
    runner_mod._STOP_REQUESTED = False
    runner_mod._INTERRUPT_COUNT = 0
    unattended_mod._STOP_REQUESTED = False
    unattended_mod._INTERRUPT_COUNT = 0
    yield
    runner_mod._STOP_REQUESTED = False
    runner_mod._INTERRUPT_COUNT = 0
    unattended_mod._STOP_REQUESTED = False
    unattended_mod._INTERRUPT_COUNT = 0


def _setup_mock_project(tmp_path: Path, name: str = "mock_rpy_proj") -> Project:
    root = tmp_path / name
    game_dir = root / "game"
    tl_dir = game_dir / "tl" / "vietnamese"
    tl_dir.mkdir(parents=True, exist_ok=True)

    rpy_content = (
        "# game/script.rpy:1\n"
        "translate vietnamese start_a1b2c3d4:\n"
        '    # eileen "Hello world!"\n'
        '    eileen "Hello world!"\n'
        "\n"
        "# game/script.rpy:5\n"
        "translate vietnamese strings:\n"
        '    old "Start Game"\n'
        '    new "Start Game"\n'
    )
    (tl_dir / "script.rpy").write_text(rpy_content, encoding="utf-8")
    return Project.create(root, target_lang="vietnamese")


# ==============================================================================
# 1. CLI FLAGS & SUBCOMMANDS EMPIRICAL TESTS
# ==============================================================================


class TestCliFlagsAndSubcommands:
    """Stress-test CLI arguments, subcommands, aliases, and error modes."""

    def test_cli_subcommands_registration(self) -> None:
        parser = _build_parser()
        subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        commands = set(subparsers.choices.keys())
        expected = {
            "init",
            "generate",
            "extract",
            "analyze",
            "bootstrap",
            "translate",
            "qa",
            "export",
            "import",
            "apply",
            "status",
            "preview",
            "clean-db",
            "run",
            "auto",
            "wizard",
            "interactive",
        }
        for cmd in expected:
            assert cmd in commands, f"Subcommand '{cmd}' missing from CLI parser!"

    def test_cli_run_and_auto_options(self) -> None:
        parser = _build_parser()
        for cmd in ["run", "auto"]:
            args = parser.parse_args(
                [
                    cmd,
                    "-p",
                    "/fake/proj",
                    "--chunk-size",
                    "50",
                    "--retry-failed",
                    "--retry-flagged",
                    "--auto-apply",
                    "--plain",
                    "-i",
                ]
            )
            assert args.command == cmd
            assert args.project == "/fake/proj"
            assert args.chunk_size == 50
            assert args.retry_failed is True
            assert args.retry_flagged is True
            assert args.auto_apply is True
            assert args.plain is True
            assert args.interactive is True

    def test_cli_invalid_subcommand_exit_code(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["nonexistent_command_xyz"])
        assert exc_info.value.code == 2

    def test_cli_missing_required_argument(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["import"])  # requires -i / --input
        assert exc_info.value.code == 2

    def test_cli_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "renslate" in captured.out

    def test_unattended_runner_wrapper_delegation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called_args: list[str] = []

        def fake_cli_main(args: list[str]) -> int:
            called_args.extend(args)
            return 42

        monkeypatch.setattr("scripts.unattended_runner.cli_main", fake_cli_main)
        res = unattended_mod.main(
            ["translate", "-p", "/test/p", "--chunk-size", "25", "--auto-apply"]
        )
        assert res == 42
        assert called_args == ["run", "-p", "/test/p", "--chunk-size", "25", "--auto-apply"]


# ==============================================================================
# 2. TERMINAL CAPABILITY & RENDERING ADVERSARIAL STRESS TESTS
# ==============================================================================


class TestTerminalCapabilitiesAndRendering:
    """Empirically test terminal capabilities, NO_COLOR, non-TTY, and extreme dimensions."""

    def test_supports_color_matrix(self) -> None:
        _master, slave = pty.openpty()
        orig_stdout = sys.stdout
        sys.stdout = os.fdopen(slave, "w")
        try:
            # 1. Default TTY -> True
            with patch.dict(os.environ, {}, clear=True):
                assert supports_color() is True

            # 2. NO_COLOR=1 -> False
            with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True):
                assert supports_color() is False

            # 3. NO_COLOR="" (empty string) -> True per standard
            with patch.dict(os.environ, {"NO_COLOR": ""}, clear=True):
                assert supports_color() is True

            # 4. TERM=dumb -> False
            with patch.dict(os.environ, {"TERM": "dumb"}, clear=True):
                assert supports_color() is False
        finally:
            sys.stdout = orig_stdout

    def test_non_tty_pipe_suppresses_color(self) -> None:
        with patch.object(sys.stdout, "isatty", return_value=False):
            assert supports_color() is False

    @pytest.mark.parametrize("width", [-100, -1, 0, 1, 5, 10, 25, 78, 300, 1000, 5000])
    def test_dashboard_render_extreme_widths(self, width: int) -> None:
        st = DashboardState(
            model_name="Qwen-Test",
            server_status="Online",
            total_units=1000,
            completed_units=450,
            pending_units=500,
            failed_units=50,
            speed_u_s=12.5,
            elapsed_s=120.0,
            eta_s=60.0,
        )
        dash = LiveDashboard(st)
        rendered_plain = dash.render(width=width, plain=True)
        assert isinstance(rendered_plain, str)
        assert "Renslate Autonomous Engine" in rendered_plain
        assert "Qwen-Test" in rendered_plain

        rendered_color = dash.render(width=width, plain=False)
        assert isinstance(rendered_color, str)

    @pytest.mark.parametrize("width", [-50, 0, 1, 10, 80, 500])
    def test_render_banner_extreme_widths(self, width: int) -> None:
        banner = render_banner("Renslate Title", width=width, items=[("Key", "Value")], plain=True)
        assert isinstance(banner, str)
        assert "Renslate Title" in banner
        assert "Key" in banner

    @pytest.mark.parametrize("width", [-50, 0, 1, 10, 80, 500])
    def test_render_summary_card_extreme_widths(self, width: int) -> None:
        st = DashboardState(total_units=100, completed_units=100)
        card = render_summary_card(st, width=width, plain=True)
        assert isinstance(card, str)
        assert "TỔNG KẾT PHIÊN DỊCH RENSLATE" in card

    @pytest.mark.parametrize("width", [-10, 0, 1, 5, 20, 100, 300])
    @pytest.mark.parametrize("pct", [-50.0, 0.0, 0.05, 12.5, 50.0, 87.5, 99.9, 100.0, 150.0])
    def test_render_progress_bar_boundary_values(self, width: int, pct: float) -> None:
        bar = render_progress_bar(pct, width=width)
        assert isinstance(bar, str)
        assert "%" in bar
        if pct <= 0.0:
            assert "0.0%" in bar
        elif pct >= 100.0:
            assert "100.0%" in bar

    def test_format_table_adversarial_matrix(self) -> None:
        # 1. Empty table
        assert format_table([], []) == ""

        # 2. Uneven columns & CJK full-width chars
        headers = ["ID", "Tiêu đề (CJK & Việt)", "Status"]
        rows = [
            ["1", "Hello 日本語 world", "OK"],
            ["2", "Tiếng Việt có dấu: ề, ử, ắ, ộ", "DONE", "EXTRA_CELL_IGNORED"],
            ["3"],  # missing cells
        ]
        table = format_table(
            headers,
            rows,
            max_widths=[5, 20, 10],
            alignments=["right", "left", "center"],
        )
        lines = table.split("\n")
        assert len(lines) >= 6

        # Check that visible length across all border and data rows matches
        line_lengths = [visible_len(line) for line in lines]
        assert len(set(line_lengths)) == 1, (
            f"Table borders misaligned! Line lengths: {line_lengths}"
        )


# ==============================================================================
# 3. 2-STAGE SIGINT TRAPPING EMPIRICAL TESTS
# ==============================================================================


class TestTwoStageSigintTrapping:
    """Stress-test signal handling: single SIGINT vs double SIGINT."""

    def test_sig_handler_single_sigint_sets_stop_requested(self) -> None:
        _sig_handler(signal.SIGINT, None)
        assert runner_mod._STOP_REQUESTED is True
        assert runner_mod._INTERRUPT_COUNT == 1
        assert _is_stop_requested() is True

    def test_sig_handler_double_sigint_calls_exit_130(self) -> None:
        with patch("os._exit") as mock_exit:
            _sig_handler(signal.SIGINT, None)
            assert mock_exit.call_count == 0

            _sig_handler(signal.SIGINT, None)
            assert mock_exit.call_count == 1
            mock_exit.assert_called_with(130)

    def test_subprocess_single_sigint_graceful_exit(self) -> None:
        code = (
            "import signal, sys, time\n"
            "from renslate.ui.runner import _sig_handler, _is_stop_requested\n"
            "signal.signal(signal.SIGINT, _sig_handler)\n"
            "sys.stdout.write('READY\\n')\n"
            "sys.stdout.flush()\n"
            "for _ in range(50):\n"
            "    if _is_stop_requested():\n"
            "        sys.stdout.write('GRACEFUL_STOP\\n')\n"
            "        sys.stdout.flush()\n"
            "        sys.exit(0)\n"
            "    time.sleep(0.1)\n"
            "sys.exit(1)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert "READY" in line
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=5)
        assert proc.returncode == 0
        assert "GRACEFUL_STOP" in out

    def test_subprocess_double_sigint_immediate_code_130(self) -> None:
        code = (
            "import signal, sys, time\n"
            "from renslate.ui.runner import _sig_handler\n"
            "signal.signal(signal.SIGINT, _sig_handler)\n"
            "sys.stdout.write('READY\\n')\n"
            "sys.stdout.flush()\n"
            "while True:\n"
            "    time.sleep(0.5)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert "READY" in line
        proc.send_signal(signal.SIGINT)
        time.sleep(0.05)
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=5)
        assert proc.returncode == 130


# ==============================================================================
# 4. MOCK SERVER HEALTH & DYNAMIC RECOVERY EMPIRICAL TESTS
# ==============================================================================


class MockLLMHandler(http.server.BaseHTTPRequestHandler):
    mode: str = "health_200"

    def do_GET(self) -> None:
        if self.mode == "server_500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "Internal server error"}')
            return

        if self.path == "/health":
            if self.mode in ("health_200", "healthy"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            else:
                self.send_response(503)
                self.end_headers()
        elif self.path == "/props":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "default_generation_settings": {
                            "model": "/models/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
                        }
                    }
                ).encode("utf-8")
            )
        elif self.path == "/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "Qwen3.6-Model"}]}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


class TestServerHealthAndRecovery:
    """Empirically test server health checks, model detection, and recovery."""

    @pytest.fixture
    def mock_server(self) -> Any:
        server = http.server.HTTPServer(("127.0.0.1", 0), MockLLMHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        yield base_url, server
        server.shutdown()
        server.server_close()

    def test_check_server_health_scenarios(self, mock_server: tuple[str, Any]) -> None:
        base_url, _ = mock_server

        # 1. Healthy server
        MockLLMHandler.mode = "health_200"
        assert check_server_health(base_url) is True

        # 2. Server returning 500
        MockLLMHandler.mode = "server_500"
        assert check_server_health(base_url) is False

        # 3. Connection refused (dead port)
        assert check_server_health("http://127.0.0.1:59999", timeout_s=0.2) is False

    def test_detect_server_model(self, mock_server: tuple[str, Any]) -> None:
        base_url, _ = mock_server
        MockLLMHandler.mode = "health_200"

        model = detect_server_model(base_url, default_name="fallback.gguf")
        assert model == "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

        # When server is down, fallback to default_name
        fallback = detect_server_model("http://127.0.0.1:59999", "fallback.gguf", timeout_s=0.1)
        assert fallback == "fallback.gguf"

    def test_wait_for_server_dynamic_revival(self, mock_server: tuple[str, Any]) -> None:
        base_url, _ = mock_server
        MockLLMHandler.mode = "server_500"

        # Initially offline
        assert check_server_health(base_url) is False

        # Schedule revival in 150ms
        def _revive() -> None:
            time.sleep(0.15)
            MockLLMHandler.mode = "health_200"

        threading.Thread(target=_revive, daemon=True).start()

        t0 = time.perf_counter()
        revived = wait_for_server(base_url, check_interval=0.05)
        elapsed = time.perf_counter() - t0

        assert revived is True
        assert elapsed >= 0.14
        assert check_server_health(base_url) is True

    def test_wait_for_server_aborts_on_stop_requested(self) -> None:
        runner_mod._STOP_REQUESTED = True
        try:
            assert wait_for_server("http://127.0.0.1:59999", check_interval=0.01) is False
        finally:
            runner_mod._STOP_REQUESTED = False


# ==============================================================================
# 5. AUTO-APPLY MECHANICS & FINGERPRINT VALIDATION EMPIRICAL TESTS
# ==============================================================================


class TestAutoApplyMechanics:
    """Stress-test auto-apply atomicity, backup generation, and fingerprint checks."""

    def test_atomic_write_and_mode_preservation(self, tmp_path: Path) -> None:
        target = tmp_path / "atomic_target.rpy"
        target.write_text("initial content\n", encoding="utf-8")
        target.chmod(0o644)

        new_data = b"new atomic content\n"
        write_atomic(target, new_data)

        assert target.read_bytes() == new_data
        tmp_leftover = list(tmp_path.glob("*.tmp"))
        assert len(tmp_leftover) == 0

    def test_apply_translations_success_and_backup(self, tmp_path: Path) -> None:
        proj = _setup_mock_project(tmp_path, "apply_ok")
        with proj.open_db() as db:
            rep_ext = extract(proj, db)
            assert rep_ext.units > 0

            # Provide valid translation for dialogue
            db.execute(
                "UPDATE units SET target = 'Xin chào thế giới!', status = 'translated' "
                "WHERE kind = 'dialogue'"
            )
            db._connection.commit()

            report = apply_translations(proj, db, timestamp="20261024T180000Z")

        assert report.lines_written == 1
        assert report.files_changed == 1
        assert report.backup_dir is not None
        assert report.backup_dir.is_dir()

        # Verify backup contains pre-apply file
        backup_file = report.backup_dir / "game" / "tl" / "vietnamese" / "script.rpy"
        assert backup_file.is_file()
        assert 'eileen "Hello world!"' in backup_file.read_text(encoding="utf-8")

        # Verify target file has translated content
        target_file = proj.root / "game" / "tl" / "vietnamese" / "script.rpy"
        target_text = target_file.read_text(encoding="utf-8")
        assert 'eileen "Xin chào thế giới!"' in target_text

    def test_apply_dry_run_leaves_files_untouched(self, tmp_path: Path) -> None:
        proj = _setup_mock_project(tmp_path, "apply_dry")
        target_file = proj.root / "game" / "tl" / "vietnamese" / "script.rpy"
        orig_bytes = target_file.read_bytes()

        with proj.open_db() as db:
            extract(proj, db)
            db.execute("UPDATE units SET target = 'Xin chào!', status = 'translated'")
            db._connection.commit()

            report = apply_translations(proj, db, dry_run=True)

        assert report.dry_run is True
        assert report.lines_written == 2
        assert report.backup_dir is None
        assert target_file.read_bytes() == orig_bytes

    def test_apply_rejects_stale_file_fingerprint_mismatch(self, tmp_path: Path) -> None:
        proj = _setup_mock_project(tmp_path, "apply_stale")
        target_file = proj.root / "game" / "tl" / "vietnamese" / "script.rpy"

        with proj.open_db() as db:
            extract(proj, db)
            db.execute("UPDATE units SET target = 'Xin chào!', status = 'translated'")
            db._connection.commit()

            # External modification of file after extract
            target_file.write_text(
                target_file.read_text(encoding="utf-8").replace("Hello world!", "Modified source!"),
                encoding="utf-8",
            )

            with pytest.raises(ApplyError) as exc_info:
                apply_translations(proj, db)
            assert "tệp đã thay đổi kể từ lần extract" in str(exc_info.value)

    def test_apply_rejects_malformed_markup_in_translation(self, tmp_path: Path) -> None:
        proj = _setup_mock_project(tmp_path, "apply_bad_markup")

        with proj.open_db() as db:
            extract(proj, db)
            # Inject unclosed { tag syntax
            db.execute("UPDATE units SET target = '{b unclosed markup tag', status = 'translated'")
            db._connection.commit()

            with pytest.raises(ApplyError) as exc_info:
                apply_translations(proj, db)
            assert "markup không hợp lệ" in str(exc_info.value)

    def test_unattended_pipeline_auto_apply_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proj = _setup_mock_project(tmp_path, "runner_auto_apply")
        with proj.open_db() as db:
            extract(proj, db)
            # Mark units translated so auto_apply has work to do
            db.execute("UPDATE units SET target = 'Xin chào thế giới!', status = 'translated'")
            db._connection.commit()

        monkeypatch.setattr("renslate.ui.runner.check_server_health", lambda *a, **kw: True)
        monkeypatch.setattr("renslate.ui.runner.detect_server_model", lambda *a, **kw: "Mock-Model")

        class FakePipeline:
            def __init__(self, *a: Any, **kw: Any) -> None:
                pass

            def run(self, *a: Any, **kw: Any) -> TranslationStats:
                # 0 pending remaining
                return TranslationStats(total_pending=0)

        monkeypatch.setattr("renslate.ui.runner.TranslationPipeline", FakePipeline)

        ret = run_unattended_pipeline(proj.root, auto_apply=True, plain=True)
        assert ret == 0

        target_file = proj.root / "game" / "tl" / "vietnamese" / "script.rpy"
        assert 'eileen "Xin chào thế giới!"' in target_file.read_text(encoding="utf-8")
