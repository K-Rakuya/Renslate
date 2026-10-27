"""Comprehensive unit and integration tests for renslate.ui.wizard and game inspection.

Tests Requirement 2 (R2):
- Interactive zero-command launch & CLI integration
- Ren'Py 6, 7, 8 game inspection (library detection, bytecode pickle protocols, GUI files)
- Script mode composition (RPY_SOURCE, RPYC_BYTECODE_ONLY, HYBRID, EMPTY)
- Configured translation languages and inspection report card formatting
- 6-phase state machine transitions (forward, backward, cancel, retry)
- Pure stdlib cross-platform terminal navigation (arrow keys, numeric shortcuts, fallbacks)
- InteractiveWizard execution and non-interactive fallbacks
"""

from __future__ import annotations

import pickle
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from renslate.cli import main
from renslate.core.project import Project
from renslate.ui.wizard import (
    DEFAULT_TRANSITIONS,
    GameInspectionReport,
    GameInspector,
    InteractiveWizard,
    RenpyVersion,
    ScriptMode,
    WizardContext,
    WizardEvent,
    WizardPhase,
    WizardState,
    inspect_game,
    read_single_key,
    render_inspection_card,
    run_interactive_wizard,
    run_wizard,
    select_menu,
)


def _make_synthetic_rpyc(pickle_protocol: int = 2) -> bytes:
    """Create a minimal valid Ren'Py RPC2 file with the specified pickle protocol."""
    header = b"RENPY RPC2"
    payload = pickle.dumps(["test_node"], protocol=pickle_protocol)
    compressed = zlib.compress(payload, level=3)

    slot_offset = len(header) + 24
    slot_entry = struct.pack("<III", 1, slot_offset, len(compressed))
    slot_end = struct.pack("<III", 0, 0, 0)
    return header + slot_entry + slot_end + compressed + (b"\x00" * 16)


# ==============================================================================
# 1. GameInspectionReport & Dataclass Properties
# ==============================================================================


class TestGameInspectionReport:
    """Tests for GameInspectionReport data structure and properties."""

    def test_report_properties_and_accessors(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        report = GameInspectionReport(
            root=tmp_path,
            game_dir=game_dir,
            version="8",
            engine_generation="Ren'Py 8 (Python 3)",
            version_details="Runtime lib/py3",
            script_mode="HYBRID",
            rpy_count=12,
            rpyc_count=15,
            rpa_count=2,
            languages=["vietnamese", "japanese"],
            target_lang="vietnamese",
            target_lang_exists=True,
            target_lang_files_count=5,
            recommendation="Ready for extraction.",
        )
        assert report.rpy_files_count == 12
        assert report.rpyc_files_count == 15
        assert report.rpa_files_count == 2
        assert report.configured_languages == ["vietnamese", "japanese"]
        assert report.target_lang_exists is True
        assert report.target_lang_files_count == 5


# ==============================================================================
# 2. Ren'Py Engine Version Detection (Ren'Py 6, 7, 8)
# ==============================================================================


class TestInspectGameRenpyVersions:
    """Tests for multi-tier Ren'Py engine generation detection."""

    def test_detect_renpy8_via_py3_linux(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "lib" / "py3-linux-x86_64"
        lib_dir.mkdir(parents=True)
        report = inspect_game(tmp_path)
        assert report.version == RenpyVersion.V8.value
        assert "Ren'Py 8" in report.engine_generation
        assert "py3" in report.version_details

    def test_detect_renpy8_via_py3_windows(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "lib" / "py3-windows-x86_64"
        lib_dir.mkdir(parents=True)
        report = inspect_game(tmp_path)
        assert report.version == "8"
        assert "Ren'Py 8" in report.engine_generation

    def test_detect_renpy8_via_python3_name(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "lib" / "python3.10"
        lib_dir.mkdir(parents=True)
        report = inspect_game(tmp_path)
        assert report.version == "8"

    def test_detect_renpy7_via_py2_windows(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "lib" / "py2-windows-x86_64"
        lib_dir.mkdir(parents=True)
        report = inspect_game(tmp_path)
        assert report.version == RenpyVersion.V7.value
        assert "Ren'Py 7" in report.engine_generation

    def test_detect_renpy7_via_windows_x86_64(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "lib" / "windows-x86_64"
        lib_dir.mkdir(parents=True)
        report = inspect_game(tmp_path)
        assert report.version == "7"

    def test_detect_renpy6_via_legacy_windows_i686(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "lib" / "windows-i686"
        lib_dir.mkdir(parents=True)
        report = inspect_game(tmp_path)
        assert report.version == RenpyVersion.V6.value
        assert "Ren'Py 6" in report.engine_generation

    def test_detect_renpy8_via_rpyc_pickle_protocol_3(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "script.rpyc").write_bytes(_make_synthetic_rpyc(pickle_protocol=3))
        report = inspect_game(tmp_path)
        assert report.version == "8"
        assert "pickle protocol >= 3" in report.version_details

    def test_detect_renpy8_via_rpyc_pickle_protocol_4(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "script.rpyc").write_bytes(_make_synthetic_rpyc(pickle_protocol=4))
        report = inspect_game(tmp_path)
        assert report.version == "8"

    def test_detect_renpy7_via_rpyc_protocol_2_and_gui_rpy(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "gui.rpy").write_text("# gui\n", encoding="utf-8")
        (game_dir / "script.rpyc").write_bytes(_make_synthetic_rpyc(pickle_protocol=2))
        report = inspect_game(tmp_path)
        assert report.version == "7"
        assert "gui.rpy" in report.version_details

    def test_detect_renpy6_via_rpyc_protocol_2_without_gui_rpy(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "options.rpy").write_text("# options\n", encoding="utf-8")
        (game_dir / "script.rpyc").write_bytes(_make_synthetic_rpyc(pickle_protocol=2))
        report = inspect_game(tmp_path)
        assert report.version == "6"

    def test_detect_version_via_vc_version_file(self, tmp_path: Path) -> None:
        renpy_dir = tmp_path / "renpy"
        renpy_dir.mkdir()
        (renpy_dir / "vc_version.py").write_text(
            "version_tuple = (7, 4, 11, 2266)\n", encoding="utf-8"
        )
        report = inspect_game(tmp_path)
        assert report.version == "7"
        assert "vc_version.py" in report.version_details

    def test_detect_renpy7_via_gui_rpy_without_lib(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "gui.rpy").write_text("# modern gui\n", encoding="utf-8")
        report = inspect_game(tmp_path)
        assert report.version == "7"
        assert "gui.rpy" in report.version_details

    def test_detect_renpy6_via_classic_options_rpy(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "options.rpy").write_text("# classic options\n", encoding="utf-8")
        report = inspect_game(tmp_path)
        assert report.version == "6"
        assert "options.rpy" in report.version_details

    def test_default_version_fallback(self, tmp_path: Path) -> None:
        report = inspect_game(tmp_path)
        assert report.version == "8"


# ==============================================================================
# 3. Script Mode & File Composition
# ==============================================================================


class TestInspectGameScriptModes:
    """Tests for script composition analysis: RPY_SOURCE, RPYC_BYTECODE_ONLY, HYBRID, EMPTY."""

    def test_script_mode_empty(self, tmp_path: Path) -> None:
        report = inspect_game(tmp_path)
        assert report.script_mode == ScriptMode.EMPTY.value
        assert report.rpy_count == 0
        assert report.rpyc_count == 0
        assert report.rpa_count == 0
        assert "Chưa phát hiện kịch bản" in report.recommendation

    def test_script_mode_rpy_source_only(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "script.rpy").write_text('label start:\n    "Hello"\n', encoding="utf-8")
        report = inspect_game(tmp_path)
        assert report.script_mode == ScriptMode.RPY_SOURCE.value
        assert report.rpy_count == 1
        assert report.rpyc_count == 0
        assert "sẵn sàng" in report.recommendation

    def test_script_mode_rpyc_bytecode_only(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "script.rpyc").write_bytes(b"RENPY RPC2\x00")
        report = inspect_game(tmp_path)
        assert report.script_mode == ScriptMode.RPYC_BYTECODE_ONLY.value
        assert report.rpy_count == 0
        assert report.rpyc_count == 1
        assert "Slot-2" in report.recommendation

    def test_script_mode_hybrid(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "script.rpy").write_text("# rpy\n", encoding="utf-8")
        (game_dir / "script.rpyc").write_bytes(b"RENPY RPC2\x00")
        (game_dir / "archive.rpa").write_bytes(b"RPA-3.0 0000000000000000 00000000\n")
        report = inspect_game(tmp_path)
        assert report.script_mode == ScriptMode.HYBRID.value
        assert report.rpy_count == 1
        assert report.rpyc_count == 1
        assert report.rpa_count == 1
        assert "Đồng bộ" in report.recommendation or "đồng bộ" in report.recommendation

    def test_inspect_from_game_subdirectory(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir()
        (game_dir / "script.rpy").write_text("# rpy\n", encoding="utf-8")
        report = inspect_game(game_dir)
        assert report.root == tmp_path
        assert report.game_dir == game_dir
        assert report.rpy_count == 1


# ==============================================================================
# 4. Configured Target Languages Detection
# ==============================================================================


class TestInspectGameLanguages:
    """Tests for detecting existing language folders in game/tl/."""

    def test_languages_detection_and_none_exclusion(self, tmp_path: Path) -> None:
        tl_dir = tmp_path / "game" / "tl"
        (tl_dir / "vietnamese").mkdir(parents=True)
        (tl_dir / "japanese").mkdir(parents=True)
        (tl_dir / "russian").mkdir(parents=True)
        (tl_dir / "None").mkdir(parents=True)

        report = inspect_game(tmp_path, target_lang="vietnamese")
        assert report.languages == ["japanese", "russian", "vietnamese"]
        assert "None" not in report.languages
        assert report.target_lang == "vietnamese"
        assert report.target_lang_exists is True

    def test_target_language_file_count(self, tmp_path: Path) -> None:
        tl_target = tmp_path / "game" / "tl" / "vietnamese"
        tl_target.mkdir(parents=True)
        (tl_target / "script.rpy").write_text("# vi\n", encoding="utf-8")
        (tl_target / "common.rpy").write_text("# vi common\n", encoding="utf-8")

        report = inspect_game(tmp_path, target_lang="vietnamese")
        assert report.target_lang_files_count == 2

    def test_target_language_missing(self, tmp_path: Path) -> None:
        (tmp_path / "game").mkdir()
        report = inspect_game(tmp_path, target_lang="french")
        assert report.target_lang_exists is False
        assert report.target_lang_files_count == 0


# ==============================================================================
# 5. GameInspector Class & Unicode Inspection Card
# ==============================================================================


class TestGameInspectorClassAndCard:
    """Tests for GameInspector wrapper and Unicode light box rendering."""

    def test_game_inspector_class_and_instance(self, tmp_path: Path) -> None:
        (tmp_path / "game").mkdir()
        rep1 = GameInspector.inspect(tmp_path)
        assert isinstance(rep1, GameInspectionReport)

        inspector = GameInspector(tmp_path)
        rep2 = inspector.run()
        assert rep2.version == rep1.version
        assert rep2.script_mode == rep1.script_mode

    def test_render_inspection_card_layout(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        report = GameInspectionReport(
            root=tmp_path,
            game_dir=game_dir,
            version="8",
            engine_generation="Ren'Py 8 (Python 3)",
            version_details="Phát hiện qua runtime lib/py3",
            script_mode="HYBRID",
            rpy_count=5,
            rpyc_count=10,
            rpa_count=1,
            languages=["vietnamese"],
            target_lang="vietnamese",
            target_lang_exists=True,
            target_lang_files_count=2,
            recommendation="Sẵn sàng trích xuất và đồng bộ.",
        )
        card = render_inspection_card(report)
        assert "KẾT QUẢ KIỂM TRA DỰ ÁN REN'PY" in card
        assert "Ren'Py 8 (Python 3)" in card
        assert "HYBRID" in card
        assert "5 .rpy" in card
        assert "vietnamese" in card
        assert "Sẵn sàng" in card
        for line in card.splitlines():
            assert line.startswith(("┌", "│", "└"))
            assert line.endswith(("┐", "│", "┘"))


# ==============================================================================
# 6. 6-Phase State Machine & Transitions
# ==============================================================================


class TestWizardStateMachine:
    """Tests for 6-phase state machine transitions, events, and context."""

    def test_phases_and_state_alias(self) -> None:
        assert WizardState is WizardPhase
        phases = [
            WizardPhase.INIT,
            WizardPhase.DETECT_GAME,
            WizardPhase.EXTRACT_OR_REPAIR,
            WizardPhase.LLM_PROBE,
            WizardPhase.TRANSLATE_DASHBOARD,
            WizardPhase.APPLY_VERIFY,
            WizardPhase.EXIT,
        ]
        assert len(phases) == 7

    def test_sequential_forward_step_transitions(self, tmp_path: Path) -> None:
        wiz = InteractiveWizard(tmp_path)
        assert wiz.current_phase == WizardPhase.INIT

        assert wiz.step() == WizardPhase.DETECT_GAME
        assert wiz.step() == WizardPhase.EXTRACT_OR_REPAIR
        assert wiz.step() == WizardPhase.LLM_PROBE
        assert wiz.step() == WizardPhase.TRANSLATE_DASHBOARD
        assert wiz.step() == WizardPhase.APPLY_VERIFY
        assert wiz.step() == WizardPhase.EXIT

    def test_backward_transitions(self, tmp_path: Path) -> None:
        wiz = InteractiveWizard(tmp_path)
        wiz.transition(WizardPhase.APPLY_VERIFY)

        assert wiz.send_event(WizardEvent.BACK) == WizardPhase.TRANSLATE_DASHBOARD
        assert wiz.send_event(WizardEvent.BACK) == WizardPhase.LLM_PROBE
        assert wiz.send_event(WizardEvent.BACK) == WizardPhase.EXTRACT_OR_REPAIR
        assert wiz.send_event(WizardEvent.BACK) == WizardPhase.DETECT_GAME
        assert wiz.send_event(WizardEvent.BACK) == WizardPhase.INIT

    def test_cancel_event_exits_from_all_phases(self, tmp_path: Path) -> None:
        for phase in (
            WizardPhase.INIT,
            WizardPhase.DETECT_GAME,
            WizardPhase.EXTRACT_OR_REPAIR,
            WizardPhase.LLM_PROBE,
            WizardPhase.TRANSLATE_DASHBOARD,
            WizardPhase.APPLY_VERIFY,
        ):
            wiz = InteractiveWizard(tmp_path)
            wiz.transition(phase)
            assert wiz.send_event(WizardEvent.CANCEL) == WizardPhase.EXIT

    def test_retry_event_behavior(self, tmp_path: Path) -> None:
        wiz = InteractiveWizard(tmp_path)
        wiz.transition(WizardPhase.EXTRACT_OR_REPAIR)
        assert wiz.send_event(WizardEvent.RETRY) == WizardPhase.EXTRACT_OR_REPAIR

        wiz.transition(WizardPhase.LLM_PROBE)
        assert wiz.send_event(WizardEvent.RETRY) == WizardPhase.LLM_PROBE

    def test_default_transitions_mapping_integrity(self) -> None:
        for phase, events in DEFAULT_TRANSITIONS.items():
            assert isinstance(phase, WizardPhase)
            for event, target in events.items():
                assert isinstance(event, WizardEvent)
                assert isinstance(target, WizardPhase)

    def test_wizard_context_initialization(self, tmp_path: Path) -> None:
        ctx = WizardContext(root=tmp_path)
        assert ctx.root == tmp_path
        assert ctx.chunk_size == 100
        assert ctx.retry_failed is False
        assert ctx.auto_apply is True


# ==============================================================================
# 7. Pure Stdlib Terminal Navigation (termios / msvcrt)
# ==============================================================================


class TestTerminalNavigation:
    """Tests for raw key reading, arrow navigation, and numeric shortcuts."""

    def test_read_single_key_non_interactive(self) -> None:
        with patch("sys.stdin.isatty", return_value=False):
            assert read_single_key() is None

    def test_read_single_key_posix_arrows(self) -> None:
        with (
            patch("sys.platform", "linux"),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdin.fileno", return_value=0),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr", return_value=None),
            patch("tty.setcbreak", return_value=None),
            patch("select.select", side_effect=[([True], [], []), ([True], [], [])]),
            patch("sys.stdin.read", side_effect=["\x1b", "[", "A"]),
        ):
            assert read_single_key() == "UP"

    def test_read_single_key_posix_enter(self) -> None:
        with (
            patch("sys.platform", "linux"),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdin.fileno", return_value=0),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr", return_value=None),
            patch("tty.setcbreak", return_value=None),
            patch("sys.stdin.read", return_value="\n"),
        ):
            assert read_single_key() == "ENTER"

    def test_read_single_key_posix_raw_arrows(self) -> None:
        with (
            patch("sys.platform", "linux"),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdin.fileno", return_value=0),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr", return_value=None),
            patch("tty.setcbreak", return_value=None),
            patch("select.select", return_value=([0], [], [])),
            patch("os.read", side_effect=[b"\x1b", b"[A", b"\x1b", b"[B"]),
        ):
            assert read_single_key() == "UP"
            assert read_single_key() == "DOWN"

    def test_read_single_key_posix_raw_standalone_escape(self) -> None:
        with (
            patch("sys.platform", "linux"),
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdin.fileno", return_value=0),
            patch("termios.tcgetattr", return_value=[]),
            patch("termios.tcsetattr", return_value=None),
            patch("tty.setcbreak", return_value=None),
            patch("select.select", return_value=([], [], [])),
            patch("os.read", return_value=b"\x1b"),
        ):
            assert read_single_key() == "ESCAPE"

    def test_read_single_key_windows_mock(self) -> None:
        mock_msvcrt = MagicMock()
        mock_msvcrt.getch.side_effect = [b"\xe0", b"H"]
        with (
            patch("sys.platform", "win32"),
            patch.dict("sys.modules", {"msvcrt": mock_msvcrt}),
        ):
            assert read_single_key() == "UP"


# ==============================================================================
# 8. select_menu Interactive and Non-Interactive Modes
# ==============================================================================


class TestSelectMenu:
    """Tests for menu rendering, arrow keys, numeric shortcuts, and fallbacks."""

    def test_select_menu_empty_options(self) -> None:
        assert select_menu("Empty", []) == ""

    def test_select_menu_non_interactive_default_input(self) -> None:
        options = [("1", "First"), ("2", "Second")]
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("builtins.input", return_value=""),
        ):
            res = select_menu("Test", options, default_index=0)
            assert res == "1"

    def test_select_menu_non_interactive_choice_by_key(self) -> None:
        options = [("1", "First"), ("2", "Second")]
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("builtins.input", return_value="2"),
        ):
            res = select_menu("Test", options)
            assert res == "2"

    def test_select_menu_non_interactive_choice_by_index(self) -> None:
        options = [("opt_a", "First"), ("opt_b", "Second")]
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("builtins.input", return_value="2"),
        ):
            res = select_menu("Test", options)
            assert res == "opt_b"

    def test_select_menu_non_interactive_eof_handling(self) -> None:
        options = [("1", "First"), ("2", "Second")]
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("builtins.input", side_effect=EOFError),
        ):
            res = select_menu("Test", options, default_index=1)
            assert res == "2"

    def test_select_menu_interactive_arrow_navigation(self) -> None:
        options = [("1", "First"), ("2", "Second"), ("3", "Third")]
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch.dict("os.environ", {"TERM": "xterm-256color"}),
            patch("renslate.ui.wizard.read_single_key", side_effect=["DOWN", "DOWN", "ENTER"]),
        ):
            res = select_menu("Select", options, default_index=0)
            assert res == "3"

    def test_select_menu_interactive_numeric_shortcut(self) -> None:
        options = [("1", "First"), ("2", "Second")]
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch.dict("os.environ", {"TERM": "xterm-256color"}),
            patch("renslate.ui.wizard.read_single_key", side_effect=["2"]),
        ):
            res = select_menu("Select", options)
            assert res == "2"

    def test_select_menu_interactive_escape_returns_zero_or_last(self) -> None:
        options = [("1", "Start"), ("0", "Quit")]
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch.dict("os.environ", {"TERM": "xterm-256color"}),
            patch("renslate.ui.wizard.read_single_key", side_effect=["ESCAPE"]),
        ):
            res = select_menu("Select", options)
            assert res == "0"


# ==============================================================================
# 9. InteractiveWizard Execution & Non-Interactive Walkthrough
# ==============================================================================


class TestInteractiveWizardExecution:
    """Tests for InteractiveWizard prompt, confirm, and run_state_machine."""

    def test_prompt_custom_and_default(self) -> None:
        wiz = InteractiveWizard()
        with patch("builtins.input", side_effect=["custom_val", "", EOFError]):
            assert wiz.prompt("Enter name", default="default_name") == "custom_val"
            assert wiz.prompt("Enter name", default="default_name") == "default_name"
            assert wiz.prompt("Enter name", default="default_name") == "default_name"

    def test_confirm_responses(self) -> None:
        wiz = InteractiveWizard()
        with patch("builtins.input", side_effect=["y", "yes", "1", "n", "", KeyboardInterrupt]):
            assert wiz.confirm("Q1", default=False) is True
            assert wiz.confirm("Q2", default=False) is True
            assert wiz.confirm("Q3", default=False) is True
            assert wiz.confirm("Q4", default=True) is False
            assert wiz.confirm("Q5", default=True) is True
            assert wiz.confirm("Q6", default=True) is True

    def test_run_state_machine_non_interactive_fast_path(self, tmp_path: Path) -> None:
        (tmp_path / "game").mkdir()
        wiz = InteractiveWizard(tmp_path)
        with patch("sys.stdin.isatty", return_value=False):
            code = wiz.run_state_machine()
            assert code == 0
            assert (tmp_path / ".renslate").is_dir()

    def test_run_state_machine_simulated_full_loop(self, tmp_path: Path) -> None:
        (tmp_path / "game").mkdir()
        Project.create(tmp_path)
        wiz = InteractiveWizard(tmp_path)

        menu_choices = [
            "1",  # INIT: Continue with current directory
            "1",  # DETECT_GAME: Continue to extraction
            "1",  # EXTRACT_OR_REPAIR: Continue to LLM probe
            "1",  # LLM_PROBE: Start translation
            "0",  # APPLY_VERIFY: Finish and exit
        ]
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch.dict("os.environ", {"TERM": "xterm-256color"}),
            patch("renslate.ui.wizard.select_menu", side_effect=menu_choices),
            patch("renslate.ui.runner.check_server_health", return_value=True),
            patch("renslate.ui.runner.detect_server_model", return_value="qwen2.5-7b"),
            patch("renslate.ui.runner.run_unattended_pipeline", return_value=0),
        ):
            code = wiz.run_state_machine()
            assert code == 0
            assert wiz.state == WizardPhase.EXIT

    def test_run_state_machine_confirm_create_project(self, tmp_path: Path) -> None:
        (tmp_path / "game").mkdir()
        wiz = InteractiveWizard(tmp_path)
        menu_choices = ["1", "0"]
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch.dict("os.environ", {"TERM": "xterm-256color"}),
            patch("renslate.ui.wizard.select_menu", side_effect=menu_choices),
            patch("builtins.input", return_value="y"),
        ):
            code = wiz.run_state_machine()
            assert code == 0
            assert (tmp_path / ".renslate").is_dir()

    def test_run_state_machine_early_quit(self, tmp_path: Path) -> None:
        (tmp_path / "game").mkdir()
        wiz = InteractiveWizard(tmp_path)
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch.dict("os.environ", {"TERM": "xterm-256color"}),
            patch("renslate.ui.wizard.select_menu", return_value="0"),
        ):
            code = wiz.run_state_machine()
            assert code == 0
            assert wiz.state == WizardPhase.EXIT

    def test_run_interactive_wizard_aliases(self, tmp_path: Path) -> None:
        (tmp_path / "game").mkdir()
        with patch("sys.stdin.isatty", return_value=False):
            assert run_interactive_wizard(tmp_path) == 0
            assert run_wizard(tmp_path) == 0


# ==============================================================================
# 10. CLI Zero-Command & Subcommand Integration
# ==============================================================================


class TestCLIWizardZeroCommand:
    """Tests for zero-command launch on interactive TTY and non-tty safety."""

    def test_main_zero_args_non_interactive_raises_2(self) -> None:
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("sys.stdout.isatty", return_value=False),
            pytest.raises(SystemExit) as exc,
        ):
            main([])
        assert exc.value.code == 2

    def test_main_zero_args_interactive_launches_wizard(self, tmp_path: Path) -> None:
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("sys.stdout.isatty", return_value=True),
            patch("renslate.ui.wizard.run_interactive_wizard", return_value=0) as mock_wiz,
        ):
            code = main([])
            assert code == 0
            assert mock_wiz.call_count == 1

    def test_main_subcommand_wizard(self, tmp_path: Path) -> None:
        (tmp_path / "game").mkdir()
        Project.create(tmp_path)
        with patch("renslate.ui.wizard.run_interactive_wizard", return_value=0) as mock_wiz:
            code = main(["wizard", "-p", str(tmp_path)])
            assert code == 0
            assert mock_wiz.call_count == 1

    def test_main_subcommand_interactive_alias(self, tmp_path: Path) -> None:
        (tmp_path / "game").mkdir()
        Project.create(tmp_path)
        with patch("renslate.ui.wizard.run_interactive_wizard", return_value=0) as mock_wiz:
            code = main(["interactive", "-p", str(tmp_path)])
            assert code == 0
            assert mock_wiz.call_count == 1
