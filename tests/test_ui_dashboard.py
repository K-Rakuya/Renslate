"""Tests for real-time interactive terminal dashboard subsystem."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from renslate.cli import main
from renslate.core.project import Project
from renslate.ui.dashboard import (
    DashboardSnapshot,
    _box_divider,
    _box_line,
    _truncate_visible,
    query_dashboard_snapshot,
    render_dashboard_screen,
    run_live_dashboard,
)

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sample_project(tmp_path: Path) -> Project:
    path = tmp_path / "game_proj"
    shutil.copytree(DATA_DIR / "tl", path / "game" / "tl")
    assert main(["init", "-p", str(path)]) == 0
    assert main(["extract", "-p", str(path)]) == 0
    return Project.open(path)


def test_truncate_visible() -> None:
    assert _truncate_visible("Hello world", 20) == "Hello world"
    assert _truncate_visible("Xin chào bạn tôi", 10) == "Xin chà..."
    assert _truncate_visible("こんにちは世界", 8) == "こん..."


def test_box_line_and_divider() -> None:
    line = _box_line("Test", width=30)
    assert line.startswith("│ ")
    assert line.endswith(" │")
    div = _box_divider(width=30)
    assert div.startswith("├")
    assert div.endswith("┤")


def test_query_dashboard_snapshot(sample_project: Project) -> None:
    snapshot = query_dashboard_snapshot(sample_project, filter_mode="all", limit=5)
    assert isinstance(snapshot, DashboardSnapshot)
    assert snapshot.project_name == sample_project.root.name
    assert snapshot.target_lang == "vietnamese"
    assert snapshot.total_units > 0
    assert snapshot.percent_complete >= 0.0
    assert isinstance(snapshot.kind_counts, dict)
    assert isinstance(snapshot.top_speakers, list)
    assert isinstance(snapshot.recent_units, list)


def test_render_dashboard_screen_plain(sample_project: Project) -> None:
    snapshot = query_dashboard_snapshot(sample_project, filter_mode="all", limit=5)
    screen = render_dashboard_screen(
        snapshot, width=80, plain=True, filter_mode="all", interval=2.0
    )
    assert "RENSLATE LOCALIZATION STUDIO" in screen
    assert "REALTIME LIVE MONITOR" in screen
    assert sample_project.root.name in screen
    assert "TIẾN ĐỘ TOÀN DỰ ÁN" in screen
    assert "PHÂN BỔ KỊCH BẢN" in screen
    assert "DÒNG THOẠI GẦN ĐÂY" in screen
    assert "[q] Thoát" in screen
    assert "[r] Tải lại" in screen
    assert "\033[" not in screen  # Plain mode must not contain ANSI escape codes


def test_render_dashboard_screen_with_units(sample_project: Project) -> None:
    with sample_project.open_db() as db:
        db.execute(
            "UPDATE units SET target = 'Lời chào từ Eileen' "
            "WHERE uid IN (SELECT uid FROM units WHERE kind = 'dialogue' LIMIT 1)"
        )
    snapshot = query_dashboard_snapshot(sample_project, filter_mode="dialogue", limit=3)
    screen = render_dashboard_screen(
        snapshot, width=80, plain=True, filter_mode="dialogue", interval=2.0
    )
    assert "BỘ LỌC: DIALOGUE" in screen
    assert "Lời chào từ Eileen" in screen


def test_run_live_dashboard_once_mode(
    sample_project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    res = run_live_dashboard(sample_project, once=True, plain=True, limit=3)
    assert res == 0
    out = capsys.readouterr().out
    assert "RENSLATE LOCALIZATION STUDIO" in out
    assert "[q] Thoát" in out


def test_cli_dashboard_once_flag(
    sample_project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["dashboard", "-p", str(sample_project.root), "--plain", "--once"])
    assert code == 0
    out = capsys.readouterr().out
    assert "RENSLATE LOCALIZATION STUDIO" in out
    assert "REALTIME LIVE MONITOR" in out


def test_cli_dashboard_with_filters(
    sample_project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "dashboard",
            "-p",
            str(sample_project.root),
            "--plain",
            "--once",
            "-f",
            "dialogue",
            "-n",
            "3",
            "-i",
            "1.5",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "BỘ LỌC: DIALOGUE" in out


def test_cli_status_dashboard_flag(
    sample_project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["status", "-p", str(sample_project.root), "-d", "--plain"])
    assert code == 0
    out = capsys.readouterr().out
    assert "RENSLATE LOCALIZATION STUDIO" in out


def test_run_live_dashboard_interactive_mocked(sample_project: Project) -> None:
    with (
        patch("sys.stdout.isatty", return_value=True),
        patch("renslate.ui.dashboard._read_key_timeout", side_effect=["r", "1", "q"]),
    ):
        code = run_live_dashboard(sample_project, interval=0.1, once=False, plain=True)
        assert code == 0
