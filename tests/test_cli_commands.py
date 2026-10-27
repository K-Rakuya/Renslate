"""Tests for new vitrans CLI commands (analyze, qa, export, import)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from vitrans.cli import main

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    path = tmp_path / "game_proj"
    shutil.copytree(DATA_DIR / "tl", path / "game" / "tl")
    assert main(["init", "-p", str(path)]) == 0
    assert main(["extract", "-p", str(path)]) == 0
    return path


def test_cli_analyze_command(project_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["analyze", "-p", str(project_root)]) == 0
    out = capsys.readouterr().out
    assert "Đã phân tích dự án" in out
    assert "nhân vật phát hiện" in out


def test_cli_qa_command(project_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Initially some units are pending, some imported
    assert main(["qa", "-p", str(project_root)]) == 0
    out = capsys.readouterr().out
    assert "kiểm tra chất lượng" in out


def test_cli_export_and_import_command(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tsv_out = project_root / "test_review.tsv"
    assert main(["export", "-p", str(project_root), "-o", str(tsv_out)]) == 0
    out = capsys.readouterr().out
    assert "Đã xuất" in out
    assert tsv_out.is_file()

    # Import back without edits -> 0 imported, all skipped
    assert main(["import", "-p", str(project_root), "-i", str(tsv_out)]) == 0
    out2 = capsys.readouterr().out
    assert "Đã nhập 0 bản dịch" in out2


def test_cli_translate_mocked(project_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target_val = "Xin chào ⟦V1⟧! ⟦T1⟧Chào mừng⟦/T1⟧ đến câu lạc bộ."
    with patch(
        "vitrans.llm.LLMClient.complete_json",
        return_value=({"d:start_a1b2c3d4": target_val}, 50, 20),
    ):
        assert main(["translate", "-p", str(project_root), "--limit", "1"]) == 0

    out = capsys.readouterr().out
    assert "Kết quả dịch:" in out
    assert "chuỗi UI có sẵn" in out


def test_cli_preview_command(project_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Before new translation: already has 2 imported translations from fixture
    assert main(["preview", "-p", str(project_root)]) == 0
    out = capsys.readouterr().out
    assert "Xem trước" in out
    assert "ENG:" in out
    assert "VIE:" in out

    # After translating 1 unit
    target_val = "Xin chào ⟦V1⟧! ⟦T1⟧Chào mừng⟦/T1⟧ đến câu lạc bộ."
    with patch(
        "vitrans.llm.LLMClient.complete_json",
        return_value=({"d:start_a1b2c3d4": target_val}, 50, 20),
    ):
        main(["translate", "-p", str(project_root), "--limit", "1"])
    capsys.readouterr()

    # Now preview should show the translated line
    assert main(["preview", "-p", str(project_root), "-n", "5"]) == 0
    out2 = capsys.readouterr().out
    assert "Xem trước" in out2
    assert "ENG:" in out2
    assert "VIE:" in out2
