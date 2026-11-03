"""Tests for new renslate CLI commands (analyze, qa, export, import)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from renslate.cli import main

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
        "renslate.llm.LLMClient.complete_json",
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
        "renslate.llm.LLMClient.complete_json",
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


def test_cli_qa_with_decoupled_guard(project_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Test renslate qa --guard with mock HallucinationGuard
    mock_scan_resp = {
        "scanned_units": [
            {
                "uid": "d:start_a1b2c3d4",
                "verdict": "PASS",
                "issue_type": "none",
                "confidence": 0.98,
                "explanation": "Dịch chuẩn xác",
            }
        ]
    }
    with patch(
        "renslate.engine.llm.LLMClient.complete_json",
        return_value=(mock_scan_resp, 100, 30),
    ):
        assert main(["qa", "-p", str(project_root), "--guard", "--limit", "5"]) == 0

    out = capsys.readouterr().out
    assert "Bắt đầu thẩm định ảo giác bằng LLM Hallucination Guard" in out
    assert "Đang quét lô 1/1" in out


def test_cli_translate_with_optional_guard_flag(project_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target_val = "Xin chào bạn!"
    with patch(
        "renslate.engine.llm.LLMClient.complete_json",
        return_value=({"d:start_a1b2c3d4": target_val}, 50, 20),
    ):
        assert main(["translate", "-p", str(project_root), "--limit", "1", "--guard"]) == 0

    out = capsys.readouterr().out
    assert "Continuous LLM Hallucination Guard đang hoạt động" in out


def test_cli_translate_with_concise_tags_flag(project_root: Path) -> None:
    with (
        patch(
            "renslate.engine.llm.LLMClient.complete_json",
            return_value=({"d:start_a1b2c3d4": "Xin chào!"}, 50, 20),
        ),
        patch("renslate.cli.TranslationPipeline") as mock_pipeline_cls,
    ):
        mock_pipeline = mock_pipeline_cls.return_value
        from renslate.engine.translate import TranslationStats

        mock_pipeline.run.return_value = TranslationStats()
        assert main(["translate", "-p", str(project_root), "--limit", "1", "--concise-tags"]) == 0
        call_config = mock_pipeline_cls.call_args[0][2]
        assert call_config.batch.concise_tags is True


def test_cli_run_with_concise_tags_flag(project_root: Path) -> None:
    with patch("renslate.ui.runner.run_unattended_pipeline", return_value=0) as mock_runner:
        assert main(["run", "-p", str(project_root), "--concise-tags"]) == 0
        assert mock_runner.call_args.kwargs.get("concise_tags") is True

