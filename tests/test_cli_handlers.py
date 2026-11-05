"""Unit tests for decomposed domain-driven CLI command handlers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from renslate.cli.handlers.project import cmd_init
from renslate.cli.handlers.qa import cmd_clean_db
from renslate.cli.handlers.tools import cmd_status
from renslate.core.project import Project


def test_cmd_init_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "game").mkdir()
    args = argparse.Namespace(project=str(tmp_path), lang="vietnamese", sdk=None, force=False)
    ret = cmd_init(args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Đã khởi tạo dự án tại" in captured.out
    assert (tmp_path / ".renslate" / "config.json").is_file()
    assert (tmp_path / ".renslate" / "project.db").is_file()


def test_cmd_status_empty_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "game").mkdir()
    Project.create(tmp_path)
    args = argparse.Namespace(project=str(tmp_path), dashboard=False, plain=True)
    ret = cmd_status(args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Chưa có dữ liệu, hãy chạy 'renslate extract'." in captured.out


def test_cmd_clean_db_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "game").mkdir()
    proj = Project.create(tmp_path)
    proj.tl_root.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(project=str(tmp_path))
    ret = cmd_clean_db(args)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Đã làm sạch" in captured.out
