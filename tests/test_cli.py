from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.test_sdk import make_sdk
from vitrans import __version__
from vitrans.cli import main

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "demo"
    (path / "game").mkdir(parents=True)
    return path


@pytest.fixture
def translated_root(root: Path) -> Path:
    shutil.copytree(DATA_DIR / "tl", root / "game" / "tl")
    assert main(["init", "-p", str(root)]) == 0
    return root


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])
    assert raised.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_command_is_required() -> None:
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_init_creates_configuration_and_database(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["init", "-p", str(root), "--lang", "vi_VN"]) == 0
    assert (root / ".vitrans" / "config.json").is_file()
    assert (root / ".vitrans" / "project.db").is_file()
    assert "Đã khởi tạo" in capsys.readouterr().out

    assert main(["init", "-p", str(root)]) == 1
    assert "--force" in capsys.readouterr().err
    assert main(["init", "-p", str(root), "--force"]) == 0


def test_init_rejects_directories_that_are_not_projects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["init", "-p", str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("lỗi:")


def test_commands_require_an_initialised_project(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for command in ("extract", "apply", "status", "generate"):
        assert main([command, "-p", str(root)]) == 1
        assert "vitrans init" in capsys.readouterr().err


def test_extract_status_and_apply_workflow(
    translated_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["status", "-p", str(translated_root)]) == 0
    assert "vitrans extract" in capsys.readouterr().out

    assert main(["extract", "-p", str(translated_root)]) == 0
    assert "11 đơn vị dịch" in capsys.readouterr().out

    assert main(["status", "-p", str(translated_root)]) == 0
    status = capsys.readouterr().out
    assert "dialogue" in status
    assert "Tổng: 11 đơn vị, 11 vị trí ghi" in status

    assert main(["apply", "--dry-run", "-p", str(translated_root)]) == 0
    assert "Sẽ ghi 0 dòng" in capsys.readouterr().out
    assert main(["apply", "-p", str(translated_root)]) == 0
    assert "Đã ghi 0 dòng" in capsys.readouterr().out


def test_extract_exit_code_reflects_errors(
    translated_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = translated_root / "game" / "tl" / "vietnamese" / "broken.rpy"
    broken.write_text('translate vietnamese x_00000000:\n\n    # e "x"\n', encoding="utf-8")
    assert main(["extract", "-p", str(translated_root)]) == 1
    captured = capsys.readouterr()
    assert "block_mismatch" in captured.err
    assert "1 lỗi" in captured.err


def test_default_project_is_the_working_directory(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(root)
    assert main(["init"]) == 0
    assert main(["status"]) == 0
    assert str(root.resolve()) in capsys.readouterr().out


def test_generate_requires_an_sdk(
    translated_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["generate", "-p", str(translated_root)]) == 1
    assert "--sdk" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="requires a POSIX shell")
def test_generate_then_extract_with_a_fake_sdk(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = (
        '#!/bin/sh\nmkdir -p "$1/game/tl/$3"\n'
        'printf \'translate %s strings:\\n\\n    old "Hi"\\n    new "Hi"\\n\' "$3" '
        '> "$1/game/tl/$3/script.rpy"\n'
    )
    sdk = make_sdk(tmp_path, body)
    assert main(["init", "-p", str(root), "--sdk", str(sdk)]) == 0
    assert main(["generate", "-p", str(root)]) == 0
    assert "game/tl/vietnamese" in capsys.readouterr().out
    assert main(["extract", "-p", str(root)]) == 0
    assert "1 đơn vị dịch" in capsys.readouterr().out
