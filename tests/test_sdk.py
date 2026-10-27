from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from vitrans.errors import SdkError
from vitrans.sdk import build_command, find_launcher, generate_translations

LAUNCHER = "renpy.exe" if os.name == "nt" else "renpy.sh"


def make_sdk(tmp_path: Path, body: str = "") -> Path:
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    launcher = sdk / LAUNCHER
    launcher.write_text(body, encoding="utf-8")
    launcher.chmod(0o755)
    return sdk


def completed(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        runner.command = command  # type: ignore[attr-defined]
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    return runner


def test_launcher_must_exist(tmp_path: Path) -> None:
    with pytest.raises(SdkError):
        find_launcher(tmp_path)
    assert find_launcher(make_sdk(tmp_path)).name == LAUNCHER


def test_command_line() -> None:
    command = build_command(Path("/sdk/renpy.sh"), Path("/games/demo"), "vietnamese")
    assert command == [
        str(Path("/sdk/renpy.sh")),
        str(Path("/games/demo")),
        "translate",
        "vietnamese",
        "--no-todo",
    ]


def test_successful_run_returns_output(tmp_path: Path) -> None:
    runner = completed(0, stdout="done", stderr="note")
    output = generate_translations(
        make_sdk(tmp_path),
        tmp_path / "game",
        "vietnamese",
        runner=runner,
        extra_args=["--strings-only"],
    )
    assert output == "done\nnote"
    assert runner.command[-2:] == ["--no-todo", "--strings-only"]  # type: ignore[attr-defined]


def test_failure_reports_the_tail_of_the_output(tmp_path: Path) -> None:
    noisy = "\n".join(f"line {i}" for i in range(40))
    runner = completed(2, stderr=noisy)
    with pytest.raises(SdkError) as raised:
        generate_translations(make_sdk(tmp_path), tmp_path, "vietnamese", runner=runner)
    message = str(raised.value)
    assert "line 39" in message
    assert "line 0\n" not in message
    assert "2" in message


def test_launch_errors_are_wrapped(tmp_path: Path) -> None:
    def runner(*_: Any, **__: Any) -> Any:
        raise PermissionError("denied")

    with pytest.raises(SdkError, match="denied"):
        generate_translations(make_sdk(tmp_path), tmp_path, "vietnamese", runner=runner)


@pytest.mark.skipif(os.name == "nt", reason="requires a POSIX shell")
def test_real_subprocess_invocation(tmp_path: Path) -> None:
    sdk = make_sdk(
        tmp_path,
        '#!/bin/sh\nmkdir -p "$1/game/tl/$3" && echo "args: $*" && '
        'printf "# generated\\n" > "$1/game/tl/$3/script.rpy"\n',
    )
    project = tmp_path / "project"
    project.mkdir()
    output = generate_translations(sdk, project, "vietnamese")
    assert f"args: {project} translate vietnamese --no-todo" in output
    assert (project / "game" / "tl" / "vietnamese" / "script.rpy").is_file()
