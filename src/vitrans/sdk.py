"""Driver for the Ren'Py SDK's ``translate`` command, which produces the translation skeleton."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from vitrans.errors import SdkError

Runner = Callable[..., "subprocess.CompletedProcess[str]"]

_TAIL_LINES = 15


def find_launcher(sdk: Path) -> Path:
    """Locate the SDK's command-line launcher or game launcher for the current platform."""
    if sdk.is_file():
        return sdk

    primary_name = "renpy.exe" if os.name == "nt" else "renpy.sh"
    launcher = sdk / primary_name
    if launcher.is_file():
        return launcher

    if (sdk / "renpy").is_file():
        return sdk / "renpy"

    # Fallback: check for standalone game distribution launcher
    pattern = "*.exe" if os.name == "nt" else "*.sh"
    candidates = [
        f for f in sdk.glob(pattern) if f.is_file() and not f.name.lower().startswith("uninstall")
    ]
    if candidates:
        return sorted(candidates, key=lambda p: len(p.name))[0]

    raise SdkError(f"không tìm thấy trình khởi chạy Ren'Py hoặc game launcher trong: {sdk}")


def build_command(launcher: Path, project_root: Path, language: str) -> list[str]:
    return [str(launcher), str(project_root), "translate", language, "--no-todo"]


def _tail(output: str) -> str:
    lines = [line for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-_TAIL_LINES:])


def generate_translations(
    sdk: Path,
    project_root: Path,
    language: str,
    *,
    runner: Runner = subprocess.run,
    extra_args: Sequence[str] = (),
) -> str:
    """Run ``renpy translate`` for language and return its output; raise SdkError on failure."""
    command = [*build_command(find_launcher(sdk), project_root, language), *extra_args]
    try:
        result = runner(command, capture_output=True, text=True, check=False)
    except OSError as error:
        raise SdkError(f"không chạy được trình khởi chạy Ren'Py: {error}") from error
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        raise SdkError(f"renpy translate thất bại (mã thoát {result.returncode}):\n{_tail(output)}")
    return output
