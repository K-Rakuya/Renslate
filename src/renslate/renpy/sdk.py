"""Driver for the Ren'Py SDK's ``translate`` command, which produces the translation skeleton."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from renslate.core.errors import SdkError

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
    sdk: Path | None,
    project_root: Path,
    language: str,
    *,
    runner: Runner = subprocess.run,
    extra_args: Sequence[str] = (),
    fallback_rpyc: bool = True,
) -> str:
    """Run ``renpy translate`` for language and return its output; raise SdkError on failure.

    If no .rpy source files exist in the project and fallback_rpyc is True, automatically
    extracts bytecode and synthesizes translation skeletons from .rpyc files.
    """
    game_root = project_root / "game" if (project_root / "game").is_dir() else project_root
    has_rpy = any(p for p in game_root.rglob("*.rpy") if "tl" not in p.parts)
    has_rpyc = any(p for p in game_root.rglob("*.rpyc") if "tl" not in p.parts)

    if not has_rpy and has_rpyc and fallback_rpyc:
        from renslate.renpy.rpyc import synthesize_translation_skeletons

        generated = synthesize_translation_skeletons(game_root, language)
        return (
            f"Không tìm thấy mã nguồn .rpy. Đã tự động sinh {len(generated)} "
            f"tệp khung dịch từ bytecode .rpyc cho ngôn ngữ {language}"
        )

    if sdk is None:
        if has_rpyc and fallback_rpyc:
            from renslate.renpy.rpyc import synthesize_translation_skeletons

            generated = synthesize_translation_skeletons(game_root, language)
            return (
                f"Không có Ren'Py SDK. Đã tự động sinh {len(generated)} "
                f"tệp khung dịch từ bytecode .rpyc cho ngôn ngữ {language}"
            )
        raise SdkError("thiếu đường dẫn Ren'Py SDK: dùng --sdk hoặc đặt 'sdk' trong cấu hình")

    try:
        launcher = find_launcher(sdk)
        command = [*build_command(launcher, project_root, language), *extra_args]
        result = runner(command, capture_output=True, text=True, check=False)
        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode != 0:
            if has_rpyc and fallback_rpyc:
                from renslate.renpy.rpyc import synthesize_translation_skeletons

                generated = synthesize_translation_skeletons(game_root, language)
                return (
                    f"Ren'Py SDK thất bại ({_tail(output)}). Đã chuyển sang trích xuất bytecode "
                    f".rpyc và sinh {len(generated)} tệp khung dịch cho ngôn ngữ {language}"
                )
            raise SdkError(
                f"renpy translate thất bại (mã thoát {result.returncode}):\n{_tail(output)}"
            )
        return output
    except OSError as error:
        if has_rpyc and fallback_rpyc:
            from renslate.renpy.rpyc import synthesize_translation_skeletons

            generated = synthesize_translation_skeletons(game_root, language)
            return (
                f"Không chạy được trình khởi chạy Ren'Py ({error}). "
                f"Đã tự động sinh {len(generated)} "
                f"tệp khung dịch từ bytecode .rpyc cho ngôn ngữ {language}"
            )
        raise SdkError(f"không chạy được trình khởi chạy Ren'Py: {error}") from error
