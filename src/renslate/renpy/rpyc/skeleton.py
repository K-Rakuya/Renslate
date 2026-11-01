from __future__ import annotations

from pathlib import Path

from renslate.renpy.literals import quote_python_text
from renslate.renpy.rpyc.decompiler import _format_say_code
from renslate.renpy.rpyc.extractor import extract_rpyc_project


def synthesize_translation_skeletons(game_dir: Path | str, target_lang: str) -> list[Path]:
    root = Path(game_dir)
    extracted = extract_rpyc_project(root)
    if (root / "game").is_dir():
        tl_root = root / "game" / "tl" / target_lang
    else:
        tl_root = root / "tl" / target_lang
    tl_root.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []
    ui_strings: list[tuple[str, int, str]] = []
    seen_ui: set[str] = set()

    for rel_path, script in extracted.items():
        source_rpy_name = Path(rel_path).with_suffix(".rpy")
        parts = source_rpy_name.parts
        if parts and parts[0] == "game":
            source_rpy_name = Path(*parts[1:])

        target_file = tl_root / source_rpy_name
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if target_file.is_file() and target_file.stat().st_size > 0:
            generated_files.append(target_file)
            for s in script.strings:
                if s.text not in seen_ui:
                    seen_ui.add(s.text)
                    ui_strings.append((s.filename, s.linenumber, s.text))
            continue

        lines: list[str] = []
        if script.dialogue:
            for item in script.dialogue:
                clean_id = item.identifier.replace(".", "_")
                ref_file = item.filename or str(source_rpy_name)
                ref_line = f"# {ref_file}:{item.linenumber}"
                header = f"translate {target_lang} {clean_id}:"
                say_code = _format_say_code(
                    item.who,
                    item.what,
                    item.attributes,
                    item.temporary_attributes,
                    item.with_,
                )

                lines.append(ref_line)
                lines.append(header)
                lines.append("")
                lines.append(f"    # {say_code}")
                lines.append(f"    {say_code}")
                lines.append("")

            target_file.write_text("\n".join(lines), encoding="utf-8")
            generated_files.append(target_file)
        elif script.strings:
            str_lines = [f"translate {target_lang} strings:", ""]
            for s in script.strings:
                escaped = quote_python_text(s.text)
                str_lines.append(f"    # {s.filename}:{s.linenumber}")
                str_lines.append(f'    old "{escaped}"')
                str_lines.append('    new ""')
                str_lines.append("")
                seen_ui.add(s.text)
            target_file.write_text("\n".join(str_lines), encoding="utf-8")
            generated_files.append(target_file)

        if script.dialogue and script.strings:
            for s in script.strings:
                if s.text not in seen_ui:
                    seen_ui.add(s.text)
                    ui_strings.append((s.filename, s.linenumber, s.text))

    if ui_strings:
        ui_file = tl_root / "ui_strings.rpy"
        existing_text = ui_file.read_text(encoding="utf-8") if ui_file.is_file() else ""
        if existing_text:
            new_strings = [
                (fn, ln, txt)
                for fn, ln, txt in ui_strings
                if f'old "{quote_python_text(txt)}"' not in existing_text
            ]
            if new_strings:
                append_lines: list[str] = [""]
                for fn, ln, txt in new_strings:
                    escaped = quote_python_text(txt)
                    append_lines.append(f"    # {fn}:{ln}")
                    append_lines.append(f'    old "{escaped}"')
                    append_lines.append(f'    new "{escaped}"')
                    append_lines.append("")
                ui_file.write_text(
                    existing_text.rstrip() + "\n" + "\n".join(append_lines),
                    encoding="utf-8",
                )
            generated_files.append(ui_file)
        else:
            lines = [f"translate {target_lang} strings:", ""]
            for fn, ln, txt in ui_strings:
                escaped = quote_python_text(txt)
                lines.append(f"    # {fn}:{ln}")
                lines.append(f'    old "{escaped}"')
                lines.append(f'    new "{escaped}"')
                lines.append("")
            ui_file.write_text("\n".join(lines), encoding="utf-8")
            generated_files.append(ui_file)

    return generated_files


def synthesize_skeletons(game_dir: Path | str, target_lang: str) -> list[Path]:
    return synthesize_translation_skeletons(game_dir, target_lang)
