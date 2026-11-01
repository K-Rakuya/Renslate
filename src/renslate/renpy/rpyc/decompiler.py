from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from renslate.renpy.literals import encode_say_text
from renslate.renpy.rpyc.archive import (
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    RpycFile,
    _unpack_stmts,
)
from renslate.renpy.rpyc.ast_types import (
    Call,
    Default,
    Define,
    EarlyPython,
    EndTranslate,
    Hide,
    If,
    Init,
    Jump,
    Label,
    Menu,
    Pass,
    Python,
    Return,
    Say,
    Scene,
    Show,
    Translate,
    TranslateSay,
    UserStatement,
    While,
    With,
)


def _format_imspec(imspec: Any) -> str:
    if not imspec:
        return ""
    if (
        isinstance(imspec, (tuple, list))
        and len(imspec) > 0
        and isinstance(imspec[0], (tuple, list))
    ):
        parts = list(imspec[0])
        at_list = imspec[3] if len(imspec) > 3 and isinstance(imspec[3], (list, tuple)) else []
        name = " ".join(str(x) for x in parts if x is not None)
        if at_list:
            name += f" at {', '.join(str(x) for x in at_list)}"
        return name
    if isinstance(imspec, (tuple, list)):
        return " ".join(str(x) for x in imspec if x is not None)
    return str(imspec)


def format_say_statement(node: Any, translated_text: str | None = None) -> str:
    parts: list[str] = []
    who = getattr(node, "who", None)
    if who is not None and str(who).strip():
        parts.append(str(who).strip())
    attrs = getattr(node, "attributes", None)
    if attrs:
        parts.extend(str(a) for a in attrs)
    temp_attrs = getattr(node, "temporary_attributes", None)
    if temp_attrs:
        parts.append("@")
        parts.extend(str(a) for a in temp_attrs)

    text = translated_text if translated_text is not None else (getattr(node, "what", "") or "")
    escaped = encode_say_text(text)
    parts.append(f'"{escaped}"')

    if not getattr(node, "interact", True):
        parts.append("nointeract")

    args = getattr(node, "arguments", None)
    if args is not None:
        args_code = getattr(args, "get_code", lambda: str(args))()
        if args_code:
            parts.append(args_code)

    with_ = getattr(node, "with_", None)
    if with_:
        parts.append(f"with {with_}")

    if getattr(node, "explicit_identifier", False) and getattr(node, "identifier", None):
        parts.append(f"id {node.identifier}")

    return " ".join(parts)


def _format_say_code(
    who: str | None,
    what: str,
    attributes: Sequence[str] | None,
    temporary_attributes: Sequence[str] | None,
    with_: str | None,
) -> str:
    parts: list[str] = []
    if who is not None and str(who).strip():
        parts.append(str(who).strip())
    if attributes:
        parts.extend(attributes)
    if temporary_attributes:
        parts.append("@")
        parts.extend(temporary_attributes)
    escaped = encode_say_text(what)
    parts.append(f'"{escaped}"')
    if with_:
        parts.append(f"with {with_}")
    return " ".join(parts)


class _Decompiler:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._indent: int = 0

    def indent(self) -> None:
        self._indent += 4

    def dedent(self) -> None:
        self._indent = max(0, self._indent - 4)

    def write_line(self, text: str = "") -> None:
        if not text:
            self.lines.append("")
        else:
            self.lines.append(" " * self._indent + text)

    def decompile_stmts(self, stmts: Sequence[Any]) -> None:
        for stmt in stmts:
            self.decompile_node(stmt)

    def decompile_node(self, node: Any) -> None:
        if node is None:
            return

        if isinstance(node, Label):
            self.write_line()
            name = getattr(node, "name", "") or ""
            params = getattr(node, "parameters", None)
            if params is not None:
                self.write_line(f"label {name}({params}):")
            else:
                self.write_line(f"label {name}:")
            block = getattr(node, "block", None)
            if block:
                self.indent()
                self.decompile_stmts(block)
                self.dedent()
            return

        if isinstance(node, (Say, TranslateSay)):
            code = format_say_statement(node)
            self.write_line(code)
            return

        if isinstance(node, Menu):
            self.write_line("menu:")
            self.indent()
            with_ = getattr(node, "with_", None)
            if with_:
                self.write_line(f"with {with_}")
            items = getattr(node, "items", []) or []
            for item in items:
                if isinstance(item, (tuple, list)) and len(item) >= 3:
                    label, cond, block = item[0], item[1], item[2]
                    if label:
                        escaped = encode_say_text(str(label))
                        if cond and cond != "True":
                            self.write_line(f'"{escaped}" if {cond}:')
                        else:
                            self.write_line(f'"{escaped}":')
                        self.indent()
                        if block:
                            self.decompile_stmts(block)
                        else:
                            self.write_line("pass")
                        self.dedent()
                    elif block:
                        self.decompile_stmts(block)
            self.dedent()
            return

        if isinstance(node, If):
            entries = getattr(node, "entries", []) or []
            for i, entry in enumerate(entries):
                if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                    cond, block = entry[0], entry[1]
                    if i == 0:
                        self.write_line(f"if {cond}:")
                    elif cond == "True" and i == len(entries) - 1:
                        self.write_line("else:")
                    else:
                        self.write_line(f"elif {cond}:")
                    self.indent()
                    if block:
                        self.decompile_stmts(block)
                    else:
                        self.write_line("pass")
                    self.dedent()
            return

        if isinstance(node, While):
            cond = getattr(node, "condition", "True")
            self.write_line(f"while {cond}:")
            block = getattr(node, "block", []) or []
            self.indent()
            if block:
                self.decompile_stmts(block)
            else:
                self.write_line("pass")
            self.dedent()
            return

        if isinstance(node, Pass):
            self.write_line("pass")
            return

        if isinstance(node, Return):
            expr = getattr(node, "expression", None)
            if expr:
                self.write_line(f"return {expr}")
            else:
                self.write_line("return")
            return

        if isinstance(node, Jump):
            target = getattr(node, "target", "")
            if getattr(node, "expression", False):
                self.write_line(f"jump expression {target}")
            else:
                self.write_line(f"jump {target}")
            return

        if isinstance(node, Call):
            label = getattr(node, "label", "")
            expr = getattr(node, "expression", False)
            target = f"expression {label}" if expr else label
            args = getattr(node, "arguments", None)
            if args:
                self.write_line(f"call {target}({args})")
            else:
                self.write_line(f"call {target}")
            return

        if isinstance(node, Scene):
            imspec = getattr(node, "imspec", ()) or ()
            name = _format_imspec(imspec)
            self.write_line(f"scene {name}".strip())
            return

        if isinstance(node, Show):
            imspec = getattr(node, "imspec", ()) or ()
            name = _format_imspec(imspec)
            self.write_line(f"show {name}".strip())
            return

        if isinstance(node, Hide):
            imspec = getattr(node, "imspec", ()) or ()
            name = _format_imspec(imspec)
            self.write_line(f"hide {name}".strip())
            return

        if isinstance(node, With):
            expr = getattr(node, "expr", "")
            self.write_line(f"with {expr}")
            return

        if isinstance(node, Python):
            code_obj = getattr(node, "code", None)
            code_str = getattr(code_obj, "source", "") or (str(code_obj) if code_obj else "")
            lines = code_str.strip().splitlines()
            if len(lines) == 1 and not lines[0].startswith(("import ", "from ", "def ", "class ")):
                self.write_line(f"$ {lines[0]}")
            else:
                self.write_line("python:")
                self.indent()
                for line in lines:
                    self.write_line(line)
                self.dedent()
            return

        if isinstance(node, EarlyPython):
            code_obj = getattr(node, "code", None)
            code_str = getattr(code_obj, "source", "") or (str(code_obj) if code_obj else "")
            self.write_line("init python:")
            self.indent()
            for line in code_str.strip().splitlines():
                self.write_line(line)
            self.dedent()
            return

        if isinstance(node, Init):
            priority = getattr(node, "priority", 0)
            self.write_line(f"init {priority}:" if priority else "init:")
            block = getattr(node, "block", []) or []
            self.indent()
            if block:
                self.decompile_stmts(block)
            else:
                self.write_line("pass")
            self.dedent()
            return

        if isinstance(node, Define):
            varname = getattr(node, "varname", "")
            code_obj = getattr(node, "code", "")
            src = getattr(code_obj, "source", "") or str(code_obj)
            self.write_line(f"define {varname} = {src}")
            return

        if isinstance(node, Default):
            varname = getattr(node, "varname", "")
            code_obj = getattr(node, "code", "")
            src = getattr(code_obj, "source", "") or str(code_obj)
            self.write_line(f"default {varname} = {src}")
            return

        if isinstance(node, UserStatement):
            line = getattr(node, "line", "")
            if line:
                self.write_line(line)
            return

        if isinstance(node, Translate):
            lang = getattr(node, "language", None) or "None"
            ident = getattr(node, "identifier", "")
            self.write_line(f"translate {lang} {ident}:")
            block = getattr(node, "block", []) or []
            self.indent()
            if block:
                self.decompile_stmts(block)
            else:
                self.write_line("pass")
            self.dedent()
            return

        if isinstance(node, EndTranslate):
            return

        if hasattr(node, "get_code") and callable(node.get_code):
            try:
                self.write_line(str(node.get_code()))
                return
            except Exception:
                pass

        name = type(node).__name__
        self.write_line(f"# [{name}]")


def decompile_rpyc(rpyc_path: Path | str, output_rpy_path: Path | str | None = None) -> Path:
    r_path = Path(rpyc_path)
    out_path = r_path.with_suffix(".rpy") if output_rpy_path is None else Path(output_rpy_path)

    with RpycFile(r_path) as rpyc:
        loaded = rpyc.load_slot(SLOT_STATEMENTS_PRE)
        if loaded is None:
            loaded = rpyc.load_slot(SLOT_STATEMENTS_POST)

    if loaded is None:
        raise ValueError(f"Tệp .rpyc bị hỏng hoặc không có dữ liệu AST hợp lệ: {r_path}")

    stmts = _unpack_stmts(loaded)
    dec = _Decompiler()
    dec.decompile_stmts(stmts)

    content = "\n".join(dec.lines).strip() + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path
