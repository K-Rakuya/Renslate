from __future__ import annotations

import contextlib
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from renslate.renpy.literals import parse_python_string
from renslate.renpy.rpyc.archive import (
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    RpycFile,
    _unpack_stmts,
)
from renslate.renpy.rpyc.ast_types import (
    Default,
    Define,
    EarlyPython,
    Hide,
    Label,
    Menu,
    Python,
    Say,
    Scene,
    Show,
    Translate,
    TranslateSay,
    UserStatement,
)
from renslate.renpy.rpyc.decompiler import _format_imspec

_GETTEXT_REGEX = re.compile(r"""(?:_\(|__\()\s*([rubRUB]*["'](?:\\.|[^\\"'])*["'])\s*\)""")


@dataclass(slots=True)
class RenpyRawItem:
    identifier: str
    who: str | None
    what: str
    source_file: str = ""
    source_line: int = 0
    filename: str = ""
    linenumber: int = 0
    attributes: tuple[str, ...] | None = None
    temporary_attributes: tuple[str, ...] | None = None
    with_: str | None = None
    arguments: Any = None
    interact: bool = True
    explicit_identifier: bool = False
    current_label: str = ""
    scene_id: str = ""
    active_sprites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.filename and self.source_file:
            self.filename = self.source_file
        elif not self.source_file and self.filename:
            self.source_file = self.filename
        if not self.linenumber and self.source_line:
            self.linenumber = self.source_line
        elif not self.source_line and self.linenumber:
            self.source_line = self.linenumber


ExtractedDialogue = RenpyRawItem


@dataclass(slots=True)
class ExtractedUIString:
    text: str
    filename: str
    linenumber: int


@dataclass(slots=True)
class SceneFlowItem:
    kind: str
    name: str
    details: str = ""
    filename: str = ""
    linenumber: int = 0
    current_label: str = ""
    scene_id: str = ""
    active_sprites: tuple[str, ...] = ()


@dataclass(slots=True)
class ExtractedScript:
    dialogue: list[RenpyRawItem] = field(default_factory=list)
    strings: list[ExtractedUIString] = field(default_factory=list)
    scene_flow: list[SceneFlowItem] = field(default_factory=list)


def compute_canonical_say_identifier(
    label: str | None,
    filename: str,
    linenumber: int,
    what: str,
    game_root: str | None = None,
) -> str:
    norm_path = filename.replace("\\", "/")
    parts = norm_path.split("/")
    if "game" in parts:
        game_idx = len(parts) - 1 - parts[::-1].index("game")
        rel_path = "/".join(parts[game_idx:])
    elif game_root:
        with contextlib.suppress(ValueError):
            rel_path = os.path.relpath(filename, game_root).replace("\\", "/")
        if not rel_path or rel_path.startswith(".."):
            rel_path = Path(filename).name
    elif (
        not Path(filename).is_absolute()
        and not (len(norm_path) >= 2 and norm_path[1] == ":")
        and not norm_path.startswith("../")
    ):
        rel_path = norm_path
    else:
        rel_path = Path(filename).name

    hash_input = f"{rel_path}{linenumber}{what}".encode()
    digest = hashlib.md5(hash_input).hexdigest()[:8]
    prefix = label or "global"
    return f"{prefix}_{digest}"


def _extract_strings_from_text(
    text: str, filename: str, linenumber: int, out: list[ExtractedUIString]
) -> None:
    for m in _GETTEXT_REGEX.finditer(text):
        lit = m.group(1)
        parsed = parse_python_string(lit)
        if parsed and parsed.strip():
            out.append(ExtractedUIString(text=parsed, filename=filename, linenumber=linenumber))


def _extract_menu_strings(
    menu: Menu, filename: str, linenumber: int, out: list[ExtractedUIString]
) -> None:
    items = getattr(menu, "items", []) or []
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) >= 1:
            caption = item[0]
            if isinstance(caption, str) and caption.strip():
                out.append(
                    ExtractedUIString(text=caption, filename=filename, linenumber=linenumber)
                )


def _extract_from_sl_node(
    node: Any, filename: str, linenumber: int, out_strings: list[ExtractedUIString]
) -> None:
    if node is None:
        return

    pos_list = getattr(node, "positional", None)
    if isinstance(pos_list, list):
        for pos in pos_list:
            if isinstance(pos, str):
                _extract_strings_from_text(pos, filename, linenumber, out_strings)

    kw_list = getattr(node, "keyword", None)
    if isinstance(kw_list, list):
        for item in kw_list:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                val = item[1]
                if isinstance(val, str):
                    _extract_strings_from_text(val, filename, linenumber, out_strings)

    children = getattr(node, "children", None)
    if isinstance(children, list):
        for child in children:
            _extract_from_sl_node(
                child, filename, getattr(child, "linenumber", linenumber), out_strings
            )

    entries = getattr(node, "entries", None)
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                _extract_from_sl_node(entry[1], filename, linenumber, out_strings)
            else:
                _extract_from_sl_node(entry, filename, linenumber, out_strings)

    block = getattr(node, "block", None)
    if isinstance(block, list):
        for b in block:
            if hasattr(b, "screen"):
                _extract_from_sl_node(
                    b.screen, filename, getattr(b, "linenumber", linenumber), out_strings
                )
            else:
                _extract_from_sl_node(
                    b, filename, getattr(b, "linenumber", linenumber), out_strings
                )


class _ExtractorContext:
    def __init__(self, default_file: str, out: ExtractedScript) -> None:
        self.default_file = default_file
        self.out = out
        self.current_label: str = ""
        self.current_scene: str = ""
        self.active_sprites: list[str] = []

    def get_node_loc(self, node: Any, parent_file: str, parent_line: int) -> tuple[str, int]:
        filename = getattr(node, "filename", parent_file) or parent_file
        linenumber = getattr(node, "linenumber", 0) or parent_line
        if not filename or filename == self.default_file:
            loc = getattr(node, "loc", None)
            if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                filename = str(loc[0]) or self.default_file
                with contextlib.suppress(ValueError, TypeError):
                    linenumber = int(loc[1]) or linenumber
        return filename, linenumber

    def walk(self, node: Any, parent_file: str = "", parent_line: int = 0) -> None:
        if node is None:
            return

        file, line = self.get_node_loc(node, parent_file or self.default_file, parent_line)

        if isinstance(node, Label):
            self.current_label = getattr(node, "name", "") or ""
            self.out.scene_flow.append(
                SceneFlowItem(
                    kind="label",
                    name=self.current_label,
                    filename=file,
                    linenumber=line,
                    current_label=self.current_label,
                    scene_id=self.current_scene,
                    active_sprites=tuple(self.active_sprites),
                )
            )
            block = getattr(node, "block", None)
            if isinstance(block, list):
                for child in block:
                    self.walk(child, file, line)
            return

        if isinstance(node, Scene):
            imspec = getattr(node, "imspec", ()) or ()
            self.current_scene = _format_imspec(imspec)
            self.active_sprites.clear()
            self.out.scene_flow.append(
                SceneFlowItem(
                    kind="scene",
                    name=self.current_scene,
                    filename=file,
                    linenumber=line,
                    current_label=self.current_label,
                    scene_id=self.current_scene,
                    active_sprites=(),
                )
            )
            return

        if isinstance(node, Show):
            imspec = getattr(node, "imspec", ()) or ()
            tag_name = _format_imspec(imspec)
            base_tag = tag_name.split()[0] if tag_name else ""
            if base_tag:
                self.active_sprites = [s for s in self.active_sprites if s.split()[0] != base_tag]
                self.active_sprites.append(tag_name)
            self.out.scene_flow.append(
                SceneFlowItem(
                    kind="show",
                    name=tag_name,
                    filename=file,
                    linenumber=line,
                    current_label=self.current_label,
                    scene_id=self.current_scene,
                    active_sprites=tuple(self.active_sprites),
                )
            )
            return

        if isinstance(node, Hide):
            imspec = getattr(node, "imspec", ()) or ()
            tag_name = _format_imspec(imspec)
            base_tag = tag_name.split()[0] if tag_name else ""
            if base_tag:
                self.active_sprites = [s for s in self.active_sprites if s.split()[0] != base_tag]
            self.out.scene_flow.append(
                SceneFlowItem(
                    kind="hide",
                    name=tag_name,
                    filename=file,
                    linenumber=line,
                    current_label=self.current_label,
                    scene_id=self.current_scene,
                    active_sprites=tuple(self.active_sprites),
                )
            )
            return

        if isinstance(node, TranslateSay):
            ident = getattr(node, "identifier", "") or ""
            what = getattr(node, "what", "") or ""
            if not ident:
                ident = compute_canonical_say_identifier(
                    self.current_label, file, line, what
                )
            who = getattr(node, "who", None)
            item = RenpyRawItem(
                identifier=ident,
                who=who,
                what=what,
                source_file=file,
                source_line=line,
                attributes=getattr(node, "attributes", None),
                temporary_attributes=getattr(node, "temporary_attributes", None),
                with_=getattr(node, "with_", None),
                arguments=getattr(node, "arguments", None),
                interact=getattr(node, "interact", True),
                explicit_identifier=getattr(node, "explicit_identifier", False),
                current_label=self.current_label,
                scene_id=self.current_scene,
                active_sprites=tuple(self.active_sprites),
            )
            self.out.dialogue.append(item)
            self.out.scene_flow.append(
                SceneFlowItem(
                    kind="say",
                    name=who or "",
                    details=what,
                    filename=file,
                    linenumber=line,
                    current_label=self.current_label,
                    scene_id=self.current_scene,
                    active_sprites=tuple(self.active_sprites),
                )
            )
            return

        if isinstance(node, Translate):
            ident = getattr(node, "identifier", "") or ""
            block = getattr(node, "block", []) or []
            for child in block:
                child_file, child_line = self.get_node_loc(child, file, line)
                if isinstance(child, (TranslateSay, Say)):
                    what = getattr(child, "what", "") or ""
                    child_ident = (
                        ident
                        or getattr(child, "identifier", "")
                        or compute_canonical_say_identifier(
                            self.current_label, child_file, child_line, what
                        )
                    )
                    who = getattr(child, "who", None)
                    item = RenpyRawItem(
                        identifier=child_ident,
                        who=who,
                        what=what,
                        source_file=child_file,
                        source_line=child_line,
                        attributes=getattr(child, "attributes", None),
                        temporary_attributes=getattr(child, "temporary_attributes", None),
                        with_=getattr(child, "with_", None),
                        arguments=getattr(child, "arguments", None),
                        interact=getattr(child, "interact", True),
                        explicit_identifier=getattr(child, "explicit_identifier", False),
                        current_label=self.current_label,
                        scene_id=self.current_scene,
                        active_sprites=tuple(self.active_sprites),
                    )
                    self.out.dialogue.append(item)
                    self.out.scene_flow.append(
                        SceneFlowItem(
                            kind="say",
                            name=who or "",
                            details=what,
                            filename=child_file,
                            linenumber=child_line,
                            current_label=self.current_label,
                            scene_id=self.current_scene,
                            active_sprites=tuple(self.active_sprites),
                        )
                    )
                elif isinstance(child, Menu):
                    _extract_menu_strings(child, child_file, child_line, self.out.strings)
                    items = getattr(child, "items", []) or []
                    for choice_item in items:
                        if isinstance(choice_item, (tuple, list)) and len(choice_item) >= 3:
                            c_block = choice_item[2]
                            if isinstance(c_block, list):
                                for sub in c_block:
                                    self.walk(sub, child_file, child_line)
                else:
                    self.walk(child, child_file, child_line)
            return

        if isinstance(node, Say):
            what = getattr(node, "what", "") or ""
            ident = getattr(node, "identifier", "") or ""
            if not ident:
                ident = compute_canonical_say_identifier(
                    self.current_label, file, line, what
                )
            who = getattr(node, "who", None)
            item = RenpyRawItem(
                identifier=ident,
                who=who,
                what=what,
                source_file=file,
                source_line=line,
                attributes=getattr(node, "attributes", None),
                temporary_attributes=getattr(node, "temporary_attributes", None),
                with_=getattr(node, "with_", None),
                arguments=getattr(node, "arguments", None),
                interact=getattr(node, "interact", True),
                explicit_identifier=getattr(node, "explicit_identifier", False),
                current_label=self.current_label,
                scene_id=self.current_scene,
                active_sprites=tuple(self.active_sprites),
            )
            self.out.dialogue.append(item)
            self.out.scene_flow.append(
                SceneFlowItem(
                    kind="say",
                    name=who or "",
                    details=what,
                    filename=file,
                    linenumber=line,
                    current_label=self.current_label,
                    scene_id=self.current_scene,
                    active_sprites=tuple(self.active_sprites),
                )
            )
            return

        if isinstance(node, Menu):
            _extract_menu_strings(node, file, line, self.out.strings)
            items = getattr(node, "items", []) or []
            for choice_item in items:
                if isinstance(choice_item, (tuple, list)) and len(choice_item) >= 3:
                    c_block = choice_item[2]
                    if isinstance(c_block, list):
                        for sub in c_block:
                            self.walk(sub, file, line)
            return

        if isinstance(node, (Python, EarlyPython, Define, Default)):
            code_obj = getattr(node, "code", None)
            src = getattr(code_obj, "source", "") or (str(code_obj) if code_obj else "")
            if src:
                _extract_strings_from_text(src, file, line, self.out.strings)
            return

        if isinstance(node, UserStatement):
            line_str = getattr(node, "line", "")
            if line_str:
                _extract_strings_from_text(line_str, file, line, self.out.strings)
            return

        for attr in ("block", "entries"):
            val = getattr(node, attr, None)
            if isinstance(val, list):
                for child in val:
                    if isinstance(child, tuple) and len(child) == 2 and isinstance(child[1], list):
                        for sub in child[1]:
                            self.walk(sub, file, line)
                    else:
                        self.walk(child, file, line)


def _extract_from_node(node: Any, default_file: str, out: ExtractedScript) -> None:
    ctx = _ExtractorContext(default_file, out)
    ctx.walk(node)


def extract_rpyc_file(rpyc_path: Path | str) -> ExtractedScript:
    path = Path(rpyc_path)
    with RpycFile(path) as rpyc:
        loaded_post = rpyc.load_slot(SLOT_STATEMENTS_POST)
        loaded_pre = rpyc.load_slot(SLOT_STATEMENTS_PRE)

    out = ExtractedScript()
    default_name = path.with_suffix(".rpy").name

    stmts_post = _unpack_stmts(loaded_post) if loaded_post is not None else []
    stmts_pre = _unpack_stmts(loaded_pre) if loaded_pre is not None else []

    target_stmts = stmts_post if stmts_post else stmts_pre
    ctx = _ExtractorContext(default_name, out)
    for stmt in target_stmts:
        ctx.walk(stmt)

    for stmt in stmts_pre:
        if type(stmt).__name__ == "Screen":
            _extract_from_sl_node(getattr(stmt, "screen", None), default_name, 0, out.strings)
        elif (
            type(stmt).__name__ == "Init"
            and hasattr(stmt, "block")
            and isinstance(stmt.block, list)
        ):
            for child in stmt.block:
                if type(child).__name__ == "Screen":
                    _extract_from_sl_node(
                        getattr(child, "screen", None), default_name, 0, out.strings
                    )

    unique_strings: list[ExtractedUIString] = []
    seen: set[str] = set()
    for s in out.strings:
        if s.text not in seen:
            seen.add(s.text)
            unique_strings.append(s)
    out.strings = unique_strings

    return out


def extract_dialogue_from_rpyc(file_path: Path | str) -> list[RenpyRawItem]:
    script = extract_rpyc_file(file_path)
    return script.dialogue


def extract_scene_flow_from_rpyc(file_path: Path | str) -> list[SceneFlowItem]:
    script = extract_rpyc_file(file_path)
    return script.scene_flow


extract_scene_flow = extract_scene_flow_from_rpyc


def extract_rpyc_project(game_dir: Path | str) -> dict[str, ExtractedScript]:
    root = Path(game_dir)
    rpyc_files = sorted(
        [p for p in root.rglob("*.rpyc") if "tl" not in p.parts],
        key=lambda p: str(p.relative_to(root)),
    )
    result: dict[str, ExtractedScript] = {}
    for rpyc_path in rpyc_files:
        rel = str(rpyc_path.relative_to(root))
        result[rel] = extract_rpyc_file(rpyc_path)
    return result


_RE_RPYC_CHAR_DEF = re.compile(
    r'(?:define|default\s+)?(?P<tag>\w+)\s*=\s*(?:Character|DynamicCharacter)\s*\(\s*(?P<name>_[^(]*\([^)]*\)|"(?:[^"\\]|\\.)*"|\'[^\']*\'|[A-Za-z_]\w*)'
)
_RE_CHAR_CALL = re.compile(
    r'(?:Character|DynamicCharacter)\s*\(\s*(?P<name>_[^(]*\([^)]*\)|"(?:[^"\\]|\\.)*"|\'[^\']*\'|[A-Za-z_]\w*)'
)


def _clean_rpyc_char_name(raw_name: str, tag: str = "") -> str:
    raw_name = raw_name.strip()
    if (raw_name.startswith('_("') and raw_name.endswith('")')) or (
        raw_name.startswith("_('") and raw_name.endswith("')")
    ):
        raw_name = raw_name[3:-2]
    elif (raw_name.startswith('"') and raw_name.endswith('"')) or (
        raw_name.startswith("'") and raw_name.endswith("'")
    ):
        raw_name = raw_name[1:-1]

    if raw_name in ("None", ""):
        return tag.capitalize() if tag else ""
    if raw_name.endswith("_name"):
        base = raw_name[:-5].strip()
        if base.lower() in ("mc", "player", "hero", "protag", "protagonist"):
            return "Protagonist"
        return base.capitalize()
    if tag.lower() in ("mc", "player", "hero", "protag", "protagonist"):
        return "Protagonist"
    return re.sub(r"\b([A-Za-z])\s+([a-z]{2,})\b", r"\1\2", raw_name).strip()


def _extract_char_definitions_from_stmt(stmt: Any, characters: dict[str, str]) -> None:
    if stmt is None:
        return

    varname = getattr(stmt, "varname", None)
    if varname and isinstance(varname, str):
        code_obj = getattr(stmt, "code", None)
        code_str = str(code_obj) if code_obj is not None else ""
        m = _RE_CHAR_CALL.search(code_str)
        if m:
            characters[varname] = _clean_rpyc_char_name(m.group("name"), varname)
            return

    block = getattr(stmt, "block", None)
    if block and isinstance(block, (list, tuple)):
        for inner in block:
            _extract_char_definitions_from_stmt(inner, characters)
        return

    py_code = getattr(stmt, "code", None)
    if py_code is not None:
        source_str = str(py_code)
        for m in _RE_RPYC_CHAR_DEF.finditer(source_str):
            tag = m.group("tag")
            characters[tag] = _clean_rpyc_char_name(m.group("name"), tag)


def extract_character_definitions_from_rpyc(game_dir: Path | str) -> dict[str, str]:
    characters: dict[str, str] = {}
    root = Path(game_dir)
    search_dir = root / "game" if (root / "game").is_dir() else root
    rpyc_files = sorted(
        [p for p in search_dir.rglob("*.rpyc") if "tl" not in p.parts],
        key=lambda p: str(p.relative_to(root)),
    )

    for rpyc_path in rpyc_files:
        try:
            with RpycFile(rpyc_path) as rpyc:
                stmts = rpyc.get_slot_1_statements()
            for stmt in _unpack_stmts(stmts):
                _extract_char_definitions_from_stmt(stmt, characters)
        except Exception:
            continue
    return characters
