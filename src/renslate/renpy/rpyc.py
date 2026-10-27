"""Native Ren'Py RPC2 bytecode parser, virtual unpickler, AST extractor, and decompiler.

Supports reading compiled .rpyc files (RPC2 format and legacy v1), virtualizing Ren'Py
data types and AST nodes without external Ren'Py dependencies, extracting canonical
translation identifiers and dialogue strings from Slot 2 AST, synthesizing translation
skeletons, and decompiling bytecode back to readable .rpy scripts.
"""

from __future__ import annotations

import collections
import contextlib
import io
import os
import pickle
import re
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from renslate.renpy.literals import encode_say_text, parse_python_string, quote_python_text

RPYC2_HEADER: bytes = b"RENPY RPC2"
SLOT_STATEMENTS_PRE: int = 1
SLOT_STATEMENTS_POST: int = 2

_GETTEXT_REGEX = re.compile(r"""(?:_\(|__\()\s*([rubRUB]*["'](?:\\.|[^\\"'])*["'])\s*\)""")


# ==============================================================================
# Virtualized AST & Data Types
# ==============================================================================


class VirtualASTNode:
    """Base class for all virtualized Ren'Py AST nodes."""

    filename: str = ""
    linenumber: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple):
            for item in state:
                if isinstance(item, dict):
                    self.__dict__.update(item)
        if not getattr(self, "filename", "") or not getattr(self, "linenumber", 0):
            loc = self.__dict__.get("loc")
            if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                if not getattr(self, "filename", ""):
                    self.filename = str(loc[0])
                if not getattr(self, "linenumber", 0):
                    with contextlib.suppress(ValueError, TypeError):
                        self.linenumber = int(loc[1])

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return None

    def __repr__(self) -> str:
        name = self.__class__.__name__
        items: list[str] = []
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (list, tuple, dict, VirtualASTNode)):
                items.append(f"{k}=<{type(v).__name__}>")
            else:
                items.append(f"{k}={v!r}")
        return f"{name}({', '.join(items)})"


class VirtualList(list[Any], VirtualASTNode):
    """Virtualized Ren'Py RevertableList / RollbackList."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if args and isinstance(args[0], (list, tuple, set)):
            self.extend(args[0])

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple):
            if len(state) >= 1 and isinstance(state[0], list):
                self.clear()
                self.extend(state[0])
            if len(state) >= 2 and isinstance(state[1], dict):
                self.__dict__.update(state[1])
        elif isinstance(state, dict):
            self.__dict__.update(state)


class VirtualDict(dict[Any, Any], VirtualASTNode):
    """Virtualized Ren'Py RevertableDict."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple):
            if len(state) >= 1 and isinstance(state[0], dict):
                self.clear()
                self.update(state[0])
            if len(state) >= 2 and isinstance(state[1], dict):
                self.__dict__.update(state[1])
        elif isinstance(state, dict):
            self.clear()
            self.update(state)


class VirtualSet(set[Any], VirtualASTNode):
    """Virtualized Ren'Py RevertableSet."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if args and isinstance(args[0], (list, tuple, set)):
            self.update(args[0])

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple):
            if len(state) >= 1 and isinstance(state[0], (list, tuple, set)):
                self.clear()
                self.update(state[0])
            if len(state) >= 2 and isinstance(state[1], dict):
                self.__dict__.update(state[1])
        elif isinstance(state, dict):
            self.__dict__.update(state)


class VirtualPyExpr(str):
    """Virtualized Ren'Py PyExpr (subclass of str preserving loc)."""

    filename: str = ""
    linenumber: int = 0

    def __new__(cls, s: str = "", *args: Any, **kwargs: Any) -> VirtualPyExpr:
        obj = super().__new__(cls, s)
        if len(args) >= 1:
            obj.filename = str(args[0])
        elif "filename" in kwargs:
            obj.filename = str(kwargs["filename"])
        if len(args) >= 2:
            try:
                obj.linenumber = int(args[1])
            except (ValueError, TypeError):
                obj.linenumber = 0
        elif "linenumber" in kwargs:
            try:
                obj.linenumber = int(kwargs["linenumber"])
            except (ValueError, TypeError):
                obj.linenumber = 0
        return obj

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) >= 2:
            self.filename = str(state[0])
            try:
                self.linenumber = int(state[1])
            except (ValueError, TypeError):
                self.linenumber = 0


class VirtualPyCode(VirtualASTNode):
    """Virtualized Ren'Py PyCode."""

    source: str = ""
    location: tuple[str, int] = ("", 0)
    mode: str = "eval"

    def __init__(
        self,
        source: str = "",
        location: tuple[str, int] = ("", 0),
        mode: str = "eval",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source = source
        self.location = location
        self.mode = mode

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple):
            if len(state) >= 2 and isinstance(state[1], str):
                self.source = state[1]
            if len(state) >= 3:
                loc = state[2]
                if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                    try:
                        line_num = int(loc[1])
                    except (ValueError, TypeError):
                        line_num = 0
                    self.location = (str(loc[0]), line_num)
                    self.filename = str(loc[0])
                    self.linenumber = line_num
                elif isinstance(loc, str):
                    self.location = (loc, 0)
                    self.filename = loc
            if len(state) >= 4 and isinstance(state[3], str):
                self.mode = state[3]
            for item in state:
                if isinstance(item, dict):
                    self.__dict__.update(item)
        elif isinstance(state, dict):
            self.__dict__.update(state)
            if "source" in state and isinstance(state["source"], str):
                self.source = state["source"]
            if "mode" in state and isinstance(state["mode"], str):
                self.mode = state["mode"]
            if "location" in state and isinstance(state["location"], (tuple, list)):
                loc = state["location"]
                if len(loc) >= 2:
                    try:
                        line_num = int(loc[1])
                    except (ValueError, TypeError):
                        line_num = 0
                    self.location = (str(loc[0]), line_num)
                    self.filename = str(loc[0])
                    self.linenumber = line_num

    def __str__(self) -> str:
        return self.source


# Explicit AST statement classes
class Say(VirtualASTNode):
    who: str | None = None
    what: str = ""
    with_: str | None = None
    interact: bool = True
    attributes: tuple[str, ...] | None = None
    temporary_attributes: tuple[str, ...] | None = None
    arguments: Any = None


class TranslateSay(VirtualASTNode):
    identifier: str = ""
    who: str | None = None
    what: str = ""
    with_: str | None = None
    interact: bool = True
    attributes: tuple[str, ...] | None = None
    temporary_attributes: tuple[str, ...] | None = None
    arguments: Any = None


class Translate(VirtualASTNode):
    identifier: str = ""
    alternate: str | None = None
    language: str | None = None
    block: list[Any] | None = None


class EndTranslate(VirtualASTNode):
    pass


class Menu(VirtualASTNode):
    items: list[tuple[str | None, str, list[Any] | None]] | None = None
    with_: str | None = None
    set: str | None = None
    arguments: Any = None


class Label(VirtualASTNode):
    name: str = ""
    parameters: Any = None
    block: list[Any] | None = None
    hide: bool = False


class Jump(VirtualASTNode):
    target: str = ""
    expression: bool = False


class Call(VirtualASTNode):
    label: str = ""
    arguments: Any = None
    expression: bool = False


class Return(VirtualASTNode):
    expression: str | None = None


class Pass(VirtualASTNode):
    pass


class If(VirtualASTNode):
    entries: list[tuple[str, list[Any]]] | None = None


class While(VirtualASTNode):
    condition: str = ""
    block: list[Any] | None = None


class Init(VirtualASTNode):
    priority: int = 0
    block: list[Any] | None = None


class Python(VirtualASTNode):
    code: Any = None
    hide: bool = False
    store: str = "store"


class EarlyPython(VirtualASTNode):
    code: Any = None
    hide: bool = False
    store: str = "store"


class Image(VirtualASTNode):
    imgname: Any = ()
    code: Any = None
    expression: str | None = None


class Transform(VirtualASTNode):
    varname: str = ""
    atl: Any = None


class Show(VirtualASTNode):
    imspec: Any = ()
    atl: Any = None


class Scene(VirtualASTNode):
    imspec: Any = ()
    atl: Any = None
    layer: str = "master"


class Hide(VirtualASTNode):
    imspec: Any = ()


class With(VirtualASTNode):
    expr: str = ""


class Define(VirtualASTNode):
    varname: str = ""
    code: Any = None
    store: str = "store"


class Default(VirtualASTNode):
    varname: str = ""
    code: Any = None
    store: str = "store"


class Screen(VirtualASTNode):
    screen: Any = None


class Style(VirtualASTNode):
    style_name: str = ""
    parent: str | None = None
    properties: dict[str, Any] | None = None


class UserStatement(VirtualASTNode):
    line: str = ""
    block: list[Any] | None = None


_EXPLICIT_AST_CLASSES: dict[str, type] = {
    "Say": Say,
    "TranslateSay": TranslateSay,
    "Translate": Translate,
    "EndTranslate": EndTranslate,
    "Menu": Menu,
    "Label": Label,
    "Jump": Jump,
    "Call": Call,
    "Return": Return,
    "Pass": Pass,
    "If": If,
    "While": While,
    "Init": Init,
    "Python": Python,
    "EarlyPython": EarlyPython,
    "Image": Image,
    "Transform": Transform,
    "Show": Show,
    "Scene": Scene,
    "Hide": Hide,
    "With": With,
    "Define": Define,
    "Default": Default,
    "Screen": Screen,
    "Style": Style,
    "UserStatement": UserStatement,
}

_DYNAMIC_CLASSES: dict[tuple[str, str], type] = {}


def _get_dynamic_ast_class(module: str, name: str) -> type:
    key = (module, name)
    if key in _DYNAMIC_CLASSES:
        return _DYNAMIC_CLASSES[key]
    cls = type(name, (VirtualASTNode,), {"__module__": module})
    _DYNAMIC_CLASSES[key] = cls
    return cls


# ==============================================================================
# Virtualized Unpickler
# ==============================================================================


_SAFE_BUILTINS: dict[str, type] = {
    "str": str,
    "unicode": str,
    "pystr": str,
    "bytes": bytes,
    "bytearray": bytearray,
    "int": int,
    "long": int,
    "float": float,
    "bool": bool,
    "complex": complex,
    "tuple": tuple,
    "list": list,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "slice": slice,
    "range": range,
    "NoneType": type(None),
    "object": object,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
}

_SAFE_COLLECTIONS: dict[str, type] = {
    "OrderedDict": collections.OrderedDict,
    "defaultdict": collections.defaultdict,
    "deque": collections.deque,
    "Counter": collections.Counter,
}

_BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "posix",
        "nt",
        "subprocess",
        "sys",
        "shutil",
        "socket",
        "http",
        "urllib",
        "importlib",
        "commands",
        "pty",
    }
)


class RenpyUnpickler(pickle.Unpickler):
    """Subclass of pickle.Unpickler resolving Ren'Py and Python 2 types virtualized."""

    def find_class(self, module: str, name: str) -> Any:
        if module in _BLOCKED_MODULES:
            raise pickle.UnpicklingError(f"Blocked dangerous module: {module}")

        # 1. Python 2 / legacy builtins
        if module in ("__builtin__", "builtins"):
            if name in _SAFE_BUILTINS:
                return _SAFE_BUILTINS[name]
            raise pickle.UnpicklingError(f"Blocked unsafe builtin: {name}")

        # 2. Standard collections
        if module == "collections":
            if name in _SAFE_COLLECTIONS:
                return _SAFE_COLLECTIONS[name]
            raise pickle.UnpicklingError(f"Blocked unsafe collection: {name}")

        # 3. Ren'Py revertable types
        if name in ("RevertableList", "RollbackList", "CompressedList"):
            return VirtualList
        if name in ("RevertableDict", "RevertableDefaultDict"):
            return VirtualDict
        if name in ("RevertableSet",):
            return VirtualSet
        if name == "PyExpr":
            return VirtualPyExpr
        if name == "PyCode":
            return VirtualPyCode

        # 4. Explicit AST nodes
        if name in _EXPLICIT_AST_CLASSES:
            return _EXPLICIT_AST_CLASSES[name]

        # 5. Dynamic fallback for arbitrary screen/displayable/store classes
        return _get_dynamic_ast_class(module, name)


# ==============================================================================
# RPYC2 File Parser
# ==============================================================================


class RpycFile:
    """Parser for Ren'Py .rpyc bytecode files (RPC2 format and legacy v1)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.slots: dict[int, tuple[int, int]] = {}
        self.is_legacy: bool = False
        self._parse_header()

    def _parse_header(self) -> None:
        with self.path.open("rb") as f:
            header_bytes = f.read(1024)

        if not header_bytes.startswith(RPYC2_HEADER):
            self.is_legacy = True
            return

        pos = len(RPYC2_HEADER)
        while pos + 12 <= len(header_bytes):
            slot_id, start, length = struct.unpack("<III", header_bytes[pos : pos + 12])
            pos += 12
            if slot_id == 0:
                break
            self.slots[slot_id] = (start, length)

    def read_slot(self, slot_id: int) -> bytes | None:
        """Read and decompress raw bytes for the given slot id."""
        with self.path.open("rb") as f:
            if self.is_legacy:
                if slot_id != SLOT_STATEMENTS_PRE:
                    return None
                f.seek(0)
                data = f.read()
                try:
                    return zlib.decompress(data)
                except Exception:
                    return None

            if slot_id not in self.slots:
                return None

            start, length = self.slots[slot_id]
            f.seek(start)
            compressed = f.read(length)
            try:
                return zlib.decompress(compressed)
            except Exception:
                return None

    def read_slots(self) -> dict[int, bytes]:
        """Read and decompress all slots in the file, returning slot_id -> bytes."""
        result: dict[int, bytes] = {}
        for slot_id in sorted(self.slots.keys()):
            data = self.read_slot(slot_id)
            if data is not None:
                result[slot_id] = data
        if self.is_legacy:
            data = self.read_slot(SLOT_STATEMENTS_PRE)
            if data is not None:
                result[SLOT_STATEMENTS_PRE] = data
        return result

    def load_slot(self, slot_id: int) -> Any:
        """Decompress and unpickle the data in the given slot id."""
        data = self.read_slot(slot_id)
        if data is None:
            return None
        unpickler = RenpyUnpickler(io.BytesIO(data))
        return unpickler.load()

    def get_slot_1_statements(self) -> list[Any]:
        """Load and return AST statements from Slot 1 (pre-transforms)."""
        loaded = self.load_slot(SLOT_STATEMENTS_PRE)
        return _unpack_stmts(loaded)

    def get_slot_2_ast(self) -> list[Any]:
        """Load and return AST statements from Slot 2 (post-transforms)."""
        loaded = self.load_slot(SLOT_STATEMENTS_POST)
        return _unpack_stmts(loaded)

    def read_md5(self) -> bytes | None:
        """Return the 16-byte MD5 digest stored at EOF if present."""
        if self.is_legacy:
            return None
        try:
            with self.path.open("rb") as f:
                f.seek(-16, os.SEEK_END)
                return f.read(16)
        except Exception:
            return None


def create_rpyc(slots: dict[int, bytes]) -> bytes:
    """Create a valid RPYC2 byte buffer from a mapping of slot_id -> uncompressed bytes."""
    header = bytearray(RPYC2_HEADER)
    num_entries = len(slots) + 1
    dir_size = num_entries * 12
    offset = len(header) + dir_size

    dir_buf = bytearray()
    payload_buf = bytearray()

    for slot_id, data in sorted(slots.items()):
        compressed = zlib.compress(data, 3)
        dir_buf.extend(struct.pack("<III", slot_id, offset, len(compressed)))
        payload_buf.extend(compressed)
        offset += len(compressed)

    # Terminator slot = 0
    dir_buf.extend(struct.pack("<III", 0, 0, 0))
    header.extend(dir_buf)
    header.extend(payload_buf)
    return bytes(header)


def write_rpyc(path: Path | str, slots: dict[int, bytes]) -> None:
    """Write an RPYC2 file to disk."""
    data = create_rpyc(slots)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


# ==============================================================================
# Extraction Models & AST Walker
# ==============================================================================


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
class ExtractedScript:
    dialogue: list[RenpyRawItem] = field(default_factory=list)
    strings: list[ExtractedUIString] = field(default_factory=list)


def _unpack_stmts(loaded: Any) -> list[Any]:
    if loaded is None:
        return []
    if isinstance(loaded, tuple) and len(loaded) >= 2 and isinstance(loaded[1], list):
        return loaded[1]
    if isinstance(loaded, list):
        return loaded
    return [loaded]


def _extract_strings_from_text(
    text: str, filename: str, linenumber: int, out: list[ExtractedUIString]
) -> None:
    for m in _GETTEXT_REGEX.finditer(text):
        lit = m.group(1)
        parsed = parse_python_string(lit)
        if parsed and parsed.strip():
            out.append(ExtractedUIString(text=parsed, filename=filename, linenumber=linenumber))


def _extract_from_node(node: Any, default_file: str, out: ExtractedScript) -> None:
    filename = getattr(node, "filename", default_file) or default_file
    linenumber = getattr(node, "linenumber", 0) or 0
    if not filename or filename == default_file:
        loc = getattr(node, "loc", None)
        if isinstance(loc, (tuple, list)) and len(loc) >= 2:
            filename = str(loc[0]) or default_file
            with contextlib.suppress(ValueError, TypeError):
                linenumber = int(loc[1]) or linenumber

    if isinstance(node, TranslateSay):
        ident = getattr(node, "identifier", "") or ""
        what = getattr(node, "what", "") or ""
        out.dialogue.append(
            RenpyRawItem(
                identifier=ident,
                who=getattr(node, "who", None),
                what=what,
                source_file=filename,
                source_line=linenumber,
                attributes=getattr(node, "attributes", None),
                temporary_attributes=getattr(node, "temporary_attributes", None),
                with_=getattr(node, "with_", None),
                arguments=getattr(node, "arguments", None),
            )
        )
        return

    if isinstance(node, Translate):
        ident = getattr(node, "identifier", "") or ""
        block = getattr(node, "block", []) or []
        for child in block:
            child_file = getattr(child, "filename", filename) or filename
            child_line = getattr(child, "linenumber", linenumber) or linenumber
            if not child_file or child_file == filename:
                loc = getattr(child, "loc", None)
                if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                    child_file = str(loc[0]) or child_file
                    with contextlib.suppress(ValueError, TypeError):
                        child_line = int(loc[1]) or child_line

            if isinstance(child, (TranslateSay, Say)):
                what = getattr(child, "what", "") or ""
                out.dialogue.append(
                    RenpyRawItem(
                        identifier=ident,
                        who=getattr(child, "who", None),
                        what=what,
                        source_file=child_file,
                        source_line=child_line,
                        attributes=getattr(child, "attributes", None),
                        temporary_attributes=getattr(child, "temporary_attributes", None),
                        with_=getattr(child, "with_", None),
                        arguments=getattr(child, "arguments", None),
                    )
                )
            elif isinstance(child, Menu):
                _extract_menu_strings(child, child_file, child_line, out.strings)
                items = getattr(child, "items", []) or []
                for choice_item in items:
                    if isinstance(choice_item, (tuple, list)) and len(choice_item) >= 3:
                        block = choice_item[2]
                        if isinstance(block, list):
                            for sub in block:
                                _extract_from_node(sub, child_file, out)
            else:
                _extract_from_node(child, child_file, out)
        return

    if isinstance(node, Say):
        ident = getattr(node, "identifier", "") or f"say_{linenumber:04d}"
        what = getattr(node, "what", "") or ""
        out.dialogue.append(
            RenpyRawItem(
                identifier=ident,
                who=getattr(node, "who", None),
                what=what,
                source_file=filename,
                source_line=linenumber,
                attributes=getattr(node, "attributes", None),
                temporary_attributes=getattr(node, "temporary_attributes", None),
                with_=getattr(node, "with_", None),
                arguments=getattr(node, "arguments", None),
            )
        )
        return

    if isinstance(node, Menu):
        _extract_menu_strings(node, filename, linenumber, out.strings)
        items = getattr(node, "items", []) or []
        for choice_item in items:
            if isinstance(choice_item, (tuple, list)) and len(choice_item) >= 3:
                block = choice_item[2]
                if isinstance(block, list):
                    for sub in block:
                        _extract_from_node(sub, filename, out)
        return

    # Scan python code / expressions for _("...")
    if isinstance(node, (Python, EarlyPython, Define, Default)):
        code_obj = getattr(node, "code", None)
        src = getattr(code_obj, "source", "") or (str(code_obj) if code_obj else "")
        if src:
            _extract_strings_from_text(src, filename, linenumber, out.strings)
        return

    if isinstance(node, UserStatement):
        line = getattr(node, "line", "")
        if line:
            _extract_strings_from_text(line, filename, linenumber, out.strings)
        return

    # Check child blocks in If, While, Init, Label
    for attr in ("block", "entries"):
        val = getattr(node, attr, None)
        if isinstance(val, list):
            for child in val:
                if isinstance(child, tuple) and len(child) == 2 and isinstance(child[1], list):
                    for sub in child[1]:
                        _extract_from_node(sub, filename, out)
                else:
                    _extract_from_node(child, filename, out)


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


def extract_rpyc_file(rpyc_path: Path | str) -> ExtractedScript:
    """Extract translatable dialogue and UI strings directly from a .rpyc file."""
    path = Path(rpyc_path)
    rpyc = RpycFile(path)

    # Prefer Slot 2 (post static transforms with TranslateSay & canonical IDs)
    loaded = rpyc.load_slot(SLOT_STATEMENTS_POST)
    if loaded is None:
        # Fall back to Slot 1 (pre-transforms)
        loaded = rpyc.load_slot(SLOT_STATEMENTS_PRE)

    stmts = _unpack_stmts(loaded)
    out = ExtractedScript()
    default_name = path.with_suffix(".rpy").name

    for stmt in stmts:
        _extract_from_node(stmt, default_name, out)

    return out


def extract_dialogue_from_rpyc(file_path: Path | str) -> list[RenpyRawItem]:
    """Extract translatable dialogue items with canonical identifiers from a .rpyc file."""
    script = extract_rpyc_file(file_path)
    return script.dialogue


def extract_rpyc_project(game_dir: Path | str) -> dict[str, ExtractedScript]:
    """Scan all .rpyc files in a game directory and extract dialogue and strings."""
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
    """Extract character definitions from Ren'Py RPC2 bytecode Slot 1 statements."""
    characters: dict[str, str] = {}
    root = Path(game_dir)
    search_dir = root / "game" if (root / "game").is_dir() else root
    rpyc_files = sorted(
        [p for p in search_dir.rglob("*.rpyc") if "tl" not in p.parts],
        key=lambda p: str(p.relative_to(root)),
    )

    for rpyc_path in rpyc_files:
        try:
            rpyc = RpycFile(rpyc_path)
            stmts = rpyc.get_slot_1_statements()
            for stmt in _unpack_stmts(stmts):
                _extract_char_definitions_from_stmt(stmt, characters)
        except Exception:
            continue
    return characters


# ==============================================================================
# Translation Skeleton Synthesis
# ==============================================================================


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


def synthesize_translation_skeletons(game_dir: Path | str, target_lang: str) -> list[Path]:
    """Synthesize canonical game/tl/<target_lang>/*.rpy skeletons from .rpyc bytecode."""
    root = Path(game_dir)
    extracted = extract_rpyc_project(root)
    if (root / "game").is_dir():
        tl_root = root / "game" / "tl" / target_lang
    else:
        tl_root = root / "tl" / target_lang
    tl_root.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []
    ui_strings: list[tuple[str, int, str]] = []  # (filename, linenumber, text)
    seen_ui: set[str] = set()

    for rel_path, script in extracted.items():
        # Determine target .rpy file name
        source_rpy_name = Path(rel_path).with_suffix(".rpy")
        # Strip game/ prefix if present
        parts = source_rpy_name.parts
        if parts and parts[0] == "game":
            source_rpy_name = Path(*parts[1:])

        target_file = tl_root / source_rpy_name
        target_file.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        if script.dialogue:
            for item in script.dialogue:
                clean_id = item.identifier.replace(".", "_")
                # Normalize reference location
                ref_file = item.filename
                if not ref_file:
                    ref_file = str(source_rpy_name)
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

        for s in script.strings:
            if s.text not in seen_ui:
                seen_ui.add(s.text)
                ui_strings.append((s.filename, s.linenumber, s.text))

    if ui_strings:
        ui_file = tl_root / "ui_strings.rpy"
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
    """Alias for synthesize_translation_skeletons."""
    return synthesize_translation_skeletons(game_dir, target_lang)


# ==============================================================================
# Decompiler (Slot 1 -> .rpy source)
# ==============================================================================


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
            code = _format_say_code(
                getattr(node, "who", None),
                getattr(node, "what", "") or "",
                getattr(node, "attributes", None),
                getattr(node, "temporary_attributes", None),
                getattr(node, "with_", None),
            )
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

        # Generic fallback for unknown nodes
        if hasattr(node, "get_code") and callable(node.get_code):
            try:
                self.write_line(str(node.get_code()))
                return
            except Exception:
                pass

        name = type(node).__name__
        self.write_line(f"# [{name}]")


def decompile_rpyc(rpyc_path: Path | str, output_rpy_path: Path | str | None = None) -> Path:
    """Decompile a .rpyc file back into readable .rpy source script."""
    r_path = Path(rpyc_path)
    out_path = r_path.with_suffix(".rpy") if output_rpy_path is None else Path(output_rpy_path)

    rpyc = RpycFile(r_path)
    # Prefer Slot 1 (pure original AST)
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
