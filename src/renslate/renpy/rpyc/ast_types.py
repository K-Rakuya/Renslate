from __future__ import annotations

import contextlib
from typing import Any


class VirtualASTNode:
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


class Say(VirtualASTNode):
    who: str | None = None
    what: str = ""
    with_: str | None = None
    interact: bool = True
    attributes: tuple[str, ...] | None = None
    temporary_attributes: tuple[str, ...] | None = None
    arguments: Any = None
    explicit_identifier: bool = False
    identifier: str | None = None


class TranslateSay(VirtualASTNode):
    identifier: str = ""
    who: str | None = None
    what: str = ""
    with_: str | None = None
    interact: bool = True
    attributes: tuple[str, ...] | None = None
    temporary_attributes: tuple[str, ...] | None = None
    arguments: Any = None
    explicit_identifier: bool = False


class Translate(VirtualASTNode):
    identifier: str = ""
    alternate: str | None = None
    language: str | None = None
    block: list[Any] | None = None


class EndTranslate(VirtualASTNode):
    pass


class TranslateString(VirtualASTNode):
    old: str = ""
    new: str = ""
    language: str | None = None
    newloc: tuple[str, int] | None = None


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
    "TranslateString": TranslateString,
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
