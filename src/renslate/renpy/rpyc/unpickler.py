from __future__ import annotations

import collections
import pickle
from typing import Any

from renslate.renpy.rpyc.ast_types import (
    _EXPLICIT_AST_CLASSES,
    VirtualDict,
    VirtualList,
    VirtualPyCode,
    VirtualPyExpr,
    VirtualSet,
    _get_dynamic_ast_class,
)

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
    "xrange": range,
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
    def find_class(self, module: str, name: str) -> Any:
        if module in _BLOCKED_MODULES:
            raise pickle.UnpicklingError(f"Blocked dangerous module: {module}")

        if module in ("__builtin__", "builtins"):
            if name in _SAFE_BUILTINS:
                return _SAFE_BUILTINS[name]
            raise pickle.UnpicklingError(f"Blocked unsafe builtin: {name}")

        if module == "collections":
            if name in _SAFE_COLLECTIONS:
                return _SAFE_COLLECTIONS[name]
            raise pickle.UnpicklingError(f"Blocked unsafe collection: {name}")

        if name in ("RevertableList", "RollbackList", "CompressedList", "VirtualList"):
            return VirtualList
        if name in ("RevertableDict", "RevertableDefaultDict", "VirtualDict"):
            return VirtualDict
        if name in ("RevertableSet", "VirtualSet"):
            return VirtualSet
        if name in ("PyExpr", "VirtualPyExpr"):
            return VirtualPyExpr
        if name in ("PyCode", "VirtualPyCode"):
            return VirtualPyCode

        if name in _EXPLICIT_AST_CLASSES:
            return _EXPLICIT_AST_CLASSES[name]

        return _get_dynamic_ast_class(module, name)
