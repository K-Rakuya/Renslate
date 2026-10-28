"""Project configuration: a strictly validated JSON document with typed sections."""

from __future__ import annotations

import dataclasses
import json
import re
import typing
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from renslate.core.errors import ConfigError

_LANGUAGE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _bounded(default: Any, minimum: float | None = None, maximum: float | None = None) -> Any:
    return field(default=default, metadata={"min": minimum, "max": maximum})


@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
    n_ctx: int = _bounded(6144, 1024)
    temperature: float = _bounded(0.25, 0.0, 2.0)
    timeout_s: float = _bounded(600.0, 1.0)
    max_retries: int = _bounded(3, 0, 20)
    grammar: str | None = None
    json_schema: dict[str, Any] | None = None



@dataclass(frozen=True, slots=True)
class BatchConfig:
    max_lines: int = _bounded(25, 1)
    max_source_tokens: int = _bounded(700, 1)
    lookback: int = _bounded(4, 0)
    workers: int = _bounded(1, 1, 64)


@dataclass(frozen=True, slots=True)
class TMConfig:
    fuzzy_threshold: float = _bounded(0.88, 0.0, 1.0)


@dataclass(frozen=True, slots=True)
class QAConfig:
    max_length_ratio: float = _bounded(1.8, 1.0)


@dataclass(frozen=True, slots=True)
class StyleConfig:
    default_register: str = "tự nhiên, trung tính"
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Config:
    target_lang: str = "vietnamese"
    tl_dir: str = "game/tl"
    sdk: str | None = None
    llm: LLMConfig = field(default_factory=LLMConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    tm: TMConfig = field(default_factory=TMConfig)
    qa: QAConfig = field(default_factory=QAConfig)
    style: StyleConfig = field(default_factory=StyleConfig)

    def __post_init__(self) -> None:
        if not _LANGUAGE.fullmatch(self.target_lang) or self.target_lang == "None":
            raise ConfigError(
                f"target_lang '{self.target_lang}' không phải tên ngôn ngữ hợp lệ của Ren'Py"
            )
        tl_dir = PurePosixPath(self.tl_dir)
        if tl_dir.is_absolute() or ".." in tl_dir.parts or not tl_dir.parts:
            raise ConfigError("tl_dir phải là đường dẫn tương đối nằm trong dự án")
        if not self.llm.base_url.startswith(("http://", "https://")):
            raise ConfigError("llm.base_url phải bắt đầu bằng http:// hoặc https://")
        if not self.llm.model:
            raise ConfigError("llm.model không được để trống")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: object) -> Config:
        data = _expect_object(raw, "config")
        _reject_unknown(data, (f.name for f in dataclasses.fields(cls)), "config")
        hints = typing.get_type_hints(cls)
        values: dict[str, Any] = {}
        for key, value in data.items():
            hint = hints[key]
            if dataclasses.is_dataclass(hint):
                values[key] = _load_section(typing.cast("type[Any]", hint), value, key)
            else:
                values[key] = _coerce(value, hint, {}, key)
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> Config:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ConfigError(f"không tìm thấy tệp cấu hình: {path}") from None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConfigError(f"không đọc được cấu hình {path}: {error}") from error
        return cls.from_dict(raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        path.write_text(text + "\n", encoding="utf-8")


def _expect_object(raw: object, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"'{name}' phải là đối tượng JSON")
    return raw


def _reject_unknown(data: dict[str, Any], known: Iterable[str], name: str) -> None:
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"khóa không hợp lệ trong '{name}': {', '.join(unknown)}")


def _load_section(cls: type[Any], raw: object, name: str) -> Any:
    data = _expect_object(raw, name)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    _reject_unknown(data, fields, name)
    hints = typing.get_type_hints(cls)
    values = {
        key: _coerce(value, hints[key], dict(fields[key].metadata), f"{name}.{key}")
        for key, value in data.items()
    }
    return cls(**values)


def _coerce(value: object, hint: Any, metadata: dict[str, Any], name: str) -> Any:
    allowed = typing.get_args(hint) or (hint,)
    if value is None and type(None) in allowed:
        return None
    types = tuple(typing.get_origin(t) or t for t in allowed if t is not type(None))
    if isinstance(value, bool) and bool not in types:
        raise ConfigError(f"'{name}' phải có kiểu {_describe(types)}")
    if float in types and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if not isinstance(value, types):
        raise ConfigError(f"'{name}' phải có kiểu {_describe(types)}")
    minimum, maximum = metadata.get("min"), metadata.get("max")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if minimum is not None and value < minimum:
            raise ConfigError(f"'{name}' phải >= {minimum}")
        if maximum is not None and value > maximum:
            raise ConfigError(f"'{name}' phải <= {maximum}")
    return value


def _describe(types: tuple[type, ...]) -> str:
    return " or ".join(t.__name__ for t in types)
