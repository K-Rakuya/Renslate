"""Lossless reading and atomic writing of UTF-8 text files with their line-ending style."""

from __future__ import annotations

import codecs
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TextFile:
    lines: list[str]
    eol: str = "\n"
    bom: bool = False

    @classmethod
    def from_bytes(cls, data: bytes) -> TextFile:
        """Decode data; raise UnicodeDecodeError if it is not valid UTF-8."""
        text = data.decode("utf-8-sig")
        crlf = text.count("\r\n")
        eol = "\r\n" if crlf > text.count("\n") - crlf else "\n"
        lines = text.split("\n")
        if eol == "\r\n":
            lines = [line[:-1] if line.endswith("\r") else line for line in lines]
        return cls(lines, eol, data.startswith(codecs.BOM_UTF8))

    def to_bytes(self) -> bytes:
        payload = self.eol.join(self.lines).encode("utf-8")
        return codecs.BOM_UTF8 + payload if self.bom else payload


def read_text_file(path: Path) -> TextFile:
    return TextFile.from_bytes(path.read_bytes())


def write_atomic(path: Path, data: bytes) -> None:
    """Replace path with data so that readers never observe a partial file."""
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_bytes(data)
        shutil.copymode(path, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
