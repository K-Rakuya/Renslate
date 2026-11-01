from __future__ import annotations

import io
import os
import struct
import threading
import types
import zlib
from pathlib import Path
from typing import Any, BinaryIO

from renslate.renpy.rpyc.unpickler import RenpyUnpickler

RPYC2_HEADER: bytes = b"RENPY RPC2"
SLOT_STATEMENTS_PRE: int = 1
SLOT_STATEMENTS_POST: int = 2


def _unpack_stmts(loaded: Any) -> list[Any]:
    if loaded is None:
        return []
    if isinstance(loaded, tuple) and len(loaded) >= 2 and isinstance(loaded[1], list):
        return loaded[1]
    if isinstance(loaded, list):
        return loaded
    return [loaded]


def create_rpyc(slots: dict[int, bytes], rpydigest: bytes | None = None) -> bytes:
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

    dir_buf.extend(struct.pack("<III", 0, 0, 0))
    header.extend(dir_buf)
    header.extend(payload_buf)

    if rpydigest is not None:
        if len(rpydigest) != 16:
            raise ValueError(f"rpydigest must be exactly 16 bytes, got {len(rpydigest)}")
        header.extend(rpydigest)
    else:
        header.extend(b"\x00" * 16)

    return bytes(header)


def write_rpyc(
    path: Path | str, slots: dict[int, bytes], rpydigest: bytes | None = None
) -> None:
    data = create_rpyc(slots, rpydigest=rpydigest)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def read_rpyc_data(source: Path | str | BinaryIO, slot_id: int) -> bytes | None:
    if isinstance(source, (str, Path)):
        with RpycFile(source) as rpyc:
            return rpyc.read_slot(slot_id)

    curr_pos = source.tell()
    try:
        source.seek(0)
        header_bytes = source.read(1024)
        if not header_bytes.startswith(RPYC2_HEADER):
            if slot_id != SLOT_STATEMENTS_PRE:
                return None
            source.seek(0)
            data = source.read()
            try:
                return zlib.decompress(data)
            except Exception:
                return None

        pos = len(RPYC2_HEADER)
        slots: dict[int, tuple[int, int]] = {}
        while pos + 12 <= len(header_bytes):
            sid, start, length = struct.unpack("<III", header_bytes[pos : pos + 12])
            pos += 12
            if sid == 0:
                break
            slots[sid] = (start, length)

        if slot_id not in slots:
            return None

        start, length = slots[slot_id]
        source.seek(start)
        compressed = source.read(length)
        try:
            return zlib.decompress(compressed)
        except (zlib.error, struct.error, Exception):
            return None
    finally:
        source.seek(curr_pos)


def write_rpyc_data(
    dest: Path | str | BinaryIO,
    slots: dict[int, bytes],
    rpydigest: bytes | None = None,
) -> None:
    data = create_rpyc(slots, rpydigest=rpydigest)
    if isinstance(dest, (str, Path)):
        p = Path(dest)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    else:
        dest.write(data)


class RpycFile:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.slots: dict[int, tuple[int, int]] = {}
        self.is_legacy: bool = False
        self._fh: BinaryIO | None = None
        self._lock: threading.Lock = threading.Lock()
        self._context_depth: int = 0
        try:
            self._fh = self.path.open("rb")
            self._parse_header()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> RpycFile:
        with self._lock:
            if self._fh is None or self._fh.closed:
                self._fh = self.path.open("rb")
            self._context_depth += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        with self._lock:
            if self._context_depth > 0:
                self._context_depth -= 1
            if self._context_depth == 0:
                self._close_handle_locked()

    def close(self) -> None:
        with self._lock:
            self._context_depth = 0
            self._close_handle_locked()

    def _close_handle_locked(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __del__(self) -> None:
        try:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()
        except Exception:
            pass

    def _read_bytes(
        self, offset: int, length: int | None = None, whence: int = os.SEEK_SET
    ) -> bytes:
        with self._lock:
            fh = self._fh
            if fh is not None and not fh.closed:
                fh.seek(offset, whence)
                return fh.read() if length is None else fh.read(length)
            with self.path.open("rb") as f:
                f.seek(offset, whence)
                return f.read() if length is None else f.read(length)

    def _parse_header(self) -> None:
        header_bytes = self._read_bytes(0, 1024)

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
        if self.is_legacy:
            if slot_id != SLOT_STATEMENTS_PRE:
                return None
            try:
                data = self._read_bytes(0)
                return zlib.decompress(data)
            except Exception:
                return None

        if slot_id not in self.slots:
            return None

        start, length = self.slots[slot_id]
        try:
            compressed = self._read_bytes(start, length)
            return zlib.decompress(compressed)
        except Exception:
            return None

    def read_slots(self) -> dict[int, bytes]:
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
        data = self.read_slot(slot_id)
        if data is None:
            return None
        unpickler = RenpyUnpickler(io.BytesIO(data))
        return unpickler.load()

    def get_slot_1_statements(self) -> list[Any]:
        loaded = self.load_slot(SLOT_STATEMENTS_PRE)
        return _unpack_stmts(loaded)

    def get_slot_2_ast(self) -> list[Any]:
        loaded = self.load_slot(SLOT_STATEMENTS_POST)
        return _unpack_stmts(loaded)

    def read_md5(self) -> bytes | None:
        if self.is_legacy:
            return None
        try:
            return self._read_bytes(-16, 16, os.SEEK_END)
        except Exception:
            return None

    def extract_dialogues(self) -> list[Any]:
        from renslate.renpy.rpyc.extractor import extract_dialogue_from_rpyc

        return extract_dialogue_from_rpyc(self.path)

    def extract_scene_flow(self) -> list[Any]:
        from renslate.renpy.rpyc.extractor import extract_scene_flow_from_rpyc

        return extract_scene_flow_from_rpyc(self.path)


RpycArchive = RpycFile
