"""Tests for vitrans.textio."""

from __future__ import annotations

from pathlib import Path

from vitrans.textio import TextFile, read_text_file, write_atomic


def test_write_atomic_on_new_and_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "new_file.txt"
    target.parent.mkdir(parents=True)

    # 1. New file
    write_atomic(target, b"initial content\n")
    assert target.is_file()
    assert target.read_bytes() == b"initial content\n"

    # 2. Existing file overwrite
    write_atomic(target, b"updated content\n")
    assert target.read_bytes() == b"updated content\n"

    # 3. Read back using read_text_file
    tf = read_text_file(target)
    assert tf.lines == ["updated content", ""]


def test_text_file_eol_and_bom_preservation(tmp_path: Path) -> None:
    # CRLF with BOM
    crlf_content = "\ufeffLine 1\r\nLine 2\r\n".encode("utf-8")
    tf = TextFile.from_bytes(crlf_content)
    assert tf.bom is True
    assert tf.eol == "\r\n"
    assert tf.lines == ["Line 1", "Line 2", ""]
    assert tf.to_bytes() == crlf_content

    # LF without BOM
    lf_content = b"Line 1\nLine 2\n"
    tf_lf = TextFile.from_bytes(lf_content)
    assert tf_lf.bom is False
    assert tf_lf.eol == "\n"
    assert tf_lf.to_bytes() == lf_content
