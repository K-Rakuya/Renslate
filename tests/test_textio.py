"""Tests for renslate.textio."""

from __future__ import annotations

from pathlib import Path

from renslate.textio import TextFile, read_text_file, write_atomic


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


def test_normalize_nfc_ascii_fast_path() -> None:
    from renslate.core.textio import normalize_nfc as core_normalize_nfc
    from renslate.textio import normalize_nfc

    assert normalize_nfc is core_normalize_nfc

    ascii_str = "Hello [player]! {b}Welcome{/b}"
    assert normalize_nfc(ascii_str) is ascii_str
    assert normalize_nfc("") == ""

    ctrl_str = "\x00\t\r\n"
    assert normalize_nfc(ctrl_str) is ctrl_str


def test_normalize_nfc_decomposed_unicode() -> None:
    import unicodedata

    from renslate.textio import normalize_nfc

    nfd_text = unicodedata.normalize("NFD", "Tiếng Việt chuẩn xác")
    assert unicodedata.is_normalized("NFC", nfd_text) is False
    nfc_text = normalize_nfc(nfd_text)
    assert nfc_text == "Tiếng Việt chuẩn xác"
    assert unicodedata.is_normalized("NFC", nfc_text) is True

    cjk = "日本語テキスト"
    assert normalize_nfc(cjk) == cjk
