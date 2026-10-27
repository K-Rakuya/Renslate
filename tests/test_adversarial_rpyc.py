"""Adversarial stress harness for renslate.renpy.rpyc.

Tests extreme edge cases, malformed headers, pickle security vulnerabilities,
deep AST recursion, circular references, non-ASCII/Unicode syntax, and file permission boundaries.
"""

from __future__ import annotations

import io
import os
import pickle
import struct
import sys
import zlib
from pathlib import Path

import pytest

from renslate.renpy.parser import TranslationScanner
from renslate.renpy.rpyc import (
    RPYC2_HEADER,
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    Init,
    Menu,
    RenpyUnpickler,
    RpycFile,
    Say,
    Translate,
    TranslateSay,
    _extract_strings_from_text,
    _format_say_code,
    decompile_rpyc,
    extract_rpyc_file,
    extract_rpyc_project,
    synthesize_translation_skeletons,
    write_rpyc,
)

# ==============================================================================
# Dimension 1: Malformed & Pathological Headers
# ==============================================================================


@pytest.mark.parametrize(
    "corrupt_magic",
    [
        b"RENPY RPC1",
        b"RENPY RPC3",
        b"INVALIDMAGIC",
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"renpy rpc2",
        b"RENPY RPC",
    ],
)
def test_malformed_header_bad_magic(tmp_path: Path, corrupt_magic: bytes) -> None:
    """Non-matching RPYC2 header should safely fall back to legacy mode."""
    target = tmp_path / "bad_magic.rpyc"
    target.write_bytes(corrupt_magic + b"\x00" * 32)

    rpyc = RpycFile(target)
    assert rpyc.is_legacy is True
    # Slot 2 does not exist in legacy mode
    assert rpyc.read_slot(SLOT_STATEMENTS_POST) is None
    # Slot 1 decompresses raw file payload; garbage should safely return None
    assert rpyc.read_slot(SLOT_STATEMENTS_PRE) is None
    assert rpyc.load_slot(SLOT_STATEMENTS_PRE) is None

    # Extraction must safely return empty script without crashing
    extracted = extract_rpyc_file(target)
    assert len(extracted.dialogue) == 0
    assert len(extracted.strings) == 0


@pytest.mark.parametrize("length", [0, 1, 5, 9, 10, 15, 21, 23])
def test_malformed_header_truncated_files(tmp_path: Path, length: int) -> None:
    """Arbitrarily truncated files (from 0 to 23 bytes) must not crash header parser."""
    payload = RPYC2_HEADER[:length]
    if len(payload) < length:
        payload += b"\x00" * (length - len(payload))

    target = tmp_path / f"truncated_{length}.rpyc"
    target.write_bytes(payload)

    rpyc = RpycFile(target)
    if length < len(RPYC2_HEADER):
        assert rpyc.is_legacy is True
    else:
        assert rpyc.is_legacy is False
        assert len(rpyc.slots) == 0

    assert rpyc.read_slot(1) is None
    assert rpyc.read_slot(2) is None
    if length < 16:
        # Seeking -16 from EOF fails or is legacy, so None is returned
        assert rpyc.read_md5() is None
    else:
        # File >= 16 bytes without MD5 digest returns last 16 bytes of truncated header
        md5 = rpyc.read_md5()
        assert md5 is not None and len(md5) == 16

    extracted = extract_rpyc_file(target)
    assert len(extracted.dialogue) == 0
    assert len(extracted.strings) == 0


def test_malformed_header_negative_and_massive_slot_offsets(tmp_path: Path) -> None:
    """Massive (0xFFFFFFFF) slot offset and length must safely return None without OOM."""
    header = bytearray(RPYC2_HEADER)
    # Slot 1: start=0xFFFFFFFF, length=0xFFFFFFFF
    header.extend(struct.pack("<III", 1, 0xFFFFFFFF, 0xFFFFFFFF))
    # Slot 2: start=100, length=0x7FFFFFFF
    header.extend(struct.pack("<III", 2, 100, 0x7FFFFFFF))
    header.extend(struct.pack("<III", 0, 0, 0))

    target = tmp_path / "huge_offsets.rpyc"
    target.write_bytes(header)

    rpyc = RpycFile(target)
    assert 1 in rpyc.slots
    assert 2 in rpyc.slots

    # read_slot seeks past EOF or reads empty, returning None on decompress
    assert rpyc.read_slot(1) is None
    assert rpyc.read_slot(2) is None
    assert rpyc.load_slot(1) is None


def test_malformed_header_slot_zero_terminator_at_start(tmp_path: Path) -> None:
    """Slot directory terminating immediately at slot 0 should have 0 slots."""
    header = bytearray(RPYC2_HEADER)
    header.extend(struct.pack("<III", 0, 100, 50))  # immediate terminator

    target = tmp_path / "zero_start.rpyc"
    target.write_bytes(header)

    rpyc = RpycFile(target)
    assert len(rpyc.slots) == 0
    assert rpyc.read_slot(1) is None


def test_malformed_header_no_terminator_full_buffer(tmp_path: Path) -> None:
    """Slot directory filling the 1024-byte read window without a terminator."""
    header = bytearray(RPYC2_HEADER)
    # Fill remaining bytes up to 1024 with valid non-zero slots
    slot_id = 1
    while len(header) + 12 <= 1024:
        header.extend(struct.pack("<III", slot_id, 2000, 10))
        slot_id += 1

    target = tmp_path / "no_terminator.rpyc"
    target.write_bytes(header)

    rpyc = RpycFile(target)
    assert len(rpyc.slots) == slot_id - 1
    # All point past EOF so read returns None
    assert rpyc.read_slot(1) is None


def test_corrupt_zlib_slot_payload(tmp_path: Path) -> None:
    """Slot entry pointing to non-zlib payload must safely return None."""
    header = bytearray(RPYC2_HEADER)
    header.extend(struct.pack("<III", 1, 34, 10))
    header.extend(struct.pack("<III", 0, 0, 0))
    header.extend(b"NOT_ZLIB_DATA_123")

    target = tmp_path / "corrupt_zlib.rpyc"
    target.write_bytes(header)

    rpyc = RpycFile(target)
    assert rpyc.read_slot(1) is None
    assert rpyc.load_slot(1) is None


def test_corrupt_pickle_slot_payload_unpickling_error(tmp_path: Path) -> None:
    """Valid zlib payload containing corrupt pickle bytes raises UnpicklingError."""
    # Empirical challenge: load_slot does not catch UnpicklingError
    compressed = zlib.compress(b"corrupt non-pickle payload stream")
    header = bytearray(RPYC2_HEADER)
    header.extend(struct.pack("<III", 2, 34, len(compressed)))
    header.extend(struct.pack("<III", 0, 0, 0))
    header.extend(compressed)

    target = tmp_path / "corrupt_pickle.rpyc"
    target.write_bytes(header)

    rpyc = RpycFile(target)
    assert rpyc.read_slot(2) == b"corrupt non-pickle payload stream"

    # Verifies that load_slot currently propagates pickle.UnpicklingError
    with pytest.raises(pickle.UnpicklingError):
        rpyc.load_slot(2)


# ==============================================================================
# Dimension 2: Deep AST Recursion, Cycles & Pickle Bomb Vulnerabilities
# ==============================================================================


def test_pickle_bomb_arbitrary_eval_execution() -> None:
    """Verify that RenpyUnpickler blocks builtins.eval and prevents arbitrary code execution."""

    class EvalBomb:
        def __reduce__(self) -> tuple[object, ...]:
            # Reduced to builtins.eval('1337 + 42')
            return (eval, ("1337 + 42",))

    payload = pickle.dumps(EvalBomb())
    unpickler = RenpyUnpickler(io.BytesIO(payload))
    with pytest.raises(pickle.UnpicklingError):
        unpickler.load()


def test_pickle_bomb_in_rpyc_slot(tmp_path: Path) -> None:
    """Verify that loading an RPYC slot with builtins.eval is blocked with UnpicklingError."""

    class SideEffectBomb:
        def __reduce__(self) -> tuple[object, ...]:
            return (eval, ("pow(2, 10)",))

    payload = pickle.dumps((None, [SideEffectBomb()]))
    target = tmp_path / "bomb.rpyc"
    write_rpyc(target, {SLOT_STATEMENTS_POST: payload})

    rpyc = RpycFile(target)
    with pytest.raises(pickle.UnpicklingError):
        rpyc.load_slot(SLOT_STATEMENTS_POST)


def test_circular_ast_reference_recursion_crash(tmp_path: Path) -> None:
    """Empirical proof: Circular references in AST nodes crash with RecursionError."""
    node = Init()
    node.block = [node]  # Direct cycle: node -> node

    payload = pickle.dumps((None, [node]))
    target = tmp_path / "cycle.rpyc"
    write_rpyc(target, {SLOT_STATEMENTS_POST: payload, SLOT_STATEMENTS_PRE: payload})

    # extract_rpyc_file has no visited/seen set; must trigger RecursionError
    with pytest.raises(RecursionError):
        extract_rpyc_file(target)

    # decompile_rpyc also has no visited/seen set; must trigger RecursionError
    with pytest.raises(RecursionError):
        decompile_rpyc(target)


def test_deep_ast_nesting_recursion_crash(tmp_path: Path) -> None:
    """Empirical proof: 1200 nested AST nodes exceed recursion limit and crash."""
    curr = root = Init()
    for _ in range(1200):
        child = Init()
        curr.block = [child]
        curr = child

    payload = pickle.dumps((None, [root]))
    target = tmp_path / "deep_nesting.rpyc"
    write_rpyc(target, {SLOT_STATEMENTS_POST: payload, SLOT_STATEMENTS_PRE: payload})

    with pytest.raises(RecursionError):
        extract_rpyc_file(target)

    with pytest.raises(RecursionError):
        decompile_rpyc(target)


def test_menu_inside_translate_dialogue_omission_gap(tmp_path: Path) -> None:
    """Verify that menu captions are extracted to strings and inner dialogue is preserved."""
    inner_say = Say(who="hero", what="Secret dialogue inside choice")
    menu = Menu()
    menu.items = [("Choice A", "True", [inner_say])]

    translate_node = Translate()
    translate_node.identifier = "choice_block_001"
    translate_node.block = [menu]

    payload = pickle.dumps((None, [translate_node]))
    target = tmp_path / "menu_translate.rpyc"
    write_rpyc(target, {SLOT_STATEMENTS_POST: payload})

    extracted = extract_rpyc_file(target)
    # The choice caption is extracted to UI strings, not duplicate dialogue
    assert any(s.text == "Choice A" for s in extracted.strings)
    # Inner dialogue inside the choice block is extracted
    dialogue_texts = [d.what for d in extracted.dialogue]
    assert "Secret dialogue inside choice" in dialogue_texts
    assert not any(d.identifier.endswith("_opt_0") for d in extracted.dialogue)


# ==============================================================================
# Dimension 3: Unicode, Multi-line, Tags & Synthesis Code Injection
# ==============================================================================


def test_unicode_and_special_character_synthesis(tmp_path: Path) -> None:
    """Test synthesis with Vietnamese, CJK, Emoji, quotes, and Ren'Py tags."""
    game_dir = tmp_path / "game"
    game_dir.mkdir()

    items = [
        # 1. Vietnamese speaker and text
        TranslateSay(
            identifier="vn_01",
            who="Người Dẫn Chuyện",
            what="Xin chào thế giới! Đất nước Việt Nam.",
            filename="game/story.rpy",
            linenumber=1,
        ),
        # 2. Japanese & Emoji
        TranslateSay(
            identifier="cjk_01",
            who="アイリス",
            what="魔法の世界へようこそ!🎮✨",
            filename="game/story.rpy",
            linenumber=2,
        ),
        # 3. Multiline text with escapes
        TranslateSay(
            identifier="multi_01",
            who="alice",
            what='Line 1\nLine 2: "quoted" & backslash C:\\System',
            filename="game/story.rpy",
            linenumber=3,
        ),
        # 4. Ren'Py formatting tags and interpolation
        TranslateSay(
            identifier="tags_01",
            who="bob",
            what="{b}Bold{/b} {size=+10}Large{/size} [player] 100%% {{escaped_braces}}",
            filename="game/story.rpy",
            linenumber=4,
        ),
    ]

    rpyc_path = game_dir / "story.rpyc"
    write_rpyc(rpyc_path, {SLOT_STATEMENTS_POST: pickle.dumps((None, items))})

    generated = synthesize_translation_skeletons(tmp_path, "vietnamese")
    assert len(generated) == 1
    tl_file = generated[0]
    content = tl_file.read_text(encoding="utf-8")

    # Content must retain all unicode characters
    assert "Xin chào thế giới! Đất nước Việt Nam." in content
    assert "魔法の世界へようこそ!🎮✨" in content
    assert "Line 1\\nLine 2" in content
    assert '\\"quoted\\"' in content
    assert "{b}Bold{/b}" in content

    # Test that TranslationScanner successfully scans the generated output
    scanner = TranslationScanner("vietnamese")
    scanner.scan_file("game/tl/vietnamese/story.rpy", content.splitlines())
    assert len(scanner.units) >= 3


def test_identifier_code_injection_in_skeleton_synthesis(tmp_path: Path) -> None:
    """Empirical proof: Newlines in AST identifier inject arbitrary code into synthesized .rpy."""
    game_dir = tmp_path / "game"
    game_dir.mkdir()

    injected_say = TranslateSay(
        identifier='exploit_01\n    $ dangerous_command("rm -rf")',
        who="villain",
        what="Payload executed",
        filename="game/script.rpy",
        linenumber=10,
    )

    rpyc_path = game_dir / "script.rpyc"
    write_rpyc(rpyc_path, {SLOT_STATEMENTS_POST: pickle.dumps((None, [injected_say]))})

    generated = synthesize_translation_skeletons(tmp_path, "vietnamese")
    content = generated[0].read_text(encoding="utf-8")

    # The newline is preserved, injecting a statement line before the block
    assert '$ dangerous_command("rm -rf"):' in content


def test_format_say_code_edge_cases() -> None:
    """Adversarial speaker names formatted by _format_say_code."""
    # None speaker (narrator)
    assert _format_say_code(None, "Hello", None, None, None) == '"Hello"'

    # Empty speaker string
    assert _format_say_code("", "Hello", None, None, None) == '"Hello"'
    assert _format_say_code("   ", "Hello", None, None, None) == '"Hello"'

    # Speaker with attributes
    assert (
        _format_say_code("eileen", "Hello", ("happy", "blush"), ("wink",), "dissolve")
        == 'eileen happy blush @ wink "Hello" with dissolve'
    )

    # Complex speaker name with spaces (not auto-quoted by _format_say_code)
    code = _format_say_code("Old Man", "Who is there?", None, None, None)
    assert code == 'Old Man "Who is there?"'


def test_gettext_triple_quote_omission_gap() -> None:
    """Empirical proof: _GETTEXT_REGEX omits triple-quoted UI strings in Python blocks."""
    py_code = """
    renpy.notify(_(\"\"\"Long translatable help text\"\"\"))
    normal_str = _(\"Normal translatable string\")
    """
    out: list[object] = []
    _extract_strings_from_text(py_code, "script.rpy", 10, out)  # type: ignore[arg-type]

    texts = [getattr(x, "text", "") for x in out]
    # Normal string is extracted
    assert "Normal translatable string" in texts
    # Triple-quoted string is missed by regex
    assert "Long translatable help text" not in texts


# ==============================================================================
# Dimension 4: Non-standard Extensions & Filesystem Permissions
# ==============================================================================


def test_non_standard_file_extensions(tmp_path: Path) -> None:
    """extract_rpyc_file operates on non-.rpyc extensions; project scanner filters them."""
    say = Say(who="narrator", what="Alternative extension test")
    payload = pickle.dumps((None, [say]))

    # 1. Non-standard extension .rpyb
    rpyb_file = tmp_path / "game" / "script.rpyb"
    rpyb_file.parent.mkdir(parents=True, exist_ok=True)
    write_rpyc(rpyb_file, {SLOT_STATEMENTS_PRE: payload})

    extracted = extract_rpyc_file(rpyb_file)
    assert len(extracted.dialogue) == 1
    assert extracted.dialogue[0].what == "Alternative extension test"
    # default name derived from suffix
    assert extracted.dialogue[0].source_file == "script.rpy"

    # 2. extract_rpyc_project strictly searches for *.rpyc, ignoring .rpyb
    project_result = extract_rpyc_project(tmp_path)
    assert len(project_result) == 0


def test_filesystem_permission_and_error_handling(tmp_path: Path) -> None:
    """Filesystem boundaries: non-existent file, directory, and unreadable permissions."""
    # 1. Non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        RpycFile(tmp_path / "non_existent.rpyc")

    with pytest.raises(FileNotFoundError):
        extract_rpyc_file(tmp_path / "non_existent.rpyc")

    # 2. Directory passed as RpycFile raises IsADirectoryError
    sub_dir = tmp_path / "fake_file.rpyc"
    sub_dir.mkdir()
    with pytest.raises(IsADirectoryError):
        RpycFile(sub_dir)

    # 3. Unreadable file (0000 permissions)
    if hasattr(os, "chmod") and sys.platform != "win32":
        unreadable = tmp_path / "locked.rpyc"
        unreadable.write_bytes(b"dummy")
        unreadable.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                RpycFile(unreadable)
        finally:
            unreadable.chmod(0o644)  # restore for cleanup
