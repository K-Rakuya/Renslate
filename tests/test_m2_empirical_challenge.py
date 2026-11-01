"""Empirical Adversarial Challenge Suite for Milestone 2: Ren'Py Bytecode Modernization.

Executes adversarial stress testing, boundary fuzzing, and contract verification
across all decomposed RPYC components:
- Package import parity and symbol identity
- 16-byte MD5 digest footer & stream I/O operations
- Canonical say identifier hashing and path normalization
- Scene flow extraction & sprite register state machine
- Ren'Py grammar compliance in say statement formatting
- Security boundaries of RenpyUnpickler
"""

from __future__ import annotations

import hashlib
import io
import pickle
from pathlib import Path
from typing import Any

import pytest

import renslate.renpy.rpyc as r_facade
from renslate.renpy.rpyc import (
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    Hide,
    Label,
    RenpyUnpickler,
    RpycArchive,
    RpycFile,
    Say,
    Scene,
    Show,
    VirtualASTNode,
    _format_say_code,
    compute_canonical_say_identifier,
    create_rpyc,
    decompile_rpyc,
    extract_rpyc_file,
    format_say_statement,
    read_rpyc_data,
    synthesize_translation_skeletons,
    write_rpyc,
    write_rpyc_data,
)
from renslate.renpy.rpyc.archive import (
    create_rpyc as arch_create_rpyc,
)
from renslate.renpy.rpyc.ast_types import (
    Say as ast_Say,
)
from renslate.renpy.rpyc.ast_types import (
    Scene as ast_Scene,
)
from renslate.renpy.rpyc.ast_types import (
    Show as ast_Show,
)
from renslate.renpy.rpyc.decompiler import (
    format_say_statement as dec_format_say_statement,
)
from renslate.renpy.rpyc.extractor import (
    compute_canonical_say_identifier as ext_compute_id,
)
from renslate.renpy.rpyc.unpickler import (
    RenpyUnpickler as unpick_RenpyUnpickler,
)

# ==============================================================================
# Challenge 1: Package Import Parity & Identity Verification
# ==============================================================================


class TestPackageImportParity:
    """Empirically verify that importing from facade matches submodules exactly."""

    def test_all_symbols_present_and_identical(self) -> None:
        submodules = [
            r_facade.archive,
            r_facade.ast_types,
            r_facade.decompiler,
            r_facade.extractor,
            r_facade.skeleton,
            r_facade.unpickler,
        ]
        assert len(r_facade.__all__) >= 70

        for symbol_name in r_facade.__all__:
            assert hasattr(r_facade, symbol_name), f"Missing {symbol_name} in facade"
            facade_obj = getattr(r_facade, symbol_name)

            # Match against submodule definition
            matched = False
            for mod in submodules:
                if hasattr(mod, symbol_name):
                    sub_obj = getattr(mod, symbol_name)
                    if facade_obj is sub_obj:
                        matched = True
                        break
            assert matched, f"Symbol {symbol_name} does not match submodule identity"

    def test_rpyc_archive_alias_identity(self) -> None:
        assert RpycArchive is RpycFile

    def test_ast_node_cross_references(self) -> None:
        assert ast_Say is Say
        assert ast_Scene is Scene
        assert ast_Show is Show
        assert arch_create_rpyc is create_rpyc
        assert dec_format_say_statement is format_say_statement
        assert ext_compute_id is compute_canonical_say_identifier
        assert unpick_RenpyUnpickler is RenpyUnpickler


# ==============================================================================
# Challenge 2: RPYC2 16-byte MD5 Footer & Stream I/O Operations
# ==============================================================================


class TestRpyc2BinarySpecification:
    """Stress-test RPYC2 footer compliance and binary stream I/O."""

    def test_default_16_byte_null_footer(self, tmp_path: Path) -> None:
        rpyc_file = tmp_path / "default_footer.rpyc"
        write_rpyc(rpyc_file, {1: b"dummy"})

        content = rpyc_file.read_bytes()
        assert content.endswith(b"\x00" * 16)

        with RpycFile(rpyc_file) as rpyc:
            md5 = rpyc.read_md5()
            assert md5 == b"\x00" * 16

    def test_custom_rpydigest_footer(self, tmp_path: Path) -> None:
        rpyc_file = tmp_path / "custom_footer.rpyc"
        custom_digest = hashlib.md5(b"unique_rpy_source_code").digest()
        assert len(custom_digest) == 16

        write_rpyc(rpyc_file, {1: b"data"}, rpydigest=custom_digest)

        with RpycFile(rpyc_file) as rpyc:
            assert rpyc.read_md5() == custom_digest

    @pytest.mark.parametrize("invalid_len", [0, 1, 15, 17, 32])
    def test_invalid_rpydigest_length_raises(self, invalid_len: int) -> None:
        bad_digest = b"x" * invalid_len
        with pytest.raises(ValueError, match="rpydigest must be exactly 16 bytes"):
            create_rpyc({1: b"payload"}, rpydigest=bad_digest)

    def test_stream_io_bytesio(self) -> None:
        """read_rpyc_data and write_rpyc_data must work with in-memory BytesIO streams."""
        buf = io.BytesIO()
        slots = {1: b"payload_slot_1", 2: b"payload_slot_2"}
        custom_digest = b"A" * 16

        write_rpyc_data(buf, slots, rpydigest=custom_digest)
        raw_bytes = buf.getvalue()
        assert raw_bytes.endswith(custom_digest)

        # Seek position preservation
        buf.seek(10)
        s1 = read_rpyc_data(buf, 1)
        assert s1 == b"payload_slot_1"
        assert buf.tell() == 10  # Position restored

        s2 = read_rpyc_data(buf, 2)
        assert s2 == b"payload_slot_2"
        assert buf.tell() == 10

        s_missing = read_rpyc_data(buf, 999)
        assert s_missing is None
        assert buf.tell() == 10

    def test_legacy_v1_stream_handling(self) -> None:
        import zlib
        payload = b"legacy_code"
        compressed = zlib.compress(payload)
        buf = io.BytesIO(compressed)

        s1 = read_rpyc_data(buf, 1)
        assert s1 == payload

        s2 = read_rpyc_data(buf, 2)
        assert s2 is None


# ==============================================================================
# Challenge 3: Canonical Ren'Py Translation Hashing
# ==============================================================================


class TestCanonicalSayIdentifier:
    """Stress-test Ren'Py translation hashing under adversarial path and text scenarios."""

    def test_canonical_hash_structure(self) -> None:
        ident = compute_canonical_say_identifier(
            label="start",
            filename="/projects/game/script.rpy",
            linenumber=42,
            what="Hello world",
        )
        assert ident.startswith("start_")
        hex_part = ident[len("start_"):]
        assert len(hex_part) == 8
        int(hex_part, 16)  # Valid hex

    def test_none_or_empty_label_uses_global(self) -> None:
        id1 = compute_canonical_say_identifier(None, "game/script.rpy", 10, "Hi")
        id2 = compute_canonical_say_identifier("", "game/script.rpy", 10, "Hi")
        assert id1.startswith("global_")
        assert id2.startswith("global_")
        assert id1 == id2

    def test_windows_and_posix_path_normalization(self) -> None:
        posix = "C:/Games/VN/game/screens.rpy"
        windows = "C:\\Games\\VN\\game\\screens.rpy"
        id_posix = compute_canonical_say_identifier("scr", posix, 15, "Click me")
        id_win = compute_canonical_say_identifier("scr", windows, 15, "Click me")
        assert id_posix == id_win

    def test_unicode_and_vietnamese_hashing(self) -> None:
        viet_text = "Xin chào các bạn, đây là tiếng Việt có dấu: ừ ơ đ á à ả ã ạ."
        id1 = compute_canonical_say_identifier("label1", "game/story.rpy", 100, viet_text)
        assert id1.startswith("label1_")
        assert len(id1) == len("label1_") + 8

        # Sensitivity: 1 char change alters hash
        id2 = compute_canonical_say_identifier("label1", "game/story.rpy", 100, viet_text + "!")
        assert id1 != id2


# ==============================================================================
# Challenge 4: Scene Flow & Sprite Register State Machine
# ==============================================================================


class TestSceneFlowAndSpriteTracking:
    """Empirically test scene flow extraction and character sprite register lifecycle."""

    def test_sprite_register_lifecycle(self, tmp_path: Path) -> None:
        """Verify:
        - scene clears sprites
        - show adds / replaces sprite by base tag
        - hide removes sprite by base tag
        - say nodes inherit currently active sprites and scene
        """
        stmts: list[Any] = []

        # Label start
        lbl = Label()
        lbl.name = "chapter_1"
        stmts.append(lbl)

        # Scene bg
        sc = Scene()
        sc.imspec = (("bg", "classroom"), None, None, None, None)
        stmts.append(sc)

        # Show Alice happy at left
        sh1 = Show()
        sh1.imspec = (("alice", "happy"), None, None, ["left"], None)
        stmts.append(sh1)

        # Say 1: Alice speaking
        say1 = Say()
        say1.who = "alice"
        say1.what = "Good morning!"
        say1.filename = "game/script.rpy"
        say1.linenumber = 10
        stmts.append(say1)

        # Show Bob neutral at right
        sh2 = Show()
        sh2.imspec = (("bob", "neutral"), None, None, ["right"], None)
        stmts.append(sh2)

        # Say 2: Bob speaking
        say2 = Say()
        say2.who = "bob"
        say2.what = "Morning Alice."
        say2.filename = "game/script.rpy"
        say2.linenumber = 15
        stmts.append(say2)

        # Show Alice sad (replacing Alice happy)
        sh3 = Show()
        sh3.imspec = (("alice", "sad"), None, None, ["left"], None)
        stmts.append(sh3)

        # Hide Bob
        hd = Hide()
        hd.imspec = (("bob",), None, None, None, None)
        stmts.append(hd)

        # Say 3: Alice alone sad
        say3 = Say()
        say3.who = "alice"
        say3.what = "Where did he go?"
        say3.filename = "game/script.rpy"
        say3.linenumber = 25
        stmts.append(say3)

        # Scene change (clears all)
        sc2 = Scene()
        sc2.imspec = (("bg", "hallway"), None, None, None, None)
        stmts.append(sc2)

        # Say 4: Narrator in empty hallway
        say4 = Say()
        say4.who = None
        say4.what = "The hallway was silent."
        say4.filename = "game/script.rpy"
        say4.linenumber = 30
        stmts.append(say4)

        # Build RPYC with Slot 1
        rpyc_path = tmp_path / "flow_test.rpyc"
        write_rpyc(rpyc_path, {SLOT_STATEMENTS_PRE: pickle.dumps((None, stmts))})

        script = extract_rpyc_file(rpyc_path)
        dialogues = script.dialogue

        assert len(dialogues) == 4

        # Dialogue 1: Alice happy alone
        assert dialogues[0].current_label == "chapter_1"
        assert dialogues[0].scene_id == "bg classroom"
        assert dialogues[0].active_sprites == ("alice happy at left",)

        # Dialogue 2: Alice + Bob both present
        assert dialogues[1].scene_id == "bg classroom"
        assert "alice happy at left" in dialogues[1].active_sprites
        assert "bob neutral at right" in dialogues[1].active_sprites
        assert len(dialogues[1].active_sprites) == 2

        # Dialogue 3: Alice replaced with sad, Bob hidden
        assert dialogues[2].scene_id == "bg classroom"
        assert dialogues[2].active_sprites == ("alice sad at left",)

        # Dialogue 4: Scene reset, hallway
        assert dialogues[3].scene_id == "bg hallway"
        assert dialogues[3].active_sprites == ()


# ==============================================================================
# Challenge 5: Say Statement Ren'Py Grammar Alignment
# ==============================================================================


class TestSayStatementGrammar:
    """Verify strict Ren'Py grammar alignment: with before id."""

    def test_with_strictly_precedes_id(self) -> None:
        node = Say()
        node.who = "lucy"
        node.what = "See you tomorrow."
        node.attributes = ("smile",)
        node.temporary_attributes = ("blush",)
        node.with_ = "dissolve"
        node.identifier = "start_abcd1234"
        node.explicit_identifier = True
        node.interact = False

        code = format_say_statement(node)

        # Required order: who attrs @temp text nointeract with id
        expected = (
            'lucy smile @ blush "See you tomorrow." '
            'nointeract with dissolve id start_abcd1234'
        )
        assert code == expected
        with_idx = code.index("with dissolve")
        id_idx = code.index("id start_abcd1234")
        assert with_idx < id_idx, "Grammar violation: 'with' must precede 'id'"

    def test_translated_text_override(self) -> None:
        node = Say()
        node.who = "eileen"
        node.what = "Hello"

        code = format_say_statement(node, translated_text="Xin chào")
        assert code == 'eileen "Xin chào"'

    def test_backward_compatible_format_say_code(self) -> None:
        """_format_say_code 5-argument helper remains fully functional."""
        code = _format_say_code("eileen", "Hello", ("happy",), ("blush",), "dissolve")
        assert code == 'eileen happy @ blush "Hello" with dissolve'


# ==============================================================================
# Challenge 6: Security & Hardened Unpickler Boundaries
# ==============================================================================


class TestUnpicklerSecurityBoundaries:
    """Stress-test RenpyUnpickler sandbox guarantees."""

    @pytest.mark.parametrize(
        "module,name",
        [
            ("os", "system"),
            ("subprocess", "Popen"),
            ("sys", "exit"),
            ("shutil", "rmtree"),
            ("socket", "socket"),
            ("importlib", "import_module"),
        ],
    )
    def test_blocked_modules_raise_unpickling_error(self, module: str, name: str) -> None:
        payload = pickle.dumps(pickle.loads(f"c{module}\n{name}\n.".encode()))
        unpickler = RenpyUnpickler(io.BytesIO(payload))
        with pytest.raises(pickle.UnpicklingError, match="Blocked dangerous module"):
            unpickler.load()

    @pytest.mark.parametrize("unsafe_builtin", ["eval", "exec", "open", "compile", "__import__"])
    def test_unsafe_builtins_blocked(self, unsafe_builtin: str) -> None:
        payload = f"cbuiltins\n{unsafe_builtin}\n.".encode()
        unpickler = RenpyUnpickler(io.BytesIO(payload))
        with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe builtin"):
            unpickler.load()

    def test_dynamic_unknown_class_virtualized_safely(self) -> None:
        """Non-blocked custom Ren'Py classes become virtual AST nodes without code execution."""
        payload = b"crenpy.ast\nCustomPluginNode\n)\x81."
        unpickler = RenpyUnpickler(io.BytesIO(payload))
        node = unpickler.load()
        assert isinstance(node, VirtualASTNode)
        assert type(node).__name__ == "CustomPluginNode"
        assert type(node).__module__ == "renpy.ast"


# ==============================================================================
# Challenge 7: Decompiler Statement Fidelity & Fallback Logic
# ==============================================================================


class TestDecompilerFidelity:
    """Stress-test decompile_rpyc across various AST node types and fallback paths."""

    def test_decompile_all_node_types(self, tmp_path: Path) -> None:
        from renslate.renpy.rpyc import (
            Call,
            Default,
            Define,
            EarlyPython,
            If,
            Jump,
            Menu,
            Pass,
            Python,
            Return,
            UserStatement,
            While,
            With,
        )

        stmts: list[Any] = []

        # Define & Default
        d1 = Define()
        d1.varname = "config.name"
        d1.code = '"My Novel"'
        stmts.append(d1)

        d2 = Default()
        d2.varname = "points"
        d2.code = "0"
        stmts.append(d2)

        # EarlyPython & Python
        ep = EarlyPython()
        ep.code = "import math"
        stmts.append(ep)

        py = Python()
        py.code = "points += 10"
        stmts.append(py)

        # Label with nested blocks
        lbl = Label()
        lbl.name = "start"

        # Say statement
        s = Say()
        s.who = "narrator"
        s.what = "A long story begins."

        # If-elif-else
        if_node = If()
        if_node.entries = [
            ("points > 5", [s]),
            ("points == 0", [Pass()]),
            ("True", [Return()]),
        ]

        # While loop
        wh = While()
        wh.condition = "points < 100"
        wh.block = [Pass()]

        # Menu
        m = Menu()
        m.items = [
            ("Option A", "True", [Jump(target="path_a")]),
            ("Option B", "points > 10", [Call(label="path_b")]),
        ]

        # UserStatement & With
        us = UserStatement()
        us.line = "play music 'bgm.ogg'"
        w = With()
        w.expr = "dissolve"

        lbl.block = [s, if_node, wh, m, us, w]
        stmts.append(lbl)

        rpyc_path = tmp_path / "comprehensive.rpyc"
        write_rpyc(rpyc_path, {SLOT_STATEMENTS_PRE: pickle.dumps((None, stmts))})

        out_rpy = decompile_rpyc(rpyc_path)
        assert out_rpy.is_file()
        decompiled_text = out_rpy.read_text(encoding="utf-8")

        # Verify statements exist in decompiled output
        assert 'define config.name = "My Novel"' in decompiled_text
        assert "default points = 0" in decompiled_text
        assert "init python:" in decompiled_text
        assert "$ points += 10" in decompiled_text
        assert "label start:" in decompiled_text
        assert 'narrator "A long story begins."' in decompiled_text
        assert "if points > 5:" in decompiled_text
        assert "elif points == 0:" in decompiled_text
        assert "else:" in decompiled_text
        assert "while points < 100:" in decompiled_text
        assert 'menu:' in decompiled_text
        assert '"Option A":' in decompiled_text
        assert '"Option B" if points > 10:' in decompiled_text
        assert "play music 'bgm.ogg'" in decompiled_text
        assert "with dissolve" in decompiled_text

    def test_decompile_missing_ast_raises(self, tmp_path: Path) -> None:
        """RPYC file without Slot 1 or Slot 2 raises ValueError."""
        rpyc_path = tmp_path / "empty_slots.rpyc"
        # Only slot 99, no slot 1 or 2
        write_rpyc(rpyc_path, {99: b"unrelated"})

        with pytest.raises(ValueError, match="bị hỏng hoặc không có dữ liệu AST"):
            decompile_rpyc(rpyc_path)

    def test_decompile_slot2_fallback_when_slot1_missing(self, tmp_path: Path) -> None:
        """Decompiler falls back to Slot 2 if Slot 1 is not present."""
        rpyc_path = tmp_path / "slot2_only.rpyc"
        s = Say()
        s.who = "eileen"
        s.what = "From slot 2 only."
        write_rpyc(rpyc_path, {SLOT_STATEMENTS_POST: pickle.dumps((None, [s]))})

        out_rpy = decompile_rpyc(rpyc_path)
        content = out_rpy.read_text(encoding="utf-8")
        assert 'eileen "From slot 2 only."' in content


# ==============================================================================
# Challenge 8: Incremental Translation Skeletons & Deduplication
# ==============================================================================


class TestTranslationSkeletonSynthesis:
    """Stress-test synthesize_translation_skeletons edge cases."""

    def test_incremental_synthesis_preserves_existing_files(self, tmp_path: Path) -> None:
        game_dir = tmp_path / "game"
        game_dir.mkdir(parents=True)

        s1 = Say()
        s1.who = "alice"
        s1.what = "Hello"
        s1.filename = "game/ch1.rpy"
        s1.linenumber = 5

        rpyc_file = game_dir / "ch1.rpyc"
        write_rpyc(rpyc_file, {SLOT_STATEMENTS_PRE: pickle.dumps((None, [s1]))})

        # Synthesize first time
        gen1 = synthesize_translation_skeletons(tmp_path, "vi")
        assert len(gen1) == 1
        target_tl = tmp_path / "game" / "tl" / "vi" / "ch1.rpy"
        assert target_tl.is_file()

        # Modify existing translation file to simulate manual edits
        custom_content = "# User custom edit\ntranslate vi ch1_001:\n    alice \"Xin chào!\"\n"
        target_tl.write_text(custom_content, encoding="utf-8")

        # Synthesize second time
        gen2 = synthesize_translation_skeletons(tmp_path, "vi")
        assert target_tl in gen2
        # Verify content was NOT overwritten
        assert target_tl.read_text(encoding="utf-8") == custom_content
