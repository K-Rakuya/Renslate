from __future__ import annotations

import io
import pickle
from pathlib import Path

import pytest

from renslate.renpy.rpyc import (
    RPYC2_HEADER,
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    Hide,
    Label,
    RpycFile,
    Say,
    Scene,
    Show,
    compute_canonical_say_identifier,
    create_rpyc,
    extract_rpyc_file,
    format_say_statement,
    read_rpyc_data,
    write_rpyc_data,
)


class TestEmpiricalRPYC:
    def test_round_trip_serialization_and_16byte_md5_footer(self, tmp_path: Path) -> None:
        """
        Verify round-trip serialization: create an .rpyc file using create_rpyc,
        verify the 16-byte MD5 footer at the end of the file.
        """
        raw_payload = pickle.dumps(["stmt1", "stmt2"], protocol=2)
        slots = {SLOT_STATEMENTS_PRE: raw_payload}
        custom_digest = b"1234567890abcdef"

        # 1. create_rpyc with custom digest
        data = create_rpyc(slots, rpydigest=custom_digest)
        assert data.startswith(RPYC2_HEADER)
        assert len(data) >= len(RPYC2_HEADER) + 24 + 16
        # Footer must be exactly custom_digest
        assert data[-16:] == custom_digest

        # Verify parsing via RpycFile
        test_file = tmp_path / "test_custom.rpyc"
        test_file.write_bytes(data)
        with RpycFile(test_file) as rpyc:
            assert rpyc.read_md5() == custom_digest
            recovered_slot1 = rpyc.read_slot(SLOT_STATEMENTS_PRE)
            assert recovered_slot1 == raw_payload
            stmts = rpyc.get_slot_1_statements()
            assert stmts == ["stmt1", "stmt2"]

        # 2. create_rpyc with default digest (None -> 16 zero bytes)
        data_default = create_rpyc(slots, rpydigest=None)
        assert data_default[-16:] == b"\x00" * 16
        test_file_default = tmp_path / "test_default.rpyc"
        test_file_default.write_bytes(data_default)
        with RpycFile(test_file_default) as rpyc:
            assert rpyc.read_md5() == b"\x00" * 16
            assert rpyc.read_slot(SLOT_STATEMENTS_PRE) == raw_payload

        # 3. Invalid digest length must raise ValueError
        with pytest.raises(ValueError, match="16 bytes"):
            create_rpyc(slots, rpydigest=b"short")

        with pytest.raises(ValueError, match="16 bytes"):
            create_rpyc(slots, rpydigest=b"this_is_too_long_digest_1234567890")

        # 4. read_rpyc_data and write_rpyc_data
        out_buf = io.BytesIO()
        write_rpyc_data(out_buf, slots, rpydigest=custom_digest)
        out_bytes = out_buf.getvalue()
        assert out_bytes[-16:] == custom_digest

        read_back = read_rpyc_data(io.BytesIO(out_bytes), SLOT_STATEMENTS_PRE)
        assert read_back == raw_payload

    def test_format_say_statement_grammar_ordering(self) -> None:
        """
        Verify that `with <with_>` is placed before `id <identifier>`.
        Ren'Py grammar:
        [ who ] [ attrs ] [ "@" temp ] string [ "nointeract" ] [ args ] [ "with" expr ] [ "id" name ]
        """
        class DummyArgs:
            def get_code(self) -> str:
                return "(arg1=True)"

        node = Say()
        node.who = "eileen"
        node.what = "Hello, world!"
        node.attributes = ("happy", "blush")
        node.temporary_attributes = ("surprised",)
        node.interact = False
        node.arguments = DummyArgs()
        node.with_ = "dissolve"
        node.identifier = "start_1a2b3c4d"
        node.explicit_identifier = True

        formatted = format_say_statement(node)
        expected = 'eileen happy blush @ surprised "Hello, world!" nointeract (arg1=True) with dissolve id start_1a2b3c4d'
        assert formatted == expected

        # Check relative index: with dissolve must precede id start_1a2b3c4d
        idx_with = formatted.index("with dissolve")
        idx_id = formatted.index("id start_1a2b3c4d")
        assert idx_with < idx_id

        # Verify translated_text override
        formatted_trans = format_say_statement(node, translated_text="Xin chào!")
        assert 'Xin chào!' in formatted_trans
        assert formatted_trans.endswith("with dissolve id start_1a2b3c4d")

        # Verify without explicit_identifier, id is omitted even if node.identifier is present
        node_no_explicit = Say()
        node_no_explicit.who = "lucy"
        node_no_explicit.what = "Hi"
        node_no_explicit.with_ = "fade"
        node_no_explicit.identifier = "secret_id"
        node_no_explicit.explicit_identifier = False
        res = format_say_statement(node_no_explicit)
        assert res == 'lucy "Hi" with fade'
        assert "id" not in res

    def test_compute_canonical_say_identifier_baseline_paths(self) -> None:
        """
        Verify standard path normalization relative to `game/`.
        """
        # Case 1: Standard clean path containing 'game/' with forward slash
        path1 = "/home/user/RenpyGame/game/scripts/ch1.rpy"
        id1 = compute_canonical_say_identifier("start", path1, 42, "Welcome!")
        assert id1.startswith("start_")
        assert len(id1) == len("start_") + 8

        # Case 2: Standard clean path containing 'game\\' with backslash (Windows style)
        path2 = r"C:\Games\RenpyProject\game\scripts\ch1.rpy"
        id2 = compute_canonical_say_identifier("start", path2, 42, "Welcome!")
        assert id1 == id2

        # Case 3: None or empty label defaults to "global"
        id_global = compute_canonical_say_identifier(None, "game/main.rpy", 100, "Text")
        assert id_global.startswith("global_")
        id_empty = compute_canonical_say_identifier("", "game/main.rpy", 100, "Text")
        assert id_empty.startswith("global_")

    def test_extract_scene_flow_and_active_sprite_register(self, tmp_path: Path) -> None:
        """
        Verify extraction of SceneFlowItem with active character sprite register:
        - scene clears active_sprites
        - show adds/updates sprite by base tag
        - hide removes sprite by base tag
        - say inherits current active_sprites and current_scene and current_label
        """
        lbl = Label()
        lbl.name = "chapter_1"
        lbl.filename = "game/script.rpy"
        lbl.linenumber = 10

        sc = Scene()
        sc.imspec = (("bg", "room"), None, None, None)
        sc.filename = "game/script.rpy"
        sc.linenumber = 11

        sh1 = Show()
        sh1.imspec = (("eileen", "happy"), None, None, None)
        sh1.filename = "game/script.rpy"
        sh1.linenumber = 12

        sh2 = Show()
        sh2.imspec = (("bob", "neutral"), None, None, None)
        sh2.filename = "game/script.rpy"
        sh2.linenumber = 13

        sy1 = Say()
        sy1.who = "eileen"
        sy1.what = "Hi Bob!"
        sy1.filename = "game/script.rpy"
        sy1.linenumber = 14

        # Eileen changes expression (same base tag 'eileen')
        sh3 = Show()
        sh3.imspec = (("eileen", "surprised"), None, None, None)
        sh3.filename = "game/script.rpy"
        sh3.linenumber = 15

        sy2 = Say()
        sy2.who = "bob"
        sy2.what = "What happened?"
        sy2.filename = "game/script.rpy"
        sy2.linenumber = 16

        # Hide bob
        hd = Hide()
        hd.imspec = (("bob",), None, None, None)
        hd.filename = "game/script.rpy"
        hd.linenumber = 17

        sy3 = Say()
        sy3.who = "eileen"
        sy3.what = "He left."
        sy3.filename = "game/script.rpy"
        sy3.linenumber = 18

        # New scene clears everything
        sc2 = Scene()
        sc2.imspec = (("bg", "corridor"), None, None, None)
        sc2.filename = "game/script.rpy"
        sc2.linenumber = 19

        sy4 = Say()
        sy4.who = None
        sy4.what = "Now in the corridor."
        sy4.filename = "game/script.rpy"
        sy4.linenumber = 20

        lbl.block = [sc, sh1, sh2, sy1, sh3, sy2, hd, sy3, sc2, sy4]

        # Package into RPYC
        ast_payload = pickle.dumps([lbl], protocol=2)
        rpyc_data = create_rpyc({SLOT_STATEMENTS_PRE: ast_payload, SLOT_STATEMENTS_POST: ast_payload})
        rpyc_file = tmp_path / "game" / "script.rpyc"
        rpyc_file.parent.mkdir(parents=True, exist_ok=True)
        rpyc_file.write_bytes(rpyc_data)

        # Extract
        script = extract_rpyc_file(rpyc_file)
        flow = script.scene_flow
        dialogue = script.dialogue

        assert len(dialogue) == 4

        # Check dialogue 1: sy1
        assert dialogue[0].who == "eileen"
        assert dialogue[0].what == "Hi Bob!"
        assert dialogue[0].scene_id == "bg room"
        assert dialogue[0].current_label == "chapter_1"
        assert set(dialogue[0].active_sprites) == {"eileen happy", "bob neutral"}

        # Check dialogue 2: sy2 (eileen updated to surprised)
        assert dialogue[1].who == "bob"
        assert dialogue[1].what == "What happened?"
        assert set(dialogue[1].active_sprites) == {"eileen surprised", "bob neutral"}
        assert len(dialogue[1].active_sprites) == 2

        # Check dialogue 3: sy3 (bob hidden)
        assert dialogue[2].who == "eileen"
        assert dialogue[2].what == "He left."
        assert dialogue[2].active_sprites == ("eileen surprised",)

        # Check dialogue 4: sy4 (new scene clears sprites)
        assert dialogue[3].who is None
        assert dialogue[3].what == "Now in the corridor."
        assert dialogue[3].scene_id == "bg corridor"
        assert dialogue[3].active_sprites == ()

        # Check scene flow items
        kinds = [item.kind for item in flow]
        assert kinds == ["label", "scene", "show", "show", "say", "show", "say", "hide", "say", "scene", "say"]


class TestEmpiricalChallengerAdversarialFindings:
    """
    Stress-testing and failure mode discovery for Milestone 2.
    """

    @pytest.mark.xfail(
        reason="BUG: norm_path.index('game/') slices at substring 'game/' inside parent dir name (e.g. 'mygame/'), yielding 'game/game/script.rpy' and mismatched canonical hash."
    )
    def test_adversarial_path_with_game_suffix_directory(self) -> None:
        """
        Adversarial test demonstrating that directories ending with 'game'
        (e.g. /home/user/mygame/game/script.rpy or renpy_game/game/script.rpy)
        corrupt the relative path slice to 'game/game/script.rpy' instead of 'game/script.rpy'.
        """
        expected = compute_canonical_say_identifier("start", "game/script.rpy", 10, "Hello")
        # In a real project named 'renpy_game' or 'mygame'
        adversarial_path = "/home/user/renpy_game/game/script.rpy"
        actual = compute_canonical_say_identifier("start", adversarial_path, 10, "Hello")
        assert actual == expected

    @pytest.mark.xfail(
        reason="BUG: read_rpyc_data(BinaryIO) does not wrap zlib.decompress in try-except, raising zlib.error on corrupt streams instead of returning None like RpycFile."
    )
    def test_adversarial_read_rpyc_data_corrupt_zlib_stream(self) -> None:
        """
        Adversarial test demonstrating that read_rpyc_data raises unhandled zlib.error
        when passed a BinaryIO source with corrupted slot data, diverging from RpycFile.read_slot.
        """
        corrupt_rpyc = bytearray(create_rpyc({1: b"valid payload"}))
        corrupt_rpyc[40:50] = b"\xff" * 10
        bio = io.BytesIO(corrupt_rpyc)
        # Should safely return None (like RpycFile does)
        result = read_rpyc_data(bio, 1)
        assert result is None
