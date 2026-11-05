"""Tier 5 White-Box Adversarial Coverage Hardening for Bytecode & Core Subsystems.

Empirical verification harness targeting:
1. renslate.renpy.rpyc:
   - Bytecode unpickler security: blocked dangerous modules, blocked dangerous builtins, safe collections, dynamic AST classes.
   - Deeply nested and malformed VirtualASTNode / VirtualList / VirtualDict / VirtualPyCode structures.
   - Slot 2 canonical hash calculation: multi-game paths, Windows paths, Unicode/emojis, whitespace labels.
   - 16-byte MD5 footer: valid, custom, invalid lengths, truncated streams, corrupt zlib payloads, multithreaded reads.
   - AST scene node extraction: active sprite state machine (scene, show, hide), consecutive scenes, SL2 screen gettext extraction.
   - Decompiler and skeleton synthesis edge cases.
2. renslate.core.config:
   - AppConfig environment variable resolution, precedence, and server_base_url stripping.
   - Config validation: target_lang, tl_dir path traversal, bounded numerics, bool-as-int rejection, unknown keys.
3. renslate.core.db:
   - SQLite migration v5, user_version pragma, normalized scenes columns.
   - Automated .db.v4.bak pre-migration backup for disk databases.
   - Backfill of normalized scenes from legacy meta (scene_ctx:*) records.
   - qa_audit_log CRUD, foreign key enforcement, and cascade deletion upon unit removal.
   - Concurrent multi-threaded writes in WAL mode with BEGIN IMMEDIATE.
   - Character dual-persistence bidirectional synchronization edge cases.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import threading
from pathlib import Path
from typing import Any

import pytest

from renslate.core.config import (
    AppConfig,
    Config,
)
from renslate.core.db import (
    Database,
    create_connection,
    get_schema_version,
    run_migrations,
    sync_characters,
)
from renslate.core.db.connection import backup_before_migration
from renslate.core.db.schema import MIGRATIONS
from renslate.core.errors import ConfigError
from renslate.core.models import (
    CharacterProfile,
    Origin,
    Status,
    Unit,
    UnitKind,
)
from renslate.renpy.rpyc import (
    RPYC2_HEADER,
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    Hide,
    Label,
    RenpyUnpickler,
    RpycFile,
    Say,
    Scene,
    Show,
    VirtualASTNode,
    VirtualDict,
    VirtualList,
    VirtualPyCode,
    VirtualPyExpr,
    compute_canonical_say_identifier,
    create_rpyc,
    decompile_rpyc,
    extract_rpyc_file,
    read_rpyc_data,
    synthesize_translation_skeletons,
)
from renslate.renpy.rpyc.extractor import _extract_from_sl_node
from renslate.renpy.rpyc.unpickler import _BLOCKED_MODULES, _SAFE_BUILTINS, _SAFE_COLLECTIONS

# ==============================================================================
# GROUP 1: Bytecode Unpickler Security & Virtual AST Robustness
# ==============================================================================


class TestTier5RpycBytecodeAndUnpickler:
    """Stress tests and white-box security verification for the Ren'Py unpickler."""

    @pytest.mark.parametrize("module_name", sorted(_BLOCKED_MODULES))
    def test_unpickler_blocks_all_dangerous_modules(self, module_name: str) -> None:
        """Every module listed in _BLOCKED_MODULES must be aggressively rejected."""
        raw_pickle = f"c{module_name}\nfunc\n)R.".encode()
        bio = io.BytesIO(raw_pickle)
        with pytest.raises(pickle.UnpicklingError, match="Blocked dangerous module"):
            RenpyUnpickler(bio).load()

    @pytest.mark.parametrize(
        "dangerous_builtin",
        ["eval", "exec", "__import__", "compile", "open", "input", "breakpoint", "getattr"],
    )
    def test_unpickler_blocks_dangerous_builtins(self, dangerous_builtin: str) -> None:
        """Dangerous built-in functions must be blocked from unpickling."""
        raw_pickle = f"cbuiltins\n{dangerous_builtin}\n)R.".encode()
        bio = io.BytesIO(raw_pickle)
        with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe builtin"):
            RenpyUnpickler(bio).load()

    def test_unpickler_allows_safe_builtins_and_collections(self) -> None:
        """Permitted safe builtins and collection types must unpickle smoothly."""
        for name, cls in _SAFE_BUILTINS.items():
            if name in (
                "NoneType",
                "Exception",
                "ValueError",
                "TypeError",
                "KeyError",
                "IndexError",
                "range",
                "xrange",
            ):
                continue
            inst = slice(0, 5, 1) if name == "slice" else cls()
            payload = pickle.dumps(inst, protocol=2)
            recovered = RenpyUnpickler(io.BytesIO(payload)).load()
            assert isinstance(recovered, cls)

        for _name, col_cls in _SAFE_COLLECTIONS.items():
            inst_col = col_cls()
            payload = pickle.dumps(inst_col, protocol=2)
            recovered = RenpyUnpickler(io.BytesIO(payload)).load()
            assert isinstance(recovered, col_cls)

    def test_unpickler_maps_renpy_container_classes_to_virtual_types(self) -> None:
        """Ren'Py engine container classes must map to Virtual AST containers."""
        from renslate.renpy.rpyc.ast_types import VirtualSet

        containers = [
            ("renpy.revertable", "RevertableList", VirtualList),
            ("renpy.revertable", "RollbackList", VirtualList),
            ("renpy.revertable", "CompressedList", VirtualList),
            ("renpy.revertable", "RevertableDict", VirtualDict),
            ("renpy.revertable", "RevertableDefaultDict", VirtualDict),
            ("renpy.revertable", "RevertableSet", VirtualSet),
            ("renpy.ast", "PyExpr", VirtualPyExpr),
            ("renpy.ast", "PyCode", VirtualPyCode),
        ]

        for mod, cls_name, expected_type in containers:
            unpickler = RenpyUnpickler(io.BytesIO(b""))
            resolved_cls = unpickler.find_class(mod, cls_name)
            assert issubclass(resolved_cls, expected_type) or resolved_cls is expected_type

    def test_unpickler_dynamic_ast_class_generation(self) -> None:
        """Unknown Ren'Py AST classes must instantiate dynamically without importing code."""
        unpickler = RenpyUnpickler(io.BytesIO(b""))
        custom_cls = unpickler.find_class("renpy.custom_subsystem", "CustomRenpyNode")
        assert issubclass(custom_cls, VirtualASTNode)
        inst = custom_cls(foo="bar", num=123)
        assert inst.foo == "bar"
        assert inst.num == 123
        assert inst.non_existent_attr is None

    def test_deeply_nested_virtual_ast_structures(self) -> None:
        """Virtual containers must tolerate deep nesting without memory corruption."""
        root: list[Any] = []
        current = root
        for _i in range(80):
            nxt: list[Any] = []
            current.append(nxt)
            current = nxt
        current.append("leaf_node")

        dumped = pickle.dumps(root, protocol=2)
        recovered = RenpyUnpickler(io.BytesIO(dumped)).load()
        assert isinstance(recovered, list)

        curr = recovered
        depth = 0
        while curr and isinstance(curr, list) and isinstance(curr[0], list):
            curr = curr[0]
            depth += 1
        assert depth == 80
        assert curr[0] == "leaf_node"

    def test_adversarial_python2_xrange_blocked(self) -> None:
        """Adversarial test demonstrating that legacy Python 2 Ren'Py bytecode using xrange is resolved."""
        xrange_pickle = b"c__builtin__\nxrange\n(K\x00K\x05K\x01tR."
        recovered = RenpyUnpickler(io.BytesIO(xrange_pickle)).load()
        assert list(recovered) == list(range(5))

    def test_adversarial_virtuallist_self_roundtrip(self) -> None:
        """Adversarial test demonstrating that RpycFile / RenpyUnpickler reloads VirtualList."""
        vl = VirtualList([1, 2, 3])
        dumped = pickle.dumps(vl, protocol=2)
        recovered = RenpyUnpickler(io.BytesIO(dumped)).load()
        assert recovered == [1, 2, 3]

    def test_adversarial_virtualdict_self_roundtrip(self) -> None:
        """Adversarial test demonstrating that RpycFile / RenpyUnpickler reloads VirtualDict."""
        vd = VirtualDict({"a": 1})
        dumped = pickle.dumps(vd, protocol=2)
        recovered = RenpyUnpickler(io.BytesIO(dumped)).load()
        assert recovered == {"a": 1}

    def test_virtual_ast_node_setstate_resilience(self) -> None:
        """VirtualASTNode.__setstate__ must tolerate non-standard states without crashing."""
        node = VirtualASTNode()

        # 1. None state
        node.__setstate__(None)

        # 2. Integer state
        node.__setstate__(42)

        # 3. Tuple of non-dicts
        node.__setstate__(("string", 123, None))

        # 4. Tuple containing valid dict and malformed loc
        node.__setstate__(
            (
                {"loc": ("script.rpy", "not_a_number")},
                {"custom_val": True},
            )
        )
        assert node.filename == "script.rpy"
        assert node.linenumber == 0
        assert node.custom_val is True

        # 5. Dunder attribute access raises AttributeError
        with pytest.raises(AttributeError):
            _ = node.__unknown_dunder__


# ==============================================================================
# GROUP 2: Canonical Hash Calculation & Path Edge Cases
# ==============================================================================


class TestTier5CanonicalSayIdentifier:
    """Stress tests for compute_canonical_say_identifier."""

    def test_paths_with_multiple_game_segments(self) -> None:
        """Path with multiple 'game' segments must slice at the true game root."""
        # E.g. /home/user/mygame/game/subgame/game/ch1.rpy
        path = "/home/user/mygame/game/subgame/game/ch1.rpy"
        ident = compute_canonical_say_identifier("start", path, 10, "Hello")
        expected_hash_input = b"game/ch1.rpy10Hello"
        import hashlib

        expected_digest = hashlib.md5(expected_hash_input).hexdigest()[:8]
        assert ident == f"start_{expected_digest}"

    def test_windows_mixed_slashes_and_drive_letters(self) -> None:
        """Windows paths with backslashes and drive letters must normalize identically."""
        win_path = r"D:\RenpyProjects\game\scripts\chapter1.rpy"
        posix_path = "D:/RenpyProjects/game/scripts/chapter1.rpy"
        id_win = compute_canonical_say_identifier("label_a", win_path, 42, "Text")
        id_posix = compute_canonical_say_identifier("label_a", posix_path, 42, "Text")
        assert id_win == id_posix
        assert id_win.startswith("label_a_")

    def test_game_root_parameter_when_game_dir_absent(self) -> None:
        """When path contains no 'game' folder, game_root parameter computes relative path."""
        file_path = "/workspace/project/custom_dir/story.rpy"
        game_root = "/workspace/project"
        ident = compute_canonical_say_identifier("ch1", file_path, 15, "Greetings", game_root=game_root)
        import hashlib

        expected = "ch1_" + hashlib.md5(b"custom_dir/story.rpy15Greetings").hexdigest()[:8]
        assert ident == expected

    def test_unicode_emojis_cjk_and_newlines(self) -> None:
        """Say text with complex Unicode, emojis, CJK, and newlines must hash deterministically."""
        text = "🌸 Xin chào! 「こんにちは」 ✨\nĐoạn văn có xuống dòng."
        id1 = compute_canonical_say_identifier("unicode_lbl", "game/script.rpy", 100, text)
        id2 = compute_canonical_say_identifier("unicode_lbl", "game/script.rpy", 100, text)
        assert id1 == id2
        assert len(id1) == len("unicode_lbl_") + 8

    def test_label_prefix_fallbacks(self) -> None:
        """Empty or None labels must default to 'global' prefix."""
        id_none = compute_canonical_say_identifier(None, "game/script.rpy", 1, "test")
        id_empty = compute_canonical_say_identifier("", "game/script.rpy", 1, "test")
        assert id_none.startswith("global_")
        assert id_empty.startswith("global_")
        assert id_none == id_empty


# ==============================================================================
# GROUP 3: 16-Byte MD5 Footer & Archive Serialization
# ==============================================================================


class TestTier5RpycArchiveAndMD5Footer:
    """Stress tests for RPYC archive creation, footer verification, and concurrent access."""

    def test_custom_md5_footer_round_trip(self, tmp_path: Path) -> None:
        """Custom 16-byte MD5 digest must persist and recover exactly."""
        payload = pickle.dumps(["test_stmt"], protocol=2)
        slots = {SLOT_STATEMENTS_PRE: payload}
        digest = b"ABCDEF0123456789"

        archive_bytes = create_rpyc(slots, rpydigest=digest)
        target = tmp_path / "custom_digest.rpyc"
        target.write_bytes(archive_bytes)

        with RpycFile(target) as rpyc:
            assert rpyc.read_md5() == digest
            assert rpyc.get_slot_1_statements() == ["test_stmt"]

    @pytest.mark.parametrize("bad_digest", [b"", b"short", b"123456789012345", b"12345678901234567"])
    def test_invalid_digest_length_raises_value_error(self, bad_digest: bytes) -> None:
        """create_rpyc must reject any rpydigest not strictly 16 bytes."""
        with pytest.raises(ValueError, match="16 bytes"):
            create_rpyc({1: b"data"}, rpydigest=bad_digest)

    def test_zero_byte_and_truncated_files(self, tmp_path: Path) -> None:
        """0-byte and truncated files must handle safely without unhandled exceptions."""
        empty_file = tmp_path / "empty.rpyc"
        empty_file.write_bytes(b"")

        with RpycFile(empty_file) as rpyc:
            assert rpyc.is_legacy is True
            assert rpyc.read_slot(SLOT_STATEMENTS_PRE) is None
            assert rpyc.read_slot(SLOT_STATEMENTS_POST) is None
            assert rpyc.read_md5() is None

        # Truncated header (starts with header but cut off at 12 bytes)
        trunc_file = tmp_path / "trunc.rpyc"
        trunc_file.write_bytes(RPYC2_HEADER + b"\x01\x00")
        with RpycFile(trunc_file) as rpyc:
            assert rpyc.is_legacy is False
            assert rpyc.read_slot(1) is None
            assert rpyc.read_md5() is None

    def test_corrupted_slot_zlib_stream(self, tmp_path: Path) -> None:
        """Corrupted zlib data in a slot must return None, not raise zlib.error."""
        valid_rpyc = bytearray(create_rpyc({SLOT_STATEMENTS_PRE: b"hello world"}))
        # Corrupt the compressed payload bytes
        valid_rpyc[-25:-16] = b"\xff\xff\xff\xff\xff\xff\xff\xff\xff"

        corrupt_file = tmp_path / "corrupt_zlib.rpyc"
        corrupt_file.write_bytes(valid_rpyc)

        with RpycFile(corrupt_file) as rpyc:
            result = rpyc.read_slot(SLOT_STATEMENTS_PRE)
            assert result is None

        # Also via read_rpyc_data
        bio = io.BytesIO(valid_rpyc)
        assert read_rpyc_data(bio, SLOT_STATEMENTS_PRE) is None

    def test_multithreaded_concurrent_rpyc_reads(self, tmp_path: Path) -> None:
        """Concurrent multi-threaded slot reads on the same RpycFile instance must be thread-safe."""
        data_slot1 = pickle.dumps(["dialogue_1", "dialogue_2"], protocol=2)
        data_slot2 = pickle.dumps(["ast_slot2"], protocol=2)
        rpyc_bytes = create_rpyc(
            {SLOT_STATEMENTS_PRE: data_slot1, SLOT_STATEMENTS_POST: data_slot2},
            rpydigest=b"1122334455667788",
        )
        rpyc_path = tmp_path / "concurrent.rpyc"
        rpyc_path.write_bytes(rpyc_bytes)

        rpyc_inst = RpycFile(rpyc_path)
        errors: list[Exception] = []

        def reader_task(tid: int) -> None:
            try:
                for _ in range(50):
                    s1 = rpyc_inst.read_slot(SLOT_STATEMENTS_PRE)
                    s2 = rpyc_inst.read_slot(SLOT_STATEMENTS_POST)
                    md5 = rpyc_inst.read_md5()
                    assert s1 == data_slot1
                    assert s2 == data_slot2
                    assert md5 == b"1122334455667788"
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader_task, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rpyc_inst.close()
        assert len(errors) == 0, f"Thread safety errors: {errors}"


# ==============================================================================
# GROUP 4: AST Scene Node Extraction & Screen UI Strings
# ==============================================================================


class TestTier5ASTSceneNodeAndScreenExtraction:
    """Stress tests for scene flow tracking, active sprites state machine, and screen parsing."""

    def test_active_sprites_state_machine_with_multiple_expressions(self, tmp_path: Path) -> None:
        """Verify active sprites register properly updates, replaces by base tag, and clears on scene."""
        lbl = Label()
        lbl.name = "hall"
        lbl.filename = "game/script.rpy"
        lbl.linenumber = 1

        sc1 = Scene()
        sc1.imspec = (("bg", "hall"), None, None, None)
        sc1.filename = "game/script.rpy"
        sc1.linenumber = 2

        # Show Alice happy
        sh_a1 = Show()
        sh_a1.imspec = (("alice", "happy"), None, None, None)
        sh_a1.filename = "game/script.rpy"
        sh_a1.linenumber = 3

        # Show Bob neutral at right
        sh_b = Show()
        sh_b.imspec = (("bob", "neutral"), None, None, ["right"])
        sh_b.filename = "game/script.rpy"
        sh_b.linenumber = 4

        # Dialogue 1: Both alice and bob present
        sy1 = Say()
        sy1.who = "alice"
        sy1.what = "Hi Bob!"
        sy1.filename = "game/script.rpy"
        sy1.linenumber = 5

        # Alice updates expression to angry at left
        sh_a2 = Show()
        sh_a2.imspec = (("alice", "angry"), None, None, ["left"])
        sh_a2.filename = "game/script.rpy"
        sh_a2.linenumber = 6

        # Dialogue 2: Alice expression updated; bob still neutral
        sy2 = Say()
        sy2.who = "bob"
        sy2.what = "Why so angry?"
        sy2.filename = "game/script.rpy"
        sy2.linenumber = 7

        # Hide Bob
        hd_b = Hide()
        hd_b.imspec = (("bob",), None, None, None)
        hd_b.filename = "game/script.rpy"
        hd_b.linenumber = 8

        # Dialogue 3: Only Alice active
        sy3 = Say()
        sy3.who = "alice"
        sy3.what = "He is gone."
        sy3.filename = "game/script.rpy"
        sy3.linenumber = 9

        # Scene transition: Clears all sprites
        sc2 = Scene()
        sc2.imspec = (("bg", "garden"), None, None, None)
        sc2.filename = "game/script.rpy"
        sc2.linenumber = 10

        # Dialogue 4: No active sprites
        sy4 = Say()
        sy4.who = None
        sy4.what = "Quiet garden."
        sy4.filename = "game/script.rpy"
        sy4.linenumber = 11

        lbl.block = [sc1, sh_a1, sh_b, sy1, sh_a2, sy2, hd_b, sy3, sc2, sy4]

        ast_data = pickle.dumps([lbl], protocol=2)
        rpyc_data = create_rpyc({SLOT_STATEMENTS_PRE: ast_data, SLOT_STATEMENTS_POST: ast_data})
        rpyc_file = tmp_path / "game" / "script.rpyc"
        rpyc_file.parent.mkdir(parents=True, exist_ok=True)
        rpyc_file.write_bytes(rpyc_data)

        script = extract_rpyc_file(rpyc_file)
        dialogues = script.dialogue
        assert len(dialogues) == 4

        # sy1: alice happy, bob neutral at right
        assert "alice happy" in dialogues[0].active_sprites
        assert "bob neutral at right" in dialogues[0].active_sprites
        assert len(dialogues[0].active_sprites) == 2

        # sy2: alice angry at left (replaces alice happy), bob neutral at right
        assert "alice angry at left" in dialogues[1].active_sprites
        assert "bob neutral at right" in dialogues[1].active_sprites
        assert "alice happy" not in dialogues[1].active_sprites
        assert len(dialogues[1].active_sprites) == 2

        # sy3: bob hidden, only alice angry at left remains
        assert dialogues[2].active_sprites == ("alice angry at left",)

        # sy4: garden scene wiped sprites
        assert dialogues[3].active_sprites == ()
        assert dialogues[3].scene_id == "bg garden"

    def test_screen_language_sl2_gettext_extraction(self) -> None:
        """SL2 screen nodes with _() and __() must extract translatable UI strings."""
        class DummySLDisplayable:
            def __init__(
                self,
                pos: list[Any] | None = None,
                kw: list[Any] | None = None,
                children: list[Any] | None = None,
            ) -> None:
                self.positional = pos or []
                self.keyword = kw or []
                self.children = children or []

        # Build SL tree: text _("Start Game"), textbutton __("Quit")
        leaf1 = DummySLDisplayable(pos=['_("Start Game")'])
        leaf2 = DummySLDisplayable(kw=[("text", '__("Quit")')])
        leaf3 = DummySLDisplayable(pos=[123, None, 'not_gettext']) # Ignored non-strings
        parent = DummySLDisplayable(children=[leaf1, leaf2, leaf3])

        out_strings: list[Any] = []
        _extract_from_sl_node(parent, "screens.rpy", 25, out_strings)

        texts = {s.text for s in out_strings}
        assert "Start Game" in texts
        assert "Quit" in texts
        assert "not_gettext" not in texts

    def test_decompile_and_skeleton_synthesis_robustness(self, tmp_path: Path) -> None:
        """Synthesizing translation skeletons and decompiling must tolerate unusual characters."""
        sy = Say()
        sy.who = "eileen"
        sy.what = 'Dialogue with "quotes" and {b}markup{/b}.'
        sy.filename = "game/script.rpy"
        sy.linenumber = 10
        sy.attributes = ("happy",)
        sy.with_ = "dissolve"

        ast_data = pickle.dumps([sy], protocol=2)
        rpyc_data = create_rpyc({SLOT_STATEMENTS_PRE: ast_data, SLOT_STATEMENTS_POST: ast_data})
        game_dir = tmp_path / "game_proj"
        rpyc_file = game_dir / "game" / "script.rpyc"
        rpyc_file.parent.mkdir(parents=True, exist_ok=True)
        rpyc_file.write_bytes(rpyc_data)

        # 1. Synthesize skeletons
        skeletons = synthesize_translation_skeletons(game_dir, "vietnamese")
        assert len(skeletons) >= 1
        skel_content = skeletons[0].read_text(encoding="utf-8")
        assert 'translate vietnamese' in skel_content
        assert 'eileen happy "Dialogue with \\"quotes\\" and {b}markup{/b}." with dissolve' in skel_content

        # 2. Decompile RPYC
        out_rpy = decompile_rpyc(rpyc_file)
        decompiled_text = out_rpy.read_text(encoding="utf-8")
        assert 'eileen happy' in decompiled_text
        assert 'with dissolve' in decompiled_text


# ==============================================================================
# GROUP 5: Centralized Configuration Hardening
# ==============================================================================


class TestTier5CentralizedConfigHardening:
    """Stress tests for AppConfig environment overrides and Config strict validation."""

    def test_appconfig_environment_variable_precedence_and_restoration(self) -> None:
        """AppConfig must resolve RENSLATE_* before RENVI_* and fall back to default."""
        # 1. Default state
        assert AppConfig.llm_api_base_url == "http://127.0.0.1:8080/v1"
        assert AppConfig.server_base_url == "http://127.0.0.1:8080"

        # 2. RENVI_* fallback
        os.environ["RENVI_LLM_URL"] = "http://10.0.0.1:5000/v1"
        assert AppConfig.llm_api_base_url == "http://10.0.0.1:5000/v1"
        assert AppConfig.server_base_url == "http://10.0.0.1:5000"

        # 3. RENSLATE_* takes precedence over RENVI_*
        os.environ["RENSLATE_LLM_URL"] = "https://llm.prod.corp:8443/v1/"
        assert AppConfig.llm_api_base_url == "https://llm.prod.corp:8443/v1/"
        assert AppConfig.server_base_url == "https://llm.prod.corp:8443"

        # Clean up
        del os.environ["RENSLATE_LLM_URL"]
        del os.environ["RENVI_LLM_URL"]

    @pytest.mark.parametrize(
        ("raw_url", "expected_server"),
        [
            ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080"),
            ("http://127.0.0.1:8080/v1/", "http://127.0.0.1:8080"),
            ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
            ("http://127.0.0.1:8080///", "http://127.0.0.1:8080"),
            ("https://api.openai.com/v1", "https://api.openai.com"),
        ],
    )
    def test_appconfig_server_base_url_stripping(self, raw_url: str, expected_server: str) -> None:
        """server_base_url and server_url must reliably strip /v1 and trailing slashes."""
        cfg = AppConfig(llm_api_base_url=raw_url)
        assert cfg.server_base_url == expected_server
        assert cfg.server_url == expected_server

    @pytest.mark.parametrize("invalid_lang", ["viet-nam", "123viet", "", "None", "lang with spaces", "tl/lang"])
    def test_config_rejects_invalid_target_lang(self, invalid_lang: str) -> None:
        """Target language must be a valid Ren'Py identifier and not 'None'."""
        with pytest.raises(ConfigError, match="không phải tên ngôn ngữ hợp lệ"):
            Config(target_lang=invalid_lang)

    @pytest.mark.parametrize("bad_tl_dir", ["/etc/passwd", "../tl", "game/../../secret", ""])
    def test_config_rejects_path_traversal_tl_dir(self, bad_tl_dir: str) -> None:
        """tl_dir must not escape the project directory."""
        with pytest.raises(ConfigError, match="đường dẫn tương đối nằm trong dự án"):
            Config(tl_dir=bad_tl_dir)

    def test_config_rejects_bool_for_numeric_fields(self) -> None:
        """Python booleans must not be accepted for numeric bounded fields."""
        # max_lines expects int; passing True must raise ConfigError
        raw_dict = {"batch": {"max_lines": True}}
        with pytest.raises(ConfigError, match="phải có kiểu int"):
            Config.from_dict(raw_dict)

    @pytest.mark.parametrize(
        ("section", "key", "val"),
        [
            ("batch", "max_lines", 0),  # min 1
            ("batch", "workers", 128),  # max 64
            ("llm", "temperature", -0.1),  # min 0.0
            ("llm", "temperature", 2.5),  # max 2.0
            ("llm", "max_retries", -1),  # min 0
            ("tm", "fuzzy_threshold", 1.5),  # max 1.0
            ("qa", "max_length_ratio", 0.5),  # min 1.0
        ],
    )
    def test_config_bounded_numeric_constraints(self, section: str, key: str, val: Any) -> None:
        """Out of bounds configuration values must be rejected."""
        raw_dict = {section: {key: val}}
        with pytest.raises(ConfigError, match="phải >=" if val < 1 else "phải <="):
            Config.from_dict(raw_dict)

    def test_config_unknown_keys_rejection(self) -> None:
        """Unexpected keys at root or sub-sections must raise ConfigError."""
        with pytest.raises(ConfigError, match="khóa không hợp lệ trong 'config'"):
            Config.from_dict({"unexpected_root_key": 123})

        with pytest.raises(ConfigError, match="khóa không hợp lệ trong 'llm'"):
            Config.from_dict({"llm": {"unsupported_option": "bad"}})


# ==============================================================================
# GROUP 6: SQLite Database Migration v5, QA Audit Log & Concurrency
# ==============================================================================


class TestTier5SQLiteDatabaseAndMigrationV5:
    """Stress tests for SQLite migrations, v5 normalized scenes, audit log, and WAL concurrency."""

    def test_fresh_database_creation_and_pragmas(self, tmp_path: Path) -> None:
        """Fresh database must execute migrations v1-v6 and enforce SQLite pragmas."""
        db_file = tmp_path / "fresh.db"
        with Database.open(db_file, create=True) as db:
            assert db.schema_version >= 6

            cur = db.execute("PRAGMA foreign_keys").fetchone()
            assert cur[0] == 1

            cur_journal = db.execute("PRAGMA journal_mode").fetchone()
            assert str(cur_journal[0]).lower() == "wal"

            cur_timeout = db.execute("PRAGMA busy_timeout").fetchone()
            assert cur_timeout[0] >= 30000

    def test_v4_to_v5_migration_with_pre_migration_backup(self, tmp_path: Path) -> None:
        """Migrating v4 disk database to v5 must create .db.v4.bak backup."""
        db_file = tmp_path / "legacy_v4.db"
        conn = create_connection(db_file)
        # Apply up to migration v4 only
        for target, script in enumerate(MIGRATIONS[:4], start=1):
            conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {target};\nCOMMIT;")
        assert get_schema_version(conn) == 4
        conn.close()

        # Migrate to v6
        conn_v5 = create_connection(db_file)
        run_migrations(conn_v5, db_path=db_file)
        assert get_schema_version(conn_v5) >= 6
        conn_v5.close()

        # Verify .db.v4.bak file existence and non-zero size
        bak_file = Path(f"{db_file}.v4.bak")
        assert bak_file.is_file()
        assert bak_file.stat().st_size > 0

        # Migrating already-at-v5 database must not recreate backup
        assert backup_before_migration(db_file, current_version=5, target_version=5) is None

    def test_migration_v5_backfill_scenes_from_meta(self, tmp_path: Path) -> None:
        """v5 migration must populate normalized scene columns from legacy meta records."""
        db_file = tmp_path / "meta_backfill.db"
        conn = create_connection(db_file)
        for target, script in enumerate(MIGRATIONS[:4], start=1):
            conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {target};\nCOMMIT;")

        # Insert legacy scene record in scenes and corresponding meta
        conn.execute(
            "INSERT INTO scenes (scene_id, label, first_seq, last_seq, summary) VALUES (?, ?, ?, ?, ?)",
            ("scene_intro", "start", 1, 50, "Introductory narrative"),
        )
        meta_val = json.dumps(
            {
                "location": "Royal Ballroom",
                "mood": "Tense",
                "atmosphere": "Opulent but ominous",
                "characters": ["King", "Queen", "Knight"],
                "time": "Evening",
                "previous_summary": "The party gathered.",
            }
        )
        conn.execute("INSERT INTO meta (key, value) VALUES ('scene_ctx:scene_intro', ?)", (meta_val,))
        conn.commit()
        conn.close()

        # Run migration 5
        conn2 = create_connection(db_file)
        run_migrations(conn2, db_path=db_file)
        row = conn2.execute(
            "SELECT location, mood, atmosphere, characters_json, temporal_anchor, previous_summary "
            "FROM scenes WHERE scene_id = 'scene_intro'"
        ).fetchone()

        assert row["location"] == "Royal Ballroom"
        assert row["mood"] == "Tense"
        assert row["atmosphere"] == "Opulent but ominous"
        assert json.loads(row["characters_json"]) == ["King", "Queen", "Knight"]
        assert row["temporal_anchor"] == "Evening"
        assert row["previous_summary"] == "The party gathered."

        # Original meta record must be preserved for zero data loss
        meta_row = conn2.execute("SELECT value FROM meta WHERE key = 'scene_ctx:scene_intro'").fetchone()
        assert meta_row is not None
        conn2.close()

    def test_qa_audit_log_crud_and_foreign_key_cascade(self, tmp_path: Path) -> None:
        """qa_audit_log must enforce foreign keys and cascade delete when unit is removed."""
        db_file = tmp_path / "qa_audit.db"
        with Database.open(db_file, create=True) as db:
            # 1. Foreign key failure: Inserting audit log for non-existent unit raises IntegrityError
            import sqlite3
            with pytest.raises(sqlite3.IntegrityError):
                db.log_qa_audit("non_existent_unit", stage="scan", verdict="FAIL")

            # 2. Insert valid unit
            unit = Unit(
                uid="say:script.rpy:1",
                kind=UnitKind.DIALOGUE,
                seq=1,
                ref="script.rpy:10",
                speaker="lucy",
                label="intro",
                source="Hello world",
                target="Chào thế giới",
                status=Status.TRANSLATED,
                origin=Origin.LLM,
            )
            db.save_unit(unit)

            # 3. Insert multiple audit logs
            id1 = db.log_qa_audit(
                unit_uid=unit.uid,
                stage="hallucination_guard",
                verdict="FAIL",
                issue_type="SEMANTIC_INVERSION",
                explanation="Meaning was inverted",
                suggested_correction="Chào thế giới",
                confidence=0.88,
            )
            id2 = db.log_qa_audit(
                unit_uid=unit.uid,
                stage="repair",
                verdict="PASS",
                confidence=1.0,
            )
            assert id1 > 0 and id2 > id1

            logs = db.get_qa_audit_logs(unit.uid)
            assert len(logs) == 2
            assert logs[0]["verdict"] == "FAIL"
            assert logs[0]["issue_type"] == "SEMANTIC_INVERSION"
            assert logs[1]["verdict"] == "PASS"

            # 4. Cascade delete
            db.execute("DELETE FROM units WHERE uid = ?", (unit.uid,))
            logs_remaining = db.get_qa_audit_logs(unit.uid)
            assert len(logs_remaining) == 0

    def test_sqlite_multithreaded_wal_write_stress(self, tmp_path: Path) -> None:
        """8 concurrent worker threads writing via BEGIN IMMEDIATE must execute without SQLITE_BUSY."""
        db_file = tmp_path / "concurrent_wal.db"
        with Database.open(db_file, create=True) as init_db:
            init_db.execute("SELECT 1")

        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                conn = create_connection(db_file)
                db = Database(conn, path=db_file)
                for i in range(15):
                    uid = f"worker_{worker_id}_unit_{i}"
                    u = Unit(
                        uid=uid,
                        kind=UnitKind.DIALOGUE,
                        seq=worker_id * 100 + i,
                        ref="script.rpy:1",
                        speaker="worker",
                        label="start",
                        source=f"Message {i} from worker {worker_id}",
                        target=f"Bản dịch {i}",
                        status=Status.TRANSLATED,
                        origin=Origin.LLM,
                    )
                    with db.transaction():
                        db.save_unit(u)
                        db.log_qa_audit(
                            unit_uid=uid,
                            stage="worker_test",
                            verdict="PASS",
                            confidence=1.0,
                        )
                db.close()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent database errors: {errors}"

        # Verify all 120 units and audit logs written
        with Database.open(db_file) as verify_db:
            all_units = verify_db.all_units()
            assert len(all_units) == 8 * 15
            for u in all_units:
                audit = verify_db.get_qa_audit_logs(u.uid)
                assert len(audit) == 1

    def test_character_dual_persistence_sync_edge_cases(self, tmp_path: Path) -> None:
        """sync_characters must handle missing JSON, conflicting updates, and field merging."""
        db_file = tmp_path / "char_sync.db"
        json_file = tmp_path / "characters.json"

        with Database.open(db_file, create=True) as db:
            # 1. Missing JSON file with existing DB character -> auto exports to JSON
            prof1 = CharacterProfile(
                tag="hero",
                display_name="Hero",
                gender="male",
                role="protagonist",
            )
            db.save_character(prof1)
            count_db, count_json = sync_characters(db, json_file, direction="auto")
            assert count_db == 1 and count_json == 1
            assert json_file.is_file()

            # 2. Modify JSON directly (adding archetype and tone)
            data = json.loads(json_file.read_text(encoding="utf-8"))
            data["hero"]["archetype"] = "Chosen One"
            data["hero"]["tone_and_voice"] = "Determined"
            # Add secondary character in JSON
            data["mentor"] = {
                "display_name": "Wise Elder",
                "gender": "male",
                "role": "mentor",
            }
            json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

            # Sync auto merges updates from JSON into DB
            count_merged_db, _ = sync_characters(db, json_file, direction="auto")
            assert count_merged_db == 2

            chars = db.get_characters()
            assert chars["hero"].archetype == "Chosen One"
            assert chars["hero"].tone_and_voice == "Determined"
            assert "mentor" in chars
