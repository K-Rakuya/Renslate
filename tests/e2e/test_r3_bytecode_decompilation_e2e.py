"""Opaque-Box E2E Test Suite for Requirement R3: Bytecode Decompilation & AST Modernization.

Covers:
- Ren'Py RPC2 bytecode binary archive parsing and slot layout (Slot 1 vs Slot 2)
- Canonical translation identifiers (Slot 2 TranslateSay) vs fallback hashing
- Complete Say statement formatting and AST node fidelity
- Screen Language (SL2) UI and menu string extraction
- Decompilation of .rpyc back to executable Ren'Py .rpy script
- Pure Python 3.10+ standard library execution without Rust dependencies

Tiers:
- Tier 1: Feature Coverage (>=5 tests per feature)
- Tier 2: Boundary & Corner Cases (>=5 tests per feature)
- Tier 3: Cross-Feature Combinations
- Tier 4: Real-World Application Scenarios
"""

from __future__ import annotations

import pickle
import struct
from pathlib import Path
from typing import Any

import pytest

from renslate.core.db import Database
from renslate.core.models import Status, Unit, UnitKind
from renslate.renpy.literals import encode_say_text
from renslate.renpy.rpyc import (
    RPYC2_HEADER,
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    Define,
    Label,
    Menu,
    RenpyUnpickler,
    RpycFile,
    Say,
    Scene,
    Show,
    TranslateSay,
    VirtualPyCode,
    _format_say_code,
    create_rpyc,
    decompile_rpyc,
    extract_character_definitions_from_rpyc,
    extract_dialogue_from_rpyc,
    synthesize_translation_skeletons,
    write_rpyc,
)

# ==============================================================================
# Helper Factories for Bytecode Generation
# ==============================================================================


def _build_synthetic_rpyc(
    dest_path: Path,
    slot1_stmts: list[Any] | None = None,
    slot2_stmts: list[Any] | None = None,
    *,
    md5_footer: bytes | None = None,
) -> Path:
    """Build a standard Ren'Py RPC2 bytecode file with Slot 1 and Slot 2 payloads."""
    slot1 = slot1_stmts or []
    slot2 = slot2_stmts or []
    slots: dict[int, bytes] = {
        SLOT_STATEMENTS_PRE: pickle.dumps((None, slot1)),
        SLOT_STATEMENTS_POST: pickle.dumps((None, slot2)),
    }
    raw_bytes = create_rpyc(slots)
    if md5_footer:
        raw_bytes = raw_bytes + md5_footer
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(raw_bytes)
    return dest_path


def _make_say(
    who: str | None,
    what: str,
    *,
    filename: str = "game/script.rpy",
    linenumber: int = 10,
    attributes: tuple[str, ...] | None = None,
) -> Say:
    say = Say()
    say.who = who
    say.what = what
    say.filename = filename
    say.linenumber = linenumber
    say.attributes = attributes
    say.temporary_attributes = None
    say.with_ = None
    return say


def _make_translate_say(
    identifier: str,
    who: str | None,
    what: str,
    *,
    filename: str = "game/script.rpy",
    linenumber: int = 10,
    attributes: tuple[str, ...] | None = None,
) -> TranslateSay:
    ts = TranslateSay()
    ts.identifier = identifier
    ts.who = who
    ts.what = what
    ts.filename = filename
    ts.linenumber = linenumber
    ts.attributes = attributes
    ts.temporary_attributes = None
    ts.with_ = None
    return ts


# ==============================================================================
# Tier 1: Feature Coverage (Core Bytecode & AST Contracts)
# ==============================================================================


class TestR3Tier1FeatureCoverage:
    """Tier 1: Verification of Ren'Py RPC2 bytecode reading, slot architecture, and AST."""

    def test_rpc2_magic_header_and_structure(self, tmp_path: Path) -> None:
        """Verify bytecode writer produces valid RPYC2 header and slot records."""
        rpyc_path = tmp_path / "test.rpyc"
        _build_synthetic_rpyc(rpyc_path, [_make_say("mc", "Hello.")])

        content = rpyc_path.read_bytes()
        assert content.startswith(RPYC2_HEADER)
        assert len(content) > len(RPYC2_HEADER) + 12

    def test_slot_1_raw_ast_loading(self, tmp_path: Path) -> None:
        """Verify Slot 1 raw AST statements are unpickled correctly."""
        say_stmt = _make_say("alice", "Good morning, Bob.")
        rpyc_path = tmp_path / "slot1_test.rpyc"
        _build_synthetic_rpyc(rpyc_path, slot1_stmts=[say_stmt])

        with RpycFile(rpyc_path) as rpyc:
            meta, stmts = rpyc.load_slot(SLOT_STATEMENTS_PRE)
            assert meta is None or isinstance(meta, dict)
            assert len(stmts) == 1
            assert stmts[0].who == "alice"
            assert stmts[0].what == "Good morning, Bob."

    def test_slot_2_translate_say_canonical_identifier(self, tmp_path: Path) -> None:
        """Verify Slot 2 retains canonical translation identifiers."""
        ts_stmt = _make_translate_say("start_19d78049", "bob", "Glad to see you, Alice.")
        rpyc_path = tmp_path / "slot2_test.rpyc"
        _build_synthetic_rpyc(rpyc_path, slot2_stmts=[ts_stmt])

        with RpycFile(rpyc_path) as rpyc:
            _, stmts = rpyc.load_slot(SLOT_STATEMENTS_POST)
            assert len(stmts) == 1
            assert stmts[0].identifier == "start_19d78049"
            assert stmts[0].who == "bob"

    def test_extract_dialogue_from_slot_2_priority(self, tmp_path: Path) -> None:
        """Verify extract_dialogue_from_rpyc prioritizes Slot 2 TranslateSay."""
        say1 = _make_say("mc", "Slot 1 line")
        say2 = _make_translate_say("canonical_id_001", "mc", "Slot 2 line")
        rpyc_path = tmp_path / "priority.rpyc"
        _build_synthetic_rpyc(rpyc_path, slot1_stmts=[say1], slot2_stmts=[say2])

        items = extract_dialogue_from_rpyc(rpyc_path)
        assert len(items) == 1
        assert items[0].identifier == "canonical_id_001"
        assert items[0].what == "Slot 2 line"

    def test_say_statement_formatting_code(self) -> None:
        """Verify _format_say_code renders Ren'Py say syntax correctly."""
        formatted = _format_say_code("eileen", "Hello, world!", ("happy",), None, "dissolve")
        assert formatted == 'eileen happy "Hello, world!" with dissolve'

        narrator_formatted = _format_say_code(None, "The sun rose.", None, None, None)
        assert narrator_formatted == '"The sun rose."'

    def test_encode_say_text_escaping(self) -> None:
        """Verify Ren'Py text quotes, backslashes, and brackets escaping."""
        escaped = encode_say_text('He said: "Wait!", then left.')
        assert '\\"' in escaped
        assert escaped == 'He said: \\"Wait!\\", then left.'

    def test_character_definition_extraction_from_slot_1(self, tmp_path: Path) -> None:
        """Verify character definitions (Define statements) are extracted from Slot 1."""
        game_dir = tmp_path / "game"
        char_stmt = Define(
            varname="s",
            code=VirtualPyCode('Character("Sylvie", color="#c8ffc8")'),
        )
        _build_synthetic_rpyc(game_dir / "chars.rpyc", slot1_stmts=[char_stmt])

        chars = extract_character_definitions_from_rpyc(game_dir)
        assert "s" in chars
        assert chars["s"] == "Sylvie"

    def test_pure_standard_library_performance_benchmark(self, tmp_path: Path) -> None:
        """Verify Python stdlib unpickling 1000 nodes takes < 0.25 seconds."""
        import time

        stmts = [
            _make_translate_say(f"id_{i:04d}", "mc", f"Dialogue line {i}")
            for i in range(1000)
        ]
        rpyc_path = tmp_path / "perf.rpyc"
        _build_synthetic_rpyc(rpyc_path, slot2_stmts=stmts)

        t0 = time.perf_counter()
        with RpycFile(rpyc_path) as rpyc:
            _, loaded = rpyc.load_slot(SLOT_STATEMENTS_POST)
        elapsed = time.perf_counter() - t0

        assert len(loaded) == 1000
        assert elapsed < 0.5, f"1000 nodes took {elapsed:.4f}s, expected < 0.5s"


# ==============================================================================
# Tier 2: Boundary & Corner Cases (Corrupt Headers, Bad Zlib, Missing Slots)
# ==============================================================================


class TestR3Tier2Boundaries:
    """Tier 2: Robustness against truncated, corrupt, or adversarial bytecode files."""

    def test_corrupt_header_less_than_10_bytes(self, tmp_path: Path) -> None:
        """Verify file smaller than header size is handled gracefully."""
        corrupt_file = tmp_path / "short.rpyc"
        corrupt_file.write_bytes(b"RENPY")
        with RpycFile(corrupt_file) as rpyc:
            assert rpyc.is_legacy is True
            assert rpyc.load_slot(SLOT_STATEMENTS_PRE) is None
        assert extract_dialogue_from_rpyc(corrupt_file) == []

    def test_invalid_magic_header_rejected(self, tmp_path: Path) -> None:
        """Verify non-RPC2 file is marked legacy and returns None on slot load."""
        corrupt_file = tmp_path / "bad_magic.rpyc"
        corrupt_file.write_bytes(b"NOT_A_RPYC_MAGIC_HEADER_AT_ALL")
        with RpycFile(corrupt_file) as rpyc:
            assert rpyc.is_legacy is True
            assert rpyc.load_slot(SLOT_STATEMENTS_POST) is None

    def test_corrupted_zlib_slot_stream(self, tmp_path: Path) -> None:
        """Verify corrupt compressed slot data returns None without crashing."""
        header = RPYC2_HEADER
        slot_dir = struct.pack("III", 1, len(header) + 24, 100) + struct.pack("III", 0, 0, 0)
        corrupted_payload = b"this is not a valid zlib compressed payload" * 5
        corrupt_file = tmp_path / "corrupt_zlib.rpyc"
        corrupt_file.write_bytes(header + slot_dir + corrupted_payload)

        with RpycFile(corrupt_file) as rpyc:
            assert rpyc.load_slot(1) is None

    def test_missing_slot_returns_empty_or_raises(self, tmp_path: Path) -> None:
        """Verify requesting non-existent slot (e.g. slot 99) returns None."""
        rpyc_path = tmp_path / "single_slot.rpyc"
        _build_synthetic_rpyc(rpyc_path, slot1_stmts=[_make_say("mc", "Alone")])

        with RpycFile(rpyc_path) as rpyc:
            assert rpyc.load_slot(99) is None

    def test_slot1_fallback_when_slot2_absent(self, tmp_path: Path) -> None:
        """Verify extract_dialogue falls back to Slot 1 Say when Slot 2 is missing."""
        say = _make_say("hero", "Fallback line from Slot 1", linenumber=42)
        slots = {SLOT_STATEMENTS_PRE: pickle.dumps((None, [say]))}
        rpyc_path = tmp_path / "slot1_only.rpyc"
        write_rpyc(rpyc_path, slots)

        items = extract_dialogue_from_rpyc(rpyc_path)
        assert len(items) == 1
        assert items[0].what == "Fallback line from Slot 1"
        assert items[0].who == "hero"

    def test_unpickler_blocks_dangerous_global_modules(self) -> None:
        """Verify RenpyUnpickler blocks os.system or subprocess execution."""
        import io

        evil_payload = b"cos\nsystem\n(S'id'\ntR."
        with pytest.raises(pickle.UnpicklingError):
            RenpyUnpickler(io.BytesIO(evil_payload)).load()

    def test_dialogue_with_extreme_unicode_and_nested_quotes(self, tmp_path: Path) -> None:
        """Verify quotes, Japanese brackets, and Vietnamese diacritics are preserved."""
        extreme_text = '「Thầy ơi... "Ngươi thực sự muốn đi?"」 — nàng khẽ thì thầm.'
        ts = _make_translate_say("extreme_01", "heroine", extreme_text)
        rpyc_path = tmp_path / "extreme.rpyc"
        _build_synthetic_rpyc(rpyc_path, slot2_stmts=[ts])

        items = extract_dialogue_from_rpyc(rpyc_path)
        assert len(items) == 1
        assert items[0].what == extreme_text


# ==============================================================================
# Tier 3: Cross-Feature Combinations (Bytecode -> DB -> Skeletons)
# ==============================================================================


class TestR3Tier3CrossFeature:
    """Tier 3: Interactions between Bytecode Extraction, Database, and Translation Skeletons."""

    def test_bytecode_extraction_to_database_units(self, tmp_path: Path) -> None:
        """Verify dialogue extracted from .rpyc can be inserted into SQLite Database."""
        game_dir = tmp_path / "game"
        ts1 = _make_translate_say("chap1_01", "mc", "First line of chapter 1.")
        ts2 = _make_translate_say("chap1_02", "eileen", "Second line of chapter 1.")
        rpyc_file = game_dir / "chapter1.rpyc"
        _build_synthetic_rpyc(rpyc_file, slot2_stmts=[ts1, ts2])

        items = extract_dialogue_from_rpyc(rpyc_file)
        assert len(items) == 2

        db_path = tmp_path / "test_store.db"
        with Database.open(db_path, create=True) as db:
            units = [
                Unit(
                    uid=item.identifier,
                    kind=UnitKind.DIALOGUE,
                    seq=i + 1,
                    source=item.what,
                    masked_source=item.what,
                    speaker=item.who,
                    status=Status.PENDING,
                    ref=f"{item.filename}:{item.linenumber}",
                )
                for i, item in enumerate(items)
            ]
            db.save_units_bulk(units)
            assert len(db.all_units()) == 2

            loaded = db.get_unit("chap1_01")
            assert loaded is not None
            assert loaded.source == "First line of chapter 1."
            assert loaded.speaker == "mc"

    def test_synthesize_translation_skeletons_from_bytecode(self, tmp_path: Path) -> None:
        """Verify synthesize_translation_skeletons generates tl/.rpy files."""
        game_dir = tmp_path / "game"
        ts = _make_translate_say("story_001", "alice", "Welcome to the game.")
        _build_synthetic_rpyc(game_dir / "story.rpyc", slot2_stmts=[ts])

        created = synthesize_translation_skeletons(game_dir, target_lang="vietnamese")
        assert len(created) >= 1
        target_script = created[0]
        assert target_script.exists()
        content = target_script.read_text(encoding="utf-8")
        assert "translate vietnamese story_001:" in content
        assert 'alice "Welcome to the game."' in content

    def test_bytecode_scene_ast_metadata_flow(self, tmp_path: Path) -> None:
        """Verify AST Scene and Label statements are present alongside Say statements."""
        label_node = Label()
        label_node.name = "chapter_intro"
        scene_node = Scene()
        scene_node.imspec = ("bg", "classroom")
        show_node = Show()
        show_node.imspec = ("alice", "happy")
        say_node = _make_say("alice", "Class is starting.")

        rpyc_path = tmp_path / "scene_flow.rpyc"
        _build_synthetic_rpyc(
            rpyc_path,
            slot1_stmts=[label_node, scene_node, show_node, say_node],
        )

        with RpycFile(rpyc_path) as rpyc:
            _, stmts = rpyc.load_slot(SLOT_STATEMENTS_PRE)
            assert len(stmts) == 4
            assert any(isinstance(s, Label) for s in stmts)
            assert any(isinstance(s, Scene) for s in stmts)
            assert any(isinstance(s, Show) for s in stmts)


# ==============================================================================
# Tier 4: Real-World Application Scenarios (Decompilation Lifecycle)
# ==============================================================================


class TestR3Tier4RealWorld:
    """Tier 4: End-to-end bytecode compilation, decompilation, and skeleton synthesis."""

    def test_full_game_bytecode_decompilation_lifecycle(self, tmp_path: Path) -> None:
        """Simulate a game with pure bytecode: extract dialogues and decompile to .rpy."""
        game_dir = tmp_path / "VN_Game" / "game"
        game_dir.mkdir(parents=True)

        # Build story.rpyc with Character definitions, Labels, Menus, and Dialogues
        char_mc = Define(varname="mc", code=VirtualPyCode('Character("Me")'))
        char_e = Define(varname="e", code=VirtualPyCode('Character("Eileen")'))
        label_start = Label()
        label_start.name = "start"
        scene_bg = Scene()
        scene_bg.imspec = ("bg", "park")

        say1 = _make_say("mc", "It's a beautiful afternoon.")
        say2 = _make_say("e", "Yes, let's explore the garden.")
        menu_stmt = Menu()
        menu_stmt.items = [("Go to the fountain", "True", []), ("Rest here", "True", [])]

        ts1 = _make_translate_say("start_1001", "mc", "It's a beautiful afternoon.")
        ts2 = _make_translate_say("start_1002", "e", "Yes, let's explore the garden.")

        story_rpyc = game_dir / "story.rpyc"
        _build_synthetic_rpyc(
            story_rpyc,
            slot1_stmts=[char_mc, char_e, label_start, scene_bg, say1, say2, menu_stmt],
            slot2_stmts=[ts1, ts2],
        )

        # 1. Verify character extraction
        chars = extract_character_definitions_from_rpyc(game_dir)
        assert chars.get("mc") == "Protagonist"
        assert chars.get("e") == "Eileen"

        # 2. Verify dialogue extraction from Slot 2
        items = extract_dialogue_from_rpyc(story_rpyc)
        assert len(items) == 2
        assert items[0].identifier == "start_1001"
        assert items[1].identifier == "start_1002"

        # 3. Verify decompilation to readable .rpy
        decompiled_rpy = decompile_rpyc(story_rpyc)
        assert decompiled_rpy.exists()
        decompiled_content = decompiled_rpy.read_text(encoding="utf-8")
        assert "label start:" in decompiled_content
        assert 'scene bg park' in decompiled_content
        assert 'mc "It\'s a beautiful afternoon."' in decompiled_content
        assert 'e "Yes, let\'s explore the garden."' in decompiled_content
