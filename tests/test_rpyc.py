from __future__ import annotations

import io
import pickle
import zlib
from pathlib import Path

import pytest

from renslate.cli import main
from renslate.core.project import Project
from renslate.renpy.extract import scan_project
from renslate.renpy.parser import TranslationScanner
from renslate.renpy.rpyc import (
    RPYC2_HEADER,
    SLOT_STATEMENTS_POST,
    SLOT_STATEMENTS_PRE,
    Call,
    Default,
    Define,
    EarlyPython,
    Hide,
    If,
    Init,
    Jump,
    Label,
    Menu,
    Pass,
    Python,
    RenpyRawItem,
    Return,
    RpycFile,
    Say,
    Scene,
    Show,
    Translate,
    TranslateSay,
    VirtualASTNode,
    VirtualDict,
    VirtualList,
    VirtualPyCode,
    VirtualPyExpr,
    VirtualSet,
    While,
    With,
    _format_imspec,
    _get_dynamic_ast_class,
    create_rpyc,
    decompile_rpyc,
    extract_dialogue_from_rpyc,
    extract_rpyc_file,
    synthesize_skeletons,
    synthesize_translation_skeletons,
    write_rpyc,
)
from renslate.renpy.sdk import generate_translations

# ==============================================================================
# Helper Utilities
# ==============================================================================


def _make_dummy_rpyc(
    dest: Path,
    slot1_stmts: list[object] | None = None,
    slot2_stmts: list[object] | None = None,
) -> Path:
    slots: dict[int, bytes] = {}
    if slot1_stmts is not None:
        slots[SLOT_STATEMENTS_PRE] = pickle.dumps((None, slot1_stmts))
    if slot2_stmts is not None:
        slots[SLOT_STATEMENTS_POST] = pickle.dumps((None, slot2_stmts))
    write_rpyc(dest, slots)
    return dest


# ==============================================================================
# Binary Format & RpycFile Tests
# ==============================================================================


def test_create_and_read_rpyc(tmp_path: Path) -> None:
    slot1_data = b"slot1 uncompressed payload"
    slot2_data = b"slot2 uncompressed payload"
    file_bytes = create_rpyc({1: slot1_data, 2: slot2_data})

    assert file_bytes.startswith(RPYC2_HEADER)

    rpyc_path = tmp_path / "test.rpyc"
    rpyc_path.write_bytes(file_bytes)

    rpyc = RpycFile(rpyc_path)
    assert not rpyc.is_legacy
    assert 1 in rpyc.slots
    assert 2 in rpyc.slots
    assert rpyc.read_slot(1) == slot1_data
    assert rpyc.read_slot(2) == slot2_data
    assert rpyc.read_slot(999) is None

    slots_dict = rpyc.read_slots()
    assert slots_dict[1] == slot1_data
    assert slots_dict[2] == slot2_data


def test_legacy_v1_rpyc(tmp_path: Path) -> None:
    payload = pickle.dumps([Say(who="narrator", what="Old Renpy")])
    compressed = zlib.compress(payload)

    legacy_path = tmp_path / "legacy.rpyc"
    legacy_path.write_bytes(compressed)

    rpyc = RpycFile(legacy_path)
    assert rpyc.is_legacy
    assert rpyc.read_md5() is None
    assert rpyc.read_slot(SLOT_STATEMENTS_POST) is None

    decompressed = rpyc.read_slot(SLOT_STATEMENTS_PRE)
    assert decompressed == payload

    stmts = rpyc.get_slot_1_statements()
    assert len(stmts) == 1
    assert stmts[0].what == "Old Renpy"


def test_rpyc_corrupted_header(tmp_path: Path) -> None:
    corrupted_path = tmp_path / "corrupted.rpyc"
    corrupted_path.write_bytes(RPYC2_HEADER + b"\x01\x00\x00\x00\xff\xff\xff\xff\x05\x00\x00\x00")

    rpyc = RpycFile(corrupted_path)
    assert not rpyc.is_legacy
    # Reading an out-of-bounds slot should safely return None
    assert rpyc.read_slot(1) is None


# ==============================================================================
# Virtualized Unpickler Tests
# ==============================================================================


def test_virtual_containers_and_pyexpr() -> None:
    vlist = VirtualList(["a", "b", "c"])
    vlist.__setstate__((["x", "y"], {"extra": 123}))
    assert list(vlist) == ["x", "y"]
    assert vlist.extra == 123

    vdict = VirtualDict({"key1": "val1"})
    vdict.__setstate__(({"k": "v"}, {"attr": 456}))
    assert dict(vdict) == {"k": "v"}
    assert vdict.attr == 456

    vset = VirtualSet({"alpha", "beta"})
    vset.__setstate__(({"gamma"}, {"meta": "ok"}))
    assert set(vset) == {"gamma"}
    assert vset.meta == "ok"

    expr1 = VirtualPyExpr("x + 1", "script.rpy", 42)
    assert expr1 == "x + 1"
    assert expr1.filename == "script.rpy"
    assert expr1.linenumber == 42

    # Extra arguments (as passed by newer Ren'Py engines)
    expr2 = VirtualPyExpr("y + 2", "other.rpy", 99, None, 0)
    assert expr2 == "y + 2"
    assert expr2.filename == "other.rpy"
    assert expr2.linenumber == 99

    pycode = VirtualPyCode()
    pycode.source = "print('hello')"
    assert pycode.source == "print('hello')"


def test_virtual_ast_node_setstate_and_dunder() -> None:
    node = VirtualASTNode()
    node.__setstate__({"loc": ("game/chapter1.rpy", 105), "custom_attr": "value"})
    assert node.filename == "game/chapter1.rpy"
    assert node.linenumber == 105
    assert node.custom_attr == "value"
    assert node.nonexistent_field is None

    with pytest.raises(AttributeError):
        _ = node.__unknown_dunder__


def test_dynamic_ast_class_generation() -> None:
    cls1 = _get_dynamic_ast_class("renpy.sl2.slast", "SLScreen")
    cls2 = _get_dynamic_ast_class("renpy.sl2.slast", "SLScreen")
    assert cls1 is cls2
    assert issubclass(cls1, VirtualASTNode)

    instance = cls1()
    instance.__setstate__({"name": "main_menu", "loc": ("screens.rpy", 12)})
    assert instance.name == "main_menu"
    assert instance.filename == "screens.rpy"
    assert instance.linenumber == 12


def test_python2_builtin_unpickling() -> None:
    from renslate.renpy.rpyc import RenpyUnpickler

    buf = io.BytesIO(b"")
    unpickler = RenpyUnpickler(buf)
    assert unpickler.find_class("__builtin__", "unicode") is str
    assert unpickler.find_class("__builtin__", "pystr") is str
    assert unpickler.find_class("__builtin__", "long") is int
    assert unpickler.find_class("__builtin__", "set") is set


# ==============================================================================
# Slot 2 Dialogue & String Extraction Tests
# ==============================================================================


def test_extract_slot_2_ast(tmp_path: Path) -> None:
    say1 = TranslateSay()
    say1.identifier = "scene1_abc123"
    say1.who = "eileen"
    say1.what = "Good morning, everyone!"
    say1.filename = "game/script.rpy"
    say1.linenumber = 14

    menu = Menu()
    menu.items = [
        ("Go to the park", "True", [Say(who="eileen", what="Let's go!")]),
        ("Stay home", "True", None),
    ]

    trans_block = Translate()
    trans_block.identifier = "choice_xyz789"
    trans_block.filename = "game/script.rpy"
    trans_block.linenumber = 25
    trans_block.block = [
        Say(who=None, what="What should we do?"),
        menu,
    ]

    py_node = Python()
    py_code = VirtualPyCode()
    py_code.source = "renpy.notify(_('Game saved successfully!'))"
    py_node.code = py_code
    py_node.filename = "game/script.rpy"
    py_node.linenumber = 50

    slot2_stmts: list[object] = [say1, trans_block, py_node]
    rpyc_file = _make_dummy_rpyc(tmp_path / "script.rpyc", slot2_stmts=slot2_stmts)

    extracted = extract_rpyc_file(rpyc_file)
    assert len(extracted.dialogue) == 3  # say1 + trans_say + inner menu say
    assert len(extracted.strings) >= 3  # 2 menu captions + 1 gettext string

    items = extract_dialogue_from_rpyc(rpyc_file)
    assert all(isinstance(i, RenpyRawItem) for i in items)

    d_map = {item.identifier: item for item in items}
    assert "scene1_abc123" in d_map
    assert d_map["scene1_abc123"].who == "eileen"
    assert d_map["scene1_abc123"].what == "Good morning, everyone!"
    assert d_map["scene1_abc123"].source_file == "game/script.rpy"
    assert d_map["scene1_abc123"].source_line == 14

    assert "choice_xyz789" in d_map
    assert d_map["choice_xyz789"].what == "What should we do?"

    # Menu captions must not be duplicated into synthetic dialogue options
    assert "choice_xyz789_opt_0" not in d_map
    assert "choice_xyz789_opt_1" not in d_map

    # Inner dialogue inside choice block is extracted
    dialogue_whats = [item.what for item in items]
    assert "Let's go!" in dialogue_whats

    ui_texts = {s.text for s in extracted.strings}
    assert "Game saved successfully!" in ui_texts
    assert "Go to the park" in ui_texts
    assert "Stay home" in ui_texts


# ==============================================================================
# Translation Skeleton Synthesis Tests
# ==============================================================================


def test_synthesize_translation_skeletons(tmp_path: Path) -> None:
    game_dir = tmp_path / "game"
    game_dir.mkdir()

    say = TranslateSay()
    say.identifier = "start_intro_9988"
    say.who = "lucy"
    say.what = 'Hello "world" & goodbye!'
    say.attributes = ("happy",)
    say.temporary_attributes = ("blush",)
    say.with_ = "dissolve"
    say.filename = "game/story.rpy"
    say.linenumber = 10

    py_node = Python()
    py_code = VirtualPyCode()
    py_code.source = "hint = _('Check the drawer')"
    py_node.code = py_code
    py_node.filename = "game/story.rpy"
    py_node.linenumber = 20

    _make_dummy_rpyc(game_dir / "story.rpyc", slot2_stmts=[say, py_node])

    generated = synthesize_translation_skeletons(tmp_path, "vietnamese")
    assert len(generated) >= 2  # story.rpy and ui_strings.rpy

    story_tl = tmp_path / "game" / "tl" / "vietnamese" / "story.rpy"
    ui_tl = tmp_path / "game" / "tl" / "vietnamese" / "ui_strings.rpy"
    assert story_tl.is_file()
    assert ui_tl.is_file()

    story_content = story_tl.read_text(encoding="utf-8")
    assert "translate vietnamese start_intro_9988:" in story_content
    assert '# lucy happy @ blush "Hello \\"world\\" & goodbye!" with dissolve' in story_content
    assert 'lucy happy @ blush "Hello \\"world\\" & goodbye!" with dissolve' in story_content

    ui_content = ui_tl.read_text(encoding="utf-8")
    assert "translate vietnamese strings:" in ui_content
    assert 'old "Check the drawer"' in ui_content
    assert 'new "Check the drawer"' in ui_content

    # Verify that TranslationScanner can parse the generated skeletons cleanly
    scanner = TranslationScanner("vietnamese")
    scanner.scan_file(
        "game/tl/vietnamese/story.rpy",
        story_content.splitlines(),
    )
    scanner.scan_file(
        "game/tl/vietnamese/ui_strings.rpy",
        ui_content.splitlines(),
    )
    assert not scanner.issues
    assert len(scanner.units) >= 2

    # Check alias
    generated_alias = synthesize_skeletons(tmp_path, "vietnamese")
    assert len(generated_alias) == len(generated)


# ==============================================================================
# Decompiler Tests
# ==============================================================================


def test_decompile_rpyc_statements(tmp_path: Path) -> None:
    label = Label()
    label.name = "start"

    say = Say()
    say.who = "eileen"
    say.what = "Welcome to the game!"

    scene = Scene()
    scene.imspec = (("bg", "room"), None, None, [], None, None, [])

    show = Show()
    show.imspec = (("eileen", "happy"), None, None, ["center"], None, None, [])

    hide = Hide()
    hide.imspec = (("eileen",), None, None, [], None, None, [])

    with_stmt = With()
    with_stmt.expr = "dissolve"

    menu = Menu()
    menu.items = [
        ("Accept quest", "True", [Say(who="eileen", what="Great!")]),
        ("Decline", "gold > 10", [Return()]),
    ]

    if_stmt = If()
    if_stmt.entries = [
        ("gold >= 100", [Say(who=None, what="You are rich!")]),
        ("True", [Say(who=None, what="Keep grinding!")]),
    ]

    while_stmt = While()
    while_stmt.condition = "count < 5"
    while_stmt.block = [Pass()]

    py_stmt = Python()
    py_code = VirtualPyCode()
    py_code.source = "score += 10"
    py_stmt.code = py_code

    early_py = EarlyPython()
    early_code = VirtualPyCode()
    early_code.source = "config.developer = True"
    early_py.code = early_code

    init_stmt = Init()
    init_stmt.priority = 5
    init_stmt.block = [Pass()]

    define_stmt = Define()
    define_stmt.varname = "e"
    define_stmt.code = "Character('Eileen')"

    default_stmt = Default()
    default_stmt.varname = "points"
    default_stmt.code = "0"

    jump_stmt = Jump()
    jump_stmt.target = "next_chapter"

    call_stmt = Call()
    call_stmt.label = "minigame"

    ret_stmt = Return()
    ret_stmt.expression = "True"

    slot1_stmts: list[object] = [
        label,
        say,
        scene,
        show,
        hide,
        with_stmt,
        menu,
        if_stmt,
        while_stmt,
        py_stmt,
        early_py,
        init_stmt,
        define_stmt,
        default_stmt,
        jump_stmt,
        call_stmt,
        ret_stmt,
    ]

    rpyc_path = _make_dummy_rpyc(tmp_path / "script.rpyc", slot1_stmts=slot1_stmts)
    out_rpy = tmp_path / "script_decompiled.rpy"

    res_path = decompile_rpyc(rpyc_path, out_rpy)
    assert res_path == out_rpy
    assert out_rpy.is_file()

    content = out_rpy.read_text(encoding="utf-8")
    assert "label start:" in content
    assert 'eileen "Welcome to the game!"' in content
    assert "scene bg room" in content
    assert "show eileen happy at center" in content
    assert "hide eileen" in content
    assert "with dissolve" in content
    assert "menu:" in content
    assert '"Accept quest":' in content
    assert '"Decline" if gold > 10:' in content
    assert "if gold >= 100:" in content
    assert "else:" in content
    assert "while count < 5:" in content
    assert "$ score += 10" in content
    assert "init python:" in content
    assert "config.developer = True" in content
    assert "init 5:" in content
    assert "define e = Character('Eileen')" in content
    assert "default points = 0" in content
    assert "jump next_chapter" in content
    assert "call minigame" in content
    assert "return True" in content


def test_format_imspec_helper() -> None:
    assert _format_imspec(None) == ""
    assert _format_imspec(("black",)) == "black"
    assert _format_imspec((("bg", "hall"), None, None, ["truecenter"], None, None, [])) == (
        "bg hall at truecenter"
    )
    assert _format_imspec("custom_image") == "custom_image"


# ==============================================================================
# Fallback Integration in extract & sdk Tests
# ==============================================================================


def test_extract_fallback_on_bytecode_only_project(tmp_path: Path) -> None:
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()
    game_dir = project_dir / "game"
    game_dir.mkdir()

    project = Project.create(project_dir, target_lang="vietnamese")

    say = TranslateSay()
    say.identifier = "intro_001"
    say.who = "hero"
    say.what = "Ready to start adventure!"
    say.filename = "game/intro.rpy"
    say.linenumber = 1

    _make_dummy_rpyc(game_dir / "intro.rpyc", slot2_stmts=[say])

    assert not project.tl_root.exists()

    scanner, count = scan_project(project)
    assert count >= 1
    assert any(u.uid == "d:intro_001" for u in scanner.units)
    assert project.tl_root.is_dir()


def test_sdk_generate_fallback_when_sdk_missing(tmp_path: Path) -> None:
    game_dir = tmp_path / "game"
    game_dir.mkdir()

    say = TranslateSay()
    say.identifier = "test_id_1"
    say.who = "alice"
    say.what = "Testing SDK fallback"
    say.filename = "game/test.rpy"
    say.linenumber = 5

    _make_dummy_rpyc(game_dir / "test.rpyc", slot2_stmts=[say])

    out = generate_translations(None, tmp_path, "vietnamese", fallback_rpyc=True)
    assert "bytecode .rpyc" in out
    assert (tmp_path / "game" / "tl" / "vietnamese" / "test.rpy").is_file()


# ==============================================================================
# Standalone CLI decompile Tests
# ==============================================================================


def test_cli_decompile_single_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    say = Say(who="narrator", what="CLI test decompilation")
    rpyc_path = _make_dummy_rpyc(tmp_path / "sample.rpyc", slot1_stmts=[say])
    out_path = tmp_path / "sample.rpy"

    ret = main(["decompile", str(rpyc_path), "-o", str(out_path)])
    assert ret == 0
    assert out_path.is_file()
    assert '"CLI test decompilation"' in out_path.read_text(encoding="utf-8")

    # Overwrite protection check
    ret_dup = main(["decompile", str(rpyc_path), "-o", str(out_path)])
    assert ret_dup == 1

    # Overwrite with flag
    ret_force = main(["decompile", str(rpyc_path), "-o", str(out_path), "--overwrite"])
    assert ret_force == 0


def test_cli_decompile_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    input_dir.mkdir()

    _make_dummy_rpyc(input_dir / "f1.rpyc", slot1_stmts=[Say(who=None, what="File 1")])
    _make_dummy_rpyc(input_dir / "f2.rpyc", slot1_stmts=[Say(who=None, what="File 2")])

    ret = main(["decompile", str(input_dir), "-o", str(out_dir)])
    assert ret == 0
    assert (out_dir / "f1.rpy").is_file()
    assert (out_dir / "f2.rpy").is_file()


def test_cli_decompile_non_rpyc_error(tmp_path: Path) -> None:
    txt_file = tmp_path / "note.txt"
    txt_file.write_text("not bytecode")
    ret = main(["decompile", str(txt_file)])
    assert ret == 1


# ==============================================================================
# Remediation & Hardening Tests (Gen 3)
# ==============================================================================


def test_security_whitelist_blocks_eval_and_dangerous_callables() -> None:
    from renslate.renpy.rpyc import RenpyUnpickler

    # 1. Block builtins.eval / exec / open / compile / __import__
    unpickler = RenpyUnpickler(io.BytesIO(b""))
    with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe builtin"):
        unpickler.find_class("__builtin__", "eval")

    with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe builtin"):
        unpickler.find_class("builtins", "exec")

    with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe builtin"):
        unpickler.find_class("builtins", "open")

    with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe builtin"):
        unpickler.find_class("builtins", "compile")

    with pytest.raises(pickle.UnpicklingError, match="Blocked unsafe builtin"):
        unpickler.find_class("builtins", "__import__")

    # 2. Block dangerous modules
    with pytest.raises(pickle.UnpicklingError, match="Blocked dangerous module"):
        unpickler.find_class("os", "system")

    with pytest.raises(pickle.UnpicklingError, match="Blocked dangerous module"):
        unpickler.find_class("subprocess", "Popen")

    # 3. Allow safe data types
    assert unpickler.find_class("builtins", "str") is str
    assert unpickler.find_class("builtins", "int") is int
    assert unpickler.find_class("builtins", "float") is float
    assert unpickler.find_class("builtins", "bool") is bool
    assert unpickler.find_class("builtins", "dict") is dict
    assert unpickler.find_class("builtins", "list") is list
    assert unpickler.find_class("builtins", "tuple") is tuple
    assert unpickler.find_class("builtins", "set") is set
    assert unpickler.find_class("builtins", "NoneType") is type(None)


def test_virtual_pycode_tuple_state_unpacking() -> None:
    # Ren'Py passes 5-element tuple: (version, source, location, mode, py_version)
    code = VirtualPyCode()
    state = (1, "daytime += 1", ("game/events.rpy", 42, 123456), "exec", 3)
    code.__setstate__(state)

    assert code.source == "daytime += 1"
    assert code.location == ("game/events.rpy", 42)
    assert code.filename == "game/events.rpy"
    assert code.linenumber == 42
    assert code.mode == "exec"
    assert str(code) == "daytime += 1"


def test_virtual_pycode_decompilation_fidelity(tmp_path: Path) -> None:
    # Test that Python and Default/Define statements decompile the actual code expression
    py_code = VirtualPyCode()
    py_code.__setstate__((1, "hero_gold += 100", ("game/script.rpy", 10), "exec", 3))
    py_stmt = Python(code=py_code)

    def_code = VirtualPyCode()
    def_code.__setstate__(
        (1, 'Character("Bandit", color="#ffffff")', ("game/script.rpy", 20), "eval", 3)
    )
    define_stmt = Define(varname="bandit", code=def_code)

    dflt_code = VirtualPyCode()
    dflt_code.__setstate__((1, "False", ("game/script.rpy", 30), "eval", 3))
    default_stmt = Default(varname="flag_seen", code=dflt_code)

    rpyc_path = _make_dummy_rpyc(
        tmp_path / "code_test.rpyc",
        slot1_stmts=[py_stmt, define_stmt, default_stmt],
    )
    out_rpy = tmp_path / "code_test.rpy"
    decompile_rpyc(rpyc_path, out_rpy)

    content = out_rpy.read_text(encoding="utf-8")
    assert "VirtualPyCode()" not in content
    assert "$ hero_gold += 100" in content
    assert 'define bandit = Character("Bandit", color="#ffffff")' in content
    assert "default flag_seen = False" in content


def test_virtual_pycode_gettext_string_extraction(tmp_path: Path) -> None:
    # Ensure _("...") in Python code, Define, and Default statements are extracted
    py_code = VirtualPyCode()
    py_code.__setstate__(
        (1, 'renpy.notify(_("Quest Completed!"))', ("game/script.rpy", 15), "exec", 3)
    )
    py_stmt = Python(code=py_code)

    def_code = VirtualPyCode()
    def_code.__setstate__((1, '_("Grand Wizard")', ("game/script.rpy", 25), "eval", 3))
    def_stmt = Define(varname="wizard_title", code=def_code)

    rpyc_path = _make_dummy_rpyc(
        tmp_path / "strings_test.rpyc",
        slot2_stmts=[py_stmt, def_stmt],
    )
    extracted = extract_rpyc_file(rpyc_path)
    string_texts = {s.text for s in extracted.strings}
    assert "Quest Completed!" in string_texts
    assert "Grand Wizard" in string_texts


def test_decompile_corrupted_rpyc_raises_error(tmp_path: Path) -> None:
    bad_rpyc = tmp_path / "corrupted.rpyc"
    bad_rpyc.write_bytes(b"corrupted binary payload that is not rpyc")
    out_rpy = tmp_path / "corrupted.rpy"

    with pytest.raises(ValueError, match="bị hỏng hoặc không có dữ liệu AST"):
        decompile_rpyc(bad_rpyc, out_rpy)


def test_cli_decompile_corrupted_file_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_rpyc = tmp_path / "bad.rpyc"
    bad_rpyc.write_bytes(b"garbage not rpyc")
    out_rpy = tmp_path / "bad.rpy"

    ret = main(["decompile", str(bad_rpyc), "-o", str(out_rpy)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Lỗi" in captured.err


def test_scan_project_incomplete_translation_fallback(tmp_path: Path) -> None:
    # Setup project with pre-existing common.rpy in tl_root, plus unextracted story.rpyc
    project_dir = tmp_path / "my_game"
    project_dir.mkdir()
    game_dir = project_dir / "game"
    game_dir.mkdir()
    tl_dir = game_dir / "tl" / "vietnamese"
    tl_dir.mkdir(parents=True)

    # Pre-existing common.rpy (e.g. from Ren'Py init)
    common_rpy = tl_dir / "common.rpy"
    common_rpy.write_text(
        'translate vietnamese strings:\n    old "OK"\n    new "OK"\n',
        encoding="utf-8",
    )

    # Story bytecode in game/
    say = TranslateSay(
        identifier="story_001",
        who="hero",
        what="The adventure continues!",
        filename="game/story.rpy",
        linenumber=10,
    )
    _make_dummy_rpyc(game_dir / "story.rpyc", slot2_stmts=[say])

    project = Project.create(project_dir, target_lang="vietnamese")
    scanner, count = scan_project(project)
    assert count >= 1

    # Must extract from both common.rpy AND the synthesized story.rpy!
    uids = {u.uid for u in scanner.units}
    assert "d:story_001" in uids
    assert (tl_dir / "story.rpy").is_file()
