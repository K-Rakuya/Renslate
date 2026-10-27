from __future__ import annotations

from pathlib import Path

import pytest

from renslate.models import UnitKind
from renslate.parser import (
    ParsedUnit,
    Severity,
    TranslationScanner,
    parse_say,
    parse_string_line,
    string_uid,
)
from renslate.textio import TextFile

DATA = Path(__file__).parent / "data" / "tl" / "vietnamese"


def scan(text: str, file: str = "game/tl/vietnamese/script.rpy") -> TranslationScanner:
    scanner = TranslationScanner("vietnamese")
    scanner.scan_file(file, TextFile.from_bytes(text.encode()).lines)
    return scanner


def by_uid(scanner: TranslationScanner) -> dict[str, ParsedUnit]:
    return {unit.uid: unit for unit in scanner.units}


def test_sample_script_units() -> None:
    lines = TextFile.from_bytes((DATA / "script.rpy").read_bytes()).lines
    scanner = TranslationScanner("vietnamese")
    scanner.scan_file("game/tl/vietnamese/script.rpy", lines)
    assert scanner.issues == []
    units = by_uid(scanner)
    assert [u.kind for u in scanner.units].count(UnitKind.DIALOGUE) == 4
    assert [u.kind for u in scanner.units].count(UnitKind.STRING) == 4

    greeting = units["d:start_a1b2c3d4"]
    assert greeting.source == "Hello, [player_name]! {i}Welcome{/i} to the club."
    assert greeting.existing is None
    assert greeting.speaker == "e"
    assert greeting.label == "start"
    assert greeting.ref == "game/script.rpy:14"
    assert (greeting.anchor_line, greeting.line) == (4, 7)

    narrator = units["d:start_0f9e8d7c"]
    assert narrator.speaker is None
    assert narrator.source == "The door creaks open.{w} Nobody is there."

    quoted = units["d:start_11223344"]
    assert quoted.source == 'Say "cheese"!'
    assert quoted.speaker == "e"

    translated = units["d:chapter_two_99aabbcc"]
    assert translated.source == "Is this already translated?"
    assert translated.existing == "Đã dịch rồi à?"
    assert translated.label == "chapter_two"

    assert units[string_uid("Cancel")].existing == "Hủy"
    assert units[string_uid("Yes")].existing is None
    assert units[string_uid('Go to the "library"')].source == 'Go to the "library"'


def test_ui_strings_are_classified_by_file_and_reference() -> None:
    lines = TextFile.from_bytes((DATA / "screens.rpy").read_bytes()).lines
    scanner = TranslationScanner("vietnamese")
    scanner.scan_file("game/tl/vietnamese/screens.rpy", lines)
    assert {u.kind for u in scanner.units} == {UnitKind.UI}
    assert [u.source for u in scanner.units] == ["Start", "Preferences", "Quit"]

    common = scan(
        "translate vietnamese strings:\n\n    # renpy/common/00gui.rpy:10\n"
        '    old "Page"\n    new "Page"\n',
        "game/tl/vietnamese/other.rpy",
    )
    assert common.units[0].kind is UnitKind.UI


def test_python_blocks_and_foreign_languages_are_skipped() -> None:
    scanner = scan(
        "translate vietnamese python:\n\n    x = 1\n\n"
        'translate french start_a1b2c3d4:\n\n    # e "Hi"\n    e "Salut"\n'
    )
    assert scanner.units == []
    assert [i.code for i in scanner.issues] == ["foreign_language"]
    assert scanner.issues[0].severity is Severity.WARNING


def test_block_mismatch_is_an_error() -> None:
    scanner = scan('translate vietnamese start_a1b2c3d4:\n\n    # e "Hi"\n    voice "x.ogg"\n')
    assert scanner.units == []
    assert [i.code for i in scanner.issues] == ["block_mismatch"]
    assert scanner.issues[0].severity is Severity.ERROR


def test_duplicate_identifier_keeps_the_first_block_and_warns() -> None:
    block = 'translate vietnamese start_a1b2c3d4:\n\n    # e "Hi"\n    e "Hi"\n\n'
    scanner = scan(block + block)
    assert len(scanner.units) == 1
    assert [(i.code, i.severity) for i in scanner.issues] == [("duplicate_id", Severity.WARNING)]


def test_extra_translated_statements_are_kept_and_reported() -> None:
    scanner = scan(
        'translate vietnamese start_a1b2c3d4:\n\n    # e "One"\n    e "Một"\n    e "Hai"\n'
    )
    assert [(u.source, u.existing) for u in scanner.units] == [("One", "Một")]
    assert [(i.code, i.severity) for i in scanner.issues] == [
        ("extra_statements", Severity.WARNING)
    ]


def test_multiple_statements_in_one_block_get_distinct_uids() -> None:
    scanner = scan(
        "translate vietnamese start_a1b2c3d4:\n\n"
        '    # e "One"\n    # e "Two"\n    e "One"\n    e "Two"\n'
    )
    assert [u.uid for u in scanner.units] == ["d:start_a1b2c3d4", "d:start_a1b2c3d4#1"]
    assert [u.source for u in scanner.units] == ["One", "Two"]


def test_unsupported_say_syntax_warns() -> None:
    scanner = scan("translate vietnamese start_a1b2c3d4:\n\n    # e 'Hi'\n    e 'Salut'\n")
    assert scanner.units == []
    assert [i.code for i in scanner.issues] == ["unsupported_syntax"] * 2
    assert {i.severity for i in scanner.issues} == {Severity.WARNING}


@pytest.mark.parametrize(
    "body",
    ['    old "A"\n    old "B"\n    new "B"\n', '    new "A"\n', '    old "A"\n', "    old A\n"],
)
def test_broken_strings_blocks_are_errors(body: str) -> None:
    scanner = scan("translate vietnamese strings:\n\n" + body)
    assert scanner.issues
    assert {i.severity for i in scanner.issues} == {Severity.ERROR}


def test_bad_header_is_an_error() -> None:
    scanner = scan('translate vietnamese start_a1b2c3d4 oops:\n\n    # e "Hi"\n    e "Hi"\n')
    assert [i.code for i in scanner.issues] == ["bad_header"]


@pytest.mark.parametrize(
    ("code", "prefix", "text", "suffix"),
    [
        ('"Hi"', "", "Hi", ""),
        ('e "Hi"', "e ", "Hi", ""),
        ('e happy @ blush "Hi" nointeract', "e happy @ blush ", "Hi", " nointeract"),
        ('e "Hi" with dissolve', "e ", "Hi", " with dissolve"),
        ('extend " more"', "extend ", " more", ""),
        (r'e "a \"b\" c"', "e ", 'a "b" c', ""),
        ('"Old Man" "Get off my lawn!"', '"Old Man" ', "Get off my lawn!", ""),
        ('"Eileen" "Hi!" with dissolve', '"Eileen" ', "Hi!", " with dissolve"),
    ],
)
def test_parse_say(code: str, prefix: str, text: str, suffix: str) -> None:
    say = parse_say(code)
    assert say is not None
    assert (say.prefix, say.text, say.suffix) == (prefix, text, suffix)
    if prefix == '"Old Man" ':
        assert say.speaker == "Old Man"


@pytest.mark.parametrize(
    "code",
    ['voice "a.ogg"', 'play music "a.ogg"', "$ x = 1", 'e "x', '"""triple"""', "show eileen", ""],
)
def test_parse_say_rejects_non_dialogue(code: str) -> None:
    assert parse_say(code) is None


def test_parse_string_line() -> None:
    assert parse_string_line('    old "A\\nB"', "old") == ("    old ", "A\nB")
    assert parse_string_line('    old "A"', "new") is None
    assert parse_string_line("    old A", "old") is None
