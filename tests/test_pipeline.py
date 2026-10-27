from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest

from renslate.db import Database
from renslate.errors import ApplyError, ProjectError
from renslate.extract import FLAG_MARKUP_INVALID, FLAG_SOURCE_CHANGED, extract
from renslate.models import Origin, Status
from renslate.parser import Severity, string_uid
from renslate.project import Project
from renslate.writer import apply_translations
from tests.helpers import read_tree

SCRIPT = "game/tl/vietnamese/script.rpy"
SCREENS = "game/tl/vietnamese/screens.rpy"
GREETING = "d:start_a1b2c3d4"
NARRATOR = "d:start_0f9e8d7c"
QUOTED = "d:start_11223344"
CHAPTER = "d:chapter_two_99aabbcc"
QUIT_BLOCK = '    # game/screens.rpy:300\n    old "Quit"\n    new "Quit"\n'


def set_target(db: Database, uid: str, text: str, origin: Origin = Origin.LLM) -> None:
    unit = db.units()[uid]
    db.save_unit(dataclasses.replace(unit, target=text, status=Status.TRANSLATED, origin=origin))


def edit(project: Project, relative: str, transform: Callable[[str], str]) -> None:
    path = project.root / relative
    path.write_text(transform(path.read_text(encoding="utf-8")), encoding="utf-8")


def test_first_extraction(project: Project, database: Database) -> None:
    report = extract(project, database, now="2026-10-23T01:00:00Z")
    assert report.issues == []
    assert (report.files, report.units, report.slots) == (2, 11, 11)
    assert (report.created, report.imported, report.obsolete) == (11, 2, 0)

    units = database.units()
    assert [u.seq for u in units.values()] == list(range(11))
    greeting = units[GREETING]
    assert greeting.status is Status.PENDING
    assert greeting.masked_source == "Hello, ⟦V1⟧! ⟦T1⟧Welcome⟦/T1⟧ to the club."
    assert greeting.updated_at == "2026-10-23T01:00:00Z"
    chapter = units[CHAPTER]
    assert (chapter.target, chapter.origin, chapter.status) == (
        "Đã dịch rồi à?",
        Origin.IMPORTED,
        Status.TRANSLATED,
    )
    assert units[string_uid("Cancel")].target == "Hủy"


def test_extraction_is_idempotent(project: Project, database: Database) -> None:
    extract(project, database, now="2026-10-23T01:00:00Z")
    before = database.units()
    report = extract(project, database, now="2026-10-23T02:00:00Z")
    assert (report.created, report.imported, report.source_changed, report.obsolete) == (0,) * 4
    assert database.units() == before
    assert database.count_slots() == 11


def test_apply_without_translations_leaves_files_untouched(
    project: Project, database: Database
) -> None:
    extract(project, database)
    before = read_tree(project)
    report = apply_translations(project, database)
    assert (report.files_changed, report.lines_written, report.unchanged) == (0, 0, 2)
    assert report.backup_dir is None
    assert read_tree(project) == before
    assert not project.backup_root.exists()


def test_apply_writes_translations_and_nothing_else(project: Project, database: Database) -> None:
    extract(project, database)
    original = read_tree(project)
    set_target(database, GREETING, "Xin chào, [player_name]! {i}Chào mừng{/i} đến câu lạc bộ.")
    set_target(database, NARRATOR, "Cánh cửa kẽo kẹt mở.{w} Không có ai ở đó.")
    set_target(database, QUOTED, 'Nói "cheese" đi!  Hai khoảng trắng.')
    set_target(database, string_uid("Yes"), "Có")
    set_target(database, string_uid('Go to the "library"'), 'Đến "thư viện"')
    set_target(database, string_uid("Quit"), "Thoát\nngay")

    report = apply_translations(project, database, timestamp="20261023T020000Z")
    assert (report.files_changed, report.lines_written) == (2, 6)

    script = original[SCRIPT].decode().split("\n")
    script[6] = '    e "Xin chào, [player_name]! {i}Chào mừng{/i} đến câu lạc bộ."'
    script[12] = '    "Cánh cửa kẽo kẹt mở.{w} Không có ai ở đó."'
    script[18] = '    e happy "Nói \\"cheese\\" đi! \\ Hai khoảng trắng." with dissolve'
    script[36] = '    new "Có"'
    script[44] = '    new "Đến \\"thư viện\\""'
    screens = original[SCREENS].decode().split("\n")
    screens[14] = '    new "Thoát\\nngay"'
    updated = read_tree(project)
    assert updated[SCRIPT].decode().split("\n") == script
    assert updated[SCREENS].decode().split("\n") == screens

    assert report.backup_dir == project.backup_root / "20261023T020000Z"
    assert (report.backup_dir / SCRIPT).read_bytes() == original[SCRIPT]
    assert (report.backup_dir / SCREENS).read_bytes() == original[SCREENS]


def test_translations_survive_re_extraction(project: Project, database: Database) -> None:
    extract(project, database)
    set_target(database, GREETING, "Xin chào, [player_name]!")
    set_target(database, string_uid("Yes"), "Có")
    apply_translations(project, database)

    report = extract(project, database)
    assert (report.created, report.imported, report.source_changed) == (0, 0, 0)
    units = database.units()
    assert units[GREETING].target == "Xin chào, [player_name]!"
    assert units[GREETING].origin is Origin.LLM
    assert apply_translations(project, database).lines_written == 0


def test_second_apply_after_apply_is_a_no_op(project: Project, database: Database) -> None:
    extract(project, database)
    set_target(database, GREETING, "Chào")
    assert apply_translations(project, database).lines_written == 1
    before = read_tree(project)
    assert apply_translations(project, database).lines_written == 0
    assert read_tree(project) == before


def test_dry_run_reports_without_writing(project: Project, database: Database) -> None:
    extract(project, database)
    set_target(database, GREETING, "Chào")
    before = read_tree(project)
    report = apply_translations(project, database, dry_run=True)
    assert (report.lines_written, report.files_changed, report.dry_run) == (1, 1, True)
    assert read_tree(project) == before
    assert not project.backup_root.exists()


def test_stale_files_abort_the_whole_operation(project: Project, database: Database) -> None:
    extract(project, database)
    set_target(database, GREETING, "Chào")
    set_target(database, string_uid("Quit"), "Thoát")
    edit(project, SCRIPT, lambda text: "# inserted\n" + text)
    before = read_tree(project)
    with pytest.raises(ApplyError, match="đã thay đổi"):
        apply_translations(project, database)
    assert read_tree(project) == before

    extract(project, database)
    assert apply_translations(project, database).lines_written == 2


def test_missing_file_aborts_apply(project: Project, database: Database) -> None:
    extract(project, database)
    set_target(database, string_uid("Quit"), "Thoát")
    (project.root / SCREENS).unlink()
    with pytest.raises(ApplyError, match="renslate extract"):
        apply_translations(project, database)


def test_malformed_markup_is_refused_before_writing(project: Project, database: Database) -> None:
    extract(project, database)
    set_target(database, GREETING, "Chào [player_name")
    set_target(database, string_uid("Quit"), "Thoát")
    before = read_tree(project)
    with pytest.raises(ApplyError, match=GREETING):
        apply_translations(project, database)
    assert read_tree(project) == before


def test_duplicate_strings_share_one_unit_and_all_slots(
    project: Project, database: Database
) -> None:
    edit(
        project,
        SCREENS,
        lambda text: text + '\n    # game/screens.rpy:400\n    old "Quit"\n    new "Quit"\n',
    )
    report = extract(project, database)
    assert (report.units, report.slots) == (11, 12)
    set_target(database, string_uid("Quit"), "Thoát")
    assert apply_translations(project, database).lines_written == 2
    assert read_tree(project)[SCREENS].decode().count('new "Thoát"') == 2


def test_changed_source_resets_machine_translations_and_flags_locked_ones(
    project: Project, database: Database
) -> None:
    extract(project, database)
    set_target(database, GREETING, "Chào")
    edit(
        project,
        SCRIPT,
        lambda text: text.replace('# e "Hello, [player_name]!', '# e "Hi, [player_name]!').replace(
            "Is this already translated?", "Is it translated?"
        ),
    )
    report = extract(project, database)
    assert (report.source_changed, report.imported) == (2, 0)

    greeting = database.units()[GREETING]
    assert (greeting.target, greeting.origin, greeting.status) == (None, None, Status.PENDING)
    assert greeting.source.startswith("Hi, ")

    chapter = database.units()[CHAPTER]
    assert chapter.target == "Đã dịch rồi à?"
    assert chapter.status is Status.QA_FLAGGED
    assert chapter.qa_flags == (FLAG_SOURCE_CHANGED,)


def test_disappearing_units_become_obsolete_and_return(
    project: Project, database: Database
) -> None:
    extract(project, database)
    set_target(database, string_uid("Quit"), "Thoát")
    path = project.root / SCREENS
    original = path.read_text(encoding="utf-8")
    path.write_text(original.replace(QUIT_BLOCK, ""), encoding="utf-8")

    report = extract(project, database)
    assert report.obsolete == 1
    assert database.units()[string_uid("Quit")].status is Status.OBSOLETE
    assert apply_translations(project, database).lines_written == 0

    path.write_text(original, encoding="utf-8")
    report = extract(project, database)
    assert (report.restored, report.obsolete) == (1, 0)
    quit_unit = database.units()[string_uid("Quit")]
    assert (quit_unit.status, quit_unit.target) == (Status.TRANSLATED, "Thoát")


def test_errors_prevent_marking_units_obsolete(project: Project, database: Database) -> None:
    extract(project, database)
    edit(project, SCREENS, lambda text: text.replace(QUIT_BLOCK, ""))
    broken = '\ntranslate vietnamese broken_00000000:\n\n    # e "x"\n'
    edit(project, SCRIPT, lambda text: text + broken)
    report = extract(project, database)
    assert report.errors == 1
    assert report.obsolete == 0
    assert database.units()[string_uid("Quit")].status is not Status.OBSOLETE


def test_invalid_source_markup_marks_unit_failed(project: Project, database: Database) -> None:
    broken = '\n    # game/screens.rpy:1\n    old "Broken [var"\n    new "Broken [var"\n'
    edit(project, SCREENS, lambda text: text + broken)
    extract(project, database)
    unit = database.units()[string_uid("Broken [var")]
    assert (unit.status, unit.masked_source, unit.qa_flags) == (
        Status.FAILED,
        None,
        (FLAG_MARKUP_INVALID,),
    )


def test_bad_encoding_is_reported_and_skipped(project: Project, database: Database) -> None:
    (project.tl_root / "binary.rpy").write_bytes(b"\xff\xfe\x00")
    report = extract(project, database)
    assert [(i.code, i.severity) for i in report.issues] == [("bad_encoding", Severity.ERROR)]
    assert report.units == 11


def test_missing_translation_directory(tmp_path: Path) -> None:
    (tmp_path / "game").mkdir()
    project = Project.create(tmp_path)
    with project.open_db() as database, pytest.raises(ProjectError, match="renslate generate"):
        extract(project, database)


def test_bom_and_crlf_line_endings_are_preserved(project: Project, database: Database) -> None:
    for path in project.tl_root.glob("*.rpy"):
        text = path.read_bytes().decode().replace("\n", "\r\n")
        path.write_bytes(b"\xef\xbb\xbf" + text.encode())
    extract(project, database)
    set_target(database, GREETING, "Chào")
    set_target(database, string_uid("Quit"), "Thoát")
    apply_translations(project, database)
    for relative, data in read_tree(project).items():
        assert data.startswith(b"\xef\xbb\xbf"), relative
        assert data.count(b"\r\n") == data.count(b"\n"), relative
    assert b'    e "Ch\xc3\xa0o"\r\n' in read_tree(project)[SCRIPT]
