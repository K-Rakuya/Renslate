"""Gender correction from script evidence and runtime knowledge refresh."""

from __future__ import annotations

from pathlib import Path

from renslate.core.models import CharacterProfile, Status, Unit, UnitKind
from renslate.core.project import Project
from renslate.engine.knowledge import collect_gender_evidence, refresh_character_genders


def _u(i: int, text: str) -> Unit:
    return Unit(f"d:{i}", UnitKind.DIALOGUE, i, text, Status.PENDING, speaker=None, label="a")


def _project(tmp_path: Path) -> Project:
    root = tmp_path / "g"
    (root / "game").mkdir(parents=True)
    return Project.create(root)


def test_evidence_counts_pronouns_in_same_clause_only() -> None:
    chars = {"ren": CharacterProfile("ren", "Ren"), "mei": CharacterProfile("mei", "Mei")}
    units = [
        _u(1, "Ren said he would come."),
        _u(2, "Ren told Mei she was late."),  # 'she' belongs to Mei, not Ren
        _u(3, "Mei lowered her eyes."),
        _u(4, "Ren, you know his plan, right?"),  # comma ends the window
    ]
    ev = collect_gender_evidence(units, chars)
    assert (ev["ren"].male, ev["ren"].female) == (1, 0)
    assert (ev["mei"].male, ev["mei"].female) == (0, 2)


def test_refresh_fills_unknown_and_respects_thresholds(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    with proj.open_db() as db:
        db.save_characters_bulk(
            [
                CharacterProfile("ren", "Ren", gender="unknown"),
                CharacterProfile("mei", "Mei", gender="unknown"),
                CharacterProfile("kai", "Kai", gender="male"),
            ]
        )
        units = (
            [_u(i, "Ren said he would come.") for i in range(1, 5)]
            + [_u(10, "Mei lowered her eyes.")]  # too little evidence
            + [_u(20 + i, "Kai smiled and she nodded.") for i in range(3)]  # weak conflict
        )
        db.save_units_bulk(units)
        changes = refresh_character_genders(db)
        assert changes == {"ren": "male"}
        chars = db.get_characters()
        assert chars["ren"].gender == "male"
        assert chars["mei"].gender == "unknown"
        assert chars["kai"].gender == "male"


def test_refresh_corrects_strong_conflict(tmp_path: Path) -> None:
    proj = _project(tmp_path)
    with proj.open_db() as db:
        db.save_characters_bulk([CharacterProfile("mei", "Mei", gender="male")])
        db.save_units_bulk([_u(i, "Mei lowered her eyes.") for i in range(1, 13)])
        assert refresh_character_genders(db) == {"mei": "female"}
        assert db.get_characters()["mei"].gender == "female"
