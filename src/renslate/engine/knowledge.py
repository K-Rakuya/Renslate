"""Evidence-based knowledge refresh run at the start of every translation.

Character gender drives Vietnamese address forms (anh/chị, cậu/cô ấy...). The persona
bootstrap guesses it from a handful of lines; the script itself usually settles it:
"Ren said he would come" / "Mei lowered her eyes". This module counts those pronouns
next to each character's name across the whole script and corrects profiles that are
unknown or clearly wrong, so the prompt never tells the model the wrong gender.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from renslate.core.db import Database
from renslate.core.models import CharacterProfile, Unit

_LOG = logging.getLogger("renslate.knowledge")

__all__ = ["GenderEvidence", "collect_gender_evidence", "refresh_character_genders"]

_MALE = re.compile(r"\b(?:he|him|his|himself)\b", re.IGNORECASE)
_FEMALE = re.compile(r"\b(?:she|her|hers|herself)\b", re.IGNORECASE)
_UNKNOWN_VALUES = frozenset({"", "unknown", "?", "n/a", "none", "neutral", "other"})
_TAG_OR_VAR = re.compile(r"\{[^{}]*\}|\[[^\[\]]*\]|⟦[^⟦⟧]*⟧")
_WINDOW = 70
_STOP = re.compile(r"[.!?;,\n]|\b[A-Z][a-z]+")


@dataclass(slots=True)
class GenderEvidence:
    male: int = 0
    female: int = 0

    @property
    def total(self) -> int:
        return self.male + self.female

    def leader(self) -> tuple[str, float]:
        if self.total == 0:
            return "", 0.0
        if self.male >= self.female:
            return "male", self.male / self.total
        return "female", self.female / self.total


def _names(profile: CharacterProfile) -> list[str]:
    out: list[str] = []
    for raw in (profile.display_name, profile.display_name.split()[0] if profile.display_name else ""):
        name = raw.strip()
        if len(name) >= 3 and name.lower() not in {n.lower() for n in out}:
            out.append(name)
    return out


def collect_gender_evidence(
    units: Sequence[Unit], characters: Mapping[str, CharacterProfile]
) -> dict[str, GenderEvidence]:
    """Count he/she pronouns that directly follow a character's name (same clause)."""
    patterns = {}
    for tag, prof in characters.items():
        names = _names(prof)
        if names:
            patterns[tag] = re.compile(
                r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE
            )
    evidence = {tag: GenderEvidence() for tag in patterns}
    for u in units:
        text = _TAG_OR_VAR.sub(" ", u.source)
        if not text.strip():
            continue
        for tag, pat in patterns.items():
            for m in pat.finditer(text):
                window = text[m.end() : m.end() + _WINDOW]
                cut = _STOP.search(window)
                if cut:
                    window = window[: cut.start()]
                ev = evidence[tag]
                ev.male += len(_MALE.findall(window))
                ev.female += len(_FEMALE.findall(window))
    return {t: e for t, e in evidence.items() if e.total}


def refresh_character_genders(
    db: Database,
    *,
    units: Sequence[Unit] | None = None,
    min_unknown: int = 3,
    min_conflict: int = 10,
) -> dict[str, str]:
    """Fix unknown/clearly wrong genders from script evidence. Returns {tag: new_gender}."""
    characters = db.get_characters()
    if not characters:
        return {}
    evidence = collect_gender_evidence(
        db.all_units() if units is None else units, characters
    )
    changes: dict[str, str] = {}
    updated: list[CharacterProfile] = []
    for tag, ev in evidence.items():
        prof = characters[tag]
        gender, ratio = ev.leader()
        current = (prof.gender or "").strip().lower()
        if current in _UNKNOWN_VALUES:
            ok = ev.total >= min_unknown and ratio >= 0.8
        elif current in ("male", "female") and current != gender:
            ok = ev.total >= min_conflict and ratio >= 0.9
        else:
            ok = False
        if ok:
            changes[tag] = gender
            updated.append(dataclasses.replace(prof, gender=gender))
    if updated:
        with db.transaction():
            db.save_characters_bulk(updated)
        _LOG.info("Cập nhật giới tính từ ngữ cảnh kịch bản: %s", changes)
    return changes
