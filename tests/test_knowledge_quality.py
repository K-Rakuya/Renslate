"""Stratified persona sampling and character voice profiling tests."""

from __future__ import annotations

from renslate.core.models import Status, Unit, UnitKind
from renslate.engine.analyze import stratified_speaker_samples


def _u(seq: int, text: str, speaker: str | None = "ren") -> Unit:
    return Unit(f"d:{seq}", UnitKind.DIALOGUE, seq, text, Status.PENDING, speaker=speaker)


def test_samples_cover_whole_script_not_just_opening() -> None:
    units = [_u(i, f"Sentence number {i:03d} " + "x" * (i % 7)) for i in range(1, 201)]
    got = stratified_speaker_samples(units, ["ren"])["ren"]
    assert len(got) == 4
    seqs = sorted(int(t.split()[2]) for t in got)
    assert seqs[0] <= 50 and seqs[-1] > 150


def test_samples_ignore_short_lines_other_speakers_and_small_sets() -> None:
    units = [_u(1, "Hi"), _u(2, "A proper long line."), _u(3, "Other person here.", "mei")]
    got = stratified_speaker_samples(units, ["ren"])
    assert got == {"ren": ["A proper long line."]}


def test_samples_empty_and_unknown_speakers() -> None:
    units = [_u(1, "A line from someone.", "alice")]
    got = stratified_speaker_samples(units, ["bob"])
    assert got == {}


def test_samples_handles_empty_units() -> None:
    got = stratified_speaker_samples([], ["ren"])
    assert got == {}
