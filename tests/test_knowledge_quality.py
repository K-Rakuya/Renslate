"""Stratified persona sampling and glossary translation with in-game examples."""

from __future__ import annotations

from typing import Any

from renslate.core.models import GlossaryEntry, Status, Unit, UnitKind
from renslate.engine.analyze import stratified_speaker_samples
from renslate.engine.glossary import collect_term_examples, translate_glossary_with_llm


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


def test_collect_term_examples_whole_word_and_limits() -> None:
    entries = [GlossaryEntry("Fall", ""), GlossaryEntry("Sect", "")]
    units = [
        _u(1, "The Fall of the empire was swift."),
        _u(2, "Fallen leaves everywhere."),
        _u(3, "She shall fall again."),
        _u(4, "Third fall line."),
        _u(5, "x " * 200),
    ]
    ex = collect_term_examples(entries, units, per_term=2)
    assert ex["Fall"] == ["The Fall of the empire was swift.", "She shall fall again."]
    assert "Sect" not in ex


class _LLM:
    def __init__(self) -> None:
        self.prompt = ""

    def complete_json(self, system: str, user: str, **_kw: Any) -> tuple[Any, int, int]:
        self.prompt = user
        return {"Fall": "Mùa thu"}, 1, 1


def test_translate_glossary_passes_examples_to_prompt() -> None:
    llm = _LLM()
    out = translate_glossary_with_llm(
        [GlossaryEntry("Fall", "")], llm, examples={"Fall": ["I love the Fall season."]}
    )
    assert "I love the Fall season." in llm.prompt
    assert out[0].translation == "Mùa thu"
    llm2 = _LLM()
    translate_glossary_with_llm([GlossaryEntry("Fall", "")], llm2)
    assert "Câu ví dụ" not in llm2.prompt
