"""Timeline dialogue sampling: stratified line selection across narrative progression."""

from __future__ import annotations

from collections.abc import Sequence

from renslate.core.models import Unit, UnitKind

__all__ = [
    "stratified_speaker_samples",
]


def stratified_speaker_samples(
    units: Sequence[Unit], tags: Sequence[str], per_speaker: int = 4
) -> dict[str, list[str]]:
    """Pick sample lines spread across the whole script, not just its opening.

    A character's voice (and gender cues, address forms) often changes or only shows up late;
    taking the first few lines describes them as they are in the prologue. Lines are bucketed
    by position and the longest line of each bucket is kept, so the samples cover the story
    and carry enough text to show style.
    """
    wanted = set(tags)
    by_speaker: dict[str, list[str]] = {}
    for u in units:
        if u.speaker in wanted and u.kind is UnitKind.DIALOGUE:
            text = u.source.strip()
            if len(text) > 8:
                by_speaker.setdefault(u.speaker, []).append(text)
    out: dict[str, list[str]] = {}
    for tag, lines in by_speaker.items():
        if len(lines) <= per_speaker:
            out[tag] = list(lines)
            continue
        picked: list[str] = []
        n = len(lines)
        for b in range(per_speaker):
            bucket = lines[b * n // per_speaker : (b + 1) * n // per_speaker]
            best = max(bucket, key=lambda t: min(len(t), 160))
            if best not in picked:
                picked.append(best)
        out[tag] = picked
    return out
