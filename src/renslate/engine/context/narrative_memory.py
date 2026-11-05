"""AST stage-grounded multi-turn dialogue memory ring buffer."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

__all__ = [
    "GroundedDialogueTurn",
    "NarrativeMemory",
]


def _format_sprite(sprite: str | Any) -> str:
    if isinstance(sprite, str):
        clean = sprite.strip()
        if not clean:
            return ""
        if "(" in clean and ")" in clean:
            return clean
        parts = clean.split(maxsplit=1)
        if len(parts) == 1:
            return parts[0]
        tag, expr = parts[0], parts[1].strip()
        return f"{tag} ({expr})" if expr else tag
    tag = getattr(sprite, "tag", "")
    expr = getattr(sprite, "expression", "")
    if expr:
        return f"{tag} ({expr})"
    return str(tag) if tag else ""


@dataclass(slots=True)
class GroundedDialogueTurn:
    """Represents an AST-grounded conversational turn with stage presence."""

    speaker: str | None = None
    listener: str | None = None
    source_text: str = ""
    target_text: str = ""
    scene_id: str = ""
    active_sprites: tuple[str, ...] = ()

    def __init__(
        self,
        speaker: str | None = None,
        listener: str | None = None,
        source_text: str = "",
        target_text: str = "",
        scene_id: str = "",
        active_sprites: tuple[str, ...] | Iterable[str] = (),
        *,
        source: str | None = None,
        target: str | None = None,
    ) -> None:
        self.speaker = speaker
        self.listener = listener
        self.source_text = source if source is not None else source_text
        self.target_text = target if target is not None else target_text
        self.scene_id = scene_id
        self.active_sprites = tuple(active_sprites)

    @property
    def source(self) -> str:
        return self.source_text

    @property
    def target(self) -> str:
        return self.target_text


class NarrativeMemory:
    """Ring buffer maintaining recent grounded dialogue turns in the active scene."""

    def __init__(self, capacity: int = 6) -> None:
        self._capacity = max(1, capacity)
        self._turns: deque[GroundedDialogueTurn] = deque(maxlen=self._capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def turns(self) -> tuple[GroundedDialogueTurn, ...]:
        return tuple(self._turns)

    def record(self, turn: GroundedDialogueTurn) -> None:
        if (
            turn.scene_id
            and self._turns
            and any(t.scene_id and t.scene_id != turn.scene_id for t in self._turns)
        ):
            self.clear()
        self._turns.append(turn)

    def format_history(self, current_scene: str = "") -> str:
        if not self._turns:
            return ""

        if current_scene and any(
            turn.scene_id and turn.scene_id != current_scene for turn in self._turns
        ):
            self.clear()
            return ""

        lines: list[str] = []
        for turn in self._turns:
            spk = turn.speaker or "narrator"
            header = f"{spk} -> {turn.listener}" if turn.listener else spk
            sprites = [_format_sprite(s) for s in turn.active_sprites if _format_sprite(s)]
            sprites_part = f" [{', '.join(sprites)}]" if sprites else ""
            lines.append(f'- {header}{sprites_part}: "{turn.source_text}" => "{turn.target_text}"')

        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self) -> Iterator[GroundedDialogueTurn]:
        return iter(self._turns)

    def __bool__(self) -> bool:
        return bool(self._turns)
