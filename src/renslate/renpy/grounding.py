"""Deterministic AST grounding engine for Ren'Py scripts and compiled bytecode.

Provides structural discourse chunking delimited by scene boundaries and label
transitions, hybrid presence tracking (ActiveSprites | SceneSpeakers), 3-way
utterance classification (spoken, thought, narration), and deterministic O(1)
dyad turn-taking resolution.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from renslate.core.project import Project
from renslate.domain.models import SpriteState
from renslate.renpy.parser import parse_say
from renslate.renpy.rpyc import extract_rpyc_file

__all__ = [
    "RENPY_UI_KEYWORDS",
    "DiscourseChunk",
    "GroundingUnit",
    "UtteranceKind",
    "classify_utterance",
    "parse_grounded_script",
    "resolve_dyad",
]

_SCENE_RE = re.compile(r"^\s*scene\b(?:\s+(?P<spec>[^#\n]+))?")
_SHOW_RE = re.compile(r"^\s*show\s+(?P<spec>[A-Za-z0-9_]+[^#\n]*)")
_HIDE_RE = re.compile(r"^\s*hide\s+(?P<tag>[A-Za-z0-9_]+)")
_LABEL_RE = re.compile(r"^\s*label\s+(?P<name>[A-Za-z0-9_]+)\s*:")
_SAY_RE = re.compile(r'^(?:(?P<speaker>[a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_.]+)*\s+)?\"(?P<text>.*)\"(?:\s+.*)?$')

_WITH_RE = re.compile(r"\s+with\s+[a-zA-Z0-9_().]+", re.IGNORECASE)
_AT_POS_RE = re.compile(r"\bat\s+(?P<pos>left|right|center|truecenter|top|bottom)\b", re.IGNORECASE)
_NON_SPRITE_TAGS = frozenset({"text", "layer", "screen", "window"})

RENPY_UI_KEYWORDS = frozenset(
    {
        "add",
        "textbutton",
        "imagebutton",
        "style_prefix",
        "idle",
        "hover",
        "selected_idle",
        "selected_hover",
        "insensitive",
        "selected_insensitive",
        "action",
        "text",
        "label",
        "frame",
        "hbox",
        "vbox",
        "grid",
        "fixed",
        "window",
        "null",
        "bar",
        "vbar",
        "viewport",
        "vpgrid",
        "button",
        "key",
        "timer",
        "mousearea",
        "input",
        "drag",
        "draggroup",
        "dismiss",
        "use",
        "has",
        "hotspot",
        "hotbar",
    }
)

_THOUGHT_TAGS = frozenset({"t", "th", "thought", "thinking", "thoughts"})
_NON_DIALOGUE_TAGS = frozenset(
    {"none", "all", "system", "ui", "menu", "narrator", "narration"}
    | RENPY_UI_KEYWORDS
)


class UtteranceKind(str, enum.Enum):
    """Classification of dialogue utterances."""

    SPOKEN = "spoken"
    THOUGHT = "thought"
    NARRATION = "narration"


@dataclass(frozen=True, slots=True)
class GroundingUnit:
    """A grounded dialogue utterance with active presence and attributed target."""

    uid: str
    seq: int
    speaker: str | None
    text: str
    kind: UtteranceKind
    active_presence: frozenset[str] = field(default_factory=frozenset)
    target: str | None = None


@dataclass(frozen=True, slots=True)
class DiscourseChunk:
    """A narrative scene block bounded by scene background or label transitions."""

    chunk_id: str
    file: str
    label: str
    scene_spec: str
    dialogue_count: int
    conversational_speakers: frozenset[str]
    visual_sprites: tuple[SpriteState, ...]
    units: tuple[GroundingUnit, ...]


def _clean_spec(raw: str) -> str:
    cleaned = _WITH_RE.sub("", raw).strip()
    return " ".join(cleaned.split())


def classify_utterance(
    speaker: str | None,
    text: str,
) -> tuple[UtteranceKind, str | None]:
    """Classifies an utterance into SPOKEN, THOUGHT, or NARRATION.

    Returns (UtteranceKind, normalized_speaker_or_None).
    """
    clean_text = text.strip()
    spk_raw = speaker.strip() if speaker else None
    spk_lower = spk_raw.lower() if spk_raw else None

    # Check for thought markers
    is_thought_text = False
    if (
        (clean_text.startswith("(") and clean_text.endswith(")"))
        or (clean_text.startswith("\uff08") and clean_text.endswith("\uff09"))
        or ((clean_text.startswith("{i}(") or clean_text.startswith("{i} (")) and (")" in clean_text))
        or ((clean_text.startswith("{i}\uff08") or clean_text.startswith("{i} \uff08")) and ("\uff09" in clean_text))
        or (clean_text.startswith("『") and clean_text.endswith("』"))
        or "{size=-6}" in clean_text
    ):
        is_thought_text = True

    if not spk_lower or spk_lower in _NON_DIALOGUE_TAGS:
        if is_thought_text:
            return UtteranceKind.THOUGHT, None
        return UtteranceKind.NARRATION, None

    if spk_lower in _THOUGHT_TAGS:
        return UtteranceKind.THOUGHT, spk_raw

    if is_thought_text:
        return UtteranceKind.THOUGHT, spk_raw

    return UtteranceKind.SPOKEN, spk_raw


def resolve_dyad(
    speaker: str,
    presence_or_speakers: frozenset[str] | set[str] | Sequence[str],
) -> str | None:
    """Deterministically resolve the listener as S \\ {s} if |S| == 2 in O(1) time."""
    spk_norm = speaker.strip().lower()
    orig_map = {
        s.strip().lower(): s.strip()
        for s in presence_or_speakers
        if s and s.strip()
    }
    pool = set(orig_map.keys()) - _NON_DIALOGUE_TAGS
    if len(pool) == 2 and spk_norm in pool:
        rem_lower = pool - {spk_norm}
        if rem_lower:
            return orig_map[next(iter(rem_lower))]
    return None


@dataclass(slots=True)
class _RawTurn:
    seq: int
    raw_speaker: str | None
    text: str
    kind: UtteranceKind
    normalized_speaker: str | None
    sprites_at_turn: dict[str, SpriteState]


@dataclass(slots=True)
class _RawSceneBlock:
    scene_spec: str
    label: str
    turns: list[_RawTurn]
    final_sprites: dict[str, SpriteState]


def _build_chunks_from_raw_blocks(
    blocks: list[_RawSceneBlock],
    filename: str,
) -> list[DiscourseChunk]:
    chunks: list[DiscourseChunk] = []

    for chunk_idx, block in enumerate(blocks):
        # 1. Compute conversational speakers across this entire chunk
        conversational_speakers: set[str] = set()
        for t in block.turns:
            if t.kind is UtteranceKind.SPOKEN and t.normalized_speaker:
                conversational_speakers.add(t.normalized_speaker)

        chunk_id = f"chunk_{chunk_idx:03d}"
        grounding_units: list[GroundingUnit] = []

        for t in block.turns:
            uid = f"u_{t.seq:04d}"
            # 2. Hybrid presence set
            if t.sprites_at_turn:
                # 2D visual presence
                sprite_tags = set(t.sprites_at_turn.keys())
                if t.normalized_speaker and t.kind is UtteranceKind.SPOKEN:
                    presence = frozenset(sprite_tags | {t.normalized_speaker})
                else:
                    presence = frozenset(sprite_tags)
            elif t.kind is UtteranceKind.SPOKEN:
                # 3D VN fallback to conversational speakers in the scene
                presence = frozenset(conversational_speakers)
            else:
                presence = frozenset()

            # 3. Deterministic Dyad turn resolution
            target: str | None = None
            if t.kind is UtteranceKind.SPOKEN and t.normalized_speaker:
                target = resolve_dyad(t.normalized_speaker, presence)

            unit = GroundingUnit(
                uid=uid,
                seq=t.seq,
                speaker=t.normalized_speaker,
                text=t.text,
                kind=t.kind,
                active_presence=presence,
                target=target,
            )
            grounding_units.append(unit)

        # Sort visual sprites deterministically by tag
        sorted_sprites = tuple(
            block.final_sprites[tag]
            for tag in sorted(block.final_sprites.keys())
        )

        chunk = DiscourseChunk(
            chunk_id=chunk_id,
            file=filename,
            label=block.label or "start",
            scene_spec=block.scene_spec or "default",
            dialogue_count=len(grounding_units),
            conversational_speakers=frozenset(conversational_speakers),
            visual_sprites=sorted_sprites,
            units=tuple(grounding_units),
        )
        chunks.append(chunk)

    return chunks


def _parse_grounded_script_text(
    script_text: str,
    filename: str = "script.rpy",
) -> list[DiscourseChunk]:
    blocks: list[_RawSceneBlock] = []
    current_scene: str = "default"
    current_label: str = "start"
    current_sprites: dict[str, SpriteState] = {}
    current_turns: list[_RawTurn] = []

    for seq, line in enumerate(script_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # 1. Scene boundary directive
        m_scene = _SCENE_RE.match(stripped)
        if m_scene:
            if current_turns or current_scene != "default":
                blocks.append(
                    _RawSceneBlock(
                        scene_spec=current_scene,
                        label=current_label,
                        turns=current_turns,
                        final_sprites=dict(current_sprites),
                    )
                )
                current_turns = []
            raw_spec = m_scene.group("spec") or ""
            current_scene = _clean_spec(raw_spec) or "default"
            current_sprites.clear()
            continue

        # 2. Label transition directive
        m_label = _LABEL_RE.match(stripped)
        if m_label:
            lbl_name = m_label.group("name")
            if current_turns:
                blocks.append(
                    _RawSceneBlock(
                        scene_spec=current_scene,
                        label=current_label,
                        turns=current_turns,
                        final_sprites=dict(current_sprites),
                    )
                )
                current_turns = []
                current_scene = "default"
                current_sprites.clear()
            current_label = lbl_name
            continue

        # 3. Show directive
        m_show = _SHOW_RE.match(stripped)
        if m_show:
            raw_spec = m_show.group("spec").strip()
            clean_spec = _WITH_RE.sub("", raw_spec).strip()
            pos = ""
            m_pos = _AT_POS_RE.search(clean_spec)
            if m_pos:
                pos = m_pos.group("pos").lower()
                clean_spec = _AT_POS_RE.sub("", clean_spec).strip()

            tokens = clean_spec.split()
            if tokens:
                raw_tag = tokens[0]
                tag_lower = raw_tag.lower()
                if tag_lower not in _NON_SPRITE_TAGS:
                    expression = " ".join(tokens[1:]) if len(tokens) > 1 else ""
                    prev = current_sprites.get(tag_lower)
                    current_sprites[tag_lower] = SpriteState(
                        tag=raw_tag,
                        expression=expression or (prev.expression if prev else ""),
                        position=pos or (prev.position if prev else ""),
                    )
            continue

        # 4. Hide directive
        m_hide = _HIDE_RE.match(stripped)
        if m_hide:
            tag = m_hide.group("tag").strip().lower()
            current_sprites.pop(tag, None)
            continue

        # 5. Dialogue say statement
        m_say = _SAY_RE.match(stripped)
        if m_say:
            spk = m_say.group("speaker")
            text = m_say.group("text")
            kind, norm_spk = classify_utterance(spk, text)
            turn = _RawTurn(
                seq=seq,
                raw_speaker=spk,
                text=text,
                kind=kind,
                normalized_speaker=norm_spk,
                sprites_at_turn=dict(current_sprites),
            )
            current_turns.append(turn)
            continue

        say = parse_say(stripped)
        if say is not None and say.text is not None:
            kind, norm_spk = classify_utterance(say.speaker, say.text)
            turn = _RawTurn(
                seq=seq,
                raw_speaker=say.speaker,
                text=say.text,
                kind=kind,
                normalized_speaker=norm_spk,
                sprites_at_turn=dict(current_sprites),
            )
            current_turns.append(turn)

    if current_turns or current_scene != "default":
        blocks.append(
            _RawSceneBlock(
                scene_spec=current_scene,
                label=current_label,
                turns=current_turns,
                final_sprites=dict(current_sprites),
            )
        )

    return _build_chunks_from_raw_blocks(blocks, filename)


def _parse_grounded_rpyc_file(
    rpyc_path: Path,
    rel_file: str,
) -> list[DiscourseChunk]:
    script = extract_rpyc_file(rpyc_path)
    blocks: list[_RawSceneBlock] = []
    current_scene: str = "default"
    current_label: str = "start"
    current_sprites: dict[str, SpriteState] = {}
    current_turns: list[_RawTurn] = []
    seq_counter: int = 1

    for item in script.scene_flow:
        if item.kind == "scene":
            if current_turns or current_scene != "default":
                blocks.append(
                    _RawSceneBlock(
                        scene_spec=current_scene,
                        label=current_label,
                        turns=current_turns,
                        final_sprites=dict(current_sprites),
                    )
                )
                current_turns = []
            current_scene = _clean_spec(item.name) or "default"
            current_sprites.clear()
            continue

        if item.kind == "label":
            if current_turns:
                blocks.append(
                    _RawSceneBlock(
                        scene_spec=current_scene,
                        label=current_label,
                        turns=current_turns,
                        final_sprites=dict(current_sprites),
                    )
                )
                current_turns = []
                current_scene = "default"
                current_sprites.clear()
            current_label = item.name or current_label
            continue

        if item.kind == "show":
            tokens = item.name.split()
            if tokens:
                raw_tag = tokens[0]
                tag_lower = raw_tag.lower()
                if tag_lower not in _NON_SPRITE_TAGS:
                    expr = " ".join(tokens[1:]) if len(tokens) > 1 else ""
                    prev = current_sprites.get(tag_lower)
                    current_sprites[tag_lower] = SpriteState(
                        tag=raw_tag,
                        expression=expr or (prev.expression if prev else ""),
                        position=prev.position if prev else "",
                    )
            continue

        if item.kind == "hide":
            tag = item.name.split()[0].lower() if item.name else ""
            if tag:
                current_sprites.pop(tag, None)
            continue

        if item.kind == "say":
            line_no = item.linenumber or seq_counter
            seq_counter += 1
            kind, norm_spk = classify_utterance(item.name or None, item.details)
            turn = _RawTurn(
                seq=line_no,
                raw_speaker=item.name or None,
                text=item.details,
                kind=kind,
                normalized_speaker=norm_spk,
                sprites_at_turn=dict(current_sprites),
            )
            current_turns.append(turn)

    if current_turns or current_scene != "default":
        blocks.append(
            _RawSceneBlock(
                scene_spec=current_scene,
                label=current_label,
                turns=current_turns,
                final_sprites=dict(current_sprites),
            )
        )

    return _build_chunks_from_raw_blocks(blocks, rel_file)


def parse_grounded_script(
    target: Any,
    filename: str = "script.rpy",
) -> list[DiscourseChunk]:
    """Deterministic unified AST grounding parser conforming to PROJECT.md §1.

    Accepts:
    - `Project`: Scans project game directory for .rpy / .rpyc files
    - `Path`: Single .rpy file, .rpyc file, or game directory
    - `str`: Raw Ren'Py script text or existing file/dir path string
    """
    # 1. Project instance
    if isinstance(target, Project):
        game_dir = target.root / "game"
        if not game_dir.is_dir():
            game_dir = target.root
        return _parse_game_directory(game_dir)

    # 2. Path instance
    if isinstance(target, Path):
        if target.is_dir():
            game_dir = target / "game" if (target / "game").is_dir() else target
            return _parse_game_directory(game_dir)
        if target.is_file():
            if target.suffix == ".rpyc":
                return _parse_grounded_rpyc_file(target, target.name)
            return _parse_grounded_script_text(
                target.read_text(encoding="utf-8", errors="replace"),
                target.name,
            )
        return []

    # 3. String instance
    if isinstance(target, str):
        if "\n" not in target:
            p = Path(target)
            if p.exists():
                return parse_grounded_script(p, p.name)
        return _parse_grounded_script_text(target, filename)

    return []


def _parse_game_directory(game_dir: Path) -> list[DiscourseChunk]:
    """Parse all script or bytecode files under a Ren'Py game directory."""
    if not game_dir.is_dir():
        return []

    chunks: list[DiscourseChunk] = []
    excluded_script_names = frozenset(
        {
            "screens.rpy",
            "gui.rpy",
            "options.rpy",
            "screens.rpyc",
            "gui.rpyc",
            "options.rpyc",
        }
    )

    # Check for uncompiled .rpy source files (excluding translations, backups, system files, and UI files)
    rpy_files = sorted(
        [
            p
            for p in game_dir.rglob("*.rpy")
            if "tl" not in p.parts
            and ".renslate" not in p.parts
            and not p.name.startswith("00")
            and p.name.lower() not in excluded_script_names
        ],
        key=lambda p: str(p.relative_to(game_dir)),
    )
    if rpy_files:
        for rpy_path in rpy_files:
            rel = str(rpy_path.relative_to(game_dir))
            try:
                text = rpy_path.read_text(encoding="utf-8", errors="replace")
                chunks.extend(_parse_grounded_script_text(text, rel))
            except Exception:
                continue
        # If rpy_files contained dialogue chunks, return them.
        # Otherwise, fall back to scanning .rpyc bytecode files.
        if any(ch.dialogue_count > 0 for ch in chunks):
            return chunks

    # Check for compiled .rpyc bytecode archives
    rpyc_files = sorted(
        [
            p
            for p in game_dir.rglob("*.rpyc")
            if "tl" not in p.parts
            and ".renslate" not in p.parts
            and not p.name.startswith("00")
            and p.name.lower() not in excluded_script_names
        ],
        key=lambda p: str(p.relative_to(game_dir)),
    )
    rpyc_chunks: list[DiscourseChunk] = []
    for rpyc_path in rpyc_files:
        rel = str(rpyc_path.relative_to(game_dir)).replace(".rpyc", ".rpy")
        try:
            rpyc_chunks.extend(_parse_grounded_rpyc_file(rpyc_path, rel))
        except Exception:
            continue

    if rpyc_chunks:
        return rpyc_chunks

    return chunks
