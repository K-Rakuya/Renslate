"""Ren'Py stage and sprite tracking engine for extracting visual context and active presence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from renslate.domain.models import SpriteState, StageSnapshot
from renslate.renpy.rpyc.ast_types import Hide, Scene, Show

__all__ = ["StageTracker", "build_stage_index"]

_SCENE_RE = re.compile(r"^\s*scene\b(?:\s+(?P<spec>[^#\n]+))?")
_SHOW_RE = re.compile(r"^\s*show\s+(?P<spec>[A-Za-z0-9_]+[^#\n]*)")
_HIDE_RE = re.compile(r"^\s*hide\s+(?P<tag>[A-Za-z0-9_]+)")
_AT_POS_RE = re.compile(r"\bat\s+(?P<pos>left|right|center|truecenter|top|bottom)\b", re.IGNORECASE)
_WITH_RE = re.compile(r"\bwith\s+[A-Za-z0-9_().]+", re.IGNORECASE)
_NON_SPRITE_TAGS = frozenset({"text", "layer", "screen", "window"})


def _format_ast_imspec(imspec: Any) -> str:
    """Format AST imspec tuple into clean string."""
    if not imspec:
        return ""
    if isinstance(imspec, tuple) and len(imspec) > 0 and isinstance(imspec[0], (tuple, list)):
        return " ".join(str(p) for p in imspec[0] if p)
    if isinstance(imspec, (tuple, list)):
        return " ".join(str(p) for p in imspec if p)
    return str(imspec)


class StageTracker:
    """Maintains active characters on stage and their emotional/positional expressions."""

    def __init__(self) -> None:
        self._scene_bg: str = ""
        self._active_sprites: dict[str, SpriteState] = {}

    @property
    def scene_bg(self) -> str:
        return self._scene_bg

    @property
    def active_sprites(self) -> dict[str, SpriteState]:
        return dict(self._active_sprites)

    def reset(self) -> None:
        """Reset the stage state."""
        self._scene_bg = ""
        self._active_sprites.clear()

    def process_node(self, node: Any) -> bool:
        """Parse stage directives from a virtual AST node (Scene, Show, Hide)."""
        if node is None:
            return False

        if isinstance(node, Scene):
            raw_spec = _format_ast_imspec(getattr(node, "imspec", ()))
            self._scene_bg = _WITH_RE.sub("", raw_spec).strip()
            self._active_sprites.clear()
            return True

        if isinstance(node, Hide):
            raw_tag = _format_ast_imspec(getattr(node, "imspec", ()))
            tag = raw_tag.split()[0].lower() if raw_tag else ""
            if tag:
                self._active_sprites.pop(tag, None)
            return True

        if isinstance(node, Show):
            raw_spec = _format_ast_imspec(getattr(node, "imspec", ()))
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
                if tag_lower in _NON_SPRITE_TAGS:
                    return False
                expression = " ".join(tokens[1:]) if len(tokens) > 1 else ""
                prev_state = self._active_sprites.get(tag_lower)
                final_pos = pos or (prev_state.position if prev_state else "")
                final_expr = expression or (prev_state.expression if prev_state else "")
                self._active_sprites[tag_lower] = SpriteState(
                    tag=raw_tag,
                    expression=final_expr,
                    position=final_pos,
                )
            return True

        return False

    def process_line(self, line: str) -> bool:
        """Parse stage directives from a raw Ren'Py script line.

        Returns True if a stage directive was matched and handled.
        """
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return False

        # 1. Handle scene directive
        m_scene = _SCENE_RE.match(stripped)
        if m_scene:
            spec = (m_scene.group("spec") or "").strip()
            spec_clean = _WITH_RE.sub("", spec).strip()
            self._scene_bg = spec_clean
            self._active_sprites.clear()
            return True

        # 2. Handle hide directive
        m_hide = _HIDE_RE.match(stripped)
        if m_hide:
            tag = m_hide.group("tag").strip().lower()
            self._active_sprites.pop(tag, None)
            return True

        # 3. Handle show directive
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
                if tag_lower in _NON_SPRITE_TAGS:
                    return False

                expression = " ".join(tokens[1:]) if len(tokens) > 1 else ""

                prev_state = self._active_sprites.get(tag_lower)
                final_pos = pos or (prev_state.position if prev_state else "")
                final_expr = expression or (prev_state.expression if prev_state else "")

                self._active_sprites[tag_lower] = SpriteState(
                    tag=raw_tag,
                    expression=final_expr,
                    position=final_pos,
                )
            return True

        return False

    def capture(self) -> StageSnapshot:
        """Capture an immutable snapshot of the current stage state."""
        sprites_tuple = tuple(self._active_sprites.values())
        return StageSnapshot(
            scene_bg=self._scene_bg,
            active_sprites=sprites_tuple,
        )


def build_stage_index(search_dir: Path) -> dict[str, StageSnapshot]:
    """Scan .rpy source files and .rpyc bytecode archives under search_dir.

    Builds an index of file:line -> StageSnapshot.
    """
    if not search_dir.is_dir():
        return {}

    from renslate.core.textio import read_text_file
    from renslate.renpy.parser import parse_say
    from renslate.renpy.rpyc import extract_rpyc_file

    stage_index: dict[str, StageSnapshot] = {}

    # 1. Scan .rpy source files if available
    rpy_files = sorted(p for p in search_dir.rglob("*.rpy") if "tl" not in p.parts)
    for path in rpy_files:
        try:
            lines = read_text_file(path).lines
        except Exception:
            continue

        try:
            rel = path.relative_to(search_dir).as_posix()
        except ValueError:
            rel = path.name

        tracker = StageTracker()
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if tracker.process_line(stripped):
                continue

            say = parse_say(stripped)
            if say is not None and say.text:
                snap = tracker.capture()
                stage_index[f"{rel}:{idx}"] = snap
                if not rel.startswith("game/"):
                    stage_index[f"game/{rel}:{idx}"] = snap
                else:
                    stage_index[rel[5:] + f":{idx}"] = snap

    # 2. Complement with compiled bytecode (.rpyc) if no source scripts exist
    if not stage_index:
        rpyc_files = sorted(p for p in search_dir.rglob("*.rpyc") if "tl" not in p.parts)
        for rpyc_path in rpyc_files:
            try:
                script = extract_rpyc_file(rpyc_path)
            except Exception:
                continue

            try:
                rel = rpyc_path.relative_to(search_dir).as_posix()
            except ValueError:
                rel = rpyc_path.name
            rel_rpy = rel.replace(".rpyc", ".rpy")

            for item in script.dialogue:
                sprites = tuple(
                    SpriteState(
                        tag=s.split()[0],
                        expression=" ".join(s.split()[1:]) if len(s.split()) > 1 else "",
                    )
                    for s in item.active_sprites
                    if s and s.split()[0].lower() not in _NON_SPRITE_TAGS
                )
                snap = StageSnapshot(
                    scene_bg=item.scene_id,
                    active_sprites=sprites,
                )
                src_file = item.source_file or rel_rpy
                line_no = item.source_line or 1
                stage_index[f"{src_file}:{line_no}"] = snap
                if not src_file.startswith("game/"):
                    stage_index[f"game/{src_file}:{line_no}"] = snap
                else:
                    stage_index[src_file[5:] + f":{line_no}"] = snap

    return stage_index
