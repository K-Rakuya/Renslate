"""AST Topology Scanning, Co-Occurrence Tensor, and Salience Window Isolation.

Pass 1 of the 2-Pass Hierarchical Bootstrap Engine:
- Traverses Ren'Py script AST / DiscourseChunks to build the N x N spatial
  co-occurrence and turn-taking interaction matrices.
- Dynamically discovers protagonists via topological network centrality,
  speaker turn ratios, and dynamic name expressions (100% data-driven).
- Isolates high-salience interaction windows using multi-factor scoring
  (vocatives, first meetings, emotional punctuation, turn density).
"""

from __future__ import annotations

import contextlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from renslate.core.models import Unit
from renslate.core.project import Project
from renslate.engine.analyze.tropes import NON_INTERACTION_TAGS
from renslate.renpy.grounding import (
    RENPY_UI_KEYWORDS,
    DiscourseChunk,
    GroundingUnit,
    UtteranceKind,
    parse_grounded_script,
)

__all__ = [
    "SalienceWindow",
    "TopologyAnalysis",
    "build_topology_analysis",
    "compute_cooccurrence_matrix",
    "detect_dynamic_name_patterns",
    "discover_topological_protagonist",
    "isolate_dyad_salience_windows",
    "isolate_salience_windows",
]

_GREETING_VOCATIVES_RE = re.compile(
    r"\b(hello|hey|hi|oh|ah|anh|em|cậu|tớ|mày|tao|bác|chú|cô|dì|ông|bà|thầy|sư phụ|"
    r"master|captain|sir|madam|lady|lord|commander|sensei|senpai|kouhai)\b",
    re.IGNORECASE,
)
_CAP_VOCATIVE_RE = re.compile(r"\b[A-Z][a-z]{1,15}\s*!")
_DYNAMIC_NAME_DEF_RE = re.compile(
    r"(?:define|default)\s+(?P<tag>[A-Za-z0-9_]+)\s*=\s*(?:DynamicCharacter\b|Character\s*\(\s*(?:\"\[[^\]]+\]\"|'\[[^\]]+\]'|\[[^\]]+\]|None\b|[a-z_][a-z0-9_]*_name\b|\"(?:Protagonist|Player|Hero|You|Nhân vật chính)\"|'(?:Protagonist|Player|Hero|You|Nhân vật chính)'))",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SalienceWindow:
    """A scored dialogue interaction window for structured distillation."""

    score: float
    units: tuple[Any, ...]
    pair: tuple[str, str] = ("", "")
    first_meeting: bool = False

    def __iter__(self) -> Any:
        # Enables tuple unpacking (score, units) for contract compatibility
        yield self.score
        yield list(self.units)

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.score
        if index == 1:
            return list(self.units)
        raise IndexError(f"SalienceWindow index {index} out of range [0, 1]")

    def __len__(self) -> int:
        return 2


@dataclass(frozen=True, slots=True)
class TopologyAnalysis:
    """Comprehensive AST topological interaction analysis."""

    speakers: list[str]
    cooccurrence_matrix: dict[str, dict[str, int]]
    turn_matrix: dict[str, dict[str, int]]
    speaker_turns: dict[str, int]
    total_turns: int
    dynamic_name_tags: frozenset[str]
    degree_centralities: dict[str, float]
    protagonist_tag: str | None
    dyad_salience_windows: dict[tuple[str, str], list[SalienceWindow]] = field(
        default_factory=dict
    )


def compute_cooccurrence_matrix(
    source: Sequence[DiscourseChunk] | Sequence[GroundingUnit] | Sequence[Unit] | Project | Path | str,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Compute symmetric N x N speaker co-occurrence tensor from AST chunks or script.

    Returns:
        (speakers, matrix) where speakers is a sorted list of speaker tags,
        and matrix[s1][s2] represents the cumulative spatial presence co-occurrence.
    """
    chunks: list[DiscourseChunk]
    if isinstance(source, (Project, Path, str)):
        chunks = parse_grounded_script(source)
    elif source and isinstance(source[0], DiscourseChunk):
        chunks = list(source)  # type: ignore[arg-type]
    elif source and (isinstance(source[0], GroundingUnit) or hasattr(source[0], "uid")):
        # Synthesize a single chunk from raw units
        raw_units = tuple(source)  # type: ignore[arg-type]
        speakers_set = {
            getattr(u, "speaker", None)
            for u in raw_units
            if getattr(u, "speaker", None)
        }
        valid_spk = frozenset(s for s in speakers_set if s)
        chunks = [
            DiscourseChunk(
                chunk_id="chunk_synth_001",
                file="synthetic.rpy",
                label="start",
                scene_spec="synthetic",
                dialogue_count=len(raw_units),
                conversational_speakers=valid_spk,
                visual_sprites=(),
                units=tuple(raw_units) if isinstance(raw_units[0], GroundingUnit) else (),
            )
        ]
    else:
        chunks = []

    all_speakers: set[str] = set()
    for ch in chunks:
        for s in ch.conversational_speakers:
            s_clean = s.strip()
            if (
                s_clean
                and s_clean.lower() not in NON_INTERACTION_TAGS
                and s_clean.lower() not in RENPY_UI_KEYWORDS
            ):
                all_speakers.add(s_clean)
    speakers = sorted(all_speakers)
    matrix = {s1: dict.fromkeys(speakers, 0) for s1 in speakers}

    for ch in chunks:
        sp_list = [s for s in ch.conversational_speakers if s in all_speakers]
        for i, s1 in enumerate(sp_list):
            for s2 in sp_list[i + 1 :]:
                matrix[s1][s2] += ch.dialogue_count
                matrix[s2][s1] += ch.dialogue_count

    return speakers, matrix


def detect_dynamic_name_patterns(
    source: Project | Path | str | Sequence[DiscourseChunk],
) -> set[str]:
    """Detect characters defined with dynamic player name expressions or interpolation."""
    tags: set[str] = set()

    script_texts: list[str] = []
    if isinstance(source, Project):
        game_dir = source.root / "game" if (source.root / "game").is_dir() else source.root
        for rpy in game_dir.rglob("*.rpy"):
            if "tl" not in rpy.parts:
                try:
                    script_texts.append(rpy.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    continue
    elif isinstance(source, Path):
        if source.is_dir():
            for rpy in source.rglob("*.rpy"):
                if "tl" not in rpy.parts:
                    try:
                        script_texts.append(rpy.read_text(encoding="utf-8", errors="replace"))
                    except Exception:
                        continue
        elif source.is_file():
            with contextlib.suppress(Exception):
                script_texts.append(source.read_text(encoding="utf-8", errors="replace"))
    elif isinstance(source, str):
        script_texts.append(source)

    for text in script_texts:
        for m in _DYNAMIC_NAME_DEF_RE.finditer(text):
            tags.add(m.group("tag").strip())

    target_dir: Path | None = None
    if isinstance(source, Project):
        target_dir = source.root
    elif isinstance(source, Path):
        target_dir = source if source.is_dir() else source.parent
    if target_dir is not None:
        with contextlib.suppress(Exception):
            from renslate.engine.analyze.pipeline import extract_character_definitions

            defs = extract_character_definitions(target_dir)
            for t, d in defs.items():
                d_low = d.lower()
                if "[" in d or d.endswith("_name") or d_low in ("protagonist", "player", "hero", "nhân vật chính", "you"):
                    tags.add(t)

    return tags


def discover_topological_protagonist(
    matrix: dict[str, dict[str, int]],
    speaker_turns: Mapping[str, int] | None = None,
    dynamic_name_tags: set[str] | Sequence[str] | None = None,
    *,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
) -> str | None:
    """Dynamically discover the protagonist tag via graph centrality and dialogue evidence.

    Formula:
        Score(v) = alpha * (Turns(v) / TotalTurns)
                 + beta * DegreeCentrality(v)
                 + gamma * Indicator(DynamicName(v))
    """
    if not matrix:
        return None

    speakers = list(matrix.keys())
    if len(speakers) == 1:
        return speakers[0]

    # 1. Degree Centrality
    degrees = {s: sum(matrix[s].values()) for s in speakers}
    max_deg = max(degrees.values()) if degrees else 0
    norm_deg = {
        s: (degrees[s] / max_deg if max_deg > 0 else 0.0)
        for s in speakers
    }

    dyn_set = set(dynamic_name_tags or ())

    # If only matrix is passed without turns or dynamic tags, match contract behavior
    if speaker_turns is None and not dyn_set:
        return max(degrees, key=lambda s: degrees[s])

    # 2. Normalized Spoken Turns
    turns_map = dict(speaker_turns or {})
    total_turns = sum(turns_map.get(s, 0) for s in speakers)
    norm_turns = {
        s: (turns_map.get(s, 0) / total_turns if total_turns > 0 else 0.0)
        for s in speakers
    }

    # 3. Dynamic Name Indicator
    norm_dyn = {s: (1.0 if s in dyn_set else 0.0) for s in speakers}

    # 4. Multi-Factor Score
    scores = {
        s: alpha * norm_turns[s] + beta * norm_deg[s] + gamma * norm_dyn[s]
        for s in speakers
    }

    return max(scores, key=lambda s: (scores[s], degrees[s]))


def isolate_salience_windows(
    units: Sequence[Any],
    window_size: int = 4,
    max_windows: int = 5,
    *,
    pair: tuple[str, str] | None = None,
    first_meeting_turn: int | None = None,
) -> list[SalienceWindow]:
    """Isolate high-salience interaction windows for distillation.

    Formula:
        Salience(W) = 3.0 * N_vocatives + 2.5 * Indicator(first_meeting)
                    + 1.5 * N_emotion + 1.0 * TurnDensity
    """
    if not units:
        return []

    scored_windows: list[SalienceWindow] = []
    step = max(1, window_size // 2)

    for i in range(0, len(units), step):
        window = list(units[i : i + window_size])
        if not window:
            continue

        n_vocatives = 0
        n_emotion = 0
        speaker_switches = 0
        prev_spk: str | None = None
        has_first_meeting = False

        for u_idx, u in enumerate(window):
            text = getattr(u, "text", "") or getattr(u, "source", "") or ""
            spk = getattr(u, "speaker", None)

            # Emotion counts (! and ?)
            n_emotion += text.count("!") * 2
            n_emotion += text.count("?") * 2

            # Vocatives and greeting cues
            if _GREETING_VOCATIVES_RE.search(text):
                n_vocatives += 1
            if _CAP_VOCATIVE_RE.search(text):
                n_vocatives += 1

            # Turn-taking density
            if spk and spk.strip():
                if prev_spk and prev_spk.lower() != spk.lower():
                    speaker_switches += 1
                prev_spk = spk

            # First meeting indicator
            seq = getattr(u, "seq", i + u_idx)
            if first_meeting_turn is not None and seq == first_meeting_turn:
                has_first_meeting = True

        turn_density = speaker_switches / max(1, len(window) - 1)
        salience_score = (
            3.0 * n_vocatives
            + (2.5 if has_first_meeting else 0.0)
            + 1.0 * n_emotion
            + 1.0 * turn_density
        )

        scored_windows.append(
            SalienceWindow(
                score=salience_score,
                units=tuple(window),
                pair=pair or ("", ""),
                first_meeting=has_first_meeting,
            )
        )

    scored_windows.sort(key=lambda w: w.score, reverse=True)
    return scored_windows[:max_windows]


def isolate_dyad_salience_windows(
    chunks: Sequence[DiscourseChunk],
    max_windows_per_dyad: int = 3,
    window_size: int = 4,
) -> dict[tuple[str, str], list[SalienceWindow]]:
    """Deterministically isolate the top high-salience windows for each interacting pair."""
    dyad_units: dict[tuple[str, str], list[GroundingUnit]] = {}
    first_meeting_seqs: dict[tuple[str, str], int] = {}

    for ch in chunks:
        speakers = sorted(ch.conversational_speakers)
        if len(speakers) < 2:
            continue
        for i, s1 in enumerate(speakers):
            for s2 in speakers[i + 1 :]:
                dyad_key = (s1, s2)
                # Filter units in this scene spoken by either character
                relevant = [
                    u for u in ch.units
                    if u.speaker and u.speaker.lower() in (s1.lower(), s2.lower())
                ]
                if relevant:
                    if dyad_key not in first_meeting_seqs:
                        first_meeting_seqs[dyad_key] = relevant[0].seq
                    dyad_units.setdefault(dyad_key, []).extend(relevant)

    results: dict[tuple[str, str], list[SalienceWindow]] = {}
    for dyad_key, units in dyad_units.items():
        fm = first_meeting_seqs.get(dyad_key)
        windows = isolate_salience_windows(
            units,
            window_size=window_size,
            max_windows=max_windows_per_dyad,
            pair=dyad_key,
            first_meeting_turn=fm,
        )
        results[dyad_key] = windows

    return results


def build_topology_analysis(
    source: Sequence[DiscourseChunk] | Project | Path | str,
) -> TopologyAnalysis:
    """Build a comprehensive Pass 1 topological interaction analysis."""
    chunks: list[DiscourseChunk]
    if isinstance(source, (Project, Path, str)):
        chunks = parse_grounded_script(source)
    else:
        chunks = list(source)

    # Exclude non-interaction UI, options, and screen scripts
    excluded_files = frozenset(
        {
            "screens.rpy",
            "gui.rpy",
            "options.rpy",
            "screens.rpyc",
            "gui.rpyc",
            "options.rpyc",
        }
    )
    chunks = [
        ch
        for ch in chunks
        if Path(ch.file).name.lower() not in excluded_files
    ]

    speakers, matrix = compute_cooccurrence_matrix(chunks)
    dynamic_name_tags = frozenset(detect_dynamic_name_patterns(source))

    speaker_turns: Counter[str] = Counter()
    turn_matrix = {s1: dict.fromkeys(speakers, 0) for s1 in speakers}
    prev_speaker: str | None = None

    for ch in chunks:
        for u in ch.units:
            if u.kind is UtteranceKind.SPOKEN and u.speaker:
                spk = u.speaker
                if spk not in speakers:
                    continue
                speaker_turns[spk] += 1
                if prev_speaker and prev_speaker in speakers and spk in speakers and prev_speaker != spk:
                    turn_matrix[prev_speaker][spk] += 1
                    turn_matrix[spk][prev_speaker] += 1
                prev_speaker = spk

    degrees = {s: sum(matrix[s].values()) for s in speakers}
    max_deg = max(degrees.values()) if degrees else 0
    degree_centralities = {
        s: (degrees[s] / max_deg if max_deg > 0 else 0.0)
        for s in speakers
    }

    protagonist = discover_topological_protagonist(
        matrix=matrix,
        speaker_turns=speaker_turns,
        dynamic_name_tags=dynamic_name_tags,
    )

    dyad_windows = isolate_dyad_salience_windows(chunks)

    return TopologyAnalysis(
        speakers=speakers,
        cooccurrence_matrix=matrix,
        turn_matrix=turn_matrix,
        speaker_turns=dict(speaker_turns),
        total_turns=sum(speaker_turns.values()),
        dynamic_name_tags=dynamic_name_tags,
        degree_centralities=degree_centralities,
        protagonist_tag=protagonist,
        dyad_salience_windows=dyad_windows,
    )
