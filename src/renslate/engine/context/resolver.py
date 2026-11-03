"""Interlocutor resolution, turn-taking adjacency, and vocative detection for renslate."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from renslate.core.models import CharacterProfile, Unit
from renslate.domain.policies.attention import DiscourseAttentionPolicy

if TYPE_CHECKING:
    from renslate.engine.graph import RelationshipGraph

__all__ = [
    "_KINSHIP_WORDS",
    "_LEADING_VOCATIVE_RE",
    "_MEDIAL_VOCATIVE_RE",
    "_NON_DIALOGUE_TAGS",
    "_PROTAGONIST_TAGS",
    "_RECIPROCAL_PRONOUN_PAIRS",
    "_TRAILING_VOCATIVE_RE",
    "InterlocutorResolver",
    "ResolvedInterlocutor",
    "_analyze_monologue_and_dialogue",
    "_extract_unit_file",
    "_is_scene_boundary",
    "_strip_outer_quotes",
    "harmonize_reciprocal_rules",
    "invert_reciprocal_pronoun_rule",
]

_NON_DIALOGUE_TAGS = frozenset(
    {
        "t",
        "th",
        "thought",
        "thinking",
        "narrator",
        "narration",
        "none",
        "all",
        "system",
        "ui",
        "menu",
    }
)

_LEADING_VOCATIVE_RE = re.compile(
    r"^[“\"'«„「『(\uff08]*\s*(?:(?:hey|hi|hello|chào|này|ơi|oi|dear)\s+)?"
    r"([A-Za-zÀ-ỹ0-9_'\- ]+?)\s*"
    r"(?:\.{2,}|…|~|\u301c|[,!?:;\-—\uff0c\uff01\uff1f\uff1a\uff1b])(?:\s+|$)",
    re.IGNORECASE,
)

_TRAILING_VOCATIVE_RE = re.compile(
    r"(?:[,;—\-\uff0c\uff1b—]|(?:\.{2,}|…|~|\u301c))\s*(?:(?:hey|hi|hello|chào|này|ơi|oi|dear)\s+)?"
    r"([A-Za-zÀ-ỹ0-9_'\- ]+?)\s*[\.!\?~…'\"”»\uff09\)」』]*$",
    re.IGNORECASE,
)

_MEDIAL_VOCATIVE_RE = re.compile(
    r"(?:[,;—\-\uff0c\uff1b—]|(?:\.{2,}|…|~|\u301c))\s*([A-Za-zÀ-ỹ0-9_'\-]+)\s*"
    r"(?:[,;—\-\uff0c\uff1b—]|(?:\.{2,}|…|~|\u301c))",
    re.IGNORECASE,
)

_NON_VOCATIVE_PRE_WORDS = frozenset(
    {
        "and",
        "or",
        "but",
        "about",
        "with",
        "to",
        "for",
        "from",
        "at",
        "by",
        "is",
        "are",
        "was",
        "were",
        "where",
        "what",
        "who",
        "whom",
        "whose",
        "how",
        "why",
        "like",
        "meet",
        "saw",
        "see",
        "ask",
        "asked",
        "call",
        "called",
    }
)

_CLAUSE_TRAILING_VOCATIVE_RE = re.compile(
    r"(?:^|(\b\w+\b)\s+)([A-Za-zÀ-ỹ0-9_'\-]+)\s*([,!?:;\-—\uff0c\uff01\uff1f~…])\s*$",
    re.IGNORECASE,
)

_KINSHIP_WORDS = frozenset(
    {
        "father",
        "dad",
        "papa",
        "mother",
        "mom",
        "mama",
        "brother",
        "sister",
        "uncle",
        "aunt",
        "cha",
        "mẹ",
        "bố",
        "chị",
        "em",
        "lord",
        "master",
        "sensei",
        "bệ hạ",
        "sư phụ",
        "tiên sinh",
        "king",
        "sir",
        "duke",
    }
)

_PROTAGONIST_TAGS = frozenset(
    {"mc", "player", "hero", "protag", "protagonist", "main", "you", "me"}
)

_RECIPROCAL_PRONOUN_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("anh", "em"), ("em", "anh"),
        ("chị", "em"), ("em", "chị"),
        ("bố", "con"), ("con", "bố"),
        ("cha", "con"), ("con", "cha"),
        ("mẹ", "con"), ("con", "mẹ"),
        ("ba", "con"), ("con", "ba"),
        ("má", "con"), ("con", "má"),
        ("ông", "cháu"), ("cháu", "ông"),
        ("bà", "cháu"), ("cháu", "bà"),
        ("chú", "cháu"), ("cháu", "chú"),
        ("cô", "cháu"), ("cháu", "cô"),
        ("dì", "cháu"), ("cháu", "dì"),
        ("bác", "cháu"), ("cháu", "bác"),
        ("thầy", "trò"), ("trò", "thầy"),
        ("thầy", "em"), ("em", "thầy"),
        ("thầy", "con"), ("con", "thầy"),
        ("sư phụ", "đồ nhi"), ("đồ nhi", "sư phụ"),
        ("sư phụ", "đệ tử"), ("đệ tử", "sư phụ"),
        ("sư phụ", "con"), ("con", "sư phụ"),
        ("tiền bối", "hậu bối"), ("hậu bối", "tiền bối"),
        ("huynh", "đệ"), ("đệ", "huynh"),
        ("huynh", "muội"), ("muội", "huynh"),
        ("tỷ", "muội"), ("muội", "tỷ"),
        ("tỷ", "đệ"), ("đệ", "tỷ"),
        ("chàng", "nàng"), ("nàng", "chàng"),
        ("ta", "con"), ("con", "ta"),
        ("tớ", "cậu"), ("cậu", "tớ"),
        ("mình", "cậu"), ("cậu", "mình"),
    }
)


def invert_reciprocal_pronoun_rule(rule: str) -> str | None:
    rule_clean = rule.strip()
    if " - " in rule_clean and not rule_clean.startswith("xưng"):
        pts = [p.strip() for p in rule_clean.split(" - ", 1)]
        if len(pts) == 2:
            key = (pts[0].lower(), pts[1].lower())
            if key in _RECIPROCAL_PRONOUN_PAIRS:
                return f"{pts[1]} - {pts[0]}"
    return None


def harmonize_reciprocal_rules(
    rule_a_to_b: str | None, rule_b_to_a: str | None
) -> tuple[str | None, str | None]:
    """Harmonize reciprocal pronoun rules between two characters.

    If one character has an intimate reciprocal rule (e.g. 'em - anh')
    and the other has a generic rule (e.g. 'tôi - em', 'tôi - cậu', or None),
    the reciprocal rule ('anh - em') overrides the generic rule to eliminate asymmetry.
    """
    inv_a = invert_reciprocal_pronoun_rule(rule_a_to_b) if rule_a_to_b else None
    inv_b = invert_reciprocal_pronoun_rule(rule_b_to_a) if rule_b_to_a else None

    if inv_a and (not rule_b_to_a or (rule_b_to_a.strip().lower().startswith(("tôi -", "ta -")) and not inv_b)):
        rule_b_to_a = inv_a
    elif inv_b and (not rule_a_to_b or (rule_a_to_b.strip().lower().startswith(("tôi -", "ta -")) and not inv_a)):
        rule_a_to_b = inv_b

    return rule_a_to_b, rule_b_to_a

_FEMALE_KINSHIP: frozenset[str] = frozenset(
    {"mother", "mom", "mama", "mẹ", "má", "sister", "chị", "em gái", "aunt", "cô", "dì", "bà", "con gái", "cháu gái"}
)
_MALE_KINSHIP: frozenset[str] = frozenset(
    {"father", "dad", "papa", "cha", "bố", "ba", "brother", "anh trai", "uncle", "chú", "bác", "ông", "con trai", "cháu trai"}
)

_FEMININE_NAMES = re.compile(
    r"^(?:charlotte|alice|eileen|diana|evelyn|luna|iris|emily|sarah|mary|anna|clara|lucy|sophia|"
    r"olivia|emma|chloe|mia|lily|grace|zoe|eva|elena|stella|amber|ruby|rose|victoria|valerie|karyn|zuri|"
    r"kaede|tomoe|yui|aoi|akane|"
    r"[a-z]*(?:ko|ka|na|ri|mi|ne|tsu|ki|yu))$",
    re.IGNORECASE,
)
_MASCULINE_NAMES = re.compile(
    r"^(?:jin|orson|kevin|john|david|michael|alex|mike|bob|tom|jack|george|arthur|"
    r"[a-z]*(?:taro|ro|shi|to|ya|suke))$",
    re.IGNORECASE,
)


def _gender_conflict(cand_term: str, target_name: str, target_tag: str) -> bool:
    cand_l = cand_term.lower()
    name_l = target_name.lower()
    tag_l = target_tag.lower()
    is_masc = bool(
        re.search(r"\b(?:he|him|his|boy|man|male|prince|king|lord|sir|mr|father|brother|uncle|son|nam|ông|chú|bác|anh|bố|cha|chàng|con trai|cháu trai)\b", name_l)
        or _MASCULINE_NAMES.match(name_l)
        or _MASCULINE_NAMES.match(tag_l)
    )
    is_fem = bool(
        re.search(r"\b(?:she|her|girl|woman|female|princess|queen|lady|miss|mrs|ms|mother|sister|aunt|daughter|nữ|bà|cô|dì|chị|mẹ|nàng|con gái|cháu gái)\b", name_l)
        or _FEMININE_NAMES.match(name_l)
        or _FEMININE_NAMES.match(tag_l)
    )
    if cand_l in _FEMALE_KINSHIP and is_masc and not is_fem:
        return True
    return bool(cand_l in _MALE_KINSHIP and is_fem and not is_masc)


def _strip_outer_quotes(text: str) -> str:
    """Strip matching pairs of outer quotation marks and Japanese brackets."""
    t = text.strip()
    quote_pairs = (
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("\u2018", "\u2019"),
        ("«", "»"),
        ("„", "”"),
        ("„", '"'),
        ("「", "」"),
    )
    changed = True
    while changed and len(t) >= 2:
        changed = False
        for op, cl in quote_pairs:
            if t.startswith(op) and t.endswith(cl):
                t = t[len(op) : -len(cl)].strip()
                changed = True
                break
    return t


def _analyze_monologue_and_dialogue(text: str) -> tuple[bool, str]:
    """Classify monologue state and extract stripped spoken dialogue in O(1) time."""
    t = text.strip()
    if t.startswith("『") and t.endswith("』"):
        return True, ""

    unquoted = _strip_outer_quotes(t)
    if (unquoted.startswith("(") and unquoted.endswith(")")) or (
        unquoted.startswith("\uff08") and unquoted.endswith("\uff09")
    ):
        return True, ""

    # Check for parenthesized stage direction: e.g. "(Laughs) Hey Liana, look at this!"
    stage_match = re.match(r"^[(\uff08][^)\uff09\n]+[)\uff09]\s*", unquoted)
    if stage_match:
        remainder = unquoted[stage_match.end() :].strip()
        if re.search(r"\w", remainder):
            return False, remainder
        return True, ""

    if unquoted.startswith("(") or unquoted.startswith("\uff08"):
        return True, ""

    return False, unquoted


@dataclass(frozen=True, slots=True)
class ResolvedInterlocutor:
    """Immutable resolution of dialogue participant roles."""

    speaker: str | None
    listener: str | None = None
    is_monologue: bool = False
    detected_emotion: str | None = None
    resolution_method: str = "default"


def _extract_unit_file(unit: Unit | None) -> str | None:
    """Safely extract filename from unit ref."""
    if not unit or not unit.ref:
        return None
    ref = unit.ref.strip()
    if ":" in ref:
        return ref.split(":", 1)[0]
    return ref


def _is_scene_boundary(curr: Unit, prev: Unit | None) -> bool:
    """Detect whether sequence transition crossed a scene boundary."""
    if prev is None:
        return True
    if abs(curr.seq - prev.seq) > 30:
        return True
    if curr.label and prev.label and curr.label != prev.label:
        return True
    curr_file = _extract_unit_file(curr)
    prev_file = _extract_unit_file(prev)
    return bool(curr_file and prev_file and curr_file != prev_file)


class InterlocutorResolver:
    """Real-time O(1) zero-token resolver for interlocutor turn-taking and vocatives."""

    def __init__(self, history_window: int = 12) -> None:
        self._aliases: dict[str, str] = {}
        self._speaker_aliases: dict[tuple[str, str], str] = {}
        self._prev_unit: Unit | None = None
        self._prev_speaker: str | None = None
        self._last_listener: str | None = None
        # Recent spoken turns (speaker, seq) of the current scene. Narration, thoughts and
        # same-speaker runs do not break the "who was I talking to" memory.
        self._history_window = max(0, history_window)
        deque_capacity = max(16, self._history_window + 4)
        self._recent_turns: deque[tuple[str, int]] = deque(maxlen=deque_capacity)

    def reset(self) -> None:
        """Clear dynamic alias associations and turn-taking state for a new scene."""
        self._aliases.clear()
        self._speaker_aliases.clear()
        self._prev_unit = None
        self._prev_speaker = None
        self._last_listener = None
        self._recent_turns.clear()

    def _register_alias(self, cand: str, target: str, spk: str | None = None) -> None:
        """Register vocative/kinship title alias globally and with speaker-scoping."""
        cand_clean = cand.strip().lower()
        self._aliases[cand_clean] = target
        if spk:
            self._speaker_aliases[(spk.strip().lower(), cand_clean)] = target

    def _remember_turn(self, speaker: str | None, unit: Unit) -> None:
        if not speaker:
            return
        if self._recent_turns and self._recent_turns[-1][0] == speaker:
            # Collapse same-speaker runs (and repeated resolves of one unit) into one entry.
            self._recent_turns[-1] = (speaker, unit.seq)
        else:
            self._recent_turns.append((speaker, unit.seq))

    def _fallback_listener(
        self,
        spk: str,
        unit: Unit,
        active_speakers: set[str] | Sequence[str] | None,
        allowed_targets: set[str] | None = None,
        prev_unit: Unit | None = None,
    ) -> tuple[str | None, str]:
        """Last-resort listener guess: recent other speaker, else the only other participant."""
        spk_lower = spk.lower()

        # Question-answer adjacency turn: if previous turn was an unanswered question by someone else
        if (
            prev_unit
            and prev_unit.speaker
            and prev_unit.speaker.lower() != spk_lower
            and prev_unit.source.strip().endswith("?")
            and (not allowed_targets or prev_unit.speaker.lower() in allowed_targets)
            and 0 <= unit.seq - prev_unit.seq <= 4
        ):
            return prev_unit.speaker, "question_adjacency"

        if self._history_window > 0:
            for other, seq in reversed(self._recent_turns):
                if other.lower() == spk_lower:
                    continue
                if 0 <= unit.seq - seq <= self._history_window:
                    if allowed_targets and other.lower() not in allowed_targets:
                        continue
                    return other, "recent_turn"
                break

        speakers_pool = (
            allowed_targets
            if allowed_targets
            else ({a.strip() for a in active_speakers} if active_speakers else set())
        )
        others = {
            a.strip()
            for a in speakers_pool
            if a
            and a.strip().lower() != spk_lower
            and a.strip().lower() not in _NON_DIALOGUE_TAGS
        }
        if len(others) == 1:
            return next(iter(others)), "dyad"

        # Same-speaker continuation fallback: if same speaker is still speaking,
        # preserve the last established listener if verified present and immediate.
        if (
            self._prev_speaker
            and self._prev_speaker.lower() == spk_lower
            and self._last_listener
            and self._last_listener.lower() != spk_lower
            and (not allowed_targets or self._last_listener.lower() in allowed_targets)
            and self._prev_unit
            and 0 <= unit.seq - self._prev_unit.seq <= 4
        ):
            return self._last_listener, "same_speaker_continuation"
        return None, "default"

    @staticmethod
    def _detect_emotion(text: str) -> str | None:
        """Lightweight O(1) punctuation and marker-based emotion detection."""
        t = text.strip()
        if "!?" in t or "?!" in t:
            return "surprised"
        if "..." in t or "…" in t:
            return "hesitant"
        if "~" in t or "\u301c" in t:
            return "affectionate"
        if t.endswith("!"):
            if t.isupper() and len(t) > 3:
                return "shouting"
            return "emphatic"
        if t.endswith("?"):
            return "inquiring"
        return None

    def _build_name_map(
        self,
        active_speakers: set[str] | Sequence[str] | None,
        known_names: dict[str, str] | None,
    ) -> dict[str, str]:
        """Construct normalized tag and display name index for O(1) vocative lookup."""
        name_map: dict[str, str] = {}
        if known_names:
            for tag, disp in known_names.items():
                t = tag.strip().lower()
                name_map[t] = tag
                d = disp.strip().lower()
                name_map[d] = tag
                words = [w.strip() for w in d.split() if len(w.strip()) > 2]
                for w in words:
                    if w not in _KINSHIP_WORDS and w not in name_map:
                        name_map[w] = tag
        if active_speakers:
            for tag in active_speakers:
                t = tag.strip().lower()
                if t not in name_map:
                    name_map[t] = tag
        return name_map

    @staticmethod
    def _clean_cand(raw: str) -> str:
        c = raw.strip().lower()
        for prefix in ("chào ", "hey ", "hi ", "hello ", "này ", "dear "):
            if c.startswith(prefix):
                c = c[len(prefix) :].strip()
        for suffix in (" ơi", " này", " ạ", " oi"):
            if c.endswith(suffix):
                c = c[: -len(suffix)].strip()
        return c

    def _extract_vocative(
        self,
        text: str,
        name_map: dict[str, str],
        speaker: str | None,
        allowed_targets: set[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Extract vocatively addressed character from text, returning (target_tag, method)."""
        clean = text.strip()
        speaker_norm = speaker.strip().lower() if speaker else None

        # 0. Normalize technical placeholders and pause markers to clause breaks
        clean_clauses = re.sub(r"⟦[^⟧]+⟧", "\n", clean)
        sentences = [
            s.strip()
            for part in re.split(r"[\n\r]+", clean_clauses)
            for s in re.split(r"(?<=[.!?])\s+", part)
            if s.strip()
        ]
        if not sentences:
            sentences = [clean]

        def _lookup(cand: str) -> str | None:
            if DiscourseAttentionPolicy.is_third_person_reference(clean, cand):
                return None
            tag = None
            if speaker_norm and (speaker_norm, cand) in self._speaker_aliases:
                tag = self._speaker_aliases[(speaker_norm, cand)]
            if not tag:
                tag = name_map.get(cand) or self._aliases.get(cand)
            if tag and tag.lower() != speaker_norm:
                if allowed_targets and tag.lower() not in allowed_targets:
                    return None
                if DiscourseAttentionPolicy.is_third_person_reference(clean, tag):
                    return None
                return tag
            return None

        # 1. Leading vocative in first sentence (highest priority)
        m = _LEADING_VOCATIVE_RE.match(sentences[0])
        if m:
            cand = self._clean_cand(m.group(1))
            tag = _lookup(cand)
            if tag:
                return tag, "vocative"

        # 2. Trailing vocative in last sentence or any sentence (in reverse)
        for s in reversed(sentences):
            m = _TRAILING_VOCATIVE_RE.search(s)
            if m:
                cand = self._clean_cand(m.group(1))
                tag = _lookup(cand)
                if tag:
                    return tag, "vocative"

        # 2b. Clause-trailing vocative across separated clauses / pause markers
        for s in reversed(sentences):
            m = _CLAUSE_TRAILING_VOCATIVE_RE.search(s)
            if m:
                pre_word = (m.group(1) or "").lower()
                punct = m.group(3)
                if pre_word not in _NON_VOCATIVE_PRE_WORDS and not (punct == "?" and not pre_word):
                    cand = self._clean_cand(m.group(2))
                    tag = _lookup(cand)
                    if tag:
                        return tag, "vocative"

        # 3. Leading vocative in subsequent sentences
        for s in sentences[1:]:
            m = _LEADING_VOCATIVE_RE.match(s)
            if m:
                cand = self._clean_cand(m.group(1))
                tag = _lookup(cand)
                if tag:
                    return tag, "vocative"

        # 4. Medial vocative (only if strictly present in allowed_targets)
        for s in sentences:
            for m in _MEDIAL_VOCATIVE_RE.finditer(s):
                cand = self._clean_cand(m.group(1))
                tag = _lookup(cand)
                if tag:
                    return tag, "vocative"

        return None, None

    def _resolve_kinship_target(
        self,
        cand: str,
        spk: str | None,
        active_speakers: set[str] | Sequence[str] | None,
        known_names: dict[str, str] | None,
        prev_unit: Unit | None,
        src: str,
        allowed_targets: set[str] | None = None,
    ) -> str | None:
        """Resolve kinship title addressee using candidate scoring."""
        spk_lower = spk.lower() if spk else ""
        cached = (
            self._speaker_aliases.get((spk_lower, cand))
            if spk_lower
            else None
        ) or self._aliases.get(cand)
        if (
            cached
            and (not allowed_targets or cached.lower() in allowed_targets)
            and (not spk or cached.lower() != spk_lower)
        ):
            return cached

        pool = allowed_targets if allowed_targets else (
            {a.strip() for a in active_speakers} if active_speakers else set()
        )
        if not pool:
            return None

        candidates = [a for a in pool if a.lower() != spk_lower]
        if not candidates:
            return None

        prev_spk = prev_unit.speaker.lower() if prev_unit and prev_unit.speaker else None

        def _score(c: str) -> float:
            score = 0.0
            disp = (known_names.get(c) if known_names else None) or c
            disp_lower = disp.lower()
            c_lower = c.lower()

            disp_words = set(re.findall(r"[\wÀ-ỹ]+", disp_lower))
            has_kinship_evidence = False
            if cand in disp_words or cand in disp_lower:
                score += 50.0
                has_kinship_evidence = True

            for kw in _KINSHIP_WORDS:
                if kw in disp_words:
                    score += 20.0
                    has_kinship_evidence = True
                    break

            name_tokens = {c_lower} | {
                w for w in disp_words if len(w) > 2 and w not in _KINSHIP_WORDS
            }
            for tok in name_tokens:
                if re.search(r"\b" + re.escape(tok) + r"\b", src, re.I):
                    score -= 100.0
                    break

            if has_kinship_evidence and prev_spk and c_lower == prev_spk:
                score += 10.0

            return score

        if len(candidates) == 1:
            target = candidates[0]
            disp = (known_names.get(target) if known_names else None) or target
            if _gender_conflict(cand, disp, target):
                return None
            if _score(target) <= 0.0:
                return None
            self._register_alias(cand, target, spk)
            return target

        scored = [(c, _score(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored[0][1] <= 0.0:
            return None
        if scored[0][1] > scored[1][1]:
            target = scored[0][0]
            disp = (known_names.get(target) if known_names else None) or target
            if _gender_conflict(cand, disp, target):
                return None
            self._register_alias(cand, target, spk)
            return target

        if prev_spk:
            for c, s in scored:
                if c.lower() == prev_spk and s > 0.0:
                    disp = (known_names.get(c) if known_names else None) or c
                    if not _gender_conflict(cand, disp, c):
                        self._register_alias(cand, c, spk)
                        return c

        for c, s in scored:
            if s > 0.0:
                disp = (known_names.get(c) if known_names else None) or c
                if not _gender_conflict(cand, disp, c) and (cand in disp.lower() or c.lower() == cand):
                    self._register_alias(cand, c, spk)
                    return c
        return None

    def resolve(
        self,
        unit: Unit,
        prev_unit: Unit | None = None,
        active_speakers: set[str] | Sequence[str] | None = None,
        known_names: dict[str, str] | None = None,
        present_characters: set[str] | Sequence[str] | None = None,
    ) -> ResolvedInterlocutor:
        """Resolve speaker, listener, and monologue state in O(1) time."""
        effective_prev = prev_unit if prev_unit is not None else self._prev_unit
        if effective_prev is not None and _is_scene_boundary(unit, effective_prev):
            self.reset()
            effective_prev = None

        spk = unit.speaker.strip() if unit.speaker else None
        spk_lower = spk.lower() if spk else ""
        src = unit.source.strip()
        emotion = self._detect_emotion(src)

        # 1. Monologue / Soliloquy classification
        is_mono, clean_dial = _analyze_monologue_and_dialogue(src)
        if not spk or spk_lower in _NON_DIALOGUE_TAGS or is_mono:
            self._prev_unit = unit
            self._prev_speaker = spk
            return ResolvedInterlocutor(
                speaker=spk,
                listener=None,
                is_monologue=True,
                detected_emotion=emotion,
                resolution_method="monologue",
            )

        name_map = self._build_name_map(active_speakers, known_names)

        # Collect allowed active presence targets (if scene presence or stage is explicitly specified)
        allowed_targets: set[str] | None = None
        if present_characters:
            allowed_targets = {
                p.strip().lower() for p in present_characters if p and p.strip()
            }
            allowed_targets.difference_update(_NON_DIALOGUE_TAGS)

        if hasattr(unit, "stage") and unit.stage is not None:
            stage_tags = getattr(unit.stage, "active_tags", None)
            if stage_tags:
                if allowed_targets is not None:
                    allowed_targets = allowed_targets.intersection(stage_tags)
                else:
                    allowed_targets = set(stage_tags)
                allowed_targets.difference_update(_NON_DIALOGUE_TAGS)

        # 2. Vocative Detection on clean dialogue
        vocative_target, _ = self._extract_vocative(
            clean_dial, name_map, spk, allowed_targets=allowed_targets
        )
        if vocative_target:
            self._prev_unit = unit
            self._prev_speaker = spk
            self._last_listener = vocative_target
            self._remember_turn(spk, unit)
            return ResolvedInterlocutor(
                speaker=spk,
                listener=vocative_target,
                is_monologue=False,
                detected_emotion=emotion,
                resolution_method="vocative",
            )

        # 3. Kinship / Title Address Fallback (leading and trailing)
        clean_dial_clauses = re.sub(r"⟦[^⟧]+⟧", "\n", clean_dial)
        sentences = [
            s.strip()
            for part in re.split(r"[\n\r]+", clean_dial_clauses)
            for s in re.split(r"(?<=[.!?])\s+", part)
            if s.strip()
        ]
        if not sentences:
            sentences = [clean_dial]

        for s in sentences:
            m_lead = _LEADING_VOCATIVE_RE.match(s)
            if m_lead:
                cand = self._clean_cand(m_lead.group(1))
                if cand in _KINSHIP_WORDS:
                    target = self._resolve_kinship_target(
                        cand=cand,
                        spk=spk,
                        active_speakers=active_speakers,
                        known_names=known_names,
                        prev_unit=effective_prev,
                        src=clean_dial,
                        allowed_targets=allowed_targets,
                    )
                    if target:
                        self._prev_unit = unit
                        self._prev_speaker = spk
                        self._last_listener = target
                        self._remember_turn(spk, unit)
                        return ResolvedInterlocutor(
                            speaker=spk,
                            listener=target,
                            is_monologue=False,
                            detected_emotion=emotion,
                            resolution_method="vocative",
                        )

            m_trail = _TRAILING_VOCATIVE_RE.search(s)
            if m_trail:
                cand = self._clean_cand(m_trail.group(1))
                if cand in _KINSHIP_WORDS:
                    target = self._resolve_kinship_target(
                        cand=cand,
                        spk=spk,
                        active_speakers=active_speakers,
                        known_names=known_names,
                        prev_unit=effective_prev,
                        src=clean_dial,
                        allowed_targets=allowed_targets,
                    )
                    if target:
                        self._prev_unit = unit
                        self._prev_speaker = spk
                        self._last_listener = target
                        self._remember_turn(spk, unit)
                        return ResolvedInterlocutor(
                            speaker=spk,
                            listener=target,
                            is_monologue=False,
                            detected_emotion=emotion,
                            resolution_method="vocative",
                        )

        # 4. Turn-Taking Adjacency
        if effective_prev and effective_prev.speaker:
            prev_spk = effective_prev.speaker.strip()
            prev_src = effective_prev.source.strip()
            prev_is_mono, _ = _analyze_monologue_and_dialogue(prev_src)
            prev_is_non_dialogue = prev_spk.lower() in _NON_DIALOGUE_TAGS or prev_is_mono
            if (
                prev_spk.lower() != spk_lower
                and not prev_is_non_dialogue
                and (not allowed_targets or prev_spk.lower() in allowed_targets)
            ):
                self._prev_unit = unit
                self._prev_speaker = spk
                self._last_listener = prev_spk
                self._remember_turn(spk, unit)
                return ResolvedInterlocutor(
                    speaker=spk,
                    listener=prev_spk,
                    is_monologue=False,
                    detected_emotion=emotion,
                    resolution_method="turn_taking",
                )

        # 5. Fallbacks: narration between turns, same-speaker runs, two-person scenes
        fb_listener, fb_method = self._fallback_listener(
            spk, unit, active_speakers, allowed_targets=allowed_targets, prev_unit=effective_prev
        )
        self._prev_unit = unit
        self._prev_speaker = spk
        if fb_listener:
            self._last_listener = fb_listener
        self._remember_turn(spk, unit)
        return ResolvedInterlocutor(
            speaker=spk,
            listener=fb_listener,
            is_monologue=False,
            detected_emotion=emotion,
            resolution_method=fb_method,
        )

    def resolve_pronouns(
        self,
        speaker: str,
        listener: str,
        graph: RelationshipGraph | None = None,
        char_map: dict[str, CharacterProfile] | None = None,
    ) -> tuple[str, str] | None:
        """Resolve dynamic symmetric and asymmetric pronoun pair.

        Returns (speaker_pronoun, listener_pronoun).
        """
        spk_clean = speaker.strip()
        spk_lower = spk_clean.lower()
        lis_clean = listener.strip()
        lis_lower = lis_clean.lower()

        cp_spk = (char_map.get(spk_clean) or char_map.get(spk_lower)) if char_map else None
        cp_lis = (char_map.get(lis_clean) or char_map.get(lis_lower)) if char_map else None

        # 1. Direct edge in RelationshipGraph: A -> B
        if graph is not None:
            edge = graph.get_edge(spk_clean, lis_clean) or graph.get_edge(spk_lower, lis_lower)
            if edge is None and cp_spk and cp_lis:
                edge = graph.get_edge(cp_spk.tag, cp_lis.tag)
            if edge and edge.speaker_pronoun and edge.listener_pronoun:
                return edge.speaker_pronoun, edge.listener_pronoun

            # 2. Reciprocal edge in RelationshipGraph: B -> A (Symmetric inversion)
            reciprocal = graph.get_edge(lis_clean, spk_clean) or graph.get_edge(
                lis_lower, spk_lower
            )
            if reciprocal is None and cp_spk and cp_lis:
                reciprocal = graph.get_edge(cp_lis.tag, cp_spk.tag)
            if reciprocal and reciprocal.speaker_pronoun and reciprocal.listener_pronoun:
                return reciprocal.listener_pronoun, reciprocal.speaker_pronoun

        # 3. Direct pronoun map from CharacterProfile: cp_spk.pronoun_map
        if cp_spk and cp_spk.pronoun_map:
            tgt_key = cp_lis.tag if cp_lis else lis_clean
            tgt_name = cp_lis.display_name if cp_lis else lis_clean
            p_rule = (
                cp_spk.pronoun_map.get(tgt_key)
                or cp_spk.pronoun_map.get(tgt_name)
                or cp_spk.pronoun_map.get(tgt_key.lower())
            )
            if not p_rule:
                is_mc = (
                    tgt_key.lower() in _PROTAGONIST_TAGS
                    or (cp_lis is not None and "protagonist" in (cp_lis.role or "").lower())
                )
                if is_mc:
                    p_rule = cp_spk.pronoun_map.get("mc") or cp_spk.pronoun_map.get("player")
            if p_rule and " - " in p_rule and not p_rule.startswith("xưng"):
                parts = [p.strip() for p in p_rule.split(" - ", 1)]
                if len(parts) == 2:
                    return parts[0], parts[1]

        # 4. Reciprocal pronoun map from CharacterProfile: cp_lis.pronoun_map
        if cp_lis and cp_lis.pronoun_map:
            spk_key = cp_spk.tag if cp_spk else spk_clean
            spk_name = cp_spk.display_name if cp_spk else spk_clean
            p_rule_rev = (
                cp_lis.pronoun_map.get(spk_key)
                or cp_lis.pronoun_map.get(spk_name)
                or cp_lis.pronoun_map.get(spk_key.lower())
            )
            if not p_rule_rev:
                is_spk_mc = (
                    spk_key.lower() in _PROTAGONIST_TAGS
                    or (cp_spk is not None and "protagonist" in (cp_spk.role or "").lower())
                )
                if is_spk_mc:
                    p_rule_rev = cp_lis.pronoun_map.get("mc") or cp_lis.pronoun_map.get("player")
            if p_rule_rev and " - " in p_rule_rev and not p_rule_rev.startswith("xưng"):
                parts = [p.strip() for p in p_rule_rev.split(" - ", 1)]
                if len(parts) == 2:
                    p0 = parts[0].strip().lower()
                    p1 = parts[1].strip().lower()
                    if (p0, p1) in _RECIPROCAL_PRONOUN_PAIRS:
                        return parts[1], parts[0]

        return None
