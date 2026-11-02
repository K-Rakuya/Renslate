"""Orchestration pipeline for character discovery, scene segmentation, and glossary mining."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from renslate.core.db import (
    Database,
    load_characters_from_json,
    save_characters_to_json,
    utc_timestamp,
)
from renslate.core.models import (
    CharacterProfile,
    GlossaryEntry,
    RelationshipEdge,
    RelationshipKind,
    SceneMacroContext,
    SceneSummary,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.analyze.persona import (
    bootstrap_characters_with_llm,
    infer_character_gender_nlp,
    infer_character_role_nlp,
    infer_persona_fallback,
    infer_semantic_relationship,
    is_collective_entity,
    is_non_human_or_meta,
    resolve_pronoun_pair_from_relationship,
)
from renslate.engine.analyze.stylometrics import extract_stylometrics
from renslate.engine.analyze.tropes import (
    NON_INTERACTION_TAGS,
    PROTAGONIST_TAGS,
)
from renslate.engine.context import infer_scene_cues
from renslate.engine.context.resolver import invert_reciprocal_pronoun_rule
from renslate.engine.glossary import (
    Glossary,
    _make_plural_aliases,
    clean_glossary,
    collect_term_examples,
    mine_glossary_candidates,
    translate_glossary_with_llm,
)
from renslate.engine.graph import RelationshipGraph

_LOG = logging.getLogger("renslate.analyze.pipeline")

_RE_CHARACTER_DEF = re.compile(
    r'(?:define|default)\s+(?P<tag>\w+)\s*=\s*(?:Character|DynamicCharacter)\s*\(\s*(?P<name>_[^(]*\([^)]*\)|"(?:[^"\\]|\\.)*"|\'[^\']*\'|[A-Za-z_]\w*)'
)


def extract_character_definitions(project_dir: Path) -> dict[str, str]:
    """Scan game script files for Ren'Py Character definitions."""
    characters: dict[str, str] = {}
    game_dir = project_dir / "game"
    if game_dir.is_dir():
        for rpy in sorted(game_dir.rglob("*.rpy")):
            if "tl" in rpy.parts:
                continue
            try:
                content = rpy.read_text(encoding="utf-8-sig")
            except (UnicodeDecodeError, OSError):
                continue

            for m in _RE_CHARACTER_DEF.finditer(content):
                tag = m.group("tag")
                raw_name = m.group("name").strip()
                if (raw_name.startswith('_("') and raw_name.endswith('")')) or (
                    raw_name.startswith("_('") and raw_name.endswith("')")
                ):
                    name = raw_name[3:-2]
                elif (raw_name.startswith('"') and raw_name.endswith('"')) or (
                    raw_name.startswith("'") and raw_name.endswith("'")
                ):
                    name = raw_name[1:-1]
                elif raw_name in ("None", ""):
                    name = tag.capitalize()
                elif raw_name.endswith("_name"):
                    base = raw_name[:-5].strip()
                    if base.lower() in PROTAGONIST_TAGS or tag.lower() in PROTAGONIST_TAGS:
                        name = "Protagonist"
                    else:
                        name = base.capitalize()
                elif tag.lower() in PROTAGONIST_TAGS:
                    name = "Protagonist"
                else:
                    name = raw_name
                characters[tag] = re.sub(r"\b([A-Za-z])\s+([a-z]{2,})\b", r"\1\2", name).strip()

    if not characters:
        with contextlib.suppress(Exception):
            from renslate.renpy.rpyc import extract_character_definitions_from_rpyc

            characters = extract_character_definitions_from_rpyc(project_dir)

    return characters


def extract_scene_macro_contexts(
    db: Database,
    units: Sequence[Unit] | None = None,
) -> list[SceneMacroContext]:
    """Group dialogue units into rich hierarchical SceneMacroContext objects."""
    all_u = db.all_units() if units is None else list(units)
    dialogues = [u for u in all_u if u.kind is UnitKind.DIALOGUE]
    if not dialogues:
        return []

    macros: list[SceneMacroContext] = []
    current_label = dialogues[0].label or "prologue"
    scene_dialogues: list[Unit] = [dialogues[0]]
    scene_idx = 1

    def _build_macro(label: str, group: list[Unit], idx: int) -> SceneMacroContext:
        first_seq = group[0].seq
        last_seq = group[-1].seq
        scene_id = f"scene_{idx}_{label}"

        speakers: list[str] = []
        for u in group:
            if (
                u.speaker
                and u.speaker.strip()
                and u.speaker.strip().lower() not in NON_INTERACTION_TAGS
            ):
                spk = u.speaker.strip()
                if spk not in speakers:
                    speakers.append(spk)

        file_path = ""
        for u in group:
            if u.ref:
                file_path = u.ref.split(":", 1)[0].strip()
                if file_path:
                    break

        loc, atmos = infer_scene_cues(label)

        label_lower = label.lower()
        temporal = ""
        if "midnight" in label_lower:
            temporal = "Nửa đêm (Midnight)"
        elif "night" in label_lower:
            temporal = "Đêm khuya (Night)"
        elif "sunset" in label_lower or "dusk" in label_lower:
            temporal = "Hoàng hôn (Sunset/Dusk)"
        elif "dawn" in label_lower:
            temporal = "Bình minh (Dawn)"
        elif "morning" in label_lower:
            temporal = "Buổi sáng (Morning)"

        summary_parts = [f"Cảnh '{label}'"]
        if loc:
            summary_parts.append(f"tại {loc}")
        if speakers:
            summary_parts.append(f"với {', '.join(speakers)}")
        summary_parts.append(f"(từ dòng {first_seq} đến {last_seq})")
        summary_text = " ".join(summary_parts)

        return SceneMacroContext(
            scene_id=scene_id,
            label=label,
            file=file_path,
            summary=summary_text,
            location=loc,
            atmosphere=atmos,
            active_characters=tuple(speakers),
            temporal_anchor=temporal,
            first_seq=first_seq,
            last_seq=last_seq,
            metadata={"dialogue_count": len(group)},
        )

    for i in range(1, len(dialogues)):
        u = dialogues[i]
        label = u.label or current_label
        last_seq = scene_dialogues[-1].seq
        if label != current_label or (u.seq - last_seq > 50):
            macros.append(_build_macro(current_label, scene_dialogues, scene_idx))
            scene_idx += 1
            current_label = label
            scene_dialogues = [u]
        else:
            scene_dialogues.append(u)

    if scene_dialogues:
        macros.append(_build_macro(current_label, scene_dialogues, scene_idx))

    return macros


def segment_scenes(db: Database) -> list[SceneSummary]:
    """Group dialogue units into coherent scenes based on labels and sequence continuity."""
    macros = extract_scene_macro_contexts(db)
    return [
        SceneSummary(
            scene_id=m.scene_id,
            label=m.label,
            first_seq=m.first_seq,
            last_seq=m.last_seq,
            summary=f"Cảnh '{m.label}' (từ dòng {m.first_seq} đến {m.last_seq})",
        )
        for m in macros
    ]


def run_analysis(
    project: Project,
    db: Database,
    *,
    mine_glossary: bool = True,
    enrich_llm: bool = False,
    llm: Any = None,
) -> dict[str, int]:
    """Run full analysis pipeline: discover characters, segment scenes, and glossary."""
    stats = {"characters": 0, "scenes": 0, "glossary_candidates": 0, "glossary_translated": 0}

    defined_chars = extract_character_definitions(project.root)
    existing_chars = db.get_characters()

    all_units = db.all_units()
    dialogue_speakers = {u.speaker for u in all_units if u.speaker}
    speaker_counts: Counter[str] = Counter(u.speaker for u in all_units if u.speaker)
    sources = [u.source for u in all_units]
    total_units_count = len(all_units)

    dialogue_by_speaker: dict[str, list[str]] = {}
    for u in all_units:
        if u.kind is UnitKind.DIALOGUE and u.speaker and u.source.strip():
            dialogue_by_speaker.setdefault(u.speaker, []).append(u.source)

    all_tags = set(defined_chars.keys()) | dialogue_speakers
    rel_graph = RelationshipGraph.sync(db, project.relationships_path)

    interactions: dict[str, Counter[str]] = {}
    prev_speaker: str | None = None
    for u in all_units:
        if u.kind is UnitKind.DIALOGUE and u.speaker:
            spk = u.speaker
            if spk.lower() not in NON_INTERACTION_TAGS:
                if prev_speaker and prev_speaker != spk:
                    interactions.setdefault(prev_speaker, Counter())[spk] += 1
                    interactions.setdefault(spk, Counter())[prev_speaker] += 1
                    rel_graph.record_interaction(prev_speaker, spk)
                prev_speaker = spk

    chars_file = project.characters_path
    file_chars: dict[str, CharacterProfile] = {}
    if chars_file.is_file():
        with contextlib.suppress(Exception):
            file_chars = load_characters_from_json(chars_file)

    with db.transaction():
        temp_chars: dict[str, dict[str, Any]] = {}
        for tag in sorted(all_tags):
            display_name = defined_chars.get(tag, tag.capitalize())
            fc = file_chars.get(tag)
            ec = existing_chars.get(tag)

            gender = (
                (fc.gender if fc and fc.gender != "unknown" else "")
                or (ec.gender if ec and ec.gender != "unknown" else "")
            )
            if not gender or gender == "unknown":
                gender = infer_character_gender_nlp(display_name, sources)

            role = (fc.role if fc and fc.role else "") or (ec.role if ec and ec.role else "")
            if not role:
                role = infer_character_role_nlp(tag, speaker_counts[tag], total_units_count)

            speech_style = (fc.speech_style if fc else "") or (ec.speech_style if ec else "")
            pronoun_map = (
                (dict(fc.pronoun_map) if fc and fc.pronoun_map else {})
                or (dict(ec.pronoun_map) if ec and ec.pronoun_map else {})
            )

            speaker_lines = dialogue_by_speaker.get(tag, [])
            sty = extract_stylometrics(speaker_lines)
            fb = infer_persona_fallback(
                tag, speaker_lines, sty, role=role, gender=gender, display_name=display_name
            )

            archetype = (
                (fc.archetype if fc and fc.archetype else "")
                or (ec.archetype if ec and ec.archetype else "")
                or fb["archetype"]
            )
            personality_traits = (
                (fc.personality_traits if fc and fc.personality_traits else ())
                or (ec.personality_traits if ec and ec.personality_traits else ())
                or fb["personality_traits"]
            )
            tone_and_voice = (
                (fc.tone_and_voice if fc and fc.tone_and_voice else "")
                or (ec.tone_and_voice if ec and ec.tone_and_voice else "")
                or fb["tone_and_voice"]
            )
            socioeconomic_status = (
                (fc.socioeconomic_status if fc and fc.socioeconomic_status else "")
                or (ec.socioeconomic_status if ec and ec.socioeconomic_status else "")
                or fb["socioeconomic_status"]
            )
            sample_dialogue = (
                (fc.sample_dialogue if fc and fc.sample_dialogue else ())
                or (ec.sample_dialogue if ec and ec.sample_dialogue else ())
                or fb["sample_dialogue"]
            )

            temp_chars[tag] = {
                "display_name": (fc.display_name if fc and fc.display_name else display_name),
                "gender": gender,
                "role": role,
                "speech_style": speech_style,
                "pronoun_map": pronoun_map,
                "archetype": archetype,
                "personality_traits": personality_traits,
                "tone_and_voice": tone_and_voice,
                "socioeconomic_status": socioeconomic_status,
                "sample_dialogue": sample_dialogue,
            }

        has_explicit_mc = any(t.lower() in PROTAGONIST_TAGS for t in temp_chars)
        for tag, tc in temp_chars.items():
            pronoun_map = dict(tc["pronoun_map"])
            if has_explicit_mc:
                is_mc = tag.lower() in PROTAGONIST_TAGS
            else:
                is_mc = tc["role"] == "Nhân vật chính (Protagonist)"

            if is_non_human_or_meta(tag, tc["display_name"]):
                profile = CharacterProfile(
                    tag=tag,
                    display_name=tc["display_name"],
                    gender=tc["gender"],
                    role=tc["role"],
                    speech_style=tc["speech_style"],
                    pronoun_map=pronoun_map,
                    archetype=tc.get("archetype", ""),
                    personality_traits=tc.get("personality_traits", ()),
                    tone_and_voice=tc.get("tone_and_voice", ""),
                    socioeconomic_status=tc.get("socioeconomic_status", ""),
                    sample_dialogue=tc.get("sample_dialogue", ()),
                )
                db.save_character(profile)
                stats["characters"] += 1
                continue

            for partner_tag, _ in interactions.get(tag, Counter()).most_common(5):
                if partner_tag.lower() == tag.lower():
                    continue
                if partner_tag in pronoun_map:
                    continue

                partner_info = temp_chars.get(partner_tag, {})
                if is_non_human_or_meta(partner_tag, partner_info.get("display_name", "")):
                    continue

                if is_collective_entity(partner_tag, partner_info.get("display_name", "")):
                    pronoun_map[partner_tag] = "tôi - mọi người"
                    continue

                if is_collective_entity(tag, tc["display_name"]):
                    pronoun_map[partner_tag] = "mọi người - bạn"
                    continue

                partner_is_mc = (
                    partner_tag.lower() in PROTAGONIST_TAGS
                    if has_explicit_mc
                    else partner_info.get("role") == "Nhân vật chính (Protagonist)"
                )

                edge_out, edge_in = rel_graph.get_bilateral(tag, partner_tag)
                spk_pro = ""
                lis_pro = ""

                if edge_out and edge_out.speaker_pronoun and edge_out.listener_pronoun:
                    spk_pro = edge_out.speaker_pronoun
                    lis_pro = edge_out.listener_pronoun
                elif edge_in and edge_in.speaker_pronoun and edge_in.listener_pronoun:
                    inv = invert_reciprocal_pronoun_rule(
                        f"{edge_in.speaker_pronoun} - {edge_in.listener_pronoun}"
                    )
                    if inv:
                        spk_pro, lis_pro = [p.strip() for p in inv.split(" - ", 1)]
                elif edge_in and edge_in.kind:
                    if edge_in.kind in (
                        RelationshipKind.RIVAL_HOSTILE,
                        RelationshipKind.FRIENDSHIP_CLOSE,
                        RelationshipKind.FRIENDSHIP_CASUAL,
                        RelationshipKind.FORMAL_ACQUAINTANCE,
                    ):
                        spk_pro, lis_pro = resolve_pronoun_pair_from_relationship(
                            edge_in.kind,
                            tc["gender"],
                            partner_info.get("gender", "unknown"),
                        )
                    else:
                        inbound_is_senior = edge_in.speaker_pronoun in (
                            "mẹ", "cha", "bố", "ba", "ông", "bà", "bác", "chú", "cô", "dì",
                            "anh", "chị", "thầy", "sư phụ", "tiền bối", "trẫm", "ta",
                        )
                        spk_pro, lis_pro = resolve_pronoun_pair_from_relationship(
                            edge_in.kind,
                            tc["gender"],
                            partner_info.get("gender", "unknown"),
                            is_speaker_senior=not inbound_is_senior,
                        )
                elif edge_out and edge_out.kind:
                    spk_pro, lis_pro = resolve_pronoun_pair_from_relationship(
                        edge_out.kind,
                        tc["gender"],
                        partner_info.get("gender", "unknown"),
                        is_speaker_senior=True,
                    )

                if not spk_pro or not lis_pro:
                    inferred = infer_semantic_relationship(
                        tag, tc, partner_tag, partner_info, is_mc=is_mc, partner_is_mc=partner_is_mc
                    )
                    if inferred is not None:
                        kind, is_senior = inferred
                        spk_pro, lis_pro = resolve_pronoun_pair_from_relationship(
                            kind,
                            tc["gender"],
                            partner_info.get("gender", "unknown"),
                            is_speaker_senior=is_senior,
                        )
                        rel_graph.add_edge(
                            RelationshipEdge(
                                source_tag=tag,
                                target_tag=partner_tag,
                                kind=kind,
                                speaker_pronoun=spk_pro,
                                listener_pronoun=lis_pro,
                                confidence=0.7,
                                source_type="heuristic",
                            )
                        )
                        if rel_graph.get_edge(partner_tag, tag) is None:
                            if kind in (
                                RelationshipKind.RIVAL_HOSTILE,
                                RelationshipKind.FRIENDSHIP_CLOSE,
                                RelationshipKind.FRIENDSHIP_CASUAL,
                                RelationshipKind.FORMAL_ACQUAINTANCE,
                            ):
                                recip_spk, recip_lis = resolve_pronoun_pair_from_relationship(
                                    kind,
                                    partner_info.get("gender", "unknown"),
                                    tc["gender"],
                                )
                            else:
                                recip_spk, recip_lis = resolve_pronoun_pair_from_relationship(
                                    kind,
                                    partner_info.get("gender", "unknown"),
                                    tc["gender"],
                                    is_speaker_senior=not is_senior,
                                )
                            rel_graph.add_edge(
                                RelationshipEdge(
                                    source_tag=partner_tag,
                                    target_tag=tag,
                                    kind=kind,
                                    speaker_pronoun=recip_spk,
                                    listener_pronoun=recip_lis,
                                    confidence=0.7,
                                    source_type="heuristic",
                                )
                            )

                if spk_pro and lis_pro:
                    pronoun_map[partner_tag] = f"{spk_pro} - {lis_pro}"

            pronoun_map.pop(tag, None)
            pronoun_map.pop(tag.lower(), None)
            if is_mc:
                pronoun_map.pop("mc", None)
                pronoun_map.pop("player", None)
            disp_self = str(tc.get("display_name", "")).strip()
            if disp_self:
                pronoun_map.pop(disp_self, None)
                pronoun_map.pop(disp_self.lower(), None)

            profile = CharacterProfile(
                tag=tag,
                display_name=tc["display_name"],
                gender=tc["gender"],
                role=tc["role"],
                speech_style=tc["speech_style"],
                pronoun_map=pronoun_map,
                archetype=tc.get("archetype", ""),
                personality_traits=tc.get("personality_traits", ()),
                tone_and_voice=tc.get("tone_and_voice", ""),
                socioeconomic_status=tc.get("socioeconomic_status", ""),
                sample_dialogue=tc.get("sample_dialogue", ()),
            )
            db.save_character(profile)
            stats["characters"] += 1

        rel_graph.save_to_db(db)
        rel_graph.save_to_json_file(project.relationships_path)

    all_profiles = db.get_characters()
    save_characters_to_json(all_profiles, chars_file)
    content = chars_file.read_bytes()
    db.set_meta("characters_json_hash", hashlib.sha256(content).hexdigest())
    db.set_meta("characters_updated_at", datetime.now(timezone.utc).isoformat())

    scenes = segment_scenes(db)
    with db.transaction():
        for sc in scenes:
            db.save_scene(sc)
            stats["scenes"] += 1

    glossary_file = project.work_dir / "glossary.json"
    exclude_character_names = {
        p.display_name.lower() for p in all_profiles.values() if p.display_name
    } | {tag.lower() for tag in all_tags}

    if mine_glossary:
        extra_terms: list[str] = []
        game_dir = project.root / "game"
        if game_dir.is_dir():
            for f in sorted(game_dir.iterdir()):
                if f.suffix in (".rpy", ".rpyc") and not f.name.startswith(
                    ("00", "gui", "options", "screens", "updater")
                ):
                    clean_stem = (
                        f.stem.replace("_events", "")
                        .replace("_event", "")
                        .replace("_store", " Store")
                        .replace("_clan", " Clan")
                        .replace("_world", " World")
                    )
                    words = clean_stem.replace("_", " ").title().split()
                    c_cand = " ".join(words)
                    if len(c_cand) >= 3 and not c_cand.isdigit():
                        extra_terms.append(c_cand)

        candidates = mine_glossary_candidates(
            sources,
            min_count=2,
            exclude_terms=exclude_character_names,
            extra_terms=extra_terms,
        )
        existing_glossary = Glossary.load_file(glossary_file)
        known_terms = {e.term.lower() for e in existing_glossary.entries}

        new_entries: list[GlossaryEntry] = list(existing_glossary.entries)
        for term, _count in candidates:
            if term.lower() not in known_terms:
                entry = GlossaryEntry(
                    term=term,
                    translation="",
                    kind="candidate",
                    case_sensitive=False,
                    aliases=tuple(_make_plural_aliases(term)),
                    notes=f"Khai thác tự động (xuất hiện {_count} lần)",
                )
                new_entries.append(entry)
                known_terms.add(term.lower())
                stats["glossary_candidates"] += 1

        cleaned_entries = clean_glossary(new_entries, exclude_terms=exclude_character_names)
        updated_glossary = Glossary(cleaned_entries)
        updated_glossary.save_file(glossary_file)
        updated_glossary.save_db(db)

    if enrich_llm and llm is not None:
        _LOG.info("Bắt đầu LLM bootstrapping cho hồ sơ nhân vật và thuật ngữ...")
        bootstrap_characters_with_llm(project, db, llm, max_chars=15)
        db.set_meta("characters_bootstrapped", utc_timestamp())
        if mine_glossary:
            cur_glossary = Glossary.load_db(db)
            cleaned = clean_glossary(cur_glossary.entries, exclude_terms=exclude_character_names)
            untranslated = [e for e in cleaned if not e.translation.strip()]
            if untranslated:

                def _freq(e: GlossaryEntry) -> int:
                    m = re.search(r"(\d+)\s*lần", e.notes)
                    return int(m.group(1)) if m else 1

                untranslated.sort(key=_freq, reverse=True)
                to_trans = untranslated[:150]
                enriched = translate_glossary_with_llm(
                    to_trans, llm, examples=collect_term_examples(to_trans, db.all_units())
                )
                trans_map = {e.term: e for e in enriched if e.translation.strip()}
                final_entries = [trans_map.get(e.term, e) for e in cleaned]
                Glossary(final_entries).save_file(glossary_file)
                Glossary(final_entries).save_db(db)
                stats["glossary_translated"] = sum(
                    1 for e in final_entries if e.translation.strip()
                )
            db.set_meta("glossary_bootstrapped", utc_timestamp())

    db.set_meta("analysis_completed", utc_timestamp())
    return stats
