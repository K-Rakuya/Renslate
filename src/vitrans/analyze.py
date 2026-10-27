"""Analysis stage: character discovery, scene segmentation, and glossary mining."""

from __future__ import annotations

import contextlib
import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vitrans.db import Database
from vitrans.glossary import (
    Glossary,
    _make_plural_aliases,
    clean_glossary,
    mine_glossary_candidates,
    translate_glossary_with_llm,
)
from vitrans.models import CharacterProfile, GlossaryEntry, SceneSummary, UnitKind
from vitrans.project import Project

_LOG = logging.getLogger("vitrans.analyze")

_RE_CHARACTER_DEF = re.compile(
    r'(?:define|default)\s+(?P<tag>\w+)\s*=\s*(?:Character|DynamicCharacter)\s*\(\s*(?P<name>_[^(]*\([^)]*\)|"(?:[^"\\]|\\.)*"|\'[^\']*\')'
)
_RE_SCENE_STMT = re.compile(r"^\s*scene\s+\w+")

_PROTAGONIST_TAGS = frozenset(
    {"mc", "player", "hero", "protag", "protagonist", "main", "narrator", "you"}
)
_FEMININE_CUES = re.compile(
    r"\b(?:she|her|hers|herself|girl|woman|lady|sister|daughter|mother|queen|empress|female|maiden|princess)\b",
    re.IGNORECASE,
)
_MASCULINE_CUES = re.compile(
    r"\b(?:he|him|his|himself|boy|man|guy|brother|son|father|king|emperor|male|prince|lord)\b",
    re.IGNORECASE,
)


def infer_character_role_nlp(tag: str, line_count: int, total_lines: int) -> str:
    """Infer character role heuristic based on dialogue volume and protagonist tags."""
    tag_lower = tag.lower()
    if tag_lower in _PROTAGONIST_TAGS:
        return "Nhân vật chính (Protagonist)"
    if total_lines <= 0 or line_count == 0:
        return ""
    ratio = line_count / total_lines
    if ratio >= 0.15:
        return "Nhân vật chính (Protagonist)"
    if ratio >= 0.04:
        return "Nhân vật chính tuyến (Main Cast)"
    if ratio >= 0.008:
        return "Nhân vật phụ (Supporting Cast)"
    return "Nhân vật quần chúng (NPC)"


def infer_character_gender_nlp(name: str, sources: Sequence[str]) -> str:
    """Infer grammatical character gender from pronoun co-occurrence across source lines."""
    # Check if the name or title itself strongly conveys gender
    name_lower = name.lower()
    if _FEMININE_CUES.search(name_lower):
        return "female"
    if _MASCULINE_CUES.search(name_lower):
        return "male"

    # Search for English pronoun co-occurrence in sentences mentioning the character name
    try:
        name_pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    except re.error:
        return "unknown"

    fem_matches = 0
    masc_matches = 0
    for text in sources:
        if name_pat.search(text):
            fem_matches += len(_FEMININE_CUES.findall(text))
            masc_matches += len(_MASCULINE_CUES.findall(text))

    if fem_matches >= 2 and fem_matches >= masc_matches * 2:
        return "female"
    if masc_matches >= 2 and masc_matches >= fem_matches * 2:
        return "male"

    return "unknown"


def extract_character_definitions(project_dir: Path) -> dict[str, str]:
    """Scan game script files for Ren'Py Character definitions."""
    characters: dict[str, str] = {}
    game_dir = project_dir / "game"
    if not game_dir.is_dir():
        return characters

    for rpy in sorted(game_dir.rglob("*.rpy")):
        # Skip translation directories
        if "tl" in rpy.parts:
            continue
        try:
            content = rpy.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            continue

        for m in _RE_CHARACTER_DEF.finditer(content):
            tag = m.group("tag")
            raw_name = m.group("name").strip()
            # Unquote string or clean _("Name")
            if raw_name.startswith('_("') and raw_name.endswith('")'):
                name = raw_name[3:-2]
            elif (raw_name.startswith('"') and raw_name.endswith('"')) or (
                raw_name.startswith("'") and raw_name.endswith("'")
            ):
                name = raw_name[1:-1]
            else:
                name = raw_name
            characters[tag] = name

    return characters


def segment_scenes(db: Database) -> list[SceneSummary]:
    """Group dialogue units into coherent scenes based on labels and sequence continuity."""
    units = db.all_units()
    dialogues = [u for u in units if u.kind is UnitKind.DIALOGUE]
    if not dialogues:
        return []

    scenes: list[SceneSummary] = []
    current_label = dialogues[0].label or "prologue"
    first_seq = dialogues[0].seq
    last_seq = dialogues[0].seq
    scene_idx = 1

    for i in range(1, len(dialogues)):
        u = dialogues[i]
        label = u.label or current_label
        # New scene if label changes or sequence jump > 50
        if label != current_label or (u.seq - last_seq > 50):
            scene_id = f"scene_{scene_idx}_{current_label}"
            scenes.append(
                SceneSummary(
                    scene_id=scene_id,
                    label=current_label,
                    first_seq=first_seq,
                    last_seq=last_seq,
                    summary=f"Cảnh '{current_label}' (từ dòng {first_seq} đến {last_seq})",
                )
            )
            scene_idx += 1
            current_label = label
            first_seq = u.seq
        last_seq = u.seq

    # Final scene
    scene_id = f"scene_{scene_idx}_{current_label}"
    scenes.append(
        SceneSummary(
            scene_id=scene_id,
            label=current_label,
            first_seq=first_seq,
            last_seq=last_seq,
            summary=f"Cảnh '{current_label}' (từ dòng {first_seq} đến {last_seq})",
        )
    )

    return scenes


def bootstrap_characters_with_llm(
    project: Project,
    db: Database,
    llm: Any,
    *,
    max_chars: int = 25,
    chunk_size: int = 6,
) -> dict[str, CharacterProfile]:
    """Bootstrap character personalities, roles, and Vietnamese pronouns using the local LLM.

    Processes prominent speakers in focused chunks for maximum depth, accuracy,
    and resilience against output truncation.
    """
    all_units = db.all_units()
    speaker_counts: Counter[str] = Counter(u.speaker for u in all_units if u.speaker)
    top_speakers = [tag for tag, _ in speaker_counts.most_common(max_chars)]
    if not top_speakers:
        return db.get_characters()

    # Collect 3-4 diverse sample lines for each speaker
    samples_by_speaker: dict[str, list[str]] = {}
    for u in all_units:
        if u.speaker in top_speakers and u.source.strip():
            lst = samples_by_speaker.setdefault(u.speaker, [])
            if len(lst) < 4 and len(u.source) > 8:
                lst.append(u.source.strip())

    existing_chars = db.get_characters()

    # Build bidirectional lookup table (by tag, lower tag, display name, lower display name)
    tag_lookup: dict[str, str] = {}
    for t, p in existing_chars.items():
        tag_lookup[t.lower()] = t
        if p.display_name:
            tag_lookup[p.display_name.lower()] = t

    system_prompt = (
        "Bạn là chuyên gia biên kịch và bản địa hóa game visual novel. "
        "Nhiệm vụ của bạn là phân tích các nhân vật dựa trên tên và các câu thoại mẫu "
        "để xây dựng hồ sơ nhân vật phục vụ dịch thuật sang tiếng Việt chuẩn xác và tự nhiên.\n"
        "Với mỗi nhân vật, hãy xác định:\n"
        "- gender: 'female', 'male', hoặc 'unknown'\n"
        "- role: Mô tả vai trò ngắn gọn (ví dụ: Nhân vật chính, Tông chủ, Sư phụ, Đệ tử...)\n"
        "- speech_style: Phong cách ăn nói (ví dụ: Uy nghiêm, Tinh nghịch, Lạnh lùng...)\n"
        "- pronoun_map: Xưng hô đối ứng tiếng Việt theo vai vế, quan hệ xã hội:\n"
        "    * 'to_elder': Nói với bề trên (VD: xưng 'con/đệ tử', gọi 'sư phụ/cha/mẹ')\n"
        "    * 'to_peer': Nói với bạn bè/ngang hàng (VD: xưng 'tôi/mình', gọi 'cậu/bạn')\n"
        "    * 'to_junior': Nói với bề dưới/con cháu (VD: xưng 'ta/thầy/bố', gọi 'con/ngươi')\n"
        "    * 'to_enemy': Nói khi đối đầu (VD: xưng 'ta', gọi 'ngươi')\n\n"
        "Trả về DUY NHẤT một đối tượng JSON có khóa là các tag nhân vật:\n"
        "{\n"
        '  "tag_name": {\n'
        '    "gender": "female",\n'
        '    "role": "...",\n'
        '    "speech_style": "...",\n'
        '    "pronoun_map": {\n'
        '      "to_elder": "xưng con, gọi sư phụ/cha/mẹ",\n'
        '      "to_peer": "xưng tôi/mình, gọi cậu/bạn",\n'
        '      "to_junior": "xưng ta/thầy, gọi con/ngươi",\n'
        '      "to_enemy": "xưng ta, gọi ngươi"\n'
        '    }\n'
        "  }\n"
        "}\n"
        "Tuyệt đối không giải thích ngoài JSON."
    )

    # Process speakers in focused chunks
    for i in range(0, len(top_speakers), chunk_size):
        chunk_tags = top_speakers[i : i + chunk_size]
        char_prompts: list[str] = []
        for tag in chunk_tags:
            prof = existing_chars.get(tag)
            disp = prof.display_name if prof else tag.capitalize()
            samples = samples_by_speaker.get(tag, [])
            sample_str = " | ".join(f'"{s}"' for s in samples[:3]) if samples else "Không có thoại"
            char_prompts.append(f"- Tag: {tag}, Tên: {disp}, Thoại mẫu: {sample_str}")

        user_prompt = "Phân tích các nhân vật sau:\n" + "\n".join(char_prompts)

        try:
            result_dict, _, _ = llm.complete_json(system_prompt, user_prompt, max_tokens=2048)
        except Exception as e:
            _LOG.warning("Bootstrapping nhóm nhân vật %s qua LLM thất bại: %s", chunk_tags, e)
            continue

        with db.transaction():
            for raw_key, data in result_dict.items():
                # Resolve tag using lookup
                tag = tag_lookup.get(raw_key.lower(), raw_key)
                if tag not in existing_chars and raw_key in existing_chars:
                    tag = raw_key

                # Defensive unpacking for stringified dicts (ast/json)
                if isinstance(data, str):
                    s = data.strip()
                    if s.startswith("{") and s.endswith("}"):
                        try:
                            import ast

                            data = ast.literal_eval(s)
                        except Exception:
                            try:
                                data = json.loads(s)
                            except Exception:
                                continue
                if not isinstance(data, dict):
                    continue

                cur = existing_chars.get(tag)
                disp = cur.display_name if cur else tag.capitalize()
                gender = data.get("gender") or (cur.gender if cur else "unknown")
                role = data.get("role") or (cur.role if cur else "")
                speech_style = data.get("speech_style") or (cur.speech_style if cur else "")
                pronoun_map = data.get("pronoun_map") or (cur.pronoun_map if cur else {})

                # Defensive unpacking for pronoun_map if stringified
                if isinstance(pronoun_map, str):
                    s_p = pronoun_map.strip()
                    if s_p.startswith("{") and s_p.endswith("}"):
                        try:
                            import ast

                            pronoun_map = ast.literal_eval(s_p)
                        except Exception:
                            try:
                                pronoun_map = json.loads(s_p)
                            except Exception:
                                pronoun_map = {}
                if not isinstance(pronoun_map, dict):
                    pronoun_map = {}

                profile = CharacterProfile(
                    tag=tag,
                    display_name=disp,
                    gender=str(gender),
                    role=str(role),
                    speech_style=str(speech_style),
                    pronoun_map={str(k): str(v) for k, v in pronoun_map.items()},
                )
                db.save_character(profile)
                existing_chars[tag] = profile

    # Update characters.json file
    chars_file = project.work_dir / "characters.json"
    json_out = {
        tag: {
            "display_name": p.display_name,
            "gender": p.gender,
            "role": p.role,
            "speech_style": p.speech_style,
            "pronoun_map": p.pronoun_map,
        }
        for tag, p in existing_chars.items()
    }
    chars_file.parent.mkdir(parents=True, exist_ok=True)
    chars_file.write_text(
        json.dumps(json_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return existing_chars


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

    # 1. Discover characters from scripts and dialogue speaker tags
    defined_chars = extract_character_definitions(project.root)
    existing_chars = db.get_characters()

    # Also collect all speakers appearing in dialogue units
    all_units = db.all_units()
    dialogue_speakers = {u.speaker for u in all_units if u.speaker}
    speaker_counts: Counter[str] = Counter(u.speaker for u in all_units if u.speaker)
    sources = [u.source for u in all_units]
    total_units_count = len(all_units)

    # Merge defined + discovered
    all_tags = set(defined_chars.keys()) | dialogue_speakers
    chars_file = project.work_dir / "characters.json"
    file_chars: dict[str, Any] = {}
    if chars_file.is_file():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            file_chars = json.loads(chars_file.read_text(encoding="utf-8"))

    with db.transaction():
        for tag in sorted(all_tags):
            display_name = defined_chars.get(tag, tag.capitalize())
            fc = file_chars.get(tag, {})
            ec = existing_chars.get(tag)

            # Prioritize: 1) manual file, 2) existing DB, 3) statistical NLP inference
            gender = fc.get("gender") or (ec.gender if ec and ec.gender != "unknown" else "")
            if not gender or gender == "unknown":
                gender = infer_character_gender_nlp(display_name, sources)

            role = fc.get("role") or (ec.role if ec and ec.role else "")
            if not role:
                role = infer_character_role_nlp(tag, speaker_counts[tag], total_units_count)

            speech_style = fc.get("speech_style") or (ec.speech_style if ec else "")
            pronoun_map = fc.get("pronoun_map") or (ec.pronoun_map if ec else {})

            profile = CharacterProfile(
                tag=tag,
                display_name=fc.get("display_name") or display_name,
                gender=gender,
                role=role,
                speech_style=speech_style,
                pronoun_map=pronoun_map,
            )
            db.save_character(profile)
            stats["characters"] += 1

    # Save to characters.json for human review/editing
    all_profiles = db.get_characters()
    json_out = {
        tag: {
            "display_name": p.display_name,
            "gender": p.gender,
            "role": p.role,
            "speech_style": p.speech_style,
            "pronoun_map": p.pronoun_map,
        }
        for tag, p in all_profiles.items()
    }
    chars_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_out, ensure_ascii=False, indent=2) + "\n"
    chars_file.write_text(payload, encoding="utf-8")

    # 2. Segment scenes
    scenes = segment_scenes(db)
    with db.transaction():
        for sc in scenes:
            db.save_scene(sc)
            stats["scenes"] += 1

    # 3. Mine glossary candidates
    glossary_file = project.work_dir / "glossary.json"
    exclude_character_names = {
        p.display_name.lower() for p in all_profiles.values() if p.display_name
    } | {tag.lower() for tag in all_tags}

    if mine_glossary:
        candidates = mine_glossary_candidates(
            sources, min_count=2, exclude_terms=exclude_character_names
        )
        existing_glossary = Glossary.load_file(glossary_file)
        known_terms = {e.term.lower() for e in existing_glossary.entries}

        new_entries: list[GlossaryEntry] = list(existing_glossary.entries)
        for term, _count in candidates:
            if term.lower() not in known_terms:
                entry = GlossaryEntry(
                    term=term,
                    translation="",  # Left for human or initial translation
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

    # 4. Optional autonomous LLM bootstrapping
    if enrich_llm and llm is not None:
        _LOG.info("Bắt đầu LLM bootstrapping cho hồ sơ nhân vật và thuật ngữ...")
        bootstrap_characters_with_llm(project, db, llm, max_chars=15)
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
                enriched = translate_glossary_with_llm(to_trans, llm)
                trans_map = {e.term: e for e in enriched if e.translation.strip()}
                final_entries = [trans_map.get(e.term, e) for e in cleaned]
                Glossary(final_entries).save_file(glossary_file)
                Glossary(final_entries).save_db(db)
                stats["glossary_translated"] = sum(
                    1 for e in final_entries if e.translation.strip()
                )

    return stats
