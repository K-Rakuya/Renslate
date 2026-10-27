"""Tests for renslate.analyze."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from renslate.core.db import (
    Database,
    load_characters_from_json,
    save_characters_to_json,
    sync_characters,
)
from renslate.core.models import (
    CharacterProfile,
    RelationshipKind,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.analyze import (
    StylometricSummary,
    bootstrap_characters_with_llm,
    build_persona_user_prompt,
    extract_character_definitions,
    extract_stylometrics,
    infer_character_gender_nlp,
    infer_character_role_nlp,
    infer_persona_fallback,
    infer_semantic_relationship,
    is_collective_entity,
    is_non_human_or_meta,
    resolve_pronoun_pair_from_relationship,
    run_analysis,
    segment_scenes,
)
from renslate.engine.graph import RelationshipGraph


def test_extract_character_definitions(tmp_path: Path) -> None:
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    script = (
        'define e = Character("Eileen")\n'
        'define m = DynamicCharacter("mc_name")\n'
        'define s = Character(_("Sylvie"))\n'
    )
    (game_dir / "script.rpy").write_text(script, encoding="utf-8")

    chars = extract_character_definitions(tmp_path)
    assert chars.get("e") == "Eileen"
    assert chars.get("m") == "mc_name"
    assert chars.get("s") == "Sylvie"


def test_segment_scenes_groups_by_label_and_gap(tmp_path: Path) -> None:
    db = Database.open(tmp_path / "test.db", create=True)
    db.save_unit(Unit("d:1", UnitKind.DIALOGUE, 1, "Hello", Status.PENDING, label="start"))
    db.save_unit(Unit("d:2", UnitKind.DIALOGUE, 2, "World", Status.PENDING, label="start"))
    db.save_unit(Unit("d:3", UnitKind.DIALOGUE, 3, "Chapter 1", Status.PENDING, label="ch1"))

    scenes = segment_scenes(db)
    assert len(scenes) == 2
    assert scenes[0].label == "start"
    assert (scenes[0].first_seq, scenes[0].last_seq) == (1, 2)
    assert scenes[1].label == "ch1"


def test_run_analysis_pipeline(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    (root / "game" / "script.rpy").write_text('define e = Character("Eileen")', encoding="utf-8")

    project = Project.create(root)
    with project.open_db() as db:
        db.save_unit(
            Unit(
                "d:1",
                UnitKind.DIALOGUE,
                1,
                "Welcome to the Grand Castle!",
                Status.PENDING,
                speaker="e",
                label="intro",
            )
        )
        db.save_unit(
            Unit(
                "d:2",
                UnitKind.DIALOGUE,
                2,
                "The Grand Castle is ancient.",
                Status.PENDING,
                speaker="e",
                label="intro",
            )
        )

        stats = run_analysis(project, db)

    assert stats["characters"] >= 1
    assert stats["scenes"] >= 1
    assert (project.work_dir / "characters.json").is_file()
    assert (project.work_dir / "glossary.json").is_file()


def test_infer_character_role_and_gender_nlp() -> None:
    # 1. Role inference
    assert infer_character_role_nlp("mc", 50, 100) == "Nhân vật chính (Protagonist)"
    assert infer_character_role_nlp("narrator", 1, 100) == "Người dẫn chuyện (Narrator)"
    assert infer_character_role_nlp("hero", 1, 100) == "Nhân vật chính (Protagonist)"
    assert infer_character_role_nlp("layla", 10, 100) == "Nhân vật chính tuyến (Main Cast)"
    assert infer_character_role_nlp("guard", 1, 100) == "Nhân vật phụ (Supporting Cast)"
    assert infer_character_role_nlp("villager", 0, 100) == ""

    # 2. Gender inference
    sources_fem = [
        "Liana is a talented mage. She cast her spell gracefully.",
        "I looked at her, and she smiled back at me.",
    ]
    assert infer_character_gender_nlp("Liana", sources_fem) == "female"

    sources_masc = [
        "Targon stood his ground. He grabbed his battleaxe firmly.",
        "The boy nodded to him with deep respect.",
    ]
    assert infer_character_gender_nlp("Targon", sources_masc) == "male"

    # Inherent keyword gender
    assert infer_character_gender_nlp("Female Deity", []) == "female"
    assert infer_character_gender_nlp("King Arthur", []) == "male"


def test_bootstrap_characters_with_llm(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)

    class MockLLM:
        def complete_json(
            self, system: str, user: str, *args: Any, **kwargs: Any
        ) -> tuple[dict[str, Any], int, int]:
            return (
                {
                    "layla": {
                        "gender": "female",
                        "role": "Sư phụ / Tông chủ",
                        "speech_style": "Uy nghiêm, điềm đạm",
                        "pronoun_map": {"self": "ta", "target": "con"},
                    }
                },
                60,
                30,
            )

    with project.open_db() as db:
        db.save_unit(
            Unit(
                "d:1",
                UnitKind.DIALOGUE,
                1,
                "I must protect the sect at all costs.",
                Status.PENDING,
                speaker="layla",
            )
        )
        chars = bootstrap_characters_with_llm(project, db, MockLLM())

    assert "layla" in chars
    p = chars["layla"]
    assert p.gender == "female"
    assert p.role == "Sư phụ / Tông chủ"
    assert p.speech_style == "Uy nghiêm, điềm đạm"
    assert p.pronoun_map.get("self") == "ta"


def test_stylometrics_extraction_empty() -> None:
    summary = extract_stylometrics([])
    assert summary.total_lines == 0
    assert summary.total_words == 0
    assert summary.unique_words == 0
    assert summary.type_token_ratio == 0.0
    assert summary.avg_sentence_length == 0.0
    assert summary.exclamation_count == 0
    assert summary.question_count == 0
    assert summary.ellipsis_count == 0
    assert summary.hesitation_count == 0
    assert summary.top_interjections == ()
    assert summary.summary_text() == "Không có dữ liệu thoại thống kê"

    d = summary.to_dict()
    assert d["total_lines"] == 0
    assert d["type_token_ratio"] == 0.0


def test_stylometrics_extraction_metrics() -> None:
    dialogues = [
        "Hello, my friend! Are you ready for our journey?",
        "Indeed... The path is treacherous, but we shall succeed!",
    ]
    summary = extract_stylometrics(dialogues)
    assert summary.total_lines == 2
    assert summary.total_words > 0
    assert summary.unique_words > 0
    assert 0.0 < summary.type_token_ratio <= 1.0
    assert summary.total_sentences >= 2
    assert summary.avg_sentence_length > 0.0
    assert summary.exclamation_count >= 2
    assert summary.question_count >= 1
    assert summary.ellipsis_count >= 1

    text = summary.summary_text()
    assert "TTR:" in text
    assert "Độ dài câu TB:" in text
    assert "Cảm thán (!):" in text
    assert "Nghi vấn (?):" in text
    assert "Chấm lửng (...):" in text


def test_stylometrics_tag_stripping_and_stutters() -> None:
    dialogues = [
        "{i}B-Baka!{/i} It's not like I [player_name] waited for you...!",
        "W-What are you staring at?! Hmph! {size=+5}Leave me alone!{/size}",
        "E-Excuse me... Is anyone there\uff1f Yes\uff01",
    ]
    summary = extract_stylometrics(dialogues)
    assert summary.hesitation_count >= 3
    has_interj = (
        any("-" in item for item in summary.top_interjections)
        or "hmph" in summary.top_interjections
    )
    assert has_interj
    assert summary.exclamation_count >= 3
    assert summary.question_count >= 2
    assert summary.ellipsis_count >= 2


def test_infer_persona_fallback_protagonist_and_meta() -> None:
    dummy_sty = extract_stylometrics([])

    p_mc = infer_persona_fallback("mc", [], dummy_sty, role="Nhân vật chính")
    assert p_mc["archetype"] == "protagonist"
    assert "quyết đoán" in p_mc["personality_traits"]
    assert "Protagonist" in p_mc["socioeconomic_status"]

    p_narr = infer_persona_fallback("narrator", [], dummy_sty)
    assert p_narr["archetype"] == "narrator"
    assert "Narrator" in p_narr["socioeconomic_status"]

    p_th = infer_persona_fallback("th", [], dummy_sty)
    assert p_th["archetype"] == "inner_monologue"
    assert "Độc thoại nội tâm" in p_th["socioeconomic_status"]


def test_infer_persona_fallback_archetypes() -> None:
    # 1. Tsundere: high exclamations and stutters/interjections
    tsun_sty = StylometricSummary(
        total_lines=5,
        total_words=40,
        unique_words=30,
        type_token_ratio=0.75,
        total_sentences=6,
        avg_sentence_length=6.5,
        exclamation_count=5,
        exclamation_rate=0.83,
        question_count=1,
        question_rate=0.17,
        ellipsis_count=2,
        ellipsis_rate=0.33,
        hesitation_count=2,
        hesitation_rate=0.33,
        top_interjections=("b-baka", "hmph"),
    )
    p_tsun = infer_persona_fallback("charlotte", [], tsun_sty, role="Noble Daughter")
    assert p_tsun["archetype"] == "tsundere"
    assert "kiêu hãnh" in p_tsun["personality_traits"]

    # 2. Dandere: high ellipsis, low exclamation, shyness cues
    dan_sty = StylometricSummary(
        total_lines=5,
        total_words=25,
        unique_words=18,
        type_token_ratio=0.72,
        total_sentences=5,
        avg_sentence_length=5.0,
        exclamation_count=0,
        exclamation_rate=0.0,
        question_count=1,
        question_rate=0.2,
        ellipsis_count=4,
        ellipsis_rate=0.8,
        hesitation_count=2,
        hesitation_rate=0.4,
        top_interjections=("eep", "um"),
    )
    p_dan = infer_persona_fallback("iris", [], dan_sty)
    assert p_dan["archetype"] == "dandere"

    # 3. Kuudere: high ellipsis, low exclamation, no stutter/shyness
    kuu_sty = StylometricSummary(
        total_lines=5,
        total_words=20,
        unique_words=18,
        type_token_ratio=0.9,
        total_sentences=5,
        avg_sentence_length=4.0,
        exclamation_count=0,
        exclamation_rate=0.0,
        question_count=0,
        question_rate=0.0,
        ellipsis_count=3,
        ellipsis_rate=0.6,
        hesitation_count=0,
        hesitation_rate=0.0,
        top_interjections=(),
    )
    p_kuu = infer_persona_fallback("rei", [], kuu_sty)
    assert p_kuu["archetype"] == "kuudere"

    # 4. Rebel warrior: high exclamation, low ellipsis
    reb_sty = StylometricSummary(
        total_lines=5,
        total_words=45,
        unique_words=35,
        type_token_ratio=0.77,
        total_sentences=5,
        avg_sentence_length=9.0,
        exclamation_count=6,
        exclamation_rate=1.2,
        question_count=0,
        question_rate=0.0,
        ellipsis_count=0,
        ellipsis_rate=0.0,
        hesitation_count=0,
        hesitation_rate=0.0,
        top_interjections=(),
    )
    p_reb = infer_persona_fallback("targon", [], reb_sty, role="Warrior")
    assert p_reb["archetype"] == "rebel_warrior"

    # 5. Elder mentor: long sentences, high TTR, low exclamation
    men_sty = StylometricSummary(
        total_lines=5,
        total_words=70,
        unique_words=55,
        type_token_ratio=0.78,
        total_sentences=5,
        avg_sentence_length=14.0,
        exclamation_count=1,
        exclamation_rate=0.2,
        question_count=1,
        question_rate=0.2,
        ellipsis_count=1,
        ellipsis_rate=0.2,
        hesitation_count=0,
        hesitation_rate=0.0,
        top_interjections=(),
    )
    p_men = infer_persona_fallback("elrond", [], men_sty, role="Wise Elder")
    assert p_men["archetype"] == "elder_mentor"

    # 6. Inquisitive apprentice: high questions
    inq_sty = StylometricSummary(
        total_lines=5,
        total_words=35,
        unique_words=25,
        type_token_ratio=0.71,
        total_sentences=5,
        avg_sentence_length=7.0,
        exclamation_count=1,
        exclamation_rate=0.2,
        question_count=4,
        question_rate=0.8,
        ellipsis_count=0,
        ellipsis_rate=0.0,
        hesitation_count=0,
        hesitation_rate=0.0,
        top_interjections=(),
    )
    p_inq = infer_persona_fallback("toby", [], inq_sty)
    assert p_inq["archetype"] == "inquisitive_apprentice"


def test_infer_persona_fallback_socioeconomic_and_samples() -> None:
    dummy_sty = extract_stylometrics([])

    p_noble = infer_persona_fallback("alice", [], dummy_sty, role="Princess of the realm")
    assert "Nobility" in p_noble["socioeconomic_status"]

    p_knight = infer_persona_fallback("arthur", [], dummy_sty, role="Knight Commander")
    assert "Military" in p_knight["socioeconomic_status"]

    p_mage = infer_persona_fallback("merlin", [], dummy_sty, role="Grand Archmage")
    assert "Scholarly" in p_mage["socioeconomic_status"]

    p_merchant = infer_persona_fallback("barkeep", [], dummy_sty, role="Innkeeper / Bartender")
    assert "Merchant" in p_merchant["socioeconomic_status"]

    p_common = infer_persona_fallback("villager_1", [], dummy_sty, role="Town resident")
    assert "Commoner" in p_common["socioeconomic_status"]

    lines = [
        "Hi!",
        "A sufficiently long representative opening line.",
        "Middle dialogue illustrating character voice and philosophy.",
        "Another filler line that is also quite long.",
        "Final memorable dialogue closing the scene.",
    ]
    p_samples = infer_persona_fallback("heroine", lines, dummy_sty)
    assert 1 <= len(p_samples["sample_dialogue"]) <= 3
    assert p_samples["sample_dialogue"][0] == "A sufficiently long representative opening line."


def test_collective_and_non_human_entities() -> None:
    assert is_collective_entity("crowd", "Townsfolk")
    assert is_collective_entity("all", "Everyone")
    assert is_collective_entity("guards", "Castle Guards")
    assert not is_collective_entity("liana", "Liana")

    assert is_non_human_or_meta("system", "System Notice")
    assert is_non_human_or_meta("announcer", "Announcer Voice")
    assert is_non_human_or_meta("narrator", "Narrator")
    assert is_non_human_or_meta("thought", "Inner Voice")
    assert not is_non_human_or_meta("charlotte", "Charlotte")


def test_resolve_pronoun_pair_from_relationship() -> None:
    # 1. FAMILY_ELDER
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FAMILY_ELDER, "female", "male", is_speaker_senior=True
    ) == ("mẹ", "con")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FAMILY_ELDER, "male", "female", is_speaker_senior=True
    ) == ("cha", "con")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FAMILY_ELDER, "male", "female", is_speaker_senior=False
    ) == ("con", "mẹ")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FAMILY_ELDER, "female", "male", is_speaker_senior=False
    ) == ("con", "cha")

    # 2. FAMILY_SIBLING
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FAMILY_SIBLING, "female", "male", is_speaker_senior=True
    ) == ("chị", "em")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FAMILY_SIBLING, "male", "female", is_speaker_senior=True
    ) == ("anh", "em")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FAMILY_SIBLING, "male", "female", is_speaker_senior=False
    ) == ("em", "chị")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FAMILY_SIBLING, "female", "male", is_speaker_senior=False
    ) == ("em", "anh")

    # 3. SUPERIOR_SUBORDINATE
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.SUPERIOR_SUBORDINATE, "male", "male", is_speaker_senior=True
    ) == ("trẫm", "khanh")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.SUPERIOR_SUBORDINATE, "male", "male", is_speaker_senior=False
    ) == ("thần", "bệ hạ")

    # 4. MENTOR_DISCIPLE
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.MENTOR_DISCIPLE, "male", "female", is_speaker_senior=True
    ) == ("ta", "con")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.MENTOR_DISCIPLE, "female", "male", is_speaker_senior=False
    ) == ("con", "thầy")

    # 5. SERVANT_MASTER
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.SERVANT_MASTER, "male", "female", is_speaker_senior=True
    ) == ("ta", "cô")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.SERVANT_MASTER, "female", "female", is_speaker_senior=False
    ) == ("em", "tiểu thư")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.SERVANT_MASTER, "male", "male", is_speaker_senior=False
    ) == ("tôi", "thiếu gia")

    # 6. RIVAL_HOSTILE
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.RIVAL_HOSTILE, "male", "male"
    ) == ("ta", "ngươi")

    # 7. FRIENDSHIP
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FRIENDSHIP_CLOSE, "female", "female"
    ) == ("mình", "cậu")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FRIENDSHIP_CASUAL, "male", "male"
    ) == ("tôi", "cậu")

    # 8. ROMANCE
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.ROMANCE, "male", "female"
    ) == ("anh", "em")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.ROMANCE, "female", "male"
    ) == ("em", "anh")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.ROMANCE, "female", "female"
    ) == ("mình", "cậu")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.ROMANCE, "male", "male"
    ) == ("tôi", "cậu")

    # 9. FORMAL_ACQUAINTANCE
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FORMAL_ACQUAINTANCE, "male", "female"
    ) == ("tôi", "cô")
    assert resolve_pronoun_pair_from_relationship(
        RelationshipKind.FORMAL_ACQUAINTANCE, "male", "male"
    ) == ("tôi", "anh")


def test_infer_semantic_relationship_hierarchy() -> None:
    # 1. Elder cue: MC to Mother
    rel_elder = infer_semantic_relationship(
        "mc",
        {"role": "Nhân vật chính", "display_name": "Hero"},
        "mom",
        {"role": "Mẹ của nhân vật chính", "display_name": "Mother", "gender": "female"},
        is_mc=True,
    )
    assert rel_elder is not None
    assert rel_elder[0] == RelationshipKind.FAMILY_ELDER
    assert not rel_elder[1]

    # 2. Sibling cue
    rel_sib = infer_semantic_relationship(
        "charlotte",
        {"role": "Elder sister", "display_name": "Charlotte", "gender": "female"},
        "mc",
        {"role": "Nhân vật chính", "display_name": "Hero", "gender": "male"},
        partner_is_mc=True,
    )
    assert rel_sib is not None
    assert rel_sib[0] == RelationshipKind.FAMILY_SIBLING
    assert rel_sib[1]

    # 3. Superior cue
    rel_sup = infer_semantic_relationship(
        "mc",
        {"role": "Knight", "display_name": "Hero"},
        "queen",
        {"role": "Nữ hoàng vương quốc", "display_name": "Queen Victoria"},
        is_mc=True,
    )
    assert rel_sup is not None
    assert rel_sup[0] == RelationshipKind.SUPERIOR_SUBORDINATE
    assert not rel_sup[1]

    # 4. Servant cue
    rel_serv = infer_semantic_relationship(
        "maid",
        {"role": "Hầu gái dinh thự", "display_name": "Rem", "gender": "female"},
        "mc",
        {"role": "Nhân vật chính", "display_name": "Hero", "gender": "male"},
        partner_is_mc=True,
    )
    assert rel_serv is not None
    assert rel_serv[0] == RelationshipKind.SERVANT_MASTER
    assert not rel_serv[1]

    # 5. Rival / Hostile cue
    rel_host = infer_semantic_relationship(
        "mc",
        {"role": "Hero", "display_name": "Hero"},
        "malakor",
        {"role": "Kẻ thù không đội trời chung (Enemy / Rival)", "display_name": "Malakor"},
        is_mc=True,
    )
    assert rel_host is not None
    assert rel_host[0] == RelationshipKind.RIVAL_HOSTILE

    # 6. Heroine vs Generic NPC
    rel_heroine = infer_semantic_relationship(
        "mc",
        {"role": "Nhân vật chính (Protagonist)", "gender": "male"},
        "liana",
        {"role": "Nhân vật chính tuyến (Main Cast)", "gender": "female"},
        is_mc=True,
    )
    assert rel_heroine is not None
    assert rel_heroine[0] == RelationshipKind.ROMANCE

    rel_npc = infer_semantic_relationship(
        "mc",
        {"role": "Nhân vật chính (Protagonist)", "gender": "male"},
        "villager_fem",
        {"role": "Nhân vật quần chúng (NPC)", "gender": "female"},
        is_mc=True,
    )
    assert rel_npc is not None
    assert rel_npc[0] == RelationshipKind.FORMAL_ACQUAINTANCE

    # 7. Non-human entity returns None
    rel_sys = infer_semantic_relationship(
        "mc",
        {"role": "Protagonist"},
        "system",
        {"display_name": "System Notice"},
        is_mc=True,
    )
    assert rel_sys is None


def test_build_persona_user_prompt() -> None:
    chars = {"c": CharacterProfile("c", "Charlotte")}
    samples = {"c": ["B-Baka!"]}
    stys = {"c": extract_stylometrics(samples["c"])}
    prompt = build_persona_user_prompt(["c"], chars, samples, stys)
    assert "Tag: c" in prompt
    assert "Charlotte" in prompt
    assert "B-Baka!" in prompt
    assert "TTR:" in prompt


def test_bootstrap_characters_with_llm_full_persona(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)

    class FullPersonaLLM:
        def complete_json(
            self, system: str, user: str, *args: Any, **kwargs: Any
        ) -> tuple[dict[str, Any], int, int]:
            return (
                {
                    "charlotte": {
                        "display_name_vi": "Charlotte",
                        "gender": "female",
                        "role": "Tiểu thư quý tộc / Nữ chính",
                        "archetype": "tsundere",
                        "personality_traits": ["kiêu hãnh", "khẩu xà tâm phật", "dễ bối rối"],
                        "tone_and_voice": "Sắc sảo, đanh đá bề ngoài",
                        "socioeconomic_status": "Tiểu thư quý tộc cấp cao",
                        "speech_style": "Kiêu kỳ, đanh thép",
                        "sample_dialogue": ["B-Baka! Don't look at me!", "Hmph!"],
                        "pronoun_map": "{'mc': 'em - anh', 'to_elder': 'con - bác'}",
                    }
                },
                120,
                80,
            )

    with project.open_db() as db:
        db.save_unit(
            Unit(
                "d:1",
                UnitKind.DIALOGUE,
                1,
                "B-Baka! It is not like I waited for you or anything!",
                Status.PENDING,
                speaker="charlotte",
            )
        )
        chars = bootstrap_characters_with_llm(project, db, FullPersonaLLM())

    assert "charlotte" in chars
    prof = chars["charlotte"]
    assert prof.archetype == "tsundere"
    assert "kiêu hãnh" in prof.personality_traits
    assert prof.socioeconomic_status == "Tiểu thư quý tộc cấp cao"
    assert "B-Baka!" in prof.sample_dialogue[0]
    assert prof.pronoun_map.get("mc") == "em - anh"

    loaded = load_characters_from_json(project.characters_path)
    assert "charlotte" in loaded
    assert loaded["charlotte"].archetype == "tsundere"


def test_sync_characters_modes_and_persona_shield(tmp_path: Path) -> None:
    db = Database.open(tmp_path / "test_sync.db", create=True)
    chars_json = tmp_path / "characters.json"

    # 1. DB to JSON when JSON does not exist
    p1 = CharacterProfile(
        tag="liana",
        display_name="Liana",
        gender="female",
        role="Mage",
        archetype="kuudere",
        personality_traits=("lãnh đạm", "thông tuệ"),
        tone_and_voice="Lạnh lùng, ngắn gọn",
        socioeconomic_status="Học giả hoàng gia",
        sample_dialogue=("The stars reveal everything.",),
    )
    db.save_character(p1)

    c_db, c_json = sync_characters(db, chars_json, direction="auto")
    assert c_db == 1
    assert c_json == 1
    assert chars_json.is_file()

    file_chars = load_characters_from_json(chars_json)
    assert "liana" in file_chars
    assert file_chars["liana"].archetype == "kuudere"

    # 2. Explicit directions
    c_db2, c_json2 = sync_characters(db, chars_json, direction="db_to_json")
    assert c_db2 == 1 and c_json2 == 1
    c_db3, c_json3 = sync_characters(db, chars_json, direction="json_to_db")
    assert c_db3 == 1 and c_json3 == 1

    # 3. Simulate user editing characters.json with legacy schema
    legacy_json = {
        "liana": {
            "display_name": "Liana Curated",
            "gender": "female",
            "role": "High Archmage",
            "speech_style": "Trang trọng",
            "pronoun_map": {"mc": "tôi - cậu"},
        }
    }
    save_characters_to_json(
        [CharacterProfile.from_dict(legacy_json["liana"], tag="liana")],
        chars_json,
    )

    sync_characters(db, chars_json, direction="auto")
    updated = db.get_characters().get("liana")
    assert updated is not None
    assert updated.display_name == "Liana Curated"
    assert updated.role == "High Archmage"
    assert updated.archetype == "kuudere"
    assert "lãnh đạm" in updated.personality_traits
    assert updated.tone_and_voice == "Lạnh lùng, ngắn gọn"


def test_run_analysis_relational_graph_and_persona(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    (root / "game" / "script.rpy").write_text(
        'define mc = Character("Hero")\n'
        'define mom = Character("Mother")\n'
        'define maid = Character("Rem")\n'
        'define enemy = Character("Malakor")\n'
        'define crowd = Character("Crowd")\n'
        'define heroine = Character("Charlotte")\n',
        encoding="utf-8",
    )

    project = Project.create(root)
    with project.open_db() as db:
        units = [
            Unit(
                "d:1",
                UnitKind.DIALOGUE,
                1,
                "Good morning mother!",
                Status.PENDING,
                speaker="mc",
                label="start",
            ),
            Unit(
                "d:2",
                UnitKind.DIALOGUE,
                2,
                "Eat your breakfast, my son.",
                Status.PENDING,
                speaker="mom",
                label="start",
            ),
            Unit(
                "d:3",
                UnitKind.DIALOGUE,
                3,
                "Young master, your tea is ready.",
                Status.PENDING,
                speaker="maid",
                label="start",
            ),
            Unit(
                "d:4",
                UnitKind.DIALOGUE,
                4,
                "Thank you, Rem.",
                Status.PENDING,
                speaker="mc",
                label="start",
            ),
            Unit(
                "d:5",
                UnitKind.DIALOGUE,
                5,
                "I will destroy you, Hero!",
                Status.PENDING,
                speaker="enemy",
                label="battle",
            ),
            Unit(
                "d:6",
                UnitKind.DIALOGUE,
                6,
                "Not on my watch, villain!",
                Status.PENDING,
                speaker="mc",
                label="battle",
            ),
            Unit(
                "d:7",
                UnitKind.DIALOGUE,
                7,
                "Long live the hero!",
                Status.PENDING,
                speaker="crowd",
                label="cheer",
            ),
            Unit(
                "d:8",
                UnitKind.DIALOGUE,
                8,
                "Thank you everyone for believing in me.",
                Status.PENDING,
                speaker="mc",
                label="cheer",
            ),
            Unit(
                "d:9",
                UnitKind.DIALOGUE,
                9,
                "B-Baka! Don't get arrogant, Hero!",
                Status.PENDING,
                speaker="heroine",
                label="epilogue",
            ),
            Unit(
                "d:10",
                UnitKind.DIALOGUE,
                10,
                "You're blushing, Charlotte.",
                Status.PENDING,
                speaker="mc",
                label="epilogue",
            ),
        ]
        for u in units:
            db.save_unit(u)

        stats = run_analysis(project, db)
        assert stats["characters"] >= 6
        chars = db.get_characters()

        # 1. MC to Mother: must NOT be "anh - em"
        mc_prof = chars["mc"]
        assert "mom" in mc_prof.pronoun_map
        assert mc_prof.pronoun_map["mom"] == "con - mẹ"

        # 2. MC to Maid: must NOT be "anh - em"
        assert "maid" in mc_prof.pronoun_map
        assert mc_prof.pronoun_map["maid"] != "anh - em"

        # 3. MC to Enemy: must NOT be "anh - em"
        assert "enemy" in mc_prof.pronoun_map
        assert mc_prof.pronoun_map["enemy"] == "ta - ngươi"

        # 4. MC to Crowd: collective neutral addressing
        assert "crowd" in mc_prof.pronoun_map
        assert mc_prof.pronoun_map["crowd"] == "tôi - mọi người"

        # 5. MC to Heroine: Romance pair
        assert "heroine" in mc_prof.pronoun_map
        assert mc_prof.pronoun_map["heroine"] == "anh - em"

        # 6. Verify RelationshipGraph persistence
        graph = RelationshipGraph.from_db(db)
        edge_mom = graph.get_edge("mc", "mom")
        assert edge_mom is not None
        assert edge_mom.kind == RelationshipKind.FAMILY_ELDER

        edge_enemy = graph.get_edge("mc", "enemy")
        assert edge_enemy is not None
        assert edge_enemy.kind == RelationshipKind.RIVAL_HOSTILE

    assert project.relationships_path.is_file()
    assert project.characters_path.is_file()
