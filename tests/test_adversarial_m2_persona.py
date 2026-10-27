"""Adversarial stress harness & empirical verification for Milestone 2.

Probing:
1. Conflict resolution when DB and characters.json are concurrently modified
2. Symmetric human curation shield and legacy schema preservation
3. Pronoun resolution: linguistic invariants, anti-inversion, non-kinship vs kinship
4. Collective and non-human/meta entity handling
5. End-to-end multi-entity relationship & persona pipeline stress tests
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

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
    is_collective_entity,
    is_non_human_or_meta,
    resolve_pronoun_pair_from_relationship,
    run_analysis,
)
from renslate.engine.graph import RelationshipGraph

# ==============================================================================
# 1. Concurrent Modification Conflicts Between DB and characters.json
# ==============================================================================


def test_concurrent_modification_conflict_resolution(tmp_path: Path) -> None:
    """Probe conflict resolution when BOTH DB and characters.json are independently modified.

    Scenario A: DB has newer modification time than JSON.
    Scenario B: JSON has newer modification time than DB.
    Scenario C: Disjoint additions in DB and JSON (union merging).
    """
    db_path = tmp_path / "conflict.db"
    json_path = tmp_path / "characters.json"

    db = Database.open(db_path, create=True)

    # Initial state
    char_a = CharacterProfile(tag="hero", display_name="Hero", role="Fighter")
    char_b = CharacterProfile(tag="mage", display_name="Mage", role="Caster")
    with db.transaction():
        db.save_characters_bulk([char_a, char_b])

    # Initial sync
    sync_characters(db, json_path, direction="auto")

    # Case 1: Disjoint additions - DB adds char_c, JSON adds char_d
    char_c = CharacterProfile(tag="rogue", display_name="Rogue", role="Thief")
    with db.transaction():
        db.save_character(char_c)

    # Directly edit JSON file to add char_d
    disk_data = load_characters_from_json(json_path)
    disk_data["cleric"] = CharacterProfile(tag="cleric", display_name="Cleric", role="Healer")
    save_characters_to_json(disk_data, json_path)

    # Sync auto: both rogue (from DB) and cleric (from JSON) must be preserved in union
    c_db, c_json = sync_characters(db, json_path, direction="auto")
    assert c_db == 4 and c_json == 4

    merged_db = db.get_characters()
    assert "hero" in merged_db
    assert "mage" in merged_db
    assert "rogue" in merged_db
    assert "cleric" in merged_db

    # Case 2: Concurrent conflict on SAME character (JSON edited with newer mtime)
    time.sleep(0.02)
    disk_data = load_characters_from_json(json_path)
    disk_data["hero"] = CharacterProfile(
        tag="hero",
        display_name="Hero JSON Newer",
        role="Paladin",
    )
    save_characters_to_json(disk_data, json_path)
    # Ensure json mtime is strictly ahead of DB timestamp
    os.utime(json_path, (time.time() + 1.0, time.time() + 1.0))

    sync_characters(db, json_path, direction="auto")
    updated_hero = db.get_characters()["hero"]
    assert updated_hero.display_name == "Hero JSON Newer"
    assert updated_hero.role == "Paladin"


# ==============================================================================
# 2. Symmetric Human Curation Shield & Schema Migration Robustness
# ==============================================================================


def test_symmetric_curation_shield_db_to_json_legacy(tmp_path: Path) -> None:
    """Verify that a legacy JSON without persona fields does NOT wipe rich DB persona."""
    db = Database.open(tmp_path / "shield_test.db", create=True)
    json_path = tmp_path / "characters.json"

    # DB has deep persona attributes
    rich_profile = CharacterProfile(
        tag="liana",
        display_name="Liana",
        gender="female",
        role="Royal High Mage",
        speech_style="Aristocratic",
        pronoun_map={"mc": "tôi - cậu"},
        archetype="kuudere",
        personality_traits=("lãnh đạm", "thông tuệ", "quyết đoán"),
        tone_and_voice="Lạnh lùng, xúc tích, đều đều",
        socioeconomic_status="Học giả quý tộc hoàng gia",
        sample_dialogue=("The stars speak truth.", "Do not disturb my research."),
    )
    with db.transaction():
        db.save_character(rich_profile)

    # First sync creates characters.json
    sync_characters(db, json_path, direction="auto")

    # User manually edits characters.json with a legacy 5-field schema (missing persona)
    legacy_json_content = {
        "liana": {
            "display_name": "Liana Curated",
            "gender": "female",
            "role": "Grand Sorceress",
            "speech_style": "Cổ điển",
            "pronoun_map": {"mc": "tôi - bạn", "enemy": "ta - ngươi"},
        }
    }
    json_path.write_text(
        json.dumps(legacy_json_content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Sync auto: JSON has newer mtime, but missing persona fields must fall back to DB
    c_db, c_json = sync_characters(db, json_path, direction="auto")
    assert c_db == 1 and c_json == 1

    updated_db = db.get_characters()["liana"]
    # Curated fields updated:
    assert updated_db.display_name == "Liana Curated"
    assert updated_db.role == "Grand Sorceress"
    assert updated_db.speech_style == "Cổ điển"
    # Deep persona fields preserved by curation shield:
    assert updated_db.archetype == "kuudere"
    assert updated_db.personality_traits == ("lãnh đạm", "thông tuệ", "quyết đoán")
    assert updated_db.tone_and_voice == "Lạnh lùng, xúc tích, đều đều"
    assert updated_db.socioeconomic_status == "Học giả quý tộc hoàng gia"
    assert len(updated_db.sample_dialogue) == 2


def test_symmetric_curation_shield_json_to_db_legacy(tmp_path: Path) -> None:
    """Verify that if DB has legacy/empty persona, rich persona in JSON is preserved."""
    db = Database.open(tmp_path / "shield_reverse.db", create=True)
    json_path = tmp_path / "characters.json"

    # JSON has rich persona
    rich_json = {
        "valeria": {
            "display_name": "Valeria",
            "gender": "female",
            "role": "Rebel Knight",
            "speech_style": "Thô ráp",
            "pronoun_map": {"mc": "tôi - cậu"},
            "archetype": "rebel",
            "personality_traits": ["kiên định", "ngang tàng"],
            "tone_and_voice": "Trầm, dứt khoát, gai góc",
            "socioeconomic_status": "Hiệp sĩ lưu vong",
            "sample_dialogue": ["I bow to no tyrant."],
        }
    }
    json_path.write_text(json.dumps(rich_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # DB has bare/legacy profile
    bare_profile = CharacterProfile(
        tag="valeria",
        display_name="Valeria Legacy",
        gender="female",
        role="Knight",
    )
    with db.transaction():
        db.save_character(bare_profile)

    # Sync auto: JSON should win and enrich DB
    sync_characters(db, json_path, direction="auto")

    updated_db = db.get_characters()["valeria"]
    assert updated_db.archetype == "rebel"
    assert updated_db.personality_traits == ("kiên định", "ngang tàng")
    assert updated_db.tone_and_voice == "Trầm, dứt khoát, gai góc"


def test_sync_characters_malformed_json_handling(tmp_path: Path) -> None:
    """Verify sync_characters raises ValueError/JSONDecodeError when file is invalid JSON."""
    db = Database.open(tmp_path / "malformed.db", create=True)
    json_path = tmp_path / "corrupt.json"

    # Write truncated/invalid JSON
    json_path.write_text('{"incomplete_key": ', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        sync_characters(db, json_path, direction="auto")


# ==============================================================================
# 3. Linguistic Invariants: Non-Kinship vs Kinship Pronouns
# ==============================================================================


def test_adversarial_linguistic_invariants_matrix() -> None:
    """Exhaustive linguistic check across all RelationshipKinds and gender configurations.

    Guarantees:
    - RIVAL_HOSTILE: ('ta', 'ngươi') - NEVER ('ngươi', 'ta')
    - FRIENDSHIP_CLOSE: ('mình', 'cậu') - NEVER ('cậu', 'mình')
    - FRIENDSHIP_CASUAL: ('tôi', 'cậu') - NEVER ('cậu', 'tôi')
    - FORMAL_ACQUAINTANCE: ('tôi', 'cô'|'anh'|'bạn') - NEVER ('cô'|'anh', 'tôi')
    - FAMILY_ELDER: ('mẹ'|'cha'|'ta', 'con') for senior; ('con', 'mẹ'|'cha'|'người') for junior
    - FAMILY_SIBLING: ('chị'|'anh', 'em') for senior; ('em', 'chị'|'anh') for junior
    - SUPERIOR_SUBORDINATE: ('trẫm'|'ta', 'khanh') for senior; ('thần', 'bệ hạ') for junior
    - MENTOR_DISCIPLE: ('ta', 'con') for senior; ('con', 'thầy'|'sư phụ') for junior
    - SERVANT_MASTER: ('ta', 'cô'|'cậu'|'ngươi') for master;
                      ('em'|'tôi', 'tiểu thư'|'thiếu gia'|'chủ nhân') for servant
    - ROMANCE: authentic gender-matched ('anh', 'em') / ('em', 'anh') / ('mình', 'cậu')
    """
    genders = ["male", "female", "unknown"]

    for spk_g in genders:
        for lis_g in genders:
            # 1. RIVAL_HOSTILE: strictly symmetric ('ta', 'ngươi')
            for senior in (True, False):
                spk, lis = resolve_pronoun_pair_from_relationship(
                    RelationshipKind.RIVAL_HOSTILE, spk_g, lis_g, is_speaker_senior=senior
                )
                assert (spk, lis) == ("ta", "ngươi"), (
                    f"RIVAL_HOSTILE produced invalid ({spk}, {lis}) for {spk_g}->{lis_g}"
                )

            # 2. FRIENDSHIP_CLOSE: strictly ('mình', 'cậu')
            for senior in (True, False):
                spk, lis = resolve_pronoun_pair_from_relationship(
                    RelationshipKind.FRIENDSHIP_CLOSE, spk_g, lis_g, is_speaker_senior=senior
                )
                assert (spk, lis) == ("mình", "cậu")

            # 3. FRIENDSHIP_CASUAL: strictly ('tôi', 'cậu')
            for senior in (True, False):
                spk, lis = resolve_pronoun_pair_from_relationship(
                    RelationshipKind.FRIENDSHIP_CASUAL, spk_g, lis_g, is_speaker_senior=senior
                )
                assert (spk, lis) == ("tôi", "cậu")

            # 4. FORMAL_ACQUAINTANCE: speaker always 'tôi'
            for senior in (True, False):
                spk, lis = resolve_pronoun_pair_from_relationship(
                    RelationshipKind.FORMAL_ACQUAINTANCE, spk_g, lis_g, is_speaker_senior=senior
                )
                assert spk == "tôi"
                assert lis in ("cô", "anh", "bạn")

            # 5. FAMILY_ELDER
            spk_sen, lis_sen = resolve_pronoun_pair_from_relationship(
                RelationshipKind.FAMILY_ELDER, spk_g, lis_g, is_speaker_senior=True
            )
            assert lis_sen == "con"
            assert spk_sen in ("mẹ", "cha", "ta")

            spk_jun, lis_jun = resolve_pronoun_pair_from_relationship(
                RelationshipKind.FAMILY_ELDER, spk_g, lis_g, is_speaker_senior=False
            )
            assert spk_jun == "con"
            assert lis_jun in ("mẹ", "cha", "người")

            # 6. FAMILY_SIBLING
            spk_sib_sen, lis_sib_sen = resolve_pronoun_pair_from_relationship(
                RelationshipKind.FAMILY_SIBLING, spk_g, lis_g, is_speaker_senior=True
            )
            assert spk_sib_sen in ("chị", "anh")
            assert lis_sib_sen == "em"

            spk_sib_jun, lis_sib_jun = resolve_pronoun_pair_from_relationship(
                RelationshipKind.FAMILY_SIBLING, spk_g, lis_g, is_speaker_senior=False
            )
            assert spk_sib_jun == "em"
            assert lis_sib_jun in ("chị", "anh")

            # 7. SUPERIOR_SUBORDINATE
            spk_sup, lis_sup = resolve_pronoun_pair_from_relationship(
                RelationshipKind.SUPERIOR_SUBORDINATE, spk_g, lis_g, is_speaker_senior=True
            )
            assert spk_sup in ("trẫm", "ta")
            assert lis_sup == "khanh"

            spk_sub, lis_sub = resolve_pronoun_pair_from_relationship(
                RelationshipKind.SUPERIOR_SUBORDINATE, spk_g, lis_g, is_speaker_senior=False
            )
            assert spk_sub == "thần"
            assert lis_sub == "bệ hạ"

            # 8. MENTOR_DISCIPLE
            spk_men, lis_men = resolve_pronoun_pair_from_relationship(
                RelationshipKind.MENTOR_DISCIPLE, spk_g, lis_g, is_speaker_senior=True
            )
            assert spk_men == "ta"
            assert lis_men == "con"

            spk_dis, lis_dis = resolve_pronoun_pair_from_relationship(
                RelationshipKind.MENTOR_DISCIPLE, spk_g, lis_g, is_speaker_senior=False
            )
            assert spk_dis == "con"
            assert lis_dis in ("thầy", "sư phụ")


# ==============================================================================
# 4. Collective and Non-Human Entities Adversarial Probing
# ==============================================================================


def test_adversarial_collective_and_non_human_classification() -> None:
    """Stress-test predicate classifications on edge-case and composite names."""
    # Collective entities
    assert is_collective_entity("crowd")
    assert is_collective_entity("mob")
    assert is_collective_entity("everyone")
    assert is_collective_entity("guards")
    assert is_collective_entity("villagers")
    assert is_collective_entity("x", "Đám đông dân làng")
    assert is_collective_entity("y", "Quần chúng thị trấn")
    assert is_collective_entity("z", "Tất cả mọi người")
    assert not is_collective_entity("hero")
    assert not is_collective_entity("liana")

    # Non-human / meta entities
    assert is_non_human_or_meta("system")
    assert is_non_human_or_meta("narrator")
    assert is_non_human_or_meta("th")
    assert is_non_human_or_meta("thought")
    assert is_non_human_or_meta("phone")
    assert is_non_human_or_meta("unknown_voice")
    assert is_non_human_or_meta("x", "Hệ thống thông báo")
    assert is_non_human_or_meta("y", "Độc thoại nội tâm")
    assert not is_non_human_or_meta("hero")
    assert not is_non_human_or_meta("charlotte")


# ==============================================================================
# 5. Full Pipeline Integration Stress Test
# ==============================================================================


def test_adversarial_multi_entity_pipeline_invariants(tmp_path: Path) -> None:
    """Comprehensive test simulating a complex visual novel cast.

    Cast:
    - mc (Hero, Male, Protagonist)
    - mom (Mother, Female, FAMILY_ELDER)
    - sister (Elder Sister, Female, FAMILY_SIBLING)
    - enemy (Dark Lord Malakor, Male, RIVAL_HOSTILE)
    - crowd (Townsfolk, Collective)
    - system (System Announcement, Non-Human Meta)
    - heroine (Charlotte, Female, ROMANCE)

    Verifies:
    1. No hardcoded 'anh - em' on mom, sister, enemy, crowd, system.
    2. Exact authentic pronoun resolution for all interactions.
    3. Bidirectional graph edges recorded and synced.
    4. Non-human tags have empty pronoun maps.
    """
    root = tmp_path / "complex_game"
    (root / "game").mkdir(parents=True)
    (root / "game" / "script.rpy").write_text(
        'define mc = Character("Protagonist")\n'
        'define mom = Character("Mother")\n'
        'define sister = Character("Older Sister")\n'
        'define enemy = Character("Malakor Enemy")\n'
        'define crowd = Character("Villagers Crowd")\n'
        'define system = Character("System Voice")\n'
        'define heroine = Character("Charlotte Heroine")\n',
        encoding="utf-8",
    )

    project = Project.create(root)
    with project.open_db() as db:
        units = [
            # mc talks with mom
            Unit(
                "d:1", UnitKind.DIALOGUE, 1, "Good morning, Mother.", Status.PENDING, speaker="mc"
            ),
            Unit(
                "d:2", UnitKind.DIALOGUE, 2, "Breakfast is ready.", Status.PENDING, speaker="mom"
            ),
            # mc talks with sister
            Unit("d:3", UnitKind.DIALOGUE, 3, "Hey sis, help me?", Status.PENDING, speaker="mc"),
            Unit(
                "d:4", UnitKind.DIALOGUE, 4, "Sure, little bro.", Status.PENDING, speaker="sister"
            ),
            # mc confronts enemy
            Unit("d:5", UnitKind.DIALOGUE, 5, "I will stop you!", Status.PENDING, speaker="mc"),
            Unit(
                "d:6", UnitKind.DIALOGUE, 6, "You will die here!", Status.PENDING, speaker="enemy"
            ),
            # mc addresses crowd
            Unit("d:7", UnitKind.DIALOGUE, 7, "Citizens, hear me!", Status.PENDING, speaker="mc"),
            Unit("d:8", UnitKind.DIALOGUE, 8, "Hurray!", Status.PENDING, speaker="crowd"),
            # system announces
            Unit("d:9", UnitKind.DIALOGUE, 9, "Quest complete.", Status.PENDING, speaker="system"),
            # mc talks with heroine
            Unit("d:10", UnitKind.DIALOGUE, 10, "Charlotte, ok?", Status.PENDING, speaker="mc"),
            Unit(
                "d:11", UnitKind.DIALOGUE, 11, "I was worried.", Status.PENDING, speaker="heroine"
            ),
        ]
        db.save_units_bulk(units)

        # Run full analysis
        stats = run_analysis(project, db)
        assert stats["characters"] >= 6

        chars = db.get_characters()

        # 1. mc assertions
        mc_map = chars["mc"].pronoun_map
        assert mc_map.get("mom") == "con - mẹ", f"mc->mom was {mc_map.get('mom')}"
        assert mc_map.get("sister") == "em - chị", f"mc->sister was {mc_map.get('sister')}"
        assert mc_map.get("enemy") == "ta - ngươi", f"mc->enemy was {mc_map.get('enemy')}"
        assert mc_map.get("crowd") == "tôi - mọi người", f"mc->crowd was {mc_map.get('crowd')}"
        assert mc_map.get("heroine") == "anh - em", f"mc->heroine was {mc_map.get('heroine')}"
        assert "system" not in mc_map

        # 2. Reciprocal assertions
        # mom -> mc
        assert chars["mom"].pronoun_map.get("mc") == "mẹ - con"
        # sister -> mc
        assert chars["sister"].pronoun_map.get("mc") == "chị - em"
        # enemy -> mc: strictly 'ta - ngươi' (NEVER 'ngươi - ta')
        assert chars["enemy"].pronoun_map.get("mc") == "ta - ngươi"
        # crowd -> mc: 'mọi người - bạn'
        assert chars["crowd"].pronoun_map.get("mc") == "mọi người - bạn"
        # heroine -> mc: 'em - anh'
        assert chars["heroine"].pronoun_map.get("mc") == "em - anh"
        # system: completely empty map
        assert chars["system"].pronoun_map == {}

        # 3. RelationshipGraph dual persistence verification
        rel_graph = RelationshipGraph.from_db(db)
        edge_mc_enemy = rel_graph.get_edge("mc", "enemy")
        edge_enemy_mc = rel_graph.get_edge("enemy", "mc")
        assert edge_mc_enemy is not None and edge_enemy_mc is not None
        assert edge_mc_enemy.speaker_pronoun == "ta" and edge_mc_enemy.listener_pronoun == "ngươi"
        assert edge_enemy_mc.speaker_pronoun == "ta" and edge_enemy_mc.listener_pronoun == "ngươi"



def test_manual_pronoun_map_curation_shield_in_analysis(tmp_path: Path) -> None:
    """Verify that custom curated pronoun pairs in pronoun_map are never overwritten by analysis."""
    root = tmp_path / "custom_game"
    (root / "game").mkdir(parents=True)
    (root / "game" / "script.rpy").write_text(
        'define mc = Character("Protagonist")\n'
        'define mom = Character("Mother")\n',
        encoding="utf-8",
    )

    project = Project.create(root)
    with project.open_db() as db:
        # Pre-seed custom regional pronoun mapping in DB
        custom_prof = CharacterProfile(
            tag="mc",
            display_name="Protagonist",
            pronoun_map={"mom": "con - u"},
        )
        db.save_character(custom_prof)

        units = [
            Unit(
                "d:1", UnitKind.DIALOGUE, 1, "Good morning, Mother.", Status.PENDING, speaker="mc"
            ),
            Unit(
                "d:2", UnitKind.DIALOGUE, 2, "Breakfast is ready.", Status.PENDING, speaker="mom"
            ),
        ]
        db.save_units_bulk(units)

        run_analysis(project, db)

        mc_after = db.get_characters()["mc"]
        # Custom "con - u" must be preserved and NOT overwritten by "con - mẹ"
        assert mc_after.pronoun_map.get("mom") == "con - u"


def test_sync_characters_boundary_conditions(tmp_path: Path) -> None:
    """Verify error contracts and boundary conditions in sync_characters."""
    db = Database.open(tmp_path / "boundary.db", create=True)
    missing_json = tmp_path / "missing.json"

    # 1. Invalid direction
    with pytest.raises(ValueError, match="Invalid direction"):
        sync_characters(db, missing_json, direction="invalid_direction")

    # 2. json_to_db with missing file
    with pytest.raises(FileNotFoundError, match="JSON file not found"):
        sync_characters(db, missing_json, direction="json_to_db")

    # 3. Empty DB and non-existent file in auto mode
    c_db, c_json = sync_characters(db, missing_json, direction="auto")
    assert c_db == 0 and c_json == 0
