"""Empirical challenge suite for Milestone 3.

Evaluates:
1. Romance Heuristic removal and persona extraction.
2. Glossary Decommissioning & externalized stopwords dictionary.
3. Modularization, backward compatibility shims, and schema migration v5.
4. Centralized configuration (AppConfig) and env var overrides.
5. UI Wizard modularization and dynamic monkeypatching.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

import renslate.core.db as root_db
import renslate.core.db.schema as db_schema
import renslate.engine.glossary as engine_glossary
import renslate.glossary as root_glossary
import renslate.ui.wizard as root_wizard
from renslate.core.config import AppConfig, LLMConfig
from renslate.core.db import Database
from renslate.core.models import (
    RelationshipKind,
)
from renslate.engine.analyze import (
    infer_semantic_relationship,
    resolve_pronoun_pair_from_relationship,
)


class TestMilestone3RomanceHeuristicRemoval:
    """Empirical challenge on the elimination of speculative romance heuristics."""

    def test_neutral_opposite_gender_defaults_to_formal_acquaintance(self) -> None:
        """Male MC and Female supporting cast without romance cues MUST yield FORMAL_ACQUAINTANCE."""
        rel = infer_semantic_relationship(
            "mc",
            {"role": "Protagonist", "gender": "male", "display_name": "Hero"},
            "alice",
            {"role": "Knight Companion", "gender": "female", "display_name": "Alice"},
            is_mc=True,
        )
        assert rel is not None
        kind, is_senior = rel
        assert kind == RelationshipKind.FORMAL_ACQUAINTANCE

        # Pronoun resolution for MC -> Alice
        mc_to_alice = resolve_pronoun_pair_from_relationship(
            kind, "male", "female", is_speaker_senior=is_senior
        )
        assert mc_to_alice == ("tôi", "cô"), f"Expected ('tôi', 'cô'), got {mc_to_alice}"

        # Pronoun resolution for Alice -> MC
        alice_to_mc = resolve_pronoun_pair_from_relationship(
            kind, "female", "male", is_speaker_senior=not is_senior
        )
        assert alice_to_mc == ("tôi", "anh"), f"Expected ('tôi', 'anh'), got {alice_to_mc}"

    def test_female_mc_and_male_neutral_defaults_to_formal_acquaintance(self) -> None:
        """Female MC and Male supporting cast without romance cues MUST yield FORMAL_ACQUAINTANCE."""
        rel = infer_semantic_relationship(
            "mc",
            {"role": "Protagonist", "gender": "female", "display_name": "Valerie"},
            "bob",
            {"role": "Merchant", "gender": "male", "display_name": "Bob"},
            is_mc=True,
        )
        assert rel is not None
        kind, is_senior = rel
        assert kind == RelationshipKind.FORMAL_ACQUAINTANCE

        mc_to_bob = resolve_pronoun_pair_from_relationship(
            kind, "female", "male", is_speaker_senior=is_senior
        )
        assert mc_to_bob == ("tôi", "anh")

        bob_to_mc = resolve_pronoun_pair_from_relationship(
            kind, "male", "female", is_speaker_senior=not is_senior
        )
        assert bob_to_mc == ("tôi", "cô")

    def test_same_gender_and_unknown_neutral_formal_acquaintance(self) -> None:
        """Neutral interactions for same-gender and unknown gender."""
        # Male to Male
        m_m = resolve_pronoun_pair_from_relationship(
            RelationshipKind.FORMAL_ACQUAINTANCE, "male", "male"
        )
        assert m_m == ("tôi", "anh")

        # Female to Female
        f_f = resolve_pronoun_pair_from_relationship(
            RelationshipKind.FORMAL_ACQUAINTANCE, "female", "female"
        )
        assert f_f == ("tôi", "cô")

        # Unknown listener
        m_u = resolve_pronoun_pair_from_relationship(
            RelationshipKind.FORMAL_ACQUAINTANCE, "male", "unknown"
        )
        assert m_u == ("tôi", "bạn")

        # Unknown speaker and unknown listener
        u_u = resolve_pronoun_pair_from_relationship(
            RelationshipKind.FORMAL_ACQUAINTANCE, "unknown", "unknown"
        )
        assert u_u == ("tôi", "bạn")

    def test_explicit_romance_cues_strictly_required_for_romance(self) -> None:
        """Only explicit romance semantic cues trigger RelationshipKind.ROMANCE."""
        # Explicit romance cue: "lover"
        rel_lover = infer_semantic_relationship(
            "mc",
            {"role": "Protagonist", "gender": "male"},
            "sarah",
            {"role": "Childhood Lover", "gender": "female"},
            is_mc=True,
        )
        assert rel_lover is not None
        assert rel_lover[0] == RelationshipKind.ROMANCE

        spk_pro, lis_pro = resolve_pronoun_pair_from_relationship(
            rel_lover[0], "male", "female", is_speaker_senior=rel_lover[1]
        )
        assert (spk_pro, lis_pro) == ("anh", "em")

        # Explicit romance cue: "vợ"
        rel_wife = infer_semantic_relationship(
            "mc",
            {"role": "Protagonist", "gender": "male"},
            "maria",
            {"role": "Vợ của nhân vật chính", "gender": "female"},
            is_mc=True,
        )
        assert rel_wife is not None
        assert rel_wife[0] == RelationshipKind.ROMANCE

        # Adversarial non-romance word boundary check: "Lovecraftian scholar"
        rel_lovecraft = infer_semantic_relationship(
            "mc",
            {"role": "Protagonist", "gender": "male"},
            "edward",
            {"role": "Lovecraftian scholar", "gender": "female"},
            is_mc=True,
        )
        assert rel_lovecraft is not None
        # Must NOT match "lover" because word boundary \blover\b is used
        assert rel_lovecraft[0] == RelationshipKind.FORMAL_ACQUAINTANCE

    def test_adversarial_female_mc_tagged_heroine_spuriously_triggers_romance(self) -> None:
        """Adversarial case: female MC with display name/role 'Heroine' must NOT default to romance with a guard."""
        rel = infer_semantic_relationship(
            "mc",
            {"role": "Protagonist", "gender": "female", "display_name": "Heroine"},
            "guard",
            {"role": "Gate Guard", "gender": "male", "display_name": "Guard"},
            is_mc=True,
        )
        assert rel is not None
        # Should be FORMAL_ACQUAINTANCE, but currently fails because 'heroine' in text_self triggers ROMANCE
        assert rel[0] == RelationshipKind.FORMAL_ACQUAINTANCE


class TestMilestone3GlossaryDecommissioning:
    """Empirical challenge on glossary data extraction and backward-compatible shims."""

    def test_stopwords_json_asset_exists_and_valid(self) -> None:
        """english_stopwords.json exists and is structured with required categories."""
        asset_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "renslate"
            / "data"
            / "dictionaries"
            / "english_stopwords.json"
        )
        assert asset_path.is_file(), f"Asset not found at {asset_path}"
        data = json.loads(asset_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        for expected_key in (
            "pronouns",
            "stop_words",
            "excluded_common_phrases",
            "contraction_starts",
            "number_or_time_words",
        ):
            assert expected_key in data
            assert isinstance(data[expected_key], list)
            assert len(data[expected_key]) > 0

    def test_glossary_source_files_do_not_contain_embedded_wordlists(self) -> None:
        """Neither engine/glossary.py nor root glossary.py embed 400+ LOC static wordlists."""
        engine_file = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "renslate"
            / "engine"
            / "glossary.py"
        )
        content = engine_file.read_text(encoding="utf-8")
        assert "_ENGLISH_STOP_WORDS = {" not in content
        assert "_ENGLISH_PRONOUNS = {" not in content
        assert "_load_stopwords_asset()" in content

        shim_file = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "renslate"
            / "glossary.py"
        )
        assert len(shim_file.read_text(encoding="utf-8").splitlines()) < 40

    def test_glossary_shim_re_exports_callable_interfaces(self) -> None:
        """Root glossary.py re-exports expected public functions and classes."""
        for name in (
            "Glossary",
            "clean_glossary",
            "is_valid_glossary_term",
            "mine_glossary_candidates",
            "translate_glossary_with_llm",
        ):
            assert hasattr(root_glossary, name)
            assert getattr(root_glossary, name) is getattr(engine_glossary, name)

    def test_adversarial_glossary_shim_preserves_all_metadata(self) -> None:
        """Adversarial check: root_glossary must expose __all__ restricting exported symbols."""
        assert hasattr(root_glossary, "__all__")
        assert "__all__" in dir(root_glossary)

    def test_term_validation_with_loaded_stopwords(self) -> None:
        """is_valid_glossary_term correctly rejects grammar noise and accepts valid terms."""
        # Stop words
        assert not engine_glossary.is_valid_glossary_term("about")
        assert not engine_glossary.is_valid_glossary_term("the")
        # Pronouns
        assert not engine_glossary.is_valid_glossary_term("herself")
        # Multi-word all stop words
        assert not engine_glossary.is_valid_glossary_term("it was that")
        # Phrase starting with contraction
        assert not engine_glossary.is_valid_glossary_term("don't think")
        # Genuine proper names/terms
        assert engine_glossary.is_valid_glossary_term("Grand Duke")
        assert engine_glossary.is_valid_glossary_term("Holy Grail")
        assert engine_glossary.is_valid_glossary_term("Excalibur")

    def test_adversarial_pyproject_package_data_includes_dictionaries(self) -> None:
        """Adversarial check: pyproject.toml must declare dictionaries in package-data."""
        pyproject_file = Path(__file__).resolve().parent.parent / "pyproject.toml"
        content = pyproject_file.read_text(encoding="utf-8")
        assert "data/dictionaries/*.json" in content or "*.json" in content


class TestMilestone3ConfigurationStandardization:
    """Empirical challenge on AppConfig centralization and runtime environment overrides."""

    def test_appconfig_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AppConfig dynamically reflects RENSLATE_* environment variables."""
        test_url = "http://192.168.1.100:9999/v1"
        test_model = "Custom-Model-V1.gguf"
        test_db = "custom_path.db"

        monkeypatch.setenv("RENSLATE_LLM_URL", test_url)
        monkeypatch.setenv("RENSLATE_MODEL", test_model)
        monkeypatch.setenv("RENSLATE_DB_PATH", test_db)

        # Class-level access
        assert AppConfig.llm_api_base_url == test_url
        assert AppConfig.llm_model_name == test_model
        assert AppConfig.db_path == test_db
        assert AppConfig.server_base_url == "http://192.168.1.100:9999"

        # LLMConfig default factory should reflect AppConfig
        llm_cfg = LLMConfig()
        assert llm_cfg.base_url == test_url
        assert llm_cfg.model == test_model

        # Instance-level overrides take precedence over env vars
        inst = AppConfig(
            llm_api_base_url="http://localhost:5000/v1",
            llm_model_name="Instance-Model.gguf",
        )
        assert inst.llm_api_base_url == "http://localhost:5000/v1"
        assert inst.llm_model_name == "Instance-Model.gguf"
        assert inst.server_url == "http://localhost:5000"

    def test_server_base_url_strips_v1_without_trailing_slash(self) -> None:
        """server_base_url strips /v1 when no trailing slash is provided."""
        inst = AppConfig(llm_api_base_url="http://host:1234/v1")
        assert inst.server_url == "http://host:1234"

    def test_adversarial_appconfig_trailing_slash_fails_to_strip_v1(self) -> None:
        """Adversarial check: URL ending with '/v1/' must be stripped to server root."""
        inst = AppConfig(llm_api_base_url="http://host:1234/v1/")
        # Should be 'http://host:1234', currently returns 'http://host:1234/v1'
        assert inst.server_url == "http://host:1234"


class TestMilestone3DatabaseModularizationAndV5Schema:
    """Empirical challenge on DB decomposition, migration v5, and repository functions."""

    def test_db_shim_forwarding(self) -> None:
        """renslate.core.db package and renslate.core.db module expose identical symbols."""
        for sym in (
            "Database",
            "UnitRepository",
            "CharacterRepository",
            "RelationshipRepository",
            "SceneRepository",
            "QAAuditRepository",
            "TMRepository",
            "BatchRepository",
            "GlossaryRepository",
        ):
            assert hasattr(root_db, sym)

    def test_schema_v5_migration_and_scene_backfill(self, tmp_path: Path) -> None:
        """Migrating legacy v4 db to v5 creates .v4.bak backup and backfills scene metadata."""
        db_file = tmp_path / "test_migrate.db"
        conn = sqlite3.connect(str(db_file))

        # Manually create schema v4
        for script in db_schema.MIGRATIONS[:4]:
            conn.executescript(script)
        conn.execute("PRAGMA user_version = 4")

        # Insert legacy scene and meta entry
        conn.execute(
            "INSERT INTO scenes (scene_id, label, first_seq, last_seq, summary) VALUES ('sc_1', 'intro', 1, 10, 'old summary')"
        )
        legacy_meta = json.dumps({
            "location": "Throne Room",
            "mood": "tense",
            "atmosphere": "tense",
            "characters": ["mc", "queen"],
            "time": "dusk",
            "previous_summary": "Introductory march",
            "updated_at": "2026-11-01T00:00:00Z",
        })
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('scene_ctx:sc_1', ?)",
            (legacy_meta,),
        )
        conn.commit()
        conn.close()

        # Run migrations to v5 via Database.open
        db = Database.open(db_file)
        assert db.schema_version == 5

        # Verify .v4.bak was created
        bak_file = Path(f"{db_file}.v4.bak")
        assert bak_file.is_file(), f"Pre-migration backup {bak_file} not found"

        # Verify scenes table columns were backfilled
        row = db._connection.execute(
            "SELECT location, mood, atmosphere, characters_json, temporal_anchor, previous_summary FROM scenes WHERE scene_id = 'sc_1'"
        ).fetchone()
        assert row is not None
        assert row["location"] == "Throne Room"
        assert row["mood"] == "tense"
        assert row["atmosphere"] == "tense"
        assert json.loads(row["characters_json"]) == ["mc", "queen"]
        assert row["temporal_anchor"] == "dusk"
        assert row["previous_summary"] == "Introductory march"

        # Verify qa_audit_log table is functional
        db._connection.execute(
            """
            INSERT INTO units (uid, kind, seq, source, status, updated_at)
            VALUES ('u1', 'dialogue', 1, 'Hello', 'translated', '2026-11-01T00:00:00Z')
            """
        )
        db._connection.execute(
            """
            INSERT INTO qa_audit_log (unit_uid, stage, verdict, issue_type, explanation, suggested_correction, confidence)
            VALUES ('u1', 'translation_hallucination', 'flagged', 'pronoun_drift', 'Pronoun changed from tôi to anh', 'tôi', 0.95)
            """
        )
        db._connection.commit()

        audit_row = db._connection.execute(
            "SELECT unit_uid, stage, verdict, issue_type, confidence FROM qa_audit_log WHERE unit_uid = 'u1'"
        ).fetchone()
        assert audit_row is not None
        assert audit_row["verdict"] == "flagged"
        assert audit_row["confidence"] == 0.95
        db.close()

    def test_get_scene_context_alias_consistency(self, tmp_path: Path) -> None:
        """Database and SceneRepository expose get_scene_context alias matching get_scene_macro_context."""
        db_file = tmp_path / "test_alias.db"
        db = Database.open(db_file, create=True)
        assert hasattr(db, "get_scene_context")
        assert hasattr(db._scenes_repo, "get_scene_context")
        assert Database.get_scene_context is Database.get_scene_macro_context
        assert db.get_scene_context == db.get_scene_macro_context
        assert db._scenes_repo.get_scene_context == db._scenes_repo.get_scene_macro_context
        db.close()


class TestMilestone3UIWizardModularization:
    """Empirical challenge on UI wizard decomposition and monkeypatch interception."""

    def test_wizard_shim_re_exports(self) -> None:
        """ui/wizard.py shim provides all expected interfaces."""
        for sym in root_wizard.__all__:
            assert hasattr(root_wizard, sym)

    def test_dynamic_module_lookup_preserves_test_monkeypatching(self) -> None:
        """Dynamic lookup in config_flow._get_select_menu intercepts package-level mocks."""
        with patch("renslate.ui.wizard.select_menu", return_value="custom_choice") as mock_menu:
            from renslate.ui.wizard.config_flow import _get_select_menu

            menu_fn = _get_select_menu()
            res = menu_fn("prompt", [("k", "desc")])
            assert res == "custom_choice"
            assert mock_menu.called
