"""Empirical adversarial stress testing for Milestone 4 Continuous LLM Hallucination Guard.

Targets:
1. 7-category error taxonomy classification and alias normalization (`taxonomy.py`).
2. Fail-Open Resilience policy under harsh faults (`resilience.py`, `scanner.py`, `translate.py`).
3. SQLite `qa_audit_log` persistence (`audit.py`, `schema.py`).
4. `RetryLadder` automated feedback loop (`feedback.py`).
"""

from __future__ import annotations

import http.client
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from renslate.core.config import Config
from renslate.core.db import Database
from renslate.core.models import (
    Origin,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.hallucination import (
    ALL_CATEGORIES,
    CONTEXT_LEAK,
    PRONOUN_ERROR,
    SEMANTIC_INVERSION,
    TONE_REGISTER_VIOLATION,
    HallucinationGuard,
    HallucinationScanResult,
    format_hallucination_feedback,
    is_fail_open_exception,
    is_valid_category,
    normalize_category,
    persist_hallucination_audit,
    scan_results_to_unit_failures,
)
from renslate.engine.repair import UnitFailure
from renslate.engine.translate import TranslationPipeline


def _make_unit(
    uid: str,
    seq: int = 1,
    source: str = "Test source",
    *,
    speaker: str | None = None,
    target: str | None = None,
    status: Status = Status.PENDING,
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        masked_source=source,
        speaker=speaker,
        target=target,
        status=status,
        origin=Origin.LLM,
    )


# ==============================================================================
# Target 1: 7-Category Taxonomy & Alias Normalization Stress Testing
# ==============================================================================


class TestTaxonomyAdversarialStress:
    """Stress tests taxonomy normalization against messy real-world strings."""

    def test_canonical_categories_exact_count_and_members(self) -> None:
        expected = {
            "SPEAKER_MISATTRIBUTION",
            "SEMANTIC_INVERSION",
            "PRONOUN_ERROR",
            "TONE_REGISTER_VIOLATION",
            "TERMINOLOGY_DRIFT",
            "CONTEXT_LEAK",
            "FOREIGN_ARTIFACT",
        }
        assert expected == ALL_CATEGORIES
        assert len(ALL_CATEGORIES) == 7

    @pytest.mark.parametrize(
        ("raw_input", "expected_canonical"),
        [
            # Exact canonical variations
            ("SPEAKER_MISATTRIBUTION", "SPEAKER_MISATTRIBUTION"),
            ("speaker_misattribution", "SPEAKER_MISATTRIBUTION"),
            ("Speaker_Misattribution", "SPEAKER_MISATTRIBUTION"),
            ("  SEMANTIC_INVERSION  ", "SEMANTIC_INVERSION"),
            ("semantic_inversion", "SEMANTIC_INVERSION"),
            ("PRONOUN_ERROR", "PRONOUN_ERROR"),
            ("pronoun_error", "PRONOUN_ERROR"),
            ("tone_register_violation", "TONE_REGISTER_VIOLATION"),
            ("terminology_drift", "TERMINOLOGY_DRIFT"),
            ("context_leak", "CONTEXT_LEAK"),
            ("foreign_artifact", "FOREIGN_ARTIFACT"),
            # RFC-001 synonym mappings
            ("factual_hallucination", "SEMANTIC_INVERSION"),
            ("FACTUAL_HALLUCINATION", "SEMANTIC_INVERSION"),
            ("Factual_Hallucination", "SEMANTIC_INVERSION"),
            ("critical_omission", "SEMANTIC_INVERSION"),
            ("CRITICAL_OMISSION", "SEMANTIC_INVERSION"),
            ("Critical_Omission", "SEMANTIC_INVERSION"),
            ("pronoun_inversion", "PRONOUN_ERROR"),
            ("PRONOUN_INVERSION", "PRONOUN_ERROR"),
            ("Pronoun_Inversion", "PRONOUN_ERROR"),
            ("pronoun_drift", "PRONOUN_ERROR"),
            ("register_breakdown", "TONE_REGISTER_VIOLATION"),
            ("REGISTER_BREAKDOWN", "TONE_REGISTER_VIOLATION"),
            ("emotional_distortion", "TONE_REGISTER_VIOLATION"),
            ("EMOTIONAL_DISTORTION", "TONE_REGISTER_VIOLATION"),
            ("continuity_breakdown", "SPEAKER_MISATTRIBUTION"),
            ("CONTINUITY_BREAKDOWN", "SPEAKER_MISATTRIBUTION"),
            ("leaked_artifacts", "FOREIGN_ARTIFACT"),
            ("LEAKED_ARTIFACTS", "FOREIGN_ARTIFACT"),
            ("leaked_artifact", "CONTEXT_LEAK"),
            # Whitespace and newline variations
            ("\n  pronoun_inversion \t\r\n", "PRONOUN_ERROR"),
            ("   FOREIGN_ARTIFACT   ", "FOREIGN_ARTIFACT"),
        ],
    )
    def test_alias_normalization_thorough(
        self, raw_input: str, expected_canonical: str
    ) -> None:
        assert normalize_category(raw_input) == expected_canonical
        assert is_valid_category(raw_input) is True

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "",
            "   ",
            "completely_bogus_category",
            "HALLUCINATION_GENERAL",
            "unknown_type",
            "12345",
            "!@#$%^&*",
        ],
    )
    def test_invalid_category_rejection(self, invalid_input: str) -> None:
        assert is_valid_category(invalid_input) is False
        normalized = normalize_category(invalid_input)
        if not invalid_input.strip():
            assert normalized == "UNKNOWN"
        else:
            assert normalized == invalid_input.strip().upper()


# ==============================================================================
# Target 2: Fail-Open Resilience Under Harsh Faults
# ==============================================================================


class TestResilienceFailOpenHarshFaults:
    """Stress tests fail-open mechanics on harsh network faults and corrupted responses."""

    @pytest.mark.parametrize(
        "exception_instance",
        [
            TimeoutError("Connection timed out after 60s"),
            http.client.RemoteDisconnected("Remote end closed connection without response"),
            ConnectionResetError("Connection reset by peer [Errno 104]"),
            ConnectionRefusedError("Connection refused by target machine"),
            json.JSONDecodeError("Unterminated string starting at line 1", "doc", 0),
            Exception("HTTP 500 Internal Server Error"),
            Exception("502 Bad Gateway - Cloudflare upstream timeout"),
            Exception("HTTP 503 Service Unavailable"),
            Exception("504 Gateway Time-out"),
            Exception("Temporary server error: capacity exhausted"),
            Exception("Upstream socket disconnected unexpectedly"),
        ],
    )
    def test_is_fail_open_exception_matrix(self, exception_instance: BaseException) -> None:
        assert is_fail_open_exception(exception_instance) is True

    def test_is_fail_open_exception_rejects_programmer_errors(self) -> None:
        assert is_fail_open_exception(ValueError("Invalid argument")) is False
        assert is_fail_open_exception(TypeError("Expected str, got int")) is False
        assert is_fail_open_exception(KeyError("missing_key")) is False

    @pytest.mark.parametrize(
        "simulated_error",
        [
            TimeoutError("LLM API timed out"),
            http.client.RemoteDisconnected("Server severed connection"),
            Exception("HTTP 502 Bad Gateway"),
            Exception("HTTP 503 Service Temporarily Unavailable"),
        ],
    )
    def test_scan_batch_network_faults_fail_open(
        self, simulated_error: BaseException
    ) -> None:
        mock_llm = MagicMock()
        mock_llm.complete_json.side_effect = simulated_error

        guard = HallucinationGuard(llm_client=mock_llm)
        units = [
            _make_unit("u1", 1, "First unit", target="Dòng thứ nhất"),
            _make_unit("u2", 2, "Second unit", target="Dòng thứ hai"),
        ]

        # Must NOT raise exception
        results = guard.scan_batch(units, context_window="Context text")

        assert len(results) == 2
        for r in results:
            assert r.has_hallucination is False
            assert r.category == "SCAN_SKIPPED_TIMEOUT"
            assert r.confidence == 0.0
            assert "Fail-open" in r.explanation
            assert r.severity == "warning"

    @pytest.mark.parametrize(
        "malformed_response",
        [
            '{"scanned_units": [{"uid": "u1", "verdict": "FAIL"',  # Truncated JSON
            "<html><head><title>502 Bad Gateway</title></head></html>",  # HTML error page
            "",  # Empty string
            "   \n\t  ",  # Blank whitespace
            "null",  # JSON null
            "12345",  # JSON integer
            '["not", "a", "dict"]',  # JSON list
            '{"status": "ok"}',  # Dict missing scanned_units
            '{"scanned_units": "invalid_type"}',  # scanned_units not a list
            '{"scanned_units": 42}',
        ],
    )
    def test_parse_guard_response_malformed_fails_open(
        self, malformed_response: str
    ) -> None:
        guard = HallucinationGuard()
        units = [_make_unit("u1", 1, "Some source", target="Một câu thoại")]

        results = guard.parse_guard_response(malformed_response, units=units)

        assert len(results) == 1
        assert results[0].unit_id == "u1"
        assert results[0].has_hallucination is False
        assert results[0].category == "SCAN_SKIPPED_MALFORMED"
        assert results[0].confidence == 0.0
        assert "Fail-open" in results[0].explanation

    @pytest.mark.parametrize(
        "corrupted_confidence_json",
        [
            (
                '{"scanned_units": [{"uid": "u1", "verdict": "FAIL", '
                '"issue_type": "PRONOUN_ERROR", "confidence": null}]}'
            ),
            (
                '{"scanned_units": [{"uid": "u1", "verdict": "FAIL", '
                '"issue_type": "PRONOUN_ERROR", "confidence": "high"}]}'
            ),
        ],
    )
    def test_parse_guard_response_item_confidence_corrupted(
        self, corrupted_confidence_json: str
    ) -> None:
        """When individual unit confidence is null or non-numeric string, it must not crash."""
        guard = HallucinationGuard()
        units = [_make_unit("u1", 1, "Some source", target="Một câu thoại")]
        results = guard.parse_guard_response(corrupted_confidence_json, units=units)
        assert len(results) == 1
        assert results[0].unit_id == "u1"

    def test_scan_batch_malformed_json_returns_skipped_malformed_category(self) -> None:
        """When scan_batch encounters JSONDecodeError, category must be
        SCAN_SKIPPED_MALFORMED, not SCAN_SKIPPED_TIMEOUT.
        """
        mock_llm = MagicMock()
        mock_llm.complete_json.side_effect = json.JSONDecodeError("Unterminated JSON doc", "doc", 0)

        guard = HallucinationGuard(llm_client=mock_llm)
        units = [_make_unit("u1", 1, "Source", target="Target")]

        results = guard.scan_batch(units)
        assert len(results) == 1
        assert results[0].has_hallucination is False
        assert results[0].category == "SCAN_SKIPPED_MALFORMED"

    def test_scan_batch_partial_response_pads_missing_units(self) -> None:
        """When LLM returns fewer units than requested, missing units must default to PASS."""
        mock_llm = MagicMock()
        # LLM only returns u1, completely forgetting u2 and u3
        mock_llm.complete_json.return_value = (
            {
                "scanned_units": [
                    {
                        "uid": "u1",
                        "verdict": "PASS",
                        "issue_type": "none",
                        "confidence": 1.0,
                        "explanation": "Valid",
                    }
                ]
            },
            50,
            20,
        )
        guard = HallucinationGuard(llm_client=mock_llm)
        units = [
            _make_unit("u1", 1, "Unit 1"),
            _make_unit("u2", 2, "Unit 2"),
            _make_unit("u3", 3, "Unit 3"),
        ]

        results = guard.scan_batch(units)

        assert len(results) == 3
        results_by_id = {r.unit_id: r for r in results}
        assert "u1" in results_by_id
        assert "u2" in results_by_id
        assert "u3" in results_by_id

        assert results_by_id["u2"].has_hallucination is False
        assert "omitted in LLM guard scan" in results_by_id["u2"].explanation
        assert results_by_id["u3"].has_hallucination is False

    def test_pipeline_survives_guard_harsh_failure(self, tmp_path: Path) -> None:
        """Full TranslationPipeline execution must not abort when HallucinationGuard fails."""
        root = tmp_path / "fail_open_proj"
        (root / "game").mkdir(parents=True)
        project = Project.create(root)
        cfg = Config()

        u1 = _make_unit("pipe_u1", 1, "Stay safe.", target=None)
        u2 = _make_unit("pipe_u2", 2, "I will.", target=None)

        with project.open_db() as db:
            db.save_unit(u1)
            db.save_unit(u2)

        # Mock LLM for normal translation
        mock_translation_client = MagicMock()
        mock_translation_client.complete_json.return_value = (
            {"pipe_u1": "Hãy cẩn thận.", "pipe_u2": "Tôi sẽ làm thế."},
            50,
            30,
        )

        # Mock Guard raising RemoteDisconnected
        mock_faulty_guard_llm = MagicMock()
        mock_faulty_guard_llm.complete_json.side_effect = (
            http.client.RemoteDisconnected("Socket closed by remote peer")
        )
        guard = HallucinationGuard(llm_client=mock_faulty_guard_llm)

        with project.open_db() as db:
            pipeline = TranslationPipeline(
                project,
                db,
                cfg,
                hallucination_guard=guard,
            )
            pipeline.llm = mock_translation_client

            # Pipeline must run to completion without raising
            pipeline.run()

            # Verify units were committed as translated
            updated_u1 = db.get_unit("pipe_u1")
            updated_u2 = db.get_unit("pipe_u2")
            assert updated_u1 is not None and updated_u1.target == "Hãy cẩn thận."
            assert updated_u1.status == Status.TRANSLATED
            assert updated_u2 is not None and updated_u2.target == "Tôi sẽ làm thế."
            assert updated_u2.status == Status.TRANSLATED

            # Verify audit log records the fail-open warning
            audit_u1 = db.get_qa_audit_logs("pipe_u1")
            assert len(audit_u1) >= 1
            assert audit_u1[0]["verdict"] == "WARNING"
            assert audit_u1[0]["issue_type"] == "SCAN_SKIPPED_TIMEOUT"


# ==============================================================================
# Target 3: SQLite `qa_audit_log` Persistence Stress Testing
# ==============================================================================


class TestAuditLogPersistenceStress:
    """Stress tests SQLite qa_audit_log persistence, constraints, and edge cases."""

    def test_schema_v5_qa_audit_log_structure(self, tmp_path: Path) -> None:
        db_path = tmp_path / "schema_test.db"
        with Database.open(db_path, create=True) as db:
            cur = db.execute("PRAGMA table_info(qa_audit_log)")
            columns = {row["name"]: row["type"].upper() for row in cur.fetchall()}

            expected_columns = {
                "id": "INTEGER",
                "unit_uid": "TEXT",
                "stage": "TEXT",
                "verdict": "TEXT",
                "issue_type": "TEXT",
                "explanation": "TEXT",
                "suggested_correction": "TEXT",
                "confidence": "REAL",
                "created_at": "TEXT",
            }
            for col, col_type in expected_columns.items():
                assert col in columns
                assert col_type in columns[col]

            cur = db.execute("PRAGMA index_list(qa_audit_log)")
            indices = {row["name"] for row in cur.fetchall()}
            assert "idx_qa_audit_unit" in indices
            assert "idx_qa_audit_stage" in indices

    def test_persist_audit_handles_special_characters_and_sql_injection(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "special_chars.db"
        with Database.open(db_path, create=True) as db:
            uid = "d:injection_test"
            db.save_unit(_make_unit(uid, 1, "Source text"))

            adversarial_explanation = (
                "DROP TABLE qa_audit_log; -- ' \" \x00 <script>alert(1)</script>"
                " Tiếng Việt có dấu, ký tự lạ 漢字 và emoji 🛡️"
            )
            adversarial_correction = "Gợi ý: [player_name] nói \"{b}Cẩn thận!{/b}\""

            results = [
                HallucinationScanResult(
                    unit_id=uid,
                    has_hallucination=True,
                    category=SEMANTIC_INVERSION,
                    confidence=0.975,
                    explanation=adversarial_explanation,
                    suggested_correction=adversarial_correction,
                    severity="error",
                )
            ]

            inserted = persist_hallucination_audit(db, results)
            assert inserted == 1

            logs = db.get_qa_audit_logs(uid)
            assert len(logs) == 1
            assert logs[0]["explanation"] == adversarial_explanation
            assert logs[0]["suggested_correction"] == adversarial_correction
            assert logs[0]["confidence"] == 0.975
            assert logs[0]["verdict"] == "FAIL"

    def test_foreign_key_cascade_deletion(self, tmp_path: Path) -> None:
        """When a Unit is removed from DB, associated qa_audit_log entries must cascade delete."""
        db_path = tmp_path / "cascade.db"
        with Database.open(db_path, create=True) as db:
            uid = "unit_to_cascade"
            db.save_unit(_make_unit(uid, 1, "Source to delete"))

            results = [
                HallucinationScanResult(
                    unit_id=uid,
                    has_hallucination=True,
                    category=PRONOUN_ERROR,
                    confidence=0.85,
                    explanation="Xưng hô sai",
                )
            ]
            persist_hallucination_audit(db, results)
            assert len(db.get_qa_audit_logs(uid)) == 1

            # Delete the parent unit
            with db.transaction():
                db.execute("DELETE FROM units WHERE uid = ?", (uid,))

            # Audit logs must have been cascaded and deleted
            cur = db.execute("SELECT * FROM qa_audit_log WHERE unit_uid = ?", (uid,))
            assert len(cur.fetchall()) == 0

    def test_empty_results_list_handled_gracefully(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty_audit.db"
        with Database.open(db_path, create=True) as db:
            inserted = persist_hallucination_audit(db, [])
            assert inserted == 0


# ==============================================================================
# Target 4: RetryLadder Feedback Loop & Confidence Thresholding
# ==============================================================================


class TestRetryLadderFeedbackLoop:
    """Stress tests confidence boundary behavior and structured feedback generation."""

    def test_confidence_threshold_boundary(self) -> None:
        """Boundary test: 0.70 is included, 0.6999 is excluded, 0.7001 is included."""
        u1 = _make_unit("u_exact", 1, "Source 1", target="Target 1")
        u2 = _make_unit("u_below", 2, "Source 2", target="Target 2")
        u3 = _make_unit("u_above", 3, "Source 3", target="Target 3")
        u4 = _make_unit("u_zero", 4, "Source 4", target="Target 4")
        u5 = _make_unit("u_max", 5, "Source 5", target="Target 5")

        units_by_id = {u.uid: u for u in (u1, u2, u3, u4, u5)}

        results = [
            HallucinationScanResult(
                unit_id="u_exact",
                has_hallucination=True,
                category=SEMANTIC_INVERSION,
                confidence=0.70,
                explanation="Exactly at boundary",
            ),
            HallucinationScanResult(
                unit_id="u_below",
                has_hallucination=True,
                category=SEMANTIC_INVERSION,
                confidence=0.6999,
                explanation="Just below boundary",
            ),
            HallucinationScanResult(
                unit_id="u_above",
                has_hallucination=True,
                category=PRONOUN_ERROR,
                confidence=0.7001,
                explanation="Just above boundary",
            ),
            HallucinationScanResult(
                unit_id="u_zero",
                has_hallucination=True,
                category=TONE_REGISTER_VIOLATION,
                confidence=0.0,
                explanation="Zero confidence",
            ),
            HallucinationScanResult(
                unit_id="u_max",
                has_hallucination=True,
                category=CONTEXT_LEAK,
                confidence=1.0,
                explanation="Max confidence",
            ),
        ]

        failures = scan_results_to_unit_failures(results, units_by_id, min_confidence=0.70)
        failure_uids = {f.unit.uid for f in failures}

        assert "u_exact" in failure_uids
        assert "u_above" in failure_uids
        assert "u_max" in failure_uids
        assert "u_below" not in failure_uids
        assert "u_zero" not in failure_uids

    def test_non_hallucination_with_high_confidence_never_creates_failure(self) -> None:
        u1 = _make_unit("u_clean", 1, "Clean source", target="Bản dịch sạch")
        units_by_id = {u1.uid: u1}

        result = HallucinationScanResult(
            unit_id="u_clean",
            has_hallucination=False,
            category="",
            confidence=1.0,
            explanation="Clean translation",
        )
        failures = scan_results_to_unit_failures([result], units_by_id)
        assert len(failures) == 0

    def test_missing_unit_id_in_map_skipped_safely(self) -> None:
        result = HallucinationScanResult(
            unit_id="orphan_uid",
            has_hallucination=True,
            category=PRONOUN_ERROR,
            confidence=0.95,
            explanation="Missing from dictionary",
        )
        failures = scan_results_to_unit_failures([result], {})
        assert len(failures) == 0

    def test_formatted_feedback_contains_all_actionable_instructions(self) -> None:
        u = _make_unit(
            "d:404", 404, "Wait, brother, do not enter!", target="Chờ đã em trai, đừng vào!"
        )
        failure = UnitFailure(
            unit=u,
            code=PRONOUN_ERROR,
            detail="Người nói gọi là 'brother' (anh trai), bản dịch dịch thành 'em trai'.",
            candidate="Chờ đã em trai, đừng vào!",
        )

        feedback_text = format_hallucination_feedback([failure])

        assert "LỖI Ở LẦN DỊCH TRƯỚC" in feedback_text
        assert "d:404" in feedback_text
        assert PRONOUN_ERROR in feedback_text
        assert "em trai" in feedback_text
        assert "Chờ đã em trai, đừng vào!" in feedback_text

    def test_pipeline_separates_high_and_low_confidence_detections(
        self, tmp_path: Path
    ) -> None:
        """High-confidence errors trigger retry, while low-confidence errors
        are flagged without retry.
        """
        root = tmp_path / "conf_proj"
        (root / "game").mkdir(parents=True)
        project = Project.create(root)
        cfg = Config()

        u_high = _make_unit("u_high_conf", 1, "High confidence test line")
        u_low = _make_unit("u_low_conf", 2, "Low confidence test line")

        with project.open_db() as db:
            db.save_unit(u_high)
            db.save_unit(u_low)

        mock_translation_client = MagicMock()
        # First pass translations
        mock_translation_client.complete_json.side_effect = [
            (
                {
                    "u_high_conf": "Bản dịch lỗi nặng",
                    "u_low_conf": "Bản dịch nghi ngờ nhẹ",
                },
                50,
                30,
            ),
            # Retry pass for u_high_conf
            (
                {"u_high_conf": "Bản dịch đã sửa chuẩn xác"},
                30,
                20,
            ),
        ]

        mock_guard = MagicMock()
        # Guard returns 1 high-confidence error (>= 0.70) and 1 low-confidence error (< 0.70)
        mock_guard.scan_batch.side_effect = [
            [
                HallucinationScanResult(
                    unit_id="u_high_conf",
                    has_hallucination=True,
                    category=SEMANTIC_INVERSION,
                    confidence=0.95,  # High confidence -> triggers retry
                    explanation="Nội dung sai lệch nghiêm trọng",
                    suggested_correction="Bản dịch đã sửa",
                ),
                HallucinationScanResult(
                    unit_id="u_low_conf",
                    has_hallucination=True,
                    category=TONE_REGISTER_VIOLATION,
                    confidence=0.55,  # Low confidence -> flagged without retry
                    explanation="Nghi ngờ giọng điệu hơi cứng",
                ),
            ],
            # Second scan on retry pass: u_high_conf passes
            [
                HallucinationScanResult(
                    unit_id="u_high_conf",
                    has_hallucination=False,
                    confidence=1.0,
                    explanation="Đạt chất lượng",
                )
            ],
        ]

        with project.open_db() as db:
            pipeline = TranslationPipeline(
                project,
                db,
                cfg,
                hallucination_guard=mock_guard,
            )
            pipeline.llm = mock_translation_client

            pipeline.run()

            # Verify low-confidence unit was flagged as QA_FLAGGED without infinite retry
            saved_low = db.get_unit("u_low_conf")
            assert saved_low is not None
            assert saved_low.status == Status.QA_FLAGGED
            assert any("qa_hallucination" in f for f in saved_low.qa_flags)

            # Verify high-confidence unit went through retry and was repaired
            saved_high = db.get_unit("u_high_conf")
            assert saved_high is not None
            assert saved_high.target == "Bản dịch đã sửa chuẩn xác"
            assert saved_high.status == Status.TRANSLATED
