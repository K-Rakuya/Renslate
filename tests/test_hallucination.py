"""Comprehensive unit tests for renslate.engine.hallucination subpackage."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from renslate.core.config import AppConfig, Config
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
    FOREIGN_ARTIFACT,
    PRONOUN_ERROR,
    SEMANTIC_INVERSION,
    SPEAKER_MISATTRIBUTION,
    TONE_REGISTER_VIOLATION,
    HallucinationBatchReport,
    HallucinationGuard,
    HallucinationScanResult,
    fail_open_results,
    format_hallucination_feedback,
    is_fail_open_exception,
    is_valid_category,
    normalize_category,
    persist_hallucination_audit,
    scan_results_to_unit_failures,
)
from renslate.engine.translate import TranslationPipeline


def _make_unit(
    uid: str,
    seq: int,
    source: str,
    *,
    speaker: str | None = None,
    target: str | None = None,
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        masked_source=source,
        speaker=speaker,
        target=target,
        status=Status.PENDING,
        origin=Origin.LLM,
    )


class TestHallucinationTaxonomy:
    """Verifies the 7 canonical categories, alias mappings, and normalization."""

    def test_7_canonical_categories(self) -> None:
        assert len(ALL_CATEGORIES) == 7
        assert "SPEAKER_MISATTRIBUTION" in ALL_CATEGORIES
        assert "SEMANTIC_INVERSION" in ALL_CATEGORIES
        assert "PRONOUN_ERROR" in ALL_CATEGORIES
        assert "TONE_REGISTER_VIOLATION" in ALL_CATEGORIES
        assert "TERMINOLOGY_DRIFT" in ALL_CATEGORIES
        assert "CONTEXT_LEAK" in ALL_CATEGORIES
        assert "FOREIGN_ARTIFACT" in ALL_CATEGORIES

    def test_alias_normalization(self) -> None:
        # RFC-001 and lowercase aliases
        assert normalize_category("factual_hallucination") == SEMANTIC_INVERSION
        assert normalize_category("critical_omission") == SEMANTIC_INVERSION
        assert normalize_category("pronoun_inversion") == PRONOUN_ERROR
        assert normalize_category("register_breakdown") == TONE_REGISTER_VIOLATION
        assert normalize_category("emotional_distortion") == TONE_REGISTER_VIOLATION
        assert normalize_category("continuity_breakdown") == SPEAKER_MISATTRIBUTION
        assert normalize_category("leaked_artifacts") == FOREIGN_ARTIFACT
        assert normalize_category("leaked_artifact") == CONTEXT_LEAK

    def test_is_valid_category(self) -> None:
        assert is_valid_category("PRONOUN_ERROR") is True
        assert is_valid_category("pronoun_inversion") is True
        assert is_valid_category("FOREIGN_ARTIFACT") is True
        assert is_valid_category("completely_bogus_category") is False


class TestHallucinationModels:
    """Verifies dataclass immutability and serialization roundtripping."""

    def test_scan_result_model(self) -> None:
        res = HallucinationScanResult(
            unit_id="d:10",
            has_hallucination=True,
            category=PRONOUN_ERROR,
            confidence=0.96,
            explanation="Tự xưng sai vai vế",
            suggested_correction="Anh yêu em",
            severity="error",
        )
        assert res.unit_id == "d:10"
        assert res.has_hallucination is True
        d = res.to_dict()
        assert d["confidence"] == 0.96
        roundtrip = HallucinationScanResult.from_dict(d)
        assert roundtrip == res

    def test_batch_report_model(self) -> None:
        r1 = HallucinationScanResult(unit_id="u1", has_hallucination=False)
        r2 = HallucinationScanResult(
            unit_id="u2", has_hallucination=True, category=SEMANTIC_INVERSION
        )
        report = HallucinationBatchReport(
            batch_id="b1",
            results=(r1, r2),
            has_hallucinations=True,
            scanned_count=2,
            flagged_count=1,
            scan_duration_ms=45.0,
        )
        assert report.flagged_count == 1
        d = report.to_dict()
        assert len(d["results"]) == 2
        assert d["has_hallucinations"] is True


class TestHallucinationResilienceFailOpen:
    """Verifies fail-open policy on timeout, unparseable JSON, and 5xx errors."""

    def test_is_fail_open_exception(self) -> None:
        assert is_fail_open_exception(TimeoutError("Connection timed out")) is True
        assert is_fail_open_exception(json.JSONDecodeError("Unterminated string", "doc", 0)) is True
        assert is_fail_open_exception(ConnectionResetError()) is True
        assert is_fail_open_exception(Exception("HTTP 502 Bad Gateway")) is True
        assert is_fail_open_exception(ValueError("Some invalid argument")) is False

    def test_fail_open_results(self) -> None:
        units = [_make_unit("u1", 1, "Hello"), _make_unit("u2", 2, "World")]
        results = fail_open_results(units, "Server connection dropped")
        assert len(results) == 2
        for r in results:
            assert r.has_hallucination is False
            assert "Fail-open" in r.explanation
            assert r.category == "SCAN_SKIPPED_TIMEOUT"


class TestHallucinationScanner:
    """Verifies HallucinationGuard prompt assembly, response parsing, and execution."""

    def test_guard_initialization(self) -> None:
        guard = HallucinationGuard()
        assert guard.llm_config.base_url == AppConfig.llm_api_base_url
        assert guard.llm_config.model == AppConfig.llm_model_name

    def test_build_guard_prompt(self) -> None:
        guard = HallucinationGuard()
        u1 = _make_unit("d:1", 1, "Who goes there?", speaker="guard", target="Ai đi đó?")
        prompt = guard.build_guard_prompt([u1], context_window="Cảnh cổng thành ban đêm.")
        assert "Cảnh cổng thành ban đêm." in prompt
        assert "d:1" in prompt
        assert "Ai đi đó?" in prompt

    def test_parse_valid_guard_response(self) -> None:
        guard = HallucinationGuard()
        raw = json.dumps(
            {
                "scanned_units": [
                    {
                        "uid": "d:1",
                        "verdict": "PASS",
                        "issue_type": "none",
                        "confidence": 1.0,
                        "explanation": "Dịch chuẩn xác",
                    },
                    {
                        "uid": "d:2",
                        "verdict": "FAIL",
                        "issue_type": "PRONOUN_ERROR",
                        "confidence": 0.95,
                        "explanation": "Đảo ngôi xưng hô",
                        "suggested_correction": "Anh sẽ bảo vệ em",
                    },
                ]
            }
        )
        results = guard.parse_guard_response(raw)
        assert len(results) == 2
        assert results[0].has_hallucination is False
        assert results[1].has_hallucination is True
        assert results[1].category == "PRONOUN_ERROR"
        assert results[1].suggested_correction == "Anh sẽ bảo vệ em"

    def test_parse_malformed_response_fails_open(self) -> None:
        guard = HallucinationGuard()
        units = [_make_unit("d:1", 1, "Test text")]
        # Truncated or malformed JSON
        malformed = '{"scanned_units": [{"uid": "d:1", "verdict": '
        results = guard.parse_guard_response(malformed, units=units)
        assert len(results) == 1
        assert results[0].has_hallucination is False
        assert results[0].category == "SCAN_SKIPPED_MALFORMED"

    def test_scan_batch_with_mock_llm(self) -> None:
        mock_llm = MagicMock()
        mock_llm.complete_json.return_value = (
            {
                "scanned_units": [
                    {
                        "uid": "u100",
                        "verdict": "FAIL",
                        "issue_type": "SEMANTIC_INVERSION",
                        "confidence": 0.99,
                        "explanation": "Bịa đặt thêm chi tiết",
                        "suggested_correction": "Bản dịch sửa",
                    }
                ]
            },
            100,
            50,
        )
        guard = HallucinationGuard(llm_client=mock_llm)
        unit = _make_unit("u100", 1, "Source", target="Target")
        results = guard.scan_batch([unit])
        assert len(results) == 1
        assert results[0].has_hallucination is True
        assert results[0].category == "SEMANTIC_INVERSION"


class TestHallucinationFeedbackAndAudit:
    """Verifies conversion to UnitFailure for RetryLadder and SQLite audit log persistence."""

    def test_scan_results_to_unit_failures(self) -> None:
        u1 = _make_unit("u1", 1, "Run!", target="Chạy đi!")
        u2 = _make_unit("u2", 2, "I will stay.", target="Em sẽ ở lại.")
        r1 = HallucinationScanResult(unit_id="u1", has_hallucination=False)
        r2 = HallucinationScanResult(
            unit_id="u2",
            has_hallucination=True,
            category=PRONOUN_ERROR,
            confidence=0.92,
            explanation="Xưng nhầm em",
            suggested_correction="Anh sẽ ở lại.",
        )
        failures = scan_results_to_unit_failures([r1, r2], {"u1": u1, "u2": u2})
        assert len(failures) == 1
        assert failures[0].unit.uid == "u2"
        assert failures[0].code == PRONOUN_ERROR
        assert "Anh sẽ ở lại." in failures[0].detail

        # Format feedback text
        feedback_text = format_hallucination_feedback(failures)
        assert "LỖI Ở LẦN DỊCH TRƯỚC" in feedback_text
        assert "u2" in feedback_text
        assert PRONOUN_ERROR in feedback_text

    def test_persist_hallucination_audit(self, tmp_path: Path) -> None:
        root = tmp_path / "proj"
        (root / "game").mkdir(parents=True)
        project = Project.create(root)
        with project.open_db() as db:
            db.save_unit(_make_unit("audit_u1", 1, "Source text"))
            results = [
                HallucinationScanResult(
                    unit_id="audit_u1",
                    has_hallucination=True,
                    category=TONE_REGISTER_VIOLATION,
                    confidence=0.88,
                    explanation="Quá suồng sã với bề trên",
                    suggested_correction="Dạ vâng bẩm ngài",
                )
            ]
            inserted = persist_hallucination_audit(db, results)
            assert inserted == 1

            logs = db.get_qa_audit_logs("audit_u1")
            assert len(logs) == 1
            assert logs[0]["stage"] == "llm_hallucination_guard"
            assert logs[0]["verdict"] == "FAIL"
            assert logs[0]["issue_type"] == TONE_REGISTER_VIOLATION
            assert logs[0]["confidence"] == 0.88


class TestPipelineHallucinationIntegration:
    """Verifies TranslationPipeline integration with HallucinationGuard and automated repair."""

    def test_pipeline_catches_hallucination_and_triggers_retry(self, tmp_path: Path) -> None:
        root = tmp_path / "pipe_test"
        (root / "game").mkdir(parents=True)
        project = Project.create(root)
        cfg = Config()

        u1 = _make_unit("u_hal_1", 1, "Do you have the key?", speaker="ren")
        u2 = _make_unit("u_hal_2", 2, "Yes, right here.", speaker="liana")

        with project.open_db() as db:
            db.save_unit(u1)
            db.save_unit(u2)

        # Mock LLM for translation: u1 translation introduces a hallucination
        mock_translation_client = MagicMock()
        mock_translation_client.complete_json.return_value = (
            {
                # Hallucinated 'cổng thiên đàng'
                "u_hal_1": "Bạn có chìa khóa mở cổng thiên đàng không?",
                "u_hal_2": "Vâng, ở ngay đây.",
            },
            60,
            40,
        )

        # Mock Guard detecting the hallucination on u_hal_1
        mock_guard = MagicMock()
        mock_guard.scan_batch.return_value = [
            HallucinationScanResult(
                unit_id="u_hal_1",
                has_hallucination=True,
                category=SEMANTIC_INVERSION,
                confidence=0.95,
                explanation="Bịa đặt chi tiết 'cổng thiên đàng' không có trong nguyên tác.",
                suggested_correction="Cậu có chìa khóa không?",
            ),
            HallucinationScanResult(unit_id="u_hal_2", has_hallucination=False),
        ]

        with project.open_db() as db:
            pipeline = TranslationPipeline(
                project,
                db,
                cfg,
                hallucination_guard=mock_guard,
            )
            pipeline.llm = mock_translation_client

            # Settle batch with retry ladder
            pipeline.run()
            # Guard was invoked
            assert mock_guard.scan_batch.called

            # Check that audit log was recorded for the batch units
            logs_u1 = db.get_qa_audit_logs("u_hal_1")
            assert len(logs_u1) >= 1
            assert logs_u1[0]["verdict"] == "FAIL"
            assert logs_u1[0]["issue_type"] == SEMANTIC_INVERSION
