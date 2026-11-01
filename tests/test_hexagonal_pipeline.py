"""Tests for the Hexagonal Architecture domain policies, use cases, and composition container."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from renslate.application.ports.outbound import (
    LLMClientPort,
)
from renslate.application.use_cases.quality_audit import QualityAuditUseCase
from renslate.application.use_cases.retry_ladder import RetryLadderUseCase
from renslate.bootstrap.container import Container, create_pipeline_container
from renslate.config import Config
from renslate.core.models import Status, Unit, UnitKind
from renslate.domain.models import (
    SpriteState,
    StageSnapshot,
    TranslationResult,
)
from renslate.domain.policies.anti_inversion import AntiInversionPolicy
from renslate.domain.policies.outer_tags import OuterTagPolicy
from renslate.domain.policies.token_bijection import TokenBijectionPolicy
from renslate.project import Project


def test_anti_inversion_policy_detects_question_inversion() -> None:
    # Speaker is 'em', listener is 'anh'. Translating 'Do you want...' as 'Em muốn...' is an inversion.
    is_valid, reason = AntiInversionPolicy.inspect_pronoun_pair(
        source_text="Do you want to die?",
        target_text="Em muốn chết sao?",
        self_pronoun="em",
        listener_pronoun="anh",
    )
    assert not is_valid
    assert reason is not None
    assert "SEMANTIC_INVERSION" in reason


def test_anti_inversion_policy_passes_valid_alignment() -> None:
    is_valid, reason = AntiInversionPolicy.inspect_pronoun_pair(
        source_text="Do you want to die?",
        target_text="Anh muốn chết sao?",
        self_pronoun="em",
        listener_pronoun="anh",
    )
    assert is_valid
    assert reason is None


def test_token_bijection_policy_checks_duplicates_and_missing() -> None:
    expected = {"T1", "/T1", "V1"}
    text_with_dup = "⟦T1⟧Hello⟦T1⟧ ⟦V1⟧"
    is_valid, issues = TokenBijectionPolicy.inspect_placeholders(expected, text_with_dup)
    assert not is_valid
    assert any("duplicate_placeholder" in issue for issue in issues)
    assert any("missing_placeholder" in issue for issue in issues)


def test_token_bijection_policy_valid_exact_match() -> None:
    expected = {"T1", "/T1"}
    text = "⟦T1⟧Xin chào⟦/T1⟧"
    is_valid, issues = TokenBijectionPolicy.inspect_placeholders(expected, text)
    assert is_valid
    assert len(issues) == 0


def test_outer_tag_policy_realigns_formatting() -> None:
    src = "{b}{i}Hello world{/i}{/b}"
    tgt = "Xin chào thế giới"
    realigned = OuterTagPolicy.realign(src, tgt)
    assert realigned == "{b}{i}Xin chào thế giới{/i}{/b}"


def test_stage_snapshot_active_tags() -> None:
    snapshot = StageSnapshot(
        scene_bg="bg classroom",
        active_sprites=(
            SpriteState(tag="Eileen", expression="happy", position="right"),
            SpriteState(tag="Player", expression="neutral", position="left"),
        ),
    )
    assert snapshot.active_tags == {"eileen", "player"}


def test_retry_ladder_use_case_success() -> None:
    u = Unit("u1", UnitKind.DIALOGUE, 1, "Hello", Status.PENDING)
    mock_llm = MagicMock(spec=LLMClientPort)
    mock_llm.complete_json.return_value = ({"u1": "Xin chào"}, 10, 5)

    ladder = RetryLadderUseCase(llm_client=mock_llm)
    ok, text, fail = ladder.execute_single_retry(
        failing_unit=u,
        prompt_builder_fn=lambda units: "prompt",
        unmask_fn=lambda unit, raw: (True, raw, None),
        system_prompt="sys",
    )
    assert ok
    assert text == "Xin chào"
    assert fail is None


def test_quality_audit_use_case_filters_high_confidence_hallucination() -> None:
    u = Unit("u1", UnitKind.DIALOGUE, 1, "Hello friend", Status.PENDING)
    candidate = TranslationResult(unit=u, target_text="Hello enemy")

    mock_guard = MagicMock()
    mock_result = MagicMock()
    mock_result.unit_id = "u1"
    mock_result.has_hallucination = True
    mock_result.confidence = 0.95
    mock_result.category = "SEMANTIC_INVERSION"
    mock_result.explanation = "Reversed meaning"
    mock_result.suggested_correction = None
    mock_guard.scan_batch.return_value = [mock_result]

    use_case = QualityAuditUseCase(guard=mock_guard)
    accepted, failures = use_case.audit_batch_results([candidate], context_prompt="context")

    assert len(accepted) == 0
    assert len(failures) == 1
    assert failures[0].unit.uid == "u1"
    assert failures[0].failure_type == "SEMANTIC_INVERSION"


def test_composition_container_wiring(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    cfg = Config()

    with project.open_db() as db:
        container = create_pipeline_container(project, db, cfg)
        assert isinstance(container, Container)
        assert container.unit_repo is not None
        assert container.llm_gateway is not None
        assert container.translate_batch_use_case is not None
