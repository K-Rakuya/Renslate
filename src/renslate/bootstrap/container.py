"""Composition root: wires adapters and use cases into an executable translation container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from renslate.adapters.outbound.db.sqlite_repo import SQLiteUnitRepositoryAdapter
from renslate.adapters.outbound.llm.openai_client import OpenAILocalClientAdapter
from renslate.adapters.outbound.telemetry.ansi_telemetry import RichAnsiTelemetryAdapter
from renslate.application.use_cases.quality_audit import QualityAuditUseCase
from renslate.application.use_cases.retry_ladder import RetryLadderUseCase
from renslate.application.use_cases.translate_batch import TranslateBatchUseCase
from renslate.core.config import Config
from renslate.core.db import Database
from renslate.core.project import Project
from renslate.engine.llm import LLMClient

__all__ = ["Container", "create_pipeline_container"]


@dataclass(slots=True)
class Container:
    """Dependency injection container managing application lifecycles."""

    unit_repo: SQLiteUnitRepositoryAdapter
    llm_gateway: OpenAILocalClientAdapter
    telemetry: RichAnsiTelemetryAdapter
    quality_audit_use_case: QualityAuditUseCase
    retry_ladder_use_case: RetryLadderUseCase
    translate_batch_use_case: TranslateBatchUseCase


def create_pipeline_container(
    project: Project,
    db: Database,
    config: Config,
    guard: Any | None = None,
    client: LLMClient | None = None,
) -> Container:
    """Build and wire the Hexagonal application container for a project."""
    llm_instance = client if client is not None else LLMClient(config.llm)

    unit_repo = SQLiteUnitRepositoryAdapter(db)
    llm_gateway = OpenAILocalClientAdapter(llm_instance)
    telemetry = RichAnsiTelemetryAdapter()

    quality_audit_use_case = QualityAuditUseCase(guard=guard)
    retry_ladder_use_case = RetryLadderUseCase(llm_client=llm_gateway, telemetry=telemetry)
    translate_batch_use_case = TranslateBatchUseCase(
        llm_client=llm_gateway,
        audit_use_case=quality_audit_use_case,
        retry_ladder_use_case=retry_ladder_use_case,
    )

    return Container(
        unit_repo=unit_repo,
        llm_gateway=llm_gateway,
        telemetry=telemetry,
        quality_audit_use_case=quality_audit_use_case,
        retry_ladder_use_case=retry_ladder_use_case,
        translate_batch_use_case=translate_batch_use_case,
    )
