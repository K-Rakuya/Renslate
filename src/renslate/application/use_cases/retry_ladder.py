"""Use case orchestrating the fallback retry ladder (cluster -> single)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from renslate.application.ports.outbound import LLMClientPort, TelemetryPort
from renslate.core.models import Unit
from renslate.domain.models import UnitFailure

_LOG = logging.getLogger("renslate.pipeline.retry")

__all__ = ["RetryLadderUseCase"]


@dataclass(slots=True)
class RetryLadderUseCase:
    """Manages progressive fallback retries for units failing unmask or QA validation."""

    llm_client: LLMClientPort
    telemetry: TelemetryPort | None = None

    def execute_single_retry(
        self,
        failing_unit: Unit,
        prompt_builder_fn: Callable[[Sequence[Unit]], str],
        unmask_fn: Callable[[Unit, str], tuple[bool, str, str | None]],
        system_prompt: str,
    ) -> tuple[bool, str, UnitFailure | None]:
        """Attempt single-line fallback translation for an isolated failing unit.

        Returns (is_success, translated_text, failure_if_unsuccessful).
        """
        user_prompt = prompt_builder_fn([failing_unit])
        schema = {
            "type": "object",
            "properties": {failing_unit.uid: {"type": "string"}},
            "required": [failing_unit.uid],
            "additionalProperties": False,
        }

        try:
            results, _, _ = self.llm_client.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
            )
            raw_text = results.get(failing_unit.uid, "")
            if not raw_text:
                return (
                    False,
                    "",
                    UnitFailure(failing_unit, "missing_key", "Phản hồi LLM thiếu key của dòng này", ""),
                )

            success, cleaned, err = unmask_fn(failing_unit, raw_text)
            if success:
                return True, cleaned, None
            else:
                return (
                    False,
                    cleaned,
                    UnitFailure(failing_unit, "unmask_error", err or "Lỗi unmask", raw_text),
                )
        except Exception as exc:
            _LOG.warning("Lỗi khi thử lại đơn lẻ dòng %s: %s", failing_unit.uid, exc)
            return (
                False,
                "",
                UnitFailure(failing_unit, "retry_exception", str(exc), ""),
            )
