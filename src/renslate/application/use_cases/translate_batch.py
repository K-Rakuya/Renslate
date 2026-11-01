"""Use case orchestrating atomic execution of a single translation batch."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from renslate.application.ports.outbound import LLMClientPort
from renslate.application.use_cases.quality_audit import QualityAuditUseCase
from renslate.application.use_cases.retry_ladder import RetryLadderUseCase
from renslate.core.models import Status, Unit
from renslate.domain.models import BatchExecutionResult, TranslationResult, UnitFailure

_LOG = logging.getLogger("renslate.pipeline.batch")

__all__ = ["TranslateBatchUseCase"]


@dataclass(slots=True)
class TranslateBatchUseCase:
    """Orchestrates translating an atomic batch through LLM, unmasking, and verification."""

    llm_client: LLMClientPort
    audit_use_case: QualityAuditUseCase
    retry_ladder_use_case: RetryLadderUseCase

    def execute_batch(
        self,
        batch_units: Sequence[Unit],
        prompt_builder_fn: Callable[[Sequence[Unit]], str],
        unmask_fn: Callable[[Unit, str], tuple[bool, str, str | None]],
        system_prompt: str,
        qa_check_fn: Callable[[str, str], tuple[str, ...]] | None = None,
    ) -> BatchExecutionResult:
        """Process a single batch and return accepted results and final failures."""
        if not batch_units:
            return BatchExecutionResult()

        user_prompt = prompt_builder_fn(batch_units)
        schema = {
            "type": "object",
            "properties": {u.uid: {"type": "string"} for u in batch_units},
            "required": [u.uid for u in batch_units],
            "additionalProperties": False,
        }

        try:
            raw_translations, in_tokens, out_tokens = self.llm_client.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
            )
        except Exception as exc:
            _LOG.error("Lỗi gọi LLM cho batch: %s", exc)
            failures = tuple(
                UnitFailure(u, "llm_batch_error", str(exc), "") for u in batch_units
            )
            return BatchExecutionResult(failures=failures)

        candidate_results: list[TranslationResult] = []
        pending_retry: list[Unit] = []

        for u in batch_units:
            raw_text = raw_translations.get(u.uid)
            if not raw_text:
                pending_retry.append(u)
                continue

            success, cleaned, _err = unmask_fn(u, raw_text)
            if success:
                flags: tuple[str, ...] = ()
                if qa_check_fn is not None:
                    flags = qa_check_fn(u.source, cleaned)
                status = Status.QA_FLAGGED if flags else Status.TRANSLATED
                candidate_results.append(
                    TranslationResult(
                        unit=u,
                        target_text=cleaned,
                        qa_flags=flags,
                        status=status,
                    )
                )
            else:
                pending_retry.append(u)

        # Audit with HallucinationGuard
        accepted_results, audit_failures = self.audit_use_case.audit_batch_results(
            candidate_results, context_prompt=user_prompt
        )

        final_failures: list[UnitFailure] = list(audit_failures)

        # Retry ladder for failed units
        for fail_u in pending_retry:
            retry_ok, retry_text, err_obj = self.retry_ladder_use_case.execute_single_retry(
                failing_unit=fail_u,
                prompt_builder_fn=prompt_builder_fn,
                unmask_fn=unmask_fn,
                system_prompt=system_prompt,
            )
            if retry_ok:
                flags = ()
                if qa_check_fn is not None:
                    flags = qa_check_fn(fail_u.source, retry_text)
                status = Status.QA_FLAGGED if flags else Status.TRANSLATED
                accepted_results.append(
                    TranslationResult(
                        unit=fail_u,
                        target_text=retry_text,
                        qa_flags=flags,
                        status=status,
                    )
                )
            else:
                if err_obj is not None:
                    final_failures.append(err_obj)
                else:
                    final_failures.append(
                        UnitFailure(fail_u, "single_retry_failed", "Thử lại đơn lẻ thất bại", "")
                    )

        return BatchExecutionResult(
            success_results=tuple(accepted_results),
            failures=tuple(final_failures),
            tokens_in=in_tokens,
            tokens_out=out_tokens,
        )
