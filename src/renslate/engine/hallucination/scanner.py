"""Continuous LLM Hallucination Guard performing batch scanning against context window."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Sequence
from typing import Any

from renslate.core.config import AppConfig, Config, LLMConfig
from renslate.core.models import Unit
from renslate.engine.hallucination.models import (
    HallucinationBatchReport,
    HallucinationScanResult,
)
from renslate.engine.hallucination.resilience import (
    fail_open_results,
)
from renslate.engine.hallucination.taxonomy import (
    ALL_CATEGORIES,
    normalize_category,
)
from renslate.engine.llm import LLMClient

_LOG = logging.getLogger("renslate.engine.hallucination.scanner")

__all__ = [
    "HALLUCINATION_GUARD_JSON_SCHEMA",
    "HALLUCINATION_GUARD_SYSTEM_PROMPT",
    "HallucinationGuard",
]

HALLUCINATION_GUARD_SYSTEM_PROMPT = (
    "Bạn là chuyên gia thẩm định chất lượng bản địa hóa visual novel "
    "(QA / Hallucination Auditor).\n"
    "Nhiệm vụ của bạn là so sánh bản dịch tiếng Việt với văn bản gốc "
    "và ngữ cảnh đối thoại để phát hiện:\n"
    "1. SPEAKER_MISATTRIBUTION (gán nhầm người nói, hiểu sai chủ thể phát ngôn).\n"
    "2. SEMANTIC_INVERSION (đảo ngược ý nghĩa, ảo giác bịa đặt chi tiết, bỏ sót ý quan trọng).\n"
    "3. PRONOUN_ERROR (sai lệch đại từ xưng hô, đảo ngôi xưng-gọi, nhầm lẫn anh-em/tôi-bạn).\n"
    "4. TONE_REGISTER_VIOLATION (lệch giọng điệu, phá vỡ phong cách nhân vật, kính ngữ sai lệch).\n"
    "5. TERMINOLOGY_DRIFT (dịch sai lệch thuật ngữ cố định trong bối cảnh).\n"
    "6. CONTEXT_LEAK (lộ nhãn ngữ cảnh kỹ thuật, tiền tố ID vào nội dung thoại).\n"
    "7. FOREIGN_ARTIFACT (chữ Hán/Kanji/tiếng Anh còn sót lại trong bản dịch).\n\n"
    "BẮT BUỘC trả về duy nhất một đối tượng JSON tuân thủ schema:\n"
    "{\n"
    '  "scanned_units": [\n'
    "    {\n"
    '      "uid": "d:123",\n'
    '      "verdict": "PASS" | "FAIL" | "WARNING",\n'
    '      "issue_type": "none" | "SPEAKER_MISATTRIBUTION" | "SEMANTIC_INVERSION" |\n'
    '                    "PRONOUN_ERROR" | "TONE_REGISTER_VIOLATION" |\n'
    '                    "TERMINOLOGY_DRIFT" | "CONTEXT_LEAK" | "FOREIGN_ARTIFACT",\n'
    '      "confidence": 0.95,\n'
    '      "explanation": "Giải thích ngắn gọn...",\n'
    '      "suggested_correction": "Bản dịch sửa đổi gợi ý..."\n'
    "    }\n"
    "  ]\n"
    "}"
)

HALLUCINATION_GUARD_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "scanned_units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["PASS", "FAIL", "WARNING"]},
                    "issue_type": {
                        "type": "string",
                        "enum": [
                            "none",
                            "SPEAKER_MISATTRIBUTION",
                            "SEMANTIC_INVERSION",
                            "PRONOUN_ERROR",
                            "TONE_REGISTER_VIOLATION",
                            "TERMINOLOGY_DRIFT",
                            "CONTEXT_LEAK",
                            "FOREIGN_ARTIFACT",
                            "factual_hallucination",
                            "critical_omission",
                            "pronoun_inversion",
                            "register_breakdown",
                            "emotional_distortion",
                            "continuity_breakdown",
                            "leaked_artifacts",
                        ],
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "explanation": {"type": "string"},
                    "suggested_correction": {"type": "string"},
                },
                "required": ["uid", "verdict", "issue_type", "confidence", "explanation"],
                "additionalProperties": True,
            },
        }
    },
    "required": ["scanned_units"],
    "additionalProperties": True,
}


class HallucinationGuard:
    """Continuous LLM batch hallucination scanning stage."""

    VALID_CATEGORIES = ALL_CATEGORIES

    def __init__(
        self,
        config: AppConfig | LLMConfig | Config | None = None,
        llm_client: Any | None = None,
    ) -> None:
        if isinstance(config, Config):
            self.llm_config = config.llm
        elif isinstance(config, LLMConfig):
            self.llm_config = config
        else:
            self.llm_config = LLMConfig(
                base_url=AppConfig.llm_api_base_url,
                model=AppConfig.llm_model_name,
            )

        if llm_client is not None:
            self._llm = llm_client
        else:
            self._llm = LLMClient(self.llm_config)

    def build_guard_prompt(self, units: Sequence[Unit], context_window: str = "") -> str:
        """Format the comparison user prompt containing context window and translation pairs."""
        sections: list[str] = []
        if context_window.strip():
            sections.append(f"NGỮ CẢNH HỘI THOẠI & BỐI CẢNH:\n{context_window.strip()}")

        lines: list[str] = ["CÁC DÒNG THOẠI CẦN THẨM ĐỊNH CHẤT LƯỢNG (SO SÁNH GỐC VÀ DỊCH):"]
        for u in units:
            spk = f"[{u.speaker}] " if u.speaker else ""
            tgt = u.target or ""
            lines.append(f"- ID: {u.uid}\n  Người nói: {spk}\n  Gốc: {u.source}\n  Dịch: {tgt}")

        sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def parse_guard_response(
        self,
        raw_response: str | dict[str, Any],
        units: Sequence[Unit] | None = None,
    ) -> list[HallucinationScanResult]:
        """Validate and parse LLM Hallucination Guard JSON output with fail-open resilience."""
        data: Any
        if isinstance(raw_response, dict):
            data = raw_response
        elif isinstance(raw_response, str):
            try:
                data = json.loads(raw_response)
            except Exception as err:
                _LOG.warning("Failed to parse guard response JSON: %s", err)
                if units:
                    return fail_open_results(
                        units, f"JSON parse error: {err}", "SCAN_SKIPPED_MALFORMED"
                    )
                return []
        else:
            if units:
                return fail_open_results(
                    units, "Unsupported response type", "SCAN_SKIPPED_MALFORMED"
                )
            return []

        if not isinstance(data, dict):
            _LOG.warning("Guard response root is not a JSON object: %s", type(data).__name__)
            if units:
                return fail_open_results(
                    units, "Response root is not a JSON object", "SCAN_SKIPPED_MALFORMED"
                )
            return []

        units_raw = data.get("scanned_units")
        if not isinstance(units_raw, list):
            if units:
                return fail_open_results(
                    units, "Missing scanned_units array", "SCAN_SKIPPED_MALFORMED"
                )
            return []

        results: list[HallucinationScanResult] = []
        parsed_uids: set[str] = set()
        valid_uids = {u.uid for u in units} if units else None

        for item in units_raw:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("uid") or item.get("unit_id") or "")
            if valid_uids is not None and uid not in valid_uids:
                _LOG.warning("HallucinationGuard output contains fabricated UID %r; ignoring", uid)
                continue
            parsed_uids.add(uid)

            verdict = str(item.get("verdict", "PASS")).upper()
            raw_issue = str(item.get("issue_type") or item.get("category") or "none")
            canonical_category = normalize_category(raw_issue) if raw_issue != "none" else ""

            has_hal = (
                verdict in ("FAIL", "WARNING")
                and raw_issue.lower() not in ("none", "", "pass")
            )
            raw_conf = item.get("confidence")
            try:
                conf = float(raw_conf) if raw_conf is not None else 1.0
                if math.isnan(conf) or math.isinf(conf):
                    conf = (
                        1.0
                        if (has_hal or item.get("has_hallucination") or item.get("category"))
                        else 0.0
                    )
            except (ValueError, TypeError):
                conf = 1.0
            expl = str(item.get("explanation", ""))
            corr = str(item.get("suggested_correction") or item.get("suggested_fix") or "")
            severity = str(item.get("severity") or ("error" if verdict == "FAIL" else "warning"))

            results.append(
                HallucinationScanResult(
                    unit_id=uid,
                    has_hallucination=has_hal,
                    category=canonical_category,
                    confidence=conf,
                    explanation=expl,
                    suggested_correction=corr,
                    severity=severity,
                )
            )

        # Ensure all expected units have a result entry
        if units:
            for u in units:
                if u.uid not in parsed_uids:
                    results.append(
                        HallucinationScanResult(
                            unit_id=u.uid,
                            has_hallucination=False,
                            category="",
                            confidence=0.0,
                            explanation="Unit omitted in LLM guard scan; passed by default",
                            suggested_correction="",
                            severity="none",
                        )
                    )

        return results

    def scan_batch(
        self,
        units: Sequence[Unit],
        context_window: str = "",
    ) -> list[HallucinationScanResult]:
        """Perform batch scanning of translation units against context window."""
        if not units:
            return []

        user_prompt = self.build_guard_prompt(units, context_window)

        try:
            resp_dict, _, _ = self._llm.complete_json(
                system_prompt=HALLUCINATION_GUARD_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_schema=HALLUCINATION_GUARD_JSON_SCHEMA,
                temperature=0.1,
            )
            return self.parse_guard_response(resp_dict, units=units)
        except Exception as err:
            _LOG.warning("Hallucination scan exception encountered (%s); failing open", err)
            is_malformed = (
                isinstance(err, (json.JSONDecodeError, ValueError))
                or "json" in str(err).lower()
                or "decode" in str(err).lower()
            )
            cat = "SCAN_SKIPPED_MALFORMED" if is_malformed else "SCAN_SKIPPED_TIMEOUT"
            return fail_open_results(units, str(err), category=cat)

    def scan_batch_report(
        self,
        batch_id: str,
        units: Sequence[Unit],
        context_window: str = "",
    ) -> HallucinationBatchReport:
        """Scan a batch and return a comprehensive audit report."""
        t0 = time.perf_counter()
        results = self.scan_batch(units, context_window)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        has_hal = any(r.has_hallucination for r in results)
        flagged = sum(1 for r in results if r.has_hallucination)

        return HallucinationBatchReport(
            batch_id=batch_id,
            results=tuple(results),
            has_hallucinations=has_hal,
            scanned_count=len(units),
            flagged_count=flagged,
            scan_duration_ms=duration_ms,
            error=None,
        )
