"""Outbound adapter connecting the application ports to LLM client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from renslate.engine.llm import LLMClient

__all__ = ["OpenAILocalClientAdapter"]


@dataclass(slots=True)
class OpenAILocalClientAdapter:
    """Implements LLMClientPort using LLMClient (OpenAI-compatible HTTP endpoint)."""

    client: LLMClient

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
        retries: int | None = None,
    ) -> tuple[dict[str, str], int, int]:
        return self.client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            retries=retries,
        )
