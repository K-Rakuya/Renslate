"""Client for local LLMs via llama.cpp server (OpenAI-compatible HTTP API)."""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any

from vitrans.config import LLMConfig
from vitrans.errors import LLMError

_LOG = logging.getLogger("vitrans.llm")

# Pattern to isolate a JSON object within response text even if wrapped in markdown
_JSON_OBJECT = re.compile(r"\{[\s\S]*\}")
_CODE_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>")


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract a dictionary of {key: value} from model response text.

    Handles clean JSON, reasoning tags (<think>), markdown code fences,
    trailing commas, and partial/truncated JSON streams. Preserves nested
    objects and lists where present.
    """
    text = raw_text.strip()

    # 1. Strip reasoning thoughts if model emits <think> ... </think>
    text = _THINK_BLOCK.sub("", text).strip()

    # 2. Strip markdown fences if present
    code_match = _CODE_BLOCK.search(text)
    if code_match:
        text = code_match.group(1).strip()

    # 3. Direct parse attempt
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    # 4. Find outer-most curly braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return {str(k): v for k, v in parsed.items()}
        except json.JSONDecodeError:
            # Try removing trailing commas
            sanitized = re.sub(r",\s*([\}\]])", r"\1", candidate)
            try:
                parsed = json.loads(sanitized)
                if isinstance(parsed, dict):
                    return {str(k): v for k, v in parsed.items()}
            except json.JSONDecodeError:
                pass

    # 5. Partial/Truncated JSON repair (if generation cut off before closing brace)
    if first_brace != -1:
        last_quote = text.rfind('"')
        if last_quote > first_brace:
            candidate_repair = text[first_brace : last_quote + 1] + "\n}"
            candidate_repair = re.sub(r",\s*\}$", "}", candidate_repair)
            try:
                parsed = json.loads(candidate_repair)
                if isinstance(parsed, dict) and parsed:
                    return {str(k): v for k, v in parsed.items()}
            except json.JSONDecodeError:
                pass

    # 6. Regex key-value pair extraction fallback (supports #, :, multiline values)
    pairs = re.findall(r'"([a-zA-Z0-9_\-#:]+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
    if pairs:
        result: dict[str, str] = {}
        for k, v in pairs:
            try:
                result[k] = json.loads(f'"{v}"')
            except json.JSONDecodeError:
                result[k] = v.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
        if result:
            return result

    raise LLMError(f"Không thể trích xuất JSON hợp lệ từ phản hồi của model:\n{raw_text[:300]}...")


class LLMClient:
    """HTTP client communicating with local llama.cpp llama-server."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.endpoint = f"{config.base_url.rstrip('/')}/chat/completions"

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Send chat completion request expecting a JSON object output.

        Supports structured JSON schema enforcement where available.
        Returns (parsed_dict, prompt_tokens, completion_tokens).
        """
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.15,
            "stop": ["\n\n//", "\n//", "<|im_end|>", "<|endoftext|>"],
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation_response",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        retries = self.config.max_retries
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            if attempt > 0:
                # Exponential backoff with jitter: (2^(attempt-1)) + [0, 1] seconds
                delay = (2 ** (attempt - 1)) + random.uniform(0.1, 1.0)
                _LOG.warning(
                    "Thử lại lần %d/%d sau %.1f giây do lỗi trước đó: %s",
                    attempt,
                    retries,
                    delay,
                    last_error,
                )
                time.sleep(delay)

            req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")

            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as response:
                    body = response.read().decode("utf-8")
                    resp_json = json.loads(body)

                choices = resp_json.get("choices")
                if not choices or not isinstance(choices, list):
                    raise LLMError(f"Phản hồi từ server thiếu trường 'choices': {body[:200]}")

                message = choices[0].get("message", {})
                content = message.get("content", "")

                usage = resp_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                result_dict = extract_json_object(content)
                return result_dict, prompt_tokens, completion_tokens

            except urllib.error.HTTPError as error:
                last_error = error
                # If server returned 400 or 500 with json_schema, fallback to json_object
                if error.code in (400, 500) and schema is not None:
                    _LOG.warning(
                        "Server lỗi schema (HTTP %d). Hạ cấp sang json_object...",
                        error.code,
                    )
                    payload["response_format"] = {"type": "json_object"}
                    data = json.dumps(payload).encode("utf-8")
                    schema = None
                    continue

                # Fatal HTTP errors that should not retry: 400 Bad Request, 401 Unauthorized
                if error.code in (400, 401, 403, 404):
                    err_body = error.read().decode("utf-8", errors="replace")
                    raise LLMError(
                        f"Lỗi HTTP {error.code} từ LLM server tại {self.endpoint}: {err_body}"
                    ) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
            except LLMError as error:
                last_error = error

        raise LLMError(f"Thất bại khi gọi LLM server sau {retries + 1} lần thử: {last_error}")
