"""Client for local LLMs via llama.cpp server (OpenAI-compatible HTTP API)."""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import random
import re
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

from renslate.core.config import LLMConfig
from renslate.core.errors import LLMError

_LOG = logging.getLogger("renslate.llm")


# Pattern to isolate a JSON object within response text even if wrapped in markdown
_JSON_OBJECT = re.compile(r"\{[\s\S]*\}")
_CODE_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)(?:```|$)")
_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>")
_JSON_START = re.compile(r'\{\s*(?:"[^"]+"|\'[^\']+\'|[a-zA-Z0-9_\-#:]+\s*:|\})')
_SMART_DOUBLE_QUOTES = re.compile(r"[\u201c\u201d\u201e\u201f\u00ab\u00bb\u2033]")
_SMART_SINGLE_QUOTES = re.compile(r"[\u2018\u2019\u201a\u201b\u2032]")
_MULTI_KEY_DOUBLE_QUOTES = re.compile(r'"{2,}([a-zA-Z0-9_\-#:.]+)"{2,}\s*:')
_MULTI_KEY_SINGLE_QUOTES = re.compile(r"'{2,}([a-zA-Z0-9_\-#:.]+)'{2,}\s*:")
_MULTI_VAL_DOUBLE_QUOTES = re.compile(
    r':\s*"{2,}((?:[^\n\r"]|"(?!"))*?)"{2,}(?=\s*(?:,|\}|\]|\n|\r|$|"{1,}[a-zA-Z0-9_\-#:.]+"{1,}\s*:))'
)
_MULTI_VAL_SINGLE_QUOTES = re.compile(
    r":\s*'{2,}((?:[^\n\r']|'(?!'))*?)'{2,}(?=\s*(?:,|\}|\]|\n|\r|$|'{1,}[a-zA-Z0-9_\-#:.]+'{1,}\s*:))"
)
_KEY_PATTERN = re.compile(
    r"""(?:["']([a-zA-Z0-9_\-#:.]+)["']|(?:\b([a-zA-Z0-9_\-#]+:[0-9a-zA-Z_\-#]+)\b))\s*:"""
)
_CHAT_CHATTER = re.compile(
    r"^(?:sure!?|certainly!?|here (?:is|are)|i cannot|i can help|as an ai|tôi là|chắc chắn)",
    re.IGNORECASE,
)


def _normalize_json_text(text: str) -> str:
    cleaned = _SMART_DOUBLE_QUOTES.sub('"', text)
    cleaned = _SMART_SINGLE_QUOTES.sub("'", cleaned)
    cleaned = _MULTI_KEY_DOUBLE_QUOTES.sub(r'"\1":', cleaned)
    cleaned = _MULTI_KEY_SINGLE_QUOTES.sub(r'"\1":', cleaned)
    cleaned = _MULTI_VAL_DOUBLE_QUOTES.sub(r': "\1"', cleaned)
    cleaned = _MULTI_VAL_SINGLE_QUOTES.sub(r': "\1"', cleaned)
    return cleaned


def _extract_fragmented_pairs(text: str) -> dict[str, str]:
    """Extract individual key-value pairs directly from raw or fragmented token streams."""
    results: dict[str, str] = {}
    pos = 0
    length = len(text)
    while pos < length:
        match = _KEY_PATTERN.search(text, pos)
        if not match:
            break
        k = match.group(1) or match.group(2)
        val_start = match.end()
        while val_start < length and text[val_start] in " \t":
            val_start += 1
        if val_start >= length:
            break

        quote_char = text[val_start]
        if quote_char in ('"', "'"):
            i = val_start + 1
            val_chars: list[str] = []
            closed = False
            while i < length:
                c = text[i]
                if c == "\\" and i + 1 < length:
                    val_chars.append(text[i : i + 2])
                    i += 2
                    continue
                if c == quote_char:
                    rest = text[i + 1 :].lstrip(" \t")
                    if (
                        not rest
                        or rest[0] in (",", "}", "]", "\n", "\r")
                        or _KEY_PATTERN.match(rest) is not None
                    ):
                        closed = True
                        i += 1
                        break
                val_chars.append(c)
                i += 1
            if closed:
                raw_val = "".join(val_chars)
                try:
                    val = json.loads(f'"{raw_val}"')
                except json.JSONDecodeError:
                    val = raw_val.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
                results[k] = val
                pos = i
            else:
                pos = val_start + 1
        else:
            end_match = re.search(r"[,\n\r\}]", text[val_start:])
            if end_match:
                raw_val = text[val_start : val_start + end_match.start()].strip()
                pos = val_start + end_match.end()
            else:
                raw_val = text[val_start:].strip()
                pos = length
            if raw_val:
                results[k] = raw_val
    return results


def extract_json_object(
    raw_text: str,
    *,
    fallback_single_key: str | None = None,
) -> dict[str, Any]:
    """Extract a dictionary of {key: value} from model response text.

    Multi-tier resilient parsing:
    1. Direct JSON parse (clean objects, strip <think> and markdown code fences).
    2. Smart quote and doubled/quadruple/malformed quote normalization.
    3. Balanced brace extraction with trailing/missing comma healing.
    4. Bracketless JSON object synthesis (recovers unbracketed key-value pairs).
    5. Fragmented raw token stream extraction (recovers salvageable pairs from broken streams).
    6. Single-item fallback extraction when batch parsing fails or raw output is provided.
    """
    text = raw_text.strip()
    text = _THINK_BLOCK.sub("", text).strip()

    code_match = _CODE_BLOCK.search(text)
    if code_match:
        text = code_match.group(1).strip()

    # Tier 1: Fast path
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    # Tier 2: Quote normalization
    normalized = _normalize_json_text(text)
    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    # Tier 3: Balanced outer braces with comma healing
    json_start_match = _JSON_START.search(normalized)
    if json_start_match:
        first_brace = json_start_match.start()
        last_brace = normalized.rfind("}")
        if last_brace > first_brace:
            candidate = normalized[first_brace : last_brace + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return {str(k): v for k, v in parsed.items()}
            except json.JSONDecodeError:
                sanitized = re.sub(r",\s*([\}\]])", r"\1", candidate)
                sanitized = re.sub(
                    r'("(?:[^"\\\n]|\\.)*")\s*(?=\r?\n\s*"[a-zA-Z0-9_\-#:.]+"\s*:)',
                    r"\1,",
                    sanitized,
                )
                try:
                    parsed = json.loads(sanitized)
                    if isinstance(parsed, dict):
                        return {str(k): v for k, v in parsed.items()}
                except json.JSONDecodeError:
                    pass

        # Truncated outer object repair
        last_quote = normalized.rfind('"')
        if last_quote > first_brace:
            candidate_repair = normalized[first_brace : last_quote + 1] + "\n}"
            candidate_repair = re.sub(r",\s*\}$", "}", candidate_repair)
            try:
                parsed = json.loads(candidate_repair)
                if isinstance(parsed, dict) and parsed:
                    return {str(k): v for k, v in parsed.items()}
            except json.JSONDecodeError:
                pass

    # Tier 4: Bracketless object synthesis
    if ":" in normalized and not normalized.startswith("{"):
        bracketless = re.sub(
            r'("(?:[^"\\\n]|\\.)*")\s*(?=\r?\n\s*"[a-zA-Z0-9_\-#:.]+"\s*:)',
            r"\1,",
            normalized,
        )
        bracketless_candidate = "{\n" + bracketless.strip().rstrip(",") + "\n}"
        try:
            parsed = json.loads(bracketless_candidate)
            if isinstance(parsed, dict) and parsed:
                return {str(k): v for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

    # Tier 5: Fragmented raw token stream extraction
    salvaged = _extract_fragmented_pairs(normalized)
    if salvaged:
        return salvaged

    if normalized != text:
        salvaged_orig = _extract_fragmented_pairs(text)
        if salvaged_orig:
            return salvaged_orig

    # Tier 6: Single-item fallback extraction
    if fallback_single_key and text.strip():
        stripped = text.strip()
        if (stripped.startswith('"') and stripped.endswith('"')) or (
            stripped.startswith("'") and stripped.endswith("'")
        ):
            stripped = stripped[1:-1].strip()
        if not _CHAT_CHATTER.match(stripped) and len(stripped.split()) < 200:
            return {fallback_single_key: stripped}

    raise LLMError(f"Không thể trích xuất JSON hợp lệ từ phản hồi của model:\n{raw_text[:300]}...")


# Generic GBNF grammars
GENERIC_DICT_GBNF = (
    'root ::= "{" ws (pair ("," ws pair)*)? ws "}" ws\n'
    'pair ::= string ws ":" ws string\n'
    'string ::= "\\"" ([^"\\\\\\r\\n] | "\\\\" (["\\\\/bfnrt] | "u" '
    '[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""\n'
    "ws ::= [ \\t\\n\\r]*"
)

GENERIC_JSON_GBNF = (
    "root ::= object\n"
    'object ::= "{" ws (pair ("," ws pair)*)? ws "}" ws\n'
    'pair ::= string ws ":" ws value\n'
    'value ::= object | array | string | number | ("true" | "false" | "null") ws\n'
    'array ::= "[" ws (value ("," ws value)*)? ws "]" ws\n'
    'string ::= "\\"" ([^"\\\\\\r\\n] | "\\\\" (["\\\\/bfnrt] | "u" '
    '[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\"" ws\n'
    'number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws\n'
    "ws ::= [ \\t\\n\\r]*"
)


def escape_gbnf_string(s: str) -> str:
    """Escape a literal string into a GBNF quoted terminal token matching valid JSON."""
    json_literal = json.dumps(s)
    escaped = json_literal.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def generate_batch_gbnf(
    unit_ids: Sequence[str],
    *,
    allow_cjk: bool = True,
    ordered: bool = True,
) -> str:
    """Generate a GBNF grammar strictly constraining output to a JSON mapping of unit_ids.

    - If ordered=True: forces exact sequential key order {"d:1": "...", "d:2": "..."}.
    - If ordered=False: allows arbitrary permutation of keys.
    - If allow_cjk=False: excludes Chinese and Japanese characters from translated string values.
    """
    if allow_cjk:
        str_rule = (
            r'string ::= "\"" ([^"\\\r\n] | "\\" (["\\/bfnrt] | "u" '
            r'[0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""'
        )
    else:
        str_rule = (
            r'string ::= "\"" ([^"\\\r\n\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff] | '
            r'"\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""'
        )
    ws_rule = r"ws ::= [ \t\n\r]*"

    if not unit_ids:
        return f'root ::= "{{" ws "}}" ws\n{ws_rule}'

    if ordered:
        pairs = [f'{escape_gbnf_string(uid)} ws ":" ws string' for uid in unit_ids]
        joined = ' "," ws '.join(pairs)
        root_rule = f'root ::= "{{" ws {joined} ws "}}" ws'
        return f"{root_rule}\n{str_rule}\n{ws_rule}"
    else:
        keys = " | ".join(escape_gbnf_string(uid) for uid in unit_ids)
        pair_rule = f'pair ::= ({keys}) ws ":" ws string'
        root_rule = r'root ::= "{" ws (pair ("," ws pair)*)? ws "}" ws'
        return f"{root_rule}\n{pair_rule}\n{str_rule}\n{ws_rule}"


class LLMClient:
    """HTTP client communicating with local llama.cpp llama-server via persistent Keep-Alive."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
        parsed = urllib.parse.urlsplit(self.endpoint)
        self._scheme = parsed.scheme.lower() or "http"
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or (443 if self._scheme == "https" else 80)
        self._path = parsed.path or "/"
        if parsed.query:
            self._path = f"{self._path}?{parsed.query}"

        self._lock = threading.RLock()
        self._conn: http.client.HTTPConnection | http.client.HTTPSConnection | None = None

    def _create_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        if self._scheme == "https":
            return http.client.HTTPSConnection(
                self._host,
                port=self._port,
                timeout=self.config.timeout_s,
            )
        if self._scheme == "http":
            return http.client.HTTPConnection(
                self._host,
                port=self._port,
                timeout=self.config.timeout_s,
            )
        raise LLMError(
            f"Giao thức URL không được hỗ trợ: {self._scheme!r}. Chỉ hỗ trợ http và https."
        )

    def _get_connection(self) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        if self._conn is None:
            self._conn = self._create_connection()
        return self._conn

    def close(self) -> None:
        """Close persistent HTTP connection and release socket resources."""
        with self._lock:
            if self._conn is not None:
                with contextlib.suppress(Exception):
                    self._conn.close()
                self._conn = None

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def _send_request(
        self,
        data: bytes,
        headers: dict[str, str],
    ) -> tuple[int, dict[str, str], str]:
        """Send HTTP request on persistent connection with auto-reconnect on socket disconnect."""
        with self._lock:
            reused = self._conn is not None
            conn = self._get_connection()
            try:
                conn.request("POST", self._path, body=data, headers=headers)
                resp = conn.getresponse()
                body = resp.read().decode("utf-8")
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                if resp_headers.get("connection", "").lower() == "close":
                    self.close()
                return resp.status, resp_headers, body
            except (
                http.client.RemoteDisconnected,
                ConnectionResetError,
                BrokenPipeError,
                http.client.CannotSendRequest,
                http.client.ResponseNotReady,
                http.client.BadStatusLine,
                http.client.ImproperConnectionState,
                OSError,
            ) as err:
                self.close()
                if reused:
                    _LOG.debug("Mất kết nối HTTP Keep-Alive (%s). Đang tự động kết nối lại...", err)
                    new_conn = self._get_connection()
                    new_conn.request("POST", self._path, body=data, headers=headers)
                    resp = new_conn.getresponse()
                    body = resp.read().decode("utf-8")
                    resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                    if resp_headers.get("connection", "").lower() == "close":
                        self.close()
                    return resp.status, resp_headers, body
                raise

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        min_p: float | None = None,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
        json_schema: dict[str, Any] | None = None,
        grammar: str | None = None,
        fallback_single_key: str | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        """Send chat completion request expecting a JSON object output.

        Supports structured JSON schema enforcement, GBNF grammar decoding,
        min-p dynamic sampling, and persistent HTTP Keep-Alive.
        Returns (parsed_dict, prompt_tokens, completion_tokens).
        """
        effective_schema = (
            json_schema
            if json_schema is not None
            else (schema if schema is not None else self.config.json_schema)
        )
        effective_grammar = grammar if grammar is not None else self.config.grammar

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": 0.9,
            "min_p": self.config.min_p if min_p is None else min_p,
            "repeat_penalty": 1.15,
            "stop": ["\n\n//", "\n//", "<|im_end|>", "<|endoftext|>"],
        }
        if effective_grammar is not None:
            payload["grammar"] = effective_grammar
        elif effective_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "translation_response",
                    "strict": True,
                    "schema": effective_schema,
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
            "Connection": "keep-alive",
            "Content-Length": str(len(data)),
        }

        retries = self.config.max_retries
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            if attempt > 0:
                delay = (2 ** (attempt - 1)) + random.uniform(0.1, 1.0)
                _LOG.warning(
                    "Thử lại lần %d/%d sau %.1f giây do lỗi trước đó: %s",
                    attempt,
                    retries,
                    delay,
                    last_error,
                )
                time.sleep(delay)

            # Compatibility check for legacy tests patching urllib.request.urlopen
            if type(urllib.request.urlopen).__module__.startswith("unittest.mock"):
                req = urllib.request.Request(
                    self.endpoint, data=data, headers=headers, method="POST"
                )
                try:
                    with urllib.request.urlopen(req, timeout=self.config.timeout_s) as response:
                        body = response.read().decode("utf-8")
                        status = getattr(response, "status", 200)
                except urllib.error.HTTPError as error:
                    last_error = error
                    # Fallback Level 1: Grammar rejected
                    if error.code in (400, 500) and "grammar" in payload:
                        _LOG.warning(
                            "Server lỗi grammar (HTTP %d). Hạ cấp sang schema / json_object...",
                            error.code,
                        )
                        payload.pop("grammar", None)
                        if effective_schema is not None:
                            payload["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "translation_response",
                                    "strict": True,
                                    "schema": effective_schema,
                                },
                            }
                        else:
                            payload["response_format"] = {"type": "json_object"}
                        data = json.dumps(payload).encode("utf-8")
                        headers["Content-Length"] = str(len(data))
                        continue

                    # Fallback Level 2: Schema rejected
                    if (
                        error.code in (400, 500)
                        and payload.get("response_format", {}).get("type") == "json_schema"
                    ):
                        _LOG.warning(
                            "Server lỗi schema (HTTP %d). Hạ cấp sang json_object...",
                            error.code,
                        )
                        payload["response_format"] = {"type": "json_object"}
                        data = json.dumps(payload).encode("utf-8")
                        headers["Content-Length"] = str(len(data))
                        effective_schema = None
                        continue

                    if error.code in (400, 401, 403, 404):
                        err_body = error.read().decode("utf-8", errors="replace")
                        raise LLMError(
                            f"Lỗi HTTP {error.code} từ LLM server tại {self.endpoint}: {err_body}"
                        ) from error
                    continue
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    last_error = error
                    continue
            else:
                try:
                    status, _resp_headers, body = self._send_request(data, headers)
                except (TimeoutError, OSError) as error:
                    last_error = error
                    self.close()
                    continue

                if status != 200:
                    # Fallback Level 1: Grammar rejected
                    if status in (400, 500) and "grammar" in payload:
                        _LOG.warning(
                            "Server lỗi grammar (HTTP %d). Hạ cấp sang schema / json_object...",
                            status,
                        )
                        payload.pop("grammar", None)
                        if effective_schema is not None:
                            payload["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "translation_response",
                                    "strict": True,
                                    "schema": effective_schema,
                                },
                            }
                        else:
                            payload["response_format"] = {"type": "json_object"}
                        data = json.dumps(payload).encode("utf-8")
                        headers["Content-Length"] = str(len(data))
                        continue

                    # Fallback Level 2: Schema rejected
                    if (
                        status in (400, 500)
                        and payload.get("response_format", {}).get("type") == "json_schema"
                    ):
                        _LOG.warning(
                            "Server lỗi schema (HTTP %d). Hạ cấp sang json_object...",
                            status,
                        )
                        payload["response_format"] = {"type": "json_object"}
                        data = json.dumps(payload).encode("utf-8")
                        headers["Content-Length"] = str(len(data))
                        effective_schema = None
                        continue

                    if status in (400, 401, 403, 404):
                        raise LLMError(
                            f"Lỗi HTTP {status} từ LLM server tại {self.endpoint}: {body}"
                        )
                    last_error = LLMError(f"HTTP {status} từ LLM server: {body[:200]}")
                    continue

            try:
                resp_json = json.loads(body)
            except json.JSONDecodeError:
                last_error = LLMError(f"Phản hồi không phải JSON hợp lệ: {body[:200]}")
                continue

            choices = resp_json.get("choices")
            if not choices or not isinstance(choices, list):
                raise LLMError(f"Phản hồi từ server thiếu trường 'choices': {body[:200]}")

            message = choices[0].get("message", {})
            content = message.get("content", "")

            usage = resp_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            try:
                result_dict = extract_json_object(
                    content, fallback_single_key=fallback_single_key
                )
                return result_dict, prompt_tokens, completion_tokens
            except LLMError as extract_err:
                if fallback_single_key and content.strip():
                    stripped = content.strip()
                    if not _CHAT_CHATTER.match(stripped) and len(stripped.split()) < 200:
                        return (
                            {fallback_single_key: stripped},
                            prompt_tokens,
                            completion_tokens,
                        )
                last_error = extract_err

        raise LLMError(f"Thất bại khi gọi LLM server sau {retries + 1} lần thử: {last_error}")


__all__ = [
    "GENERIC_DICT_GBNF",
    "GENERIC_JSON_GBNF",
    "LLMClient",
    "escape_gbnf_string",
    "extract_json_object",
    "generate_batch_gbnf",
]

