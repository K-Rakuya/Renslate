"""Tests for renslate.llm."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from renslate.config import LLMConfig
from renslate.errors import LLMError
from renslate.llm import LLMClient, extract_json_object


def test_extract_json_clean() -> None:
    raw = '{"d:1": "Xin chào", "d:2": "Tạm biệt"}'
    res = extract_json_object(raw)
    assert res == {"d:1": "Xin chào", "d:2": "Tạm biệt"}


def test_extract_json_from_markdown_code_block() -> None:
    raw = 'Here is your translation:\n```json\n{\n  "d:1": "Xin chào"\n}\n```\nHope it helps!'
    res = extract_json_object(raw)
    assert res == {"d:1": "Xin chào"}


def test_extract_json_with_trailing_commas() -> None:
    raw = '{"d:1": "Xin chào", "d:2": "Tạm biệt",}'
    res = extract_json_object(raw)
    assert res == {"d:1": "Xin chào", "d:2": "Tạm biệt"}


def test_extract_json_invalid_raises() -> None:
    with pytest.raises(LLMError):
        extract_json_object("This is just plain text without any JSON.")


def test_extract_json_regex_fallback() -> None:
    # Model broke outer brace but outputted key-value lines
    broken = 'Here are the lines:\n"d:1#1": "Xin chào bạn",\n"d:2": "Tạm biệt nhé"'
    res = extract_json_object(broken)
    assert res == {"d:1#1": "Xin chào bạn", "d:2": "Tạm biệt nhé"}


def test_extract_json_with_think_tags() -> None:
    raw = (
        '<think>I should translate this dialogue carefully.\n{"test": 123}\n</think>\n'
        '```json\n{"d:intro#1": "Chào bạn!"}\n```'
    )
    res = extract_json_object(raw)
    assert res == {"d:intro#1": "Chào bạn!"}


def test_extract_json_truncated_repair() -> None:
    truncated = '{"d:1": "Dòng 1", "d:2#1": "Dòng 2"'
    res = extract_json_object(truncated)
    assert res == {"d:1": "Dòng 1", "d:2#1": "Dòng 2"}


def test_extract_json_doubled_quotes() -> None:
    raw = (
        '{\n  ""s:298f7158edeafce173d6"": "Bạn và Liana đi bộ một lúc...",\n'
        '  ""s:123"": ""Dòng 2""\n}'
    )
    res = extract_json_object(raw)
    assert res == {
        "s:298f7158edeafce173d6": "Bạn và Liana đi bộ một lúc...",
        "s:123": "Dòng 2",
    }


def test_extract_json_ignores_renpy_tags() -> None:
    raw = "{size=60}{font=fonts/chinese-rocks.otf}Sau đó, vào ban đêm.{/font}{/size}"
    with pytest.raises(LLMError):
        extract_json_object(raw)


def test_llm_client_fallback_single_key() -> None:
    cfg = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="test-model")
    client = LLMClient(cfg)

    fake_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "{size=60}{font=fonts/f.otf}Sau đó, vào ban đêm.{/font}{/size}",
                }
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12},
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_resp).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res, p_tok, c_tok = client.complete_json(
            "system", "user", fallback_single_key="s:test_key_1"
        )

    assert res == {"s:test_key_1": "{size=60}{font=fonts/f.otf}Sau đó, vào ban đêm.{/font}{/size}"}
    assert (p_tok, c_tok) == (40, 12)


def test_llm_client_successful_response() -> None:
    cfg = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="test-model")
    client = LLMClient(cfg)

    fake_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"d:1": "Xin chào!"}',
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15},
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_resp).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res, p_tok, c_tok = client.complete_json("system", "user")

    assert res == {"d:1": "Xin chào!"}
    assert (p_tok, c_tok) == (50, 15)


def test_llm_client_with_schema() -> None:
    import urllib.request

    cfg = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="test-model")
    client = LLMClient(cfg)

    fake_resp = {
        "choices": [{"message": {"content": '{"d:1": "Xin chào!"}'}}],
        "usage": {"prompt_tokens": 60, "completion_tokens": 10},
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_resp).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    captured_requests: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, **_: Any) -> Any:
        captured_requests.append(req)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        schema = {"type": "object", "properties": {"d:1": {"type": "string"}}}
        res, _, _ = client.complete_json("system", "user", schema=schema)
        assert res == {"d:1": "Xin chào!"}
        assert len(captured_requests) == 1
        req_data = captured_requests[0].data
        assert isinstance(req_data, bytes)
        payload = json.loads(req_data.decode("utf-8"))
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["schema"] == schema


def test_llm_client_fallback_on_400() -> None:
    import io
    import urllib.error
    import urllib.request
    from email.message import Message

    cfg = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="test-model", max_retries=1)
    client = LLMClient(cfg)

    fake_resp = {
        "choices": [{"message": {"content": '{"d:1": "Xin chào fallback!"}'}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 10},
    }
    mock_resp_success = MagicMock()
    mock_resp_success.read.return_value = json.dumps(fake_resp).encode("utf-8")
    mock_resp_success.__enter__.return_value = mock_resp_success

    call_count = 0

    def fake_urlopen(req: urllib.request.Request, **_: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call with json_schema returns 400
            fp = io.BytesIO(b'{"error": "json_schema not supported"}')
            msg = Message()
            msg["Content-Type"] = "application/json"
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", msg, fp)
        # Second call with json_object succeeds
        return mock_resp_success

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        schema = {"type": "object"}
        res, _, _ = client.complete_json("system", "user", schema=schema)
        assert res == {"d:1": "Xin chào fallback!"}
        assert call_count == 2


def test_llm_client_fallback_on_500_peg_error() -> None:
    import io
    import urllib.error
    import urllib.request
    from email.message import Message

    cfg = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="test-model", max_retries=1)
    client = LLMClient(cfg)

    fake_resp = {
        "choices": [{"message": {"content": '{"d:1": "Xin chào sau lỗi 500!"}'}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 10},
    }
    mock_resp_success = MagicMock()
    mock_resp_success.read.return_value = json.dumps(fake_resp).encode("utf-8")
    mock_resp_success.__enter__.return_value = mock_resp_success

    call_count = 0

    def fake_urlopen(req: urllib.request.Request, **_: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            err_payload = json.dumps(
                {
                    "error": {
                        "code": 500,
                        "message": "The model produced output not matching peg-native format",
                    }
                }
            ).encode("utf-8")
            fp = io.BytesIO(err_payload)
            msg = Message()
            msg["Content-Type"] = "application/json"
            raise urllib.error.HTTPError(req.full_url, 500, "Internal Server Error", msg, fp)
        # Second call with json_object succeeds
        return mock_resp_success

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        schema = {"type": "object", "properties": {"d:1": {"type": "string"}}}
        res, _, _ = client.complete_json("system", "user", schema=schema)
        assert res == {"d:1": "Xin chào sau lỗi 500!"}
        assert call_count == 2


def test_extract_json_with_trailing_comments() -> None:
    raw = """{
"d:1": "Xin chào"
}

// Note on JSON formatting correction:
{
"d:1": "Xin chào đã sửa"
}"""
    res = extract_json_object(raw)
    assert "d:1" in res
