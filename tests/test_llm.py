"""Tests for renslate.llm."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from renslate.config import LLMConfig
from renslate.errors import LLMError
from renslate.llm import (
    GENERIC_DICT_GBNF,
    GENERIC_JSON_GBNF,
    LLMClient,
    escape_gbnf_string,
    extract_json_object,
    generate_batch_gbnf,
)


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


def test_escape_gbnf_string() -> None:
    assert escape_gbnf_string("d:1") == r'"\"d:1\""'
    assert escape_gbnf_string("d:with\"quote") == r'"\"d:with\\\"quote\""'


def test_generic_gbnf_constants() -> None:
    assert "root ::= " in GENERIC_DICT_GBNF
    assert "root ::= " in GENERIC_JSON_GBNF


def test_generate_batch_gbnf_ordered() -> None:
    grammar = generate_batch_gbnf(["d:1", "d:2"], ordered=True, allow_cjk=True)
    expected_pair = (
        r'root ::= "{" ws "\"d:1\"" ws ":" ws string "," ws "\"d:2\"" ws ":" ws string ws "}" ws'
    )
    assert expected_pair in grammar
    assert r'string ::= "\""' in grammar
    assert "ws ::=" in grammar


def test_generate_batch_gbnf_unordered() -> None:
    grammar = generate_batch_gbnf(["d:1", "d:2"], ordered=False)
    assert r'pair ::= ("\"d:1\"" | "\"d:2\"") ws ":" ws string' in grammar
    assert r'root ::= "{" ws (pair ("," ws pair)*)? ws "}" ws' in grammar


def test_generate_batch_gbnf_empty() -> None:
    grammar = generate_batch_gbnf([])
    assert r'root ::= "{" ws "}" ws' in grammar


def test_generate_batch_gbnf_allow_cjk_false() -> None:
    grammar = generate_batch_gbnf(["d:1"], allow_cjk=False)
    assert r"\u4e00-\u9fff" in grammar
    assert r"\u3040-\u30ff" in grammar


def test_llm_client_grammar_payload_omits_response_format() -> None:
    import urllib.request

    cfg = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="test-model")
    client = LLMClient(cfg)

    fake_resp = {
        "choices": [{"message": {"content": '{"d:1": "Xin chào!"}'}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 10},
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(fake_resp).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    captured_requests: list[urllib.request.Request] = []

    def fake_urlopen(req: urllib.request.Request, **_: Any) -> Any:
        captured_requests.append(req)
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        g = generate_batch_gbnf(["d:1"])
        res, _, _ = client.complete_json("system", "user", grammar=g, schema={"type": "object"})
        assert res == {"d:1": "Xin chào!"}
        assert len(captured_requests) == 1
        req_data = captured_requests[0].data
        assert isinstance(req_data, bytes)
        payload = json.loads(req_data.decode("utf-8"))
        assert "grammar" in payload
        assert payload["grammar"] == g
        assert "response_format" not in payload


def test_llm_client_grammar_fallback_to_schema_on_400() -> None:
    import io
    import urllib.error
    import urllib.request
    from email.message import Message

    cfg = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="test-model", max_retries=1)
    client = LLMClient(cfg)

    fake_resp = {
        "choices": [{"message": {"content": '{"d:1": "Xin chào fallback grammar!"}'}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 10},
    }
    mock_resp_success = MagicMock()
    mock_resp_success.read.return_value = json.dumps(fake_resp).encode("utf-8")
    mock_resp_success.__enter__.return_value = mock_resp_success

    call_count = 0
    captured_payloads: list[dict[str, Any]] = []

    def fake_urlopen(req: urllib.request.Request, **_: Any) -> Any:
        nonlocal call_count
        call_count += 1
        req_data = req.data
        assert isinstance(req_data, bytes)
        captured_payloads.append(json.loads(req_data.decode("utf-8")))
        if call_count == 1:
            fp = io.BytesIO(b'{"error": "grammar not supported"}')
            msg = Message()
            msg["Content-Type"] = "application/json"
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", msg, fp)
        return mock_resp_success

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        schema = {"type": "object", "properties": {"d:1": {"type": "string"}}}
        g = generate_batch_gbnf(["d:1"])
        res, _, _ = client.complete_json("system", "user", grammar=g, schema=schema)
        assert res == {"d:1": "Xin chào fallback grammar!"}
        assert call_count == 2
        assert "grammar" in captured_payloads[0]
        assert "response_format" not in captured_payloads[0]
        assert "grammar" not in captured_payloads[1]
        assert captured_payloads[1]["response_format"]["type"] == "json_schema"


def test_llm_client_persistent_socket_reuse() -> None:
    import http.server
    import threading

    class KeepAliveHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": '{"d:1": "ok"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), KeepAliveHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    cfg = LLMConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="test")
    client = LLMClient(cfg)
    try:
        ports: list[int] = []
        for _ in range(3):
            res, _, _ = client.complete_json("sys", "user")
            assert res == {"d:1": "ok"}
            assert client._conn is not None
            assert client._conn.sock is not None
            ports.append(client._conn.sock.getsockname()[1])

        assert len(ports) == 3
        assert len(set(ports)) == 1
    finally:
        client.close()
        server.shutdown()


def test_llm_client_context_manager_and_close() -> None:
    import http.server
    import threading

    class PingHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": '{"d:1": "pong"}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), PingHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    cfg = LLMConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="test")
    with LLMClient(cfg) as client:
        res, _, _ = client.complete_json("sys", "user")
        assert res == {"d:1": "pong"}
        assert client._conn is not None

    assert client._conn is None
    server.shutdown()


def test_llm_client_auto_reconnect_on_server_disconnect() -> None:
    import http.server
    import threading

    class DisconnectingHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True
        req_count = 0

        def do_POST(self) -> None:
            DisconnectingHandler.req_count += 1
            idx = DisconnectingHandler.req_count
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps({
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"d:1": f"val_{idx}"}),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if idx == 1:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            if idx == 1:
                self.close_connection = True

        def log_message(self, *_: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), DisconnectingHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    cfg = LLMConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="test")
    with LLMClient(cfg) as client:
        r1, _, _ = client.complete_json("sys", "u1")
        assert r1 == {"d:1": "val_1"}
        r2, _, _ = client.complete_json("sys", "u2")
        assert r2 == {"d:1": "val_2"}

    server.shutdown()


def test_llm_client_thread_safe_locking() -> None:
    import http.server
    import threading
    from concurrent.futures import ThreadPoolExecutor

    class ConcurrentHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True
        count = 0
        lock = threading.Lock()

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            with ConcurrentHandler.lock:
                ConcurrentHandler.count += 1
                curr = ConcurrentHandler.count
            body = json.dumps({
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"d:1": f"worker_{curr}"}),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), ConcurrentHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    cfg = LLMConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="test")
    client = LLMClient(cfg)
    try:
        def invoke_call(idx: int) -> dict[str, Any]:
            res, _, _ = client.complete_json("sys", f"call_{idx}")
            return res

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(invoke_call, range(8)))

        assert len(results) == 8
        assert all("d:1" in r for r in results)
    finally:
        client.close()
        server.shutdown()

