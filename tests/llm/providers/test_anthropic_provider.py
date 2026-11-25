"""Pure-offline tests for the novakit Anthropic provider.

Covers ``novakit.llm.providers.anthropic`` only: request shaping, message
conversion, non-streaming parsing, streaming event translation (text,
thinking/signature, tool input events, usage, errors, truncation) and
``count_tokens``. No network access; ``aiohttp`` is replaced with fakes and
request hooks are monkeypatched. Mirrors the intent of
``nova/tests/test_anthropic_provider.py`` but targets the novakit API.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest

from novakit.llm.accumulator import StreamAccumulator
from novakit.llm.provider import (
    Done,
    Error,
    LLMProvider,
    Message,
    ReasoningDelta,
    RetryNotice,
    TextDelta,
    ToolCall,
    ToolInputDelta,
    ToolInputEnd,
    ToolInputError,
    ToolInputStart,
)
from novakit.llm.providers.anthropic import (
    AnthropicProvider,
    _AnthropicStreamParser,
    _content_block_index,
    _default_max_output_tokens,
    _preserves_thinking,
    _state_to_tool_call,
    _thinking_provider_meta,
    _tool_call_parts,
)

# ---------------------------------------------------------------------------
# aiohttp fakes (module top, reused by every test)
# ---------------------------------------------------------------------------


class _FakeConnector:
    """Stands in for ``aiohttp.TCPConnector``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        self._closed = True


class _FakeStreamContent:
    """Mimics ``aiohttp.StreamReader.readline``, including its default line cap."""

    def __init__(self, lines: list[bytes], default_limit: int = 131072) -> None:
        self._lines = list(lines)
        self._index = 0
        self._default_limit = default_limit

    async def readline(self, *, max_line_length: int | None = None) -> bytes:
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        limit = max_line_length if max_line_length is not None else self._default_limit
        if len(line) > limit:
            raise aiohttp.http_exceptions.LineTooLong(line[:100] + b"...", limit)
        self._index += 1
        return line


class _FakeResponse:
    """Stands in for ``aiohttp.ClientResponse``."""

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        content_type: str = "application/json",
        json_data: dict[str, Any] | None = None,
        text_data: str | None = None,
        sse_lines: list[bytes] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.content_type = content_type
        self._json_data = json_data
        self._text_data = text_data
        self.content = _FakeStreamContent(sse_lines or [])

    async def text(self) -> str:
        if self._text_data is not None:
            return self._text_data
        if self._json_data is not None:
            return json.dumps(self._json_data)
        return ""

    async def json(self) -> dict[str, Any]:
        if self._json_data is not None:
            return self._json_data
        raise ValueError("no json_data")

    def close(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _FakeSession:
    """Stands in for ``aiohttp.ClientSession``; records every ``post`` call."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: Any = None,
    ) -> _FakeResponse:
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return self._response

    async def close(self) -> None:
        self.closed = True


class _RaisingSession(_FakeSession):
    """Fails the ``post`` itself, exercising the transport exception path."""

    async def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: Any = None,
    ) -> _FakeResponse:
        raise RuntimeError("boom")


def _sse_lines(
    events: list[dict[str, Any] | tuple[str, dict[str, Any]]],
) -> list[bytes]:
    """Turn payloads (or ``(event, payload)`` pairs) into SSE byte lines."""
    output: list[bytes] = []
    for item in events:
        if isinstance(item, tuple):
            event_name, payload = item
        else:
            payload = item
            event_name = str(payload.get("type", ""))
        output.append(f"event: {event_name}\n".encode())
        output.append(f"data: {json.dumps(payload)}\n".encode())
        output.append(b"\n")
    return output


def _install_fake(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse
) -> tuple[_FakeSession, _FakeConnector]:
    session = _FakeSession(response)
    connector = _FakeConnector()

    def _fake_session_factory(*args: Any, **kwargs: Any) -> _FakeSession:
        return session

    def _fake_connector_factory(*args: Any, **kwargs: Any) -> _FakeConnector:
        return connector

    monkeypatch.setattr(
        "novakit.llm.providers.anthropic.aiohttp.ClientSession",
        _fake_session_factory,
    )
    monkeypatch.setattr(
        "novakit.llm.providers.anthropic.aiohttp.TCPConnector",
        _fake_connector_factory,
    )
    return session, connector


def _acc() -> StreamAccumulator:
    """Fresh accumulator with the runaway guards disabled."""
    return StreamAccumulator("claude-3-opus-20240229", None, None)


# ---------------------------------------------------------------------------
# A. Endpoint, headers, timeouts
# ---------------------------------------------------------------------------


def test_endpoint_default() -> None:
    provider = AnthropicProvider()
    assert provider._endpoint() == "https://api.anthropic.com/v1/messages"


def test_endpoint_base_url_not_doubled() -> None:
    provider = AnthropicProvider(base_url="https://example.com/v1")
    assert provider._endpoint() == "https://example.com/v1/messages"


def test_endpoint_plain_base_url_gains_version_prefix() -> None:
    provider = AnthropicProvider(base_url="https://example.com")
    assert provider._endpoint() == "https://example.com/v1/messages"


def test_endpoint_trailing_slash_stripped() -> None:
    provider = AnthropicProvider(base_url="https://example.com/v1/")
    assert provider.base_url == "https://example.com/v1"
    assert provider._endpoint() == "https://example.com/v1/messages"
    provider2 = AnthropicProvider(base_url="https://example.com/")
    assert provider2._endpoint() == "https://example.com/v1/messages"


def test_build_headers_basic() -> None:
    provider = AnthropicProvider(api_key="sk-test")
    headers = provider._build_headers()
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["x-api-key"] == "sk-test"
    assert "Authorization" not in headers


def test_build_headers_omits_key_when_empty() -> None:
    provider = AnthropicProvider(api_key="")
    headers = provider._build_headers()
    assert "x-api-key" not in headers


def test_build_headers_betas_and_user_agent() -> None:
    provider = AnthropicProvider(api_key="k", betas=["beta1", "beta2"])
    headers = provider._build_headers()
    assert headers["anthropic-beta"] == "beta1,beta2"
    assert headers["User-Agent"].startswith("novakit/")

    provider2 = AnthropicProvider(api_key="k")
    headers2 = provider2._build_headers()
    assert "anthropic-beta" not in headers2
    assert headers2["User-Agent"].startswith("novakit/")


def test_build_headers_extra_headers_merged() -> None:
    provider = AnthropicProvider(
        api_key="k", extra_headers={"X-Custom": "1", "User-Agent": "evil"}
    )
    headers = provider._build_headers()
    assert headers["X-Custom"] == "1"
    # The default identity is written last: config cannot replace it.
    assert headers["User-Agent"].startswith("novakit/")


def test_build_headers_custom_user_agent() -> None:
    provider = AnthropicProvider(api_key="k", user_agent="  gateway/1.0  ")
    assert provider._build_headers()["User-Agent"] == "gateway/1.0"


def test_build_headers_request_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "novakit.llm.providers.anthropic.run_session_hook",
        lambda path, session_id: {"X-Session": "s"},
    )
    monkeypatch.setattr(
        "novakit.llm.providers.anthropic.run_request_hook",
        lambda path, session_id: {"X-Req": "r"},
    )
    provider = AnthropicProvider(
        api_key="k", request_session_hook="s.py", request_hook="r.py"
    )
    headers = provider._build_headers(session_id="abc")
    assert headers["X-Session"] == "s"
    assert headers["X-Req"] == "r"
    assert headers["User-Agent"].startswith("novakit/")


def test_build_headers_no_hooks_by_default() -> None:
    provider = AnthropicProvider(api_key="k")
    headers = provider._build_headers(session_id="abc")
    assert "X-Session" not in headers


# ---------------------------------------------------------------------------
# B. Request body: tools, tokens, system, caching
# ---------------------------------------------------------------------------


def test_build_body_max_tokens_and_system_and_stream() -> None:
    provider = AnthropicProvider(request_options={"prompt_caching": False})
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
    ]
    body = provider._build_body(
        messages=messages, model="claude-3-5-sonnet-20241022", stream=False
    )
    assert body["max_tokens"] == 8192
    assert body["system"] == "sys"
    assert "stream" not in body
    assert all(message["role"] != "system" for message in body["messages"])


def test_build_body_system_joined_and_stream_true() -> None:
    provider = AnthropicProvider(request_options={"prompt_caching": False})
    messages = [
        Message(role="system", content="a"),
        Message(role="system", content="b"),
        Message(role="user", content="hi"),
    ]
    body = provider._build_body(
        messages=messages, model="claude-3-opus-20240229", stream=True
    )
    assert body["system"] == "a\n\nb"
    assert body["stream"] is True


def test_build_body_request_options_flatten_and_max_tokens_override() -> None:
    provider = AnthropicProvider(
        request_options={"max_tokens": 123, "thinking": {"type": "enabled"}}
    )
    body = provider._build_body(
        messages=[Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert body["max_tokens"] == 123
    assert body["thinking"] == {"type": "enabled"}


def test_build_body_bad_max_tokens_falls_back_to_default() -> None:
    provider = AnthropicProvider(request_options={"max_tokens": "bogus"})
    body = provider._build_body(
        messages=[Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert body["max_tokens"] == 4096


def test_build_body_constructor_max_output_tokens_override() -> None:
    provider = AnthropicProvider(max_output_tokens=100)
    body = provider._build_body(
        messages=[Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert body["max_tokens"] == 100


def test_build_body_system_list_from_options() -> None:
    provider = AnthropicProvider(
        request_options={"system": [{"type": "text", "text": "s"}]}
    )
    body = provider._build_body(
        messages=[Message(role="user", content="hi")], model="claude-opus-4-8"
    )
    assert isinstance(body["system"], list)
    assert body["system"][0]["text"] == "s"
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_build_body_no_system_key_when_empty() -> None:
    provider = AnthropicProvider(
        request_options={"prompt_caching": False, "system": ""}
    )
    body = provider._build_body(
        messages=[Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert "system" not in body


def test_build_body_system_only_messages() -> None:
    provider = AnthropicProvider(request_options={"prompt_caching": False})
    body = provider._build_body(
        messages=[Message(role="system", content="sys")],
        model="claude-3-opus-20240229",
    )
    assert body["messages"] == []
    assert body["system"] == "sys"


def test_build_body_request_options_tools_false_omits_key() -> None:
    provider = AnthropicProvider(request_options={"tools": False})
    body = provider._build_body(
        messages=[Message(role="user", content="hi")],
        model="claude-3-opus-20240229",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )
    assert "tools" not in body


def test_build_body_tools_empty_list_omits_key() -> None:
    provider = AnthropicProvider()
    body = provider._build_body(
        messages=[Message(role="user", content="hi")],
        model="claude-3-opus-20240229",
        tools=[],
    )
    assert "tools" not in body


def test_build_body_tools_none_omits_key() -> None:
    provider = AnthropicProvider()
    body = provider._build_body(
        messages=[Message(role="user", content="hi")],
        model="claude-3-opus-20240229",
        tools=None,
    )
    assert "tools" not in body


def test_format_tools_via_build_body() -> None:
    provider = AnthropicProvider()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "r",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
        {"name": "write", "input_schema": {"type": "object"}},
        {"description": "no name"},
    ]
    body = provider._build_body(
        messages=[Message(role="user", content="hi")],
        model="claude-3-opus-20240229",
        tools=tools,
    )
    assert len(body["tools"]) == 2
    assert body["tools"][0] == {
        "name": "read",
        "description": "r",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
    assert body["tools"][1]["name"] == "write"
    assert body["tools"][1]["input_schema"] == {"type": "object"}
    assert all(tool["name"] for tool in body["tools"])


def test_format_tools_skips_non_dict_and_bare_function() -> None:
    provider = AnthropicProvider()
    assert provider._format_tools(None) == []
    assert provider._format_tools(["nope", 42, {"function": "nope"}]) == []
    assert provider._format_tools([{"function": {"description": "x"}}]) == []


def test_format_tools_description_and_schema_fallbacks() -> None:
    provider = AnthropicProvider()
    formatted = provider._format_tools(
        [
            {"name": "a"},
            {"name": "b", "parameters": {"type": "object"}},
            {"name": "c", "input_schema": "nope"},
            {"name": "d", "input_schema": {"type": "object"}, "description": 42},
        ]
    )
    assert formatted[0]["description"] == ""
    assert formatted[0]["input_schema"] == {"type": "object", "properties": {}}
    assert formatted[1]["input_schema"] == {"type": "object"}
    assert formatted[2]["input_schema"] == {"type": "object", "properties": {}}
    assert formatted[3]["description"] == 42


def test_preserves_thinking_by_model() -> None:
    assert _preserves_thinking("claude-opus-4-5") is True
    assert _preserves_thinking("claude-opus-4-8") is True
    assert _preserves_thinking("claude-sonnet-4-5") is True
    assert _preserves_thinking("claude-sonnet-4-5-20250929") is True
    assert _preserves_thinking("anthropic/claude-opus-4-6") is True
    assert _preserves_thinking("claude-opus-4-1-20250805") is False
    assert _preserves_thinking("claude-sonnet-4-20250514") is False
    assert _preserves_thinking("claude-3-7-sonnet-20250219") is False
    assert _preserves_thinking("claude-3-5-sonnet-20241022") is False
    assert _preserves_thinking("claude-3-opus-20240229") is False
    assert _preserves_thinking("claude-haiku-4-5") is False
    assert _preserves_thinking(None) is False
    assert _preserves_thinking("") is False
    assert _preserves_thinking("gpt-5") is False


def test_build_body_cache_control_on_system_and_last_message() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="system", content="sys prompt"),
        Message(role="user", content="hello"),
    ]
    body = provider._build_body(messages=messages, model="claude-opus-4-8")
    assert isinstance(body["system"], list)
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert body["system"][-1]["text"] == "sys prompt"
    assert body["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_build_body_cache_control_ttl_5m_omits_ttl_key() -> None:
    provider = AnthropicProvider(request_options={"cache_ttl": "5m"})
    body = provider._build_body(
        messages=[
            Message(role="system", content="s"),
            Message(role="user", content="hi"),
        ],
        model="claude-opus-4-8",
    )
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_build_body_cache_control_ttl_1h_included() -> None:
    provider = AnthropicProvider(request_options={"cache_ttl": "1h"})
    body = provider._build_body(
        messages=[
            Message(role="system", content="s"),
            Message(role="user", content="hi"),
        ],
        model="claude-opus-4-8",
    )
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert body["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_build_body_cache_control_disabled() -> None:
    provider = AnthropicProvider(request_options={"prompt_caching": False})
    body = provider._build_body(
        messages=[
            Message(role="system", content="s"),
            Message(role="user", content="hi"),
        ],
        model="claude-opus-4-8",
    )
    assert body["system"] == "s"
    assert all(
        "cache_control" not in block for block in body["messages"][-1]["content"]
    )


def test_build_body_cache_control_not_leaked_into_body_params() -> None:
    provider = AnthropicProvider(
        request_options={"prompt_caching": True, "cache_ttl": "1h"}
    )
    body = provider._build_body(
        messages=[Message(role="user", content="hi")],
        model="claude-opus-4-8",
    )
    assert "prompt_caching" not in body
    assert "cache_ttl" not in body


def test_build_body_cache_control_unknown_ttl_falls_back_to_default() -> None:
    provider = AnthropicProvider(request_options={"cache_ttl": "bogus"})
    body = provider._build_body(
        messages=[
            Message(role="system", content="s"),
            Message(role="user", content="hi"),
        ],
        model="claude-opus-4-8",
    )
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_default_max_output_tokens_table() -> None:
    assert _default_max_output_tokens("claude-3-opus-20240229") == 4096
    assert _default_max_output_tokens("claude-3-5-sonnet-20241022") == 8192
    assert _default_max_output_tokens("claude-3-7-sonnet-latest") == 64000
    assert _default_max_output_tokens("claude-sonnet-4-5") == 64000
    assert _default_max_output_tokens("claude-opus-4-1") == 32000
    assert _default_max_output_tokens("claude-3-haiku-20240307") == 4096
    assert _default_max_output_tokens("claude-3-5-haiku-20241022") == 8192
    assert _default_max_output_tokens("unknown-model-xyz") == 8192
    assert _default_max_output_tokens("anthropic/claude-3-5-sonnet-20241022") == 8192


# ---------------------------------------------------------------------------
# C. Small helpers: tool-call parts, block index, thinking meta
# ---------------------------------------------------------------------------


def test_tool_call_parts_flat_and_nested() -> None:
    flat = {"id": "t1", "name": "read", "arguments": '{"a": 1}'}
    assert _tool_call_parts(flat) == ("t1", "read", '{"a": 1}')
    nested = {
        "id": "t2",
        "function": {"name": "write", "arguments": '{"b": 2}'},
    }
    assert _tool_call_parts(nested) == ("t2", "write", '{"b": 2}')


def test_tool_call_parts_arguments_dict_and_none() -> None:
    with_dict = {"id": "t1", "name": "read", "arguments": {"a": 1}}
    parts = _tool_call_parts(with_dict)
    assert parts is not None
    assert parts[2] == json.dumps({"a": 1}, ensure_ascii=False)
    with_none = {"id": "t1", "name": "read", "arguments": None}
    assert _tool_call_parts(with_none) == ("t1", "read", "")
    assert _tool_call_parts({}) is None
    assert _tool_call_parts({"id": "", "name": ""}) is None


def test_tool_call_parts_non_dict_inputs() -> None:
    assert _tool_call_parts("nope") is None
    assert _tool_call_parts(42) is None

    class _Exploding:
        def model_dump(self) -> dict[str, Any]:
            raise RuntimeError("dump failed")

    assert _tool_call_parts(_Exploding()) is None

    class _NonDict:
        def model_dump(self) -> list[str]:
            return ["not", "a", "dict"]

    assert _tool_call_parts(_NonDict()) is None

    class _Dumped:
        def model_dump(self) -> dict[str, Any]:
            return {"id": "t9", "name": "read", "arguments": "{}"}

    assert _tool_call_parts(_Dumped()) == ("t9", "read", "{}")


def test_content_block_index() -> None:
    assert _content_block_index({"index": 3}) == 3
    assert _content_block_index({}) == 0
    assert _content_block_index({"index": "bad"}) == 0
    assert _content_block_index({"index": None}) == 0


def test_state_to_tool_call_prefers_arguments() -> None:
    call = _state_to_tool_call({"id": "t", "name": "n", "arguments": '{"a": 1}'})
    assert isinstance(call, ToolCall)
    assert (call.id, call.name, call.arguments) == ("t", "n", '{"a": 1}')
    via_initial = _state_to_tool_call({"id": "t", "name": "n", "initial_input": "{}"})
    assert via_initial.arguments == "{}"
    empty = _state_to_tool_call({"id": "t", "name": "n"})
    assert empty.arguments == "{}"


def test_thinking_provider_meta_empty() -> None:
    assert _thinking_provider_meta([]) is None
    assert _thinking_provider_meta([{"type": "thinking"}]) is None
    assert _thinking_provider_meta([{"type": "thinking", "signature": ""}]) is None
    assert _thinking_provider_meta([{"type": "thinking", "signature": 123}]) is None
    assert _thinking_provider_meta([{"type": "redacted_thinking"}]) is None
    assert _thinking_provider_meta([{"type": "redacted_thinking", "data": ""}]) is None
    assert _thinking_provider_meta([{"type": "text", "text": "x"}]) is None


def test_thinking_provider_meta_mixed() -> None:
    meta = _thinking_provider_meta(
        [
            {"type": "thinking", "thinking": "r", "signature": ""},
            {"type": "thinking", "thinking": "r2", "signature": "SIG2"},
            {"type": "thinking", "thinking": "r3", "signature": "SIG3"},
            {"type": "redacted_thinking", "data": ""},
            {"type": "redacted_thinking", "data": "D1"},
            {"type": "text", "text": "ignored"},
        ]
    )
    assert meta == {"thinking_signature": "SIG2", "redacted_thinking": ["D1"]}


def test_replay_thinking_blocks_edge_cases() -> None:
    provider = AnthropicProvider()
    get_attr = lambda message, key: message.get(key)  # noqa: E731

    def assistant(**fields: Any) -> dict[str, Any]:
        base: dict[str, Any] = {"role": "assistant"}
        base.update(fields)
        return base

    # Non-dict provider_meta is ignored.
    message = assistant(provider_meta="nope", model="claude-sonnet-4-5")
    assert (
        provider._replay_thinking_blocks(get_attr, message, "claude-sonnet-4-5") == []
    )
    # Dict meta without signature or redacted entries yields nothing.
    message2 = assistant(provider_meta={}, model="claude-sonnet-4-5")
    assert (
        provider._replay_thinking_blocks(get_attr, message2, "claude-sonnet-4-5") == []
    )
    # Non-string signature is ignored; falsy redacted entries are skipped.
    message3 = assistant(
        provider_meta={"thinking_signature": 123, "redacted_thinking": ["", "D"]},
        model="claude-sonnet-4-5",
    )
    assert provider._replay_thinking_blocks(
        get_attr, message3, "claude-sonnet-4-5"
    ) == [{"type": "redacted_thinking", "data": "D"}]
    # Non-list redacted_thinking is ignored.
    message4 = assistant(
        provider_meta={"redacted_thinking": "nope"}, model="claude-sonnet-4-5"
    )
    assert (
        provider._replay_thinking_blocks(get_attr, message4, "claude-sonnet-4-5") == []
    )


# ---------------------------------------------------------------------------
# D. Message conversion
# ---------------------------------------------------------------------------


def test_format_messages_full_tool_round_trip() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content="will read",
            tool_calls=[
                {
                    "id": "toolu_1",
                    "type": "tool_call",
                    "name": "read",
                    "arguments": '{"path":"a.txt"}',
                },
                {
                    "id": "toolu_2",
                    "type": "tool_call",
                    "name": "write",
                    "arguments": '{"path":"b.txt"}',
                },
            ],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="content a"),
        Message(role="tool", tool_call_id="toolu_2", content="content b"),
    ]
    formatted, system_text = provider._format_messages(messages)
    assert system_text == "sys"
    assert len(formatted) == 3
    assert [message["role"] for message in formatted] == ["user", "assistant", "user"]
    assistant = formatted[1]
    assert assistant["content"][0] == {"type": "text", "text": "will read"}
    assert assistant["content"][1]["type"] == "tool_use"
    assert assistant["content"][1]["id"] == "toolu_1"
    assert assistant["content"][1]["input"] == {"path": "a.txt"}
    assert assistant["content"][2]["input"] == {"path": "b.txt"}
    trailing = formatted[2]
    assert len(trailing["content"]) == 2
    assert all(block["type"] == "tool_result" for block in trailing["content"])
    assert trailing["content"][0]["tool_use_id"] == "toolu_1"
    assert trailing["content"][1]["tool_use_id"] == "toolu_2"


def test_format_messages_plain_dict_inputs() -> None:
    provider = AnthropicProvider()
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"id": "toolu_1", "name": "read", "arguments": '{"x":1}'}],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "out"},
    ]
    formatted, _ = provider._format_messages(messages)
    assert len(formatted) == 3


def test_format_messages_skips_roleless_and_empty() -> None:
    provider = AnthropicProvider()
    formatted, _ = provider._format_messages(
        [
            {"content": "no role"},
            Message(role="user", content=""),
            Message(role="user", content="hi"),
            Message(role="assistant", content=""),
            {"role": "weird"},
            {"role": "weird", "content": "fallback"},
        ]
    )
    # The two surviving user messages are adjacent, so they merge.
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    texts = [
        block.get("text", "")
        for block in formatted[0]["content"]
        if block.get("type") == "text"
    ]
    assert texts == ["hi", "fallback"]


def test_format_messages_non_list_tool_calls_ignored() -> None:
    provider = AnthropicProvider()
    formatted, _ = provider._format_messages(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="x", tool_calls="nope"),
        ]
    )
    assert formatted[1]["content"] == [{"type": "text", "text": "x"}]


def test_format_messages_unparseable_tool_call_shapes_skipped() -> None:
    provider = AnthropicProvider()
    formatted, _ = provider._format_messages(
        [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content="x",
                tool_calls=[{}, "nope"],
            ),
        ]
    )
    assert formatted[1]["content"] == [{"type": "text", "text": "x"}]


def test_format_messages_tool_call_without_name_skipped() -> None:
    provider = AnthropicProvider()
    formatted, _ = provider._format_messages(
        [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content="x",
                tool_calls=[{"id": "toolu_9", "name": "", "arguments": "{}"}],
            ),
            Message(role="tool", tool_call_id="toolu_9", content="out"),
        ]
    )
    assistant = next(message for message in formatted if message["role"] == "assistant")
    assert all(block.get("type") != "tool_use" for block in assistant["content"])


def test_format_messages_arguments_dict_shape() -> None:
    provider = AnthropicProvider()
    formatted, _ = provider._format_messages(
        [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content="x",
                tool_calls=[{"id": "toolu_1", "name": "read", "arguments": {"p": "a"}}],
            ),
            Message(role="tool", tool_call_id="toolu_1", content="out"),
        ]
    )
    tool_use = next(
        block
        for message in formatted
        for block in message["content"]
        if block.get("type") == "tool_use"
    )
    assert tool_use["input"] == {"p": "a"}


def test_format_messages_arguments_non_object_json_degrades() -> None:
    provider = AnthropicProvider()
    formatted, _ = provider._format_messages(
        [
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                content="x",
                tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "[1, 2]"}],
            ),
            Message(role="tool", tool_call_id="toolu_1", content="out"),
        ]
    )
    tool_use = next(
        block
        for message in formatted
        for block in message["content"]
        if block.get("type") == "tool_use"
    )
    assert tool_use["input"] == {}


def test_format_messages_tool_without_id_skipped() -> None:
    provider = AnthropicProvider()
    formatted, _ = provider._format_messages(
        [
            Message(role="user", content="hi"),
            Message(role="tool", content="orphan"),
        ]
    )
    assert len(formatted) == 1


def test_consecutive_same_role_merge() -> None:
    provider = AnthropicProvider()
    messages = [Message(role="user", content="a"), Message(role="user", content="b")]
    formatted, _ = provider._format_messages(messages)
    assert len(formatted) == 1
    assert len(formatted[0]["content"]) == 2


def test_tool_result_blocks_sorted_front() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="user", content="start"),
        Message(
            role="assistant",
            content="x",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out"),
        Message(role="user", content="follow up"),
    ]
    formatted, _ = provider._format_messages(messages)
    assert len(formatted) >= 2
    merged_user = formatted[2] if len(formatted) > 2 else None
    assert merged_user is not None
    assert merged_user["role"] == "user"
    assert merged_user["content"][0]["type"] == "tool_result"


def test_empty_tool_result_placeholder() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="x",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content=""),
    ]
    formatted, _ = provider._format_messages(messages)
    tool_results = [
        block
        for message in formatted
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert tool_results
    assert tool_results[0]["content"][0]["text"] == "(no output)"


def test_pair_integrity_drops_unmatched() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="x",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
    ]
    formatted, _ = provider._format_messages(messages)
    assistant = next(message for message in formatted if message["role"] == "assistant")
    assert all(block["type"] != "tool_use" for block in assistant["content"])

    messages2 = [
        Message(role="user", content="hi"),
        Message(role="tool", tool_call_id="toolu_99", content="out"),
    ]
    formatted2, _ = provider._format_messages(messages2)
    assert all(
        block.get("tool_use_id") != "toolu_99"
        for message in formatted2
        for block in message.get("content", [])
    )


def test_nova_flat_and_openai_nested_shapes() -> None:
    provider = AnthropicProvider()
    messages_flat = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="x",
            tool_calls=[
                {
                    "type": "tool_call",
                    "id": "toolu_1",
                    "name": "read",
                    "arguments": '{"path":"a"}',
                }
            ],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out"),
    ]
    formatted, _ = provider._format_messages(messages_flat)
    assert any(
        block.get("name") == "read"
        for message in formatted
        for block in message["content"]
        if block.get("type") == "tool_use"
    )
    messages_nested = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="x",
            tool_calls=[
                {
                    "id": "toolu_2",
                    "function": {"name": "write", "arguments": '{"path":"b"}'},
                }
            ],
        ),
        Message(role="tool", tool_call_id="toolu_2", content="out"),
    ]
    formatted2, _ = provider._format_messages(messages_nested)
    assert any(
        block.get("name") == "write"
        for message in formatted2
        for block in message["content"]
        if block.get("type") == "tool_use"
    )


def test_unparseable_arguments_degrades_to_empty_dict() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="x",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "not-json"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out"),
    ]
    formatted, _ = provider._format_messages(messages)
    tool_use = next(
        block
        for message in formatted
        for block in message["content"]
        if block.get("type") == "tool_use"
    )
    assert tool_use["input"] == {}


def test_images_user_and_tool() -> None:
    provider = AnthropicProvider()
    messages = [Message(role="user", content="hi", images=["BASE64"])]
    formatted, _ = provider._format_messages(messages)
    assert len(formatted[0]["content"]) == 2
    assert formatted[0]["content"][0] == {"type": "text", "text": "hi"}
    assert formatted[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "BASE64"},
    }

    messages2 = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="x",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out", images=["IMG2"]),
    ]
    formatted2, _ = provider._format_messages(messages2)
    tool_result = next(
        block
        for message in formatted2
        for block in message["content"]
        if block.get("type") == "tool_result"
    )
    assert any(
        content_block.get("type") == "image" for content_block in tool_result["content"]
    )


def test_image_media_type_comes_from_the_data_url() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="user", content="hi", images=["data:image/jpeg;base64,QUJD"])
    ]
    formatted, _ = provider._format_messages(messages)
    assert formatted[0]["content"][1]["source"] == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "QUJD",
    }


def test_leading_assistant_dropped() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="assistant", content="should drop"),
        Message(role="user", content="hi"),
    ]
    formatted, _ = provider._format_messages(messages)
    assert formatted[0]["role"] == "user"
    assert len(formatted) == 1


def test_leading_assistant_summary_converted_to_user() -> None:
    provider = AnthropicProvider()
    summary_content = (
        "[Previous conversation summary]\nEarlier context here\n\nContinue."
    )
    messages = [
        Message(role="system", content="SYSTEM PROMPT"),
        Message(role="assistant", content=summary_content),
        Message(role="user", content="continue"),
    ]
    formatted, system_text = provider._format_messages(messages)
    assert system_text == "SYSTEM PROMPT"
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    first_text_blocks = [
        block for block in formatted[0]["content"] if block.get("type") == "text"
    ]
    assert first_text_blocks
    combined_text = " ".join(block.get("text", "") for block in first_text_blocks)
    assert summary_content in combined_text


def test_leading_assistant_summary_merges_with_following_user() -> None:
    provider = AnthropicProvider()
    summary_content = "[Previous conversation summary]\nSummarized history"
    messages = [
        Message(role="assistant", content=summary_content),
        Message(role="user", content="continue"),
    ]
    formatted, _ = provider._format_messages(messages)
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    text_blocks = [
        block for block in formatted[0]["content"] if block.get("type") == "text"
    ]
    assert len(text_blocks) == 2
    assert text_blocks[0]["text"] == summary_content
    assert text_blocks[1]["text"] == "continue"


def test_leading_assistant_with_tool_use_dropped_with_its_tool_results() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(
            role="assistant",
            content="x",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="tool output"),
        Message(role="user", content="next"),
    ]
    formatted, _ = provider._format_messages(messages)
    for message in formatted:
        for block in message["content"]:
            assert block.get("type") != "tool_use"
            assert block.get("tool_use_id") != "toolu_1"
            if block.get("type") == "tool_result":
                assert block.get("tool_use_id") != "toolu_1"
    assert formatted
    assert formatted[0]["role"] == "user"


def test_leading_thinking_only_assistant_is_dropped() -> None:
    provider = AnthropicProvider(
        request_options={"thinking": {"type": "enabled", "budget_tokens": 1024}}
    )
    messages = [
        Message(
            role="assistant",
            content="",
            reasoning_content="r",
            provider_meta={"thinking_signature": "SIG"},
            model="claude-sonnet-4-5",
        ),
        Message(role="user", content="next"),
    ]
    formatted, _ = provider._format_messages(
        messages, include_thinking=True, model="claude-sonnet-4-5"
    )
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"


def test_leading_assistant_thinking_blocks_not_carried_into_user() -> None:
    provider = AnthropicProvider(
        request_options={"thinking": {"type": "enabled", "budget_tokens": 1024}}
    )
    messages = [
        Message(
            role="assistant",
            content="x",
            reasoning_content="should not leak",
            provider_meta={"thinking_signature": "sig123"},
            model="claude-sonnet-4-5",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out"),
        Message(role="user", content="next"),
    ]
    formatted, _ = provider._format_messages(
        messages, include_thinking=True, model="claude-sonnet-4-5"
    )
    for message in formatted:
        if message["role"] == "user":
            for block in message["content"]:
                assert block.get("type") not in ("thinking", "redacted_thinking")
    messages2 = [
        Message(
            role="assistant",
            content="y",
            provider_meta={"redacted_thinking": ["secret"]},
            model="claude-sonnet-4-5",
            tool_calls=[{"id": "toolu_2", "name": "write", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_2", content="out2"),
        Message(role="user", content="follow up"),
    ]
    formatted2, _ = provider._format_messages(
        messages2, include_thinking=True, model="claude-sonnet-4-5"
    )
    for message in formatted2:
        if message["role"] == "user":
            for block in message["content"]:
                assert block.get("type") not in ("thinking", "redacted_thinking")


def test_leading_user_message_unaffected() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
        Message(role="user", content="follow up"),
    ]
    formatted, system_text = provider._format_messages(messages)
    assert system_text == "sys"
    assert [message["role"] for message in formatted] == ["user", "assistant", "user"]
    assert formatted[0]["content"][0]["text"] == "hello"
    assert formatted[1]["content"][0]["text"] == "hi there"
    assert formatted[2]["content"][0]["text"] == "follow up"


def test_trailing_assistant_whitespace_is_stripped() -> None:
    provider = AnthropicProvider()
    formatted, _ = provider._format_messages(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="answer   "),
        ]
    )
    assert formatted[-1]["content"][0]["text"] == "answer"


def test_trailing_assistant_blank_text_block_is_removed() -> None:
    provider = AnthropicProvider()
    formatted, _ = provider._format_messages(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="   "),
        ]
    )
    assert formatted[-1]["role"] == "assistant"
    assert formatted[-1]["content"] == []


# ---------------------------------------------------------------------------
# E. Thinking replay (provider_meta)
# ---------------------------------------------------------------------------


def test_format_messages_rehydrates_thinking_signature() -> None:
    provider = AnthropicProvider(
        request_options={"thinking": {"type": "enabled", "budget_tokens": 1024}}
    )
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="will read",
            reasoning_content="reasoned",
            provider_meta={"thinking_signature": "SIG"},
            model="claude-sonnet-4-5",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": '{"path":"a"}'}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out"),
    ]
    formatted, _ = provider._format_messages(
        messages, include_thinking=True, model="claude-sonnet-4-5"
    )
    assistant_message = next(
        message for message in formatted if message["role"] == "assistant"
    )
    assert assistant_message["content"][0] == {
        "type": "thinking",
        "thinking": "reasoned",
        "signature": "SIG",
    }


def test_format_messages_empty_reasoning_content_still_emits_block() -> None:
    provider = AnthropicProvider(
        request_options={"thinking": {"type": "enabled", "budget_tokens": 1024}}
    )
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="will read",
            reasoning_content="",
            provider_meta={"thinking_signature": "SIG"},
            model="claude-sonnet-4-5",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": '{"path":"a"}'}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out"),
    ]
    formatted, _ = provider._format_messages(
        messages, include_thinking=True, model="claude-sonnet-4-5"
    )
    assistant_message = next(
        message for message in formatted if message["role"] == "assistant"
    )
    assert assistant_message["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "SIG",
    }
    messages2 = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="will read",
            reasoning_content=None,
            provider_meta={"thinking_signature": "SIG2"},
            model="claude-sonnet-4-5",
            tool_calls=[{"id": "toolu_2", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_2", content="out"),
    ]
    formatted2, _ = provider._format_messages(
        messages2, include_thinking=True, model="claude-sonnet-4-5"
    )
    assistant_message2 = next(
        message for message in formatted2 if message["role"] == "assistant"
    )
    assert assistant_message2["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "SIG2",
    }


def test_format_messages_thinking_disabled_no_block() -> None:
    provider = AnthropicProvider()
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="will read",
            reasoning_content="reasoned",
            provider_meta={"thinking_signature": "SIG"},
            model="claude-sonnet-4-5",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out"),
    ]
    formatted, _ = provider._format_messages(
        messages, include_thinking=False, model="claude-sonnet-4-5"
    )
    for message in formatted:
        for block in message["content"]:
            assert block.get("type") not in ("thinking", "redacted_thinking")


def test_format_messages_model_mismatch_no_block() -> None:
    provider = AnthropicProvider(
        request_options={"thinking": {"type": "enabled", "budget_tokens": 1024}}
    )
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="will read",
            reasoning_content="reasoned",
            provider_meta={"thinking_signature": "SIG"},
            model="claude-3-5-sonnet-20241022",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out"),
    ]
    formatted, _ = provider._format_messages(
        messages, include_thinking=True, model="claude-sonnet-4-5"
    )
    for message in formatted:
        for block in message["content"]:
            assert block.get("type") not in ("thinking", "redacted_thinking")


def test_format_messages_last_assistant_only_gets_thinking_on_legacy_model() -> None:
    provider = AnthropicProvider(
        request_options={"thinking": {"type": "enabled", "budget_tokens": 1024}}
    )
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="first",
            reasoning_content="r1",
            provider_meta={"thinking_signature": "SIG1"},
            model="claude-3-7-sonnet-20250219",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out1"),
        Message(
            role="assistant",
            content="second",
            reasoning_content="r2",
            provider_meta={"thinking_signature": "SIG2"},
            model="claude-3-7-sonnet-20250219",
            tool_calls=[{"id": "toolu_2", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_2", content="out2"),
    ]
    formatted, _ = provider._format_messages(
        messages, include_thinking=True, model="claude-3-7-sonnet-20250219"
    )
    assistant_messages = [
        message for message in formatted if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 2
    first_assistant = assistant_messages[0]
    second_assistant = assistant_messages[1]
    assert all(
        block.get("type") not in ("thinking", "redacted_thinking")
        for block in first_assistant["content"]
    )
    assert second_assistant["content"][0] == {
        "type": "thinking",
        "thinking": "r2",
        "signature": "SIG2",
    }


def test_format_messages_all_assistants_get_thinking_on_preserving_model() -> None:
    provider = AnthropicProvider(
        request_options={"thinking": {"type": "enabled", "budget_tokens": 1024}}
    )
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="first",
            reasoning_content="r1",
            provider_meta={"thinking_signature": "SIG1"},
            model="claude-sonnet-4-5",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out1"),
        Message(
            role="assistant",
            content="second",
            reasoning_content="r2",
            provider_meta={"thinking_signature": "SIG2"},
            model="claude-sonnet-4-5",
            tool_calls=[{"id": "toolu_2", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_2", content="out2"),
    ]
    formatted, _ = provider._format_messages(
        messages, include_thinking=True, model="claude-sonnet-4-5"
    )
    assistant_messages = [
        message for message in formatted if message["role"] == "assistant"
    ]
    assert len(assistant_messages) == 2
    assert assistant_messages[0]["content"][0] == {
        "type": "thinking",
        "thinking": "r1",
        "signature": "SIG1",
    }
    assert assistant_messages[1]["content"][0] == {
        "type": "thinking",
        "thinking": "r2",
        "signature": "SIG2",
    }


def test_format_messages_rehydrates_redacted_thinking() -> None:
    provider = AnthropicProvider(
        request_options={"thinking": {"type": "enabled", "budget_tokens": 1024}}
    )
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content="x",
            provider_meta={"redacted_thinking": ["secret"]},
            model="claude-sonnet-4-5",
            tool_calls=[{"id": "toolu_1", "name": "read", "arguments": "{}"}],
        ),
        Message(role="tool", tool_call_id="toolu_1", content="out"),
    ]
    formatted, _ = provider._format_messages(
        messages, include_thinking=True, model="claude-sonnet-4-5"
    )
    assistant_message = next(
        message for message in formatted if message["role"] == "assistant"
    )
    assert assistant_message["content"][0] == {
        "type": "redacted_thinking",
        "data": "secret",
    }


# ---------------------------------------------------------------------------
# F. Non-streaming chat: parsing, usage, errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_non_streaming_mixed_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "hi "},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read",
                "input": {"path": "a"},
            },
            {"type": "text", "text": "there"},
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 1,
        },
    }
    response = _FakeResponse(status=200, json_data=body, text_data=json.dumps(body))
    session, _ = _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert isinstance(result, Done)
    assert result.content == "hi there"
    assert len(result.tool_calls or []) == 1
    assert (result.tool_calls or [])[0].id == "toolu_1"
    assert (result.tool_calls or [])[0].name == "read"
    assert json.loads((result.tool_calls or [])[0].arguments) == {"path": "a"}
    assert isinstance((result.tool_calls or [])[0].arguments, str)
    assert result.tokens_input == 10 + 2 + 1
    assert result.tokens_output == 5
    assert session.calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert "Accept" not in (session.calls[0]["headers"] or {})
    assert session.calls[0]["timeout"].total == 120


@pytest.mark.asyncio
async def test_chat_non_streaming_timeout_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(status=200, json_data={"content": [], "usage": {}})
    session, _ = _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k", timeout_seconds=5)
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert isinstance(result, Done)
    assert session.calls[0]["timeout"].total == 5


@pytest.mark.asyncio
async def test_chat_non_streaming_content_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicProvider(api_key="k")
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": "not-a-list",
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    _install_fake(monkeypatch, _FakeResponse(status=200, json_data=body))
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert isinstance(result, Done)
    assert result.content == ""
    assert result.tool_calls == []

    body2 = {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "content": [
            "nope",
            {"type": "unknown_block"},
            {"type": "tool_use", "id": "t", "name": "n", "input": "nope"},
            {"type": "redacted_thinking", "data": "D"},
            {"type": "thinking", "thinking": "r"},
        ],
    }
    _install_fake(monkeypatch, _FakeResponse(status=200, json_data=body2))
    result2 = await provider.chat(
        [Message(role="user", content="hi")], model="claude-sonnet-4-5"
    )
    assert isinstance(result2, Done)
    assert result2.content == ""
    assert len(result2.tool_calls or []) == 1
    assert (result2.tool_calls or [])[0].arguments == "{}"
    assert result2.provider_meta == {"redacted_thinking": ["D"]}
    assert result2.tokens_input is None
    assert result2.tokens_output is None


@pytest.mark.asyncio
async def test_chat_non_streaming_usage_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicProvider(api_key="k")
    body = {
        "content": [{"type": "text", "text": "hi"}],
        "usage": "nope",
    }
    _install_fake(monkeypatch, _FakeResponse(status=200, json_data=body))
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert isinstance(result, Done)
    assert result.tokens_input is None
    assert result.tokens_output is None


@pytest.mark.asyncio
async def test_chat_non_streaming_provider_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "reasoned", "signature": "SIG_NONSTREAM"},
            {"type": "text", "text": "done"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read",
                "input": {"path": "a"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    response = _FakeResponse(status=200, json_data=body, text_data=json.dumps(body))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-sonnet-4-5"
    )
    assert isinstance(result, Done)
    assert result.provider_meta is not None
    assert result.provider_meta["thinking_signature"] == "SIG_NONSTREAM"
    body2 = {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    response2 = _FakeResponse(status=200, json_data=body2, text_data=json.dumps(body2))
    _install_fake(monkeypatch, response2)
    result2 = await provider.chat(
        [Message(role="user", content="hi")], model="claude-sonnet-4-5"
    )
    assert isinstance(result2, Done)
    assert result2.provider_meta is None


@pytest.mark.asyncio
async def test_chat_cache_read_tokens_parsed_non_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 4,
            "output_tokens": 2,
            "cache_read_input_tokens": 9000,
            "cache_creation_input_tokens": 100,
        },
    }
    _install_fake(
        monkeypatch,
        _FakeResponse(status=200, json_data=body, text_data=json.dumps(body)),
    )
    provider = AnthropicProvider(api_key="k")
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-opus-4-8"
    )
    assert isinstance(result, Done)
    assert result.cache_read_tokens == 9000
    assert result.tokens_input == 4 + 9000 + 100


@pytest.mark.asyncio
async def test_chat_cache_read_absent_is_none_non_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }
    _install_fake(
        monkeypatch,
        _FakeResponse(status=200, json_data=body, text_data=json.dumps(body)),
    )
    provider = AnthropicProvider(api_key="k")
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-opus-4-8"
    )
    assert isinstance(result, Done)
    assert result.cache_read_tokens is None
    assert result.tokens_input == 4


@pytest.mark.asyncio
async def test_chat_non_streaming_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error_body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "bad"},
    }
    response = _FakeResponse(
        status=400, json_data=error_body, text_data=json.dumps(error_body)
    )
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert isinstance(result, Error)
    assert "400" in result.message
    assert "bad" in result.message
    assert not isinstance(result, Done)


@pytest.mark.asyncio
async def test_chat_non_streaming_non_json_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _FakeResponse(
        status=200,
        json_data=None,
        text_data="<html>gateway page</html>",
        content_type="text/html",
    )
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert isinstance(result, Error)
    assert result.message == "unexpected response from API"


@pytest.mark.asyncio
async def test_chat_transport_exception_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _RaisingSession(_FakeResponse(status=200, json_data={}))
    connector = _FakeConnector()
    monkeypatch.setattr(
        "novakit.llm.providers.anthropic.aiohttp.ClientSession",
        lambda *args, **kwargs: session,
    )
    monkeypatch.setattr(
        "novakit.llm.providers.anthropic.aiohttp.TCPConnector",
        lambda *args, **kwargs: connector,
    )
    provider = AnthropicProvider(api_key="k")
    result = await provider.chat(
        [Message(role="user", content="hi")], model="claude-3-opus-20240229"
    )
    assert isinstance(result, Error)
    assert "boom" in result.message


@pytest.mark.asyncio
async def test_chat_abort_event(monkeypatch: pytest.MonkeyPatch) -> None:
    abort = asyncio.Event()
    abort.set()
    response = _FakeResponse(status=200, json_data={"content": [], "usage": {}})
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    result = await provider.chat(
        [Message(role="user", content="hi")],
        model="claude-3-5-sonnet-20241022",
        abort_event=abort,
    )
    assert isinstance(result, Done)
    assert result.aborted is True


# ---------------------------------------------------------------------------
# G. Streaming chat_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_text_canonical(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 5, "output_tokens": 0}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    assert isinstance(collected[0], TextDelta) and collected[0].content == "Hello "
    assert isinstance(collected[1], TextDelta) and collected[1].content == "world"
    assert isinstance(collected[2], Done)
    assert collected[2].content == "Hello world"
    assert collected[2].tokens_input == 5
    assert collected[2].tokens_output == 7
    assert len([event for event in collected if isinstance(event, TextDelta)]) == 2


@pytest.mark.asyncio
async def test_chat_stream_text_in_start_block(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "Hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    assert isinstance(collected[0], TextDelta) and collected[0].content == "Hi"
    assert isinstance(collected[-1], Done)
    assert collected[-1].content == "Hi"


@pytest.mark.asyncio
async def test_chat_stream_garbage_and_blank_lines_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _sse_lines(
        [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
            {"type": "message_stop"},
        ]
    )
    lines.insert(0, b"not json at all\n")
    lines.insert(1, b"\n")
    lines.insert(2, b"data: [1, 2]\n")
    lines.insert(3, b"\n")
    response = _FakeResponse(status=200, sse_lines=lines)
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    texts = [event.content for event in collected if isinstance(event, TextDelta)]
    assert texts == ["hi"]
    assert isinstance(collected[-1], Done)


@pytest.mark.asyncio
async def test_chat_stream_handles_oversized_sse_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    big = "x" * 200_000
    events = [
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "message_delta", "usage": {"output_tokens": 3}, "padding": big},
        {"type": "message_stop"},
    ]
    lines = _sse_lines(events)
    assert any(len(line) > 131072 for line in lines)

    response = _FakeResponse(status=200, sse_lines=lines)
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")

    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]

    assert not any(isinstance(event, Error) for event in collected), collected
    assert any(
        isinstance(event, TextDelta) and event.content == "hi" for event in collected
    )
    assert isinstance(collected[-1], Done)


@pytest.mark.asyncio
async def test_chat_stream_tool_use_chunked_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"pa'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": 'th": "a.txt"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    tool_calls = [event for event in collected if isinstance(event, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "toolu_1"
    assert tool_calls[0].name == "read"
    assert tool_calls[0].arguments == '{"path": "a.txt"}'
    input_events = [
        type(event).__name__
        for event in collected
        if isinstance(event, (ToolInputStart, ToolInputDelta, ToolInputEnd, ToolCall))
    ]
    assert input_events == [
        "ToolInputStart",
        "ToolInputDelta",
        "ToolInputDelta",
        "ToolInputEnd",
        "ToolCall",
    ]
    deltas = "".join(
        event.delta for event in collected if isinstance(event, ToolInputDelta)
    )
    assert deltas == '{"path": "a.txt"}'
    done = collected[-1]
    assert isinstance(done, Done)
    assert len(done.tool_calls or []) == 1
    assert (done.tool_calls or [])[0].arguments == '{"path": "a.txt"}'


@pytest.mark.asyncio
async def test_chat_stream_tool_use_initial_input_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_9",
                "name": "read",
                "input": {"path": "a"},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    tool_calls = [event for event in collected if isinstance(event, ToolCall)]
    assert len(tool_calls) == 1
    assert json.loads(tool_calls[0].arguments) == {"path": "a"}
    assert isinstance(collected[-1], Done)


@pytest.mark.asyncio
async def test_chat_stream_tool_use_empty_args_defaults_to_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "t", "name": "n"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    tool_calls = [event for event in collected if isinstance(event, ToolCall)]
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == "{}"


@pytest.mark.asyncio
async def test_chat_stream_reports_malformed_tool_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_bad",
                "name": "write",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"filePath": '},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    kinds = [type(event).__name__ for event in collected]
    assert "ToolInputError" in kinds
    assert any(isinstance(event, ToolInputError) for event in collected)
    assert "ToolCall" not in kinds
    done = collected[-1]
    assert isinstance(done, Done)
    assert done.tool_calls == []


@pytest.mark.asyncio
async def test_chat_stream_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": " yes"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig123"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    reasoning = [event for event in collected if isinstance(event, ReasoningDelta)]
    assert len(reasoning) == 2
    assert reasoning[0].content == "hmm"
    assert reasoning[1].content == " yes"
    assert not any(isinstance(event, TextDelta) for event in collected)
    assert isinstance(collected[-1], Error)


@pytest.mark.asyncio
async def test_chat_stream_error_event(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        },
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    assert len(collected) == 1
    assert isinstance(collected[0], Error)
    assert "overloaded_error" in collected[0].message
    assert not any(isinstance(event, Done) for event in collected)


@pytest.mark.asyncio
async def test_chat_stream_error_event_shape_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicProvider(api_key="k")
    events = [
        {"type": "error", "error": "boom"},
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    assert isinstance(collected[0], Error)
    assert collected[0].message == "error"

    events2 = [
        {"type": "error", "error": {"message": "just a message"}},
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events2)))
    collected2 = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    assert isinstance(collected2[0], Error)
    assert collected2[0].message == "error: just a message"


@pytest.mark.asyncio
async def test_chat_stream_ping_and_blank_lines_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello "},
        },
        {"type": "ping"},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "world"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    lines = _sse_lines(base_events)
    lines.insert(4, b"\n")
    lines.insert(4, b"\n")
    response = _FakeResponse(status=200, sse_lines=lines)
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    texts = [event.content for event in collected if isinstance(event, TextDelta)]
    assert texts == ["Hello ", "world"]
    assert isinstance(collected[-1], Done)
    assert collected[-1].content == "Hello world"


@pytest.mark.asyncio
async def test_chat_stream_abort_mid_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    abort = asyncio.Event()
    abort.set()
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")],
            model="claude-3-opus-20240229",
            abort_event=abort,
        )
    ]
    assert isinstance(collected[0], Done)
    assert collected[0].aborted is True


@pytest.mark.asyncio
async def test_chat_stream_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _FakeResponse(
        status=400, text_data="bad request", json_data=None, sse_lines=[]
    )
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    errors = [event for event in collected if isinstance(event, Error)]
    assert errors, collected
    assert isinstance(collected[-1], Error)
    assert "400" in collected[-1].message
    assert not any(isinstance(event, Done) for event in collected)
    assert not any(isinstance(event, RetryNotice) for event in collected)


@pytest.mark.asyncio
async def test_chat_stream_sends_accept_and_stream_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [{"type": "message_stop"}]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    session, _ = _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    _ = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-3-opus-20240229"
        )
    ]
    assert session.calls[0]["headers"]["Accept"] == "text/event-stream"
    assert session.calls[0]["json"]["stream"] is True
    assert session.calls[0]["timeout"].total is None


@pytest.mark.asyncio
async def test_chat_stream_total_timeout_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [{"type": "message_stop"}]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    session, _ = _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    _ = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")],
            model="claude-3-opus-20240229",
            total_timeout_seconds=30,
        )
    ]
    assert session.calls[0]["timeout"].total == 30


# ---------------------------------------------------------------------------
# H. Persisted thinking state (provider_meta)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_provider_meta_thinking_signature_concatenated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "reasoned"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "SIG_A"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "SIG_B"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path": "a"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-sonnet-4-5"
        )
    ]
    done = collected[-1]
    assert isinstance(done, Done)
    assert done.provider_meta is not None
    assert done.provider_meta["thinking_signature"] == "SIG_ASIG_B"


@pytest.mark.asyncio
async def test_chat_stream_provider_meta_none_when_no_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-sonnet-4-5"
        )
    ]
    done = collected[-1]
    assert isinstance(done, Done)
    assert done.provider_meta is None


@pytest.mark.asyncio
async def test_chat_stream_provider_meta_redacted_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "<data>"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    response = _FakeResponse(status=200, sse_lines=_sse_lines(events))
    _install_fake(monkeypatch, response)
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-sonnet-4-5"
        )
    ]
    final = collected[-1]
    assert isinstance(final, Error)
    assert final.provider_meta is not None
    assert final.provider_meta["redacted_thinking"] == ["<data>"]


@pytest.mark.asyncio
async def test_chat_stream_cache_read_tokens_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 1234,
                    "cache_creation_input_tokens": 10,
                }
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    done = collected[-1]
    assert isinstance(done, Done)
    assert done.cache_read_tokens == 1234
    assert done.tokens_input == 5 + 1234 + 10


@pytest.mark.asyncio
async def test_chat_stream_message_start_shape_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicProvider(api_key="k")
    events = [
        {"type": "message_start", "message": "not-a-dict"},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": "not-a-dict", "usage": "not-a-dict"},
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    done = collected[-1]
    assert isinstance(done, Done)
    assert done.content == "hi"
    # A non-dict message yields an empty usage dict, which still sums to 0.
    assert done.tokens_input == 0

    events2 = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 7, "output_tokens": "bad"}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "usage": {"output_tokens": "bad"}},
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events2)))
    collected2 = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    done2 = collected2[-1]
    assert isinstance(done2, Done)
    assert done2.tokens_input == 7
    assert done2.tokens_output is None


@pytest.mark.asyncio
async def test_chat_stream_message_start_usage_not_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "message_start", "message": {"usage": "nope"}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    done = collected[-1]
    assert isinstance(done, Done)
    assert done.tokens_input is None


@pytest.mark.asyncio
async def test_chat_stream_max_tokens_without_answer_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "reasoning..."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 64000},
        },
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    assert isinstance(collected[-1], Error)
    assert "stop_reason=max_tokens" in collected[-1].message


@pytest.mark.asyncio
async def test_chat_stream_max_tokens_with_partial_answer_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "partial"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    done = collected[-1]
    assert isinstance(done, Done)
    assert done.content == "partial"


# ---------------------------------------------------------------------------
# I. Truncated streams: EOF before message_stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_eof_before_message_stop_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "I should fetch that."},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "abc"},
        },
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    final = collected[-1]
    assert isinstance(final, Error)
    assert "message_stop" in final.message
    assert "No answer text had arrived" in final.message


@pytest.mark.asyncio
async def test_chat_stream_truncated_keeps_the_recovered_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "half an ans"},
        },
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    final = collected[-1]
    assert isinstance(final, Error)
    assert final.content == "half an ans"
    assert "11 chars of answer text were recovered" in final.message


@pytest.mark.asyncio
async def test_chat_stream_truncated_keeps_the_recovered_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "web_fetch"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"url": "https://x/"}',
            },
        },
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    final = collected[-1]
    assert isinstance(final, Error)
    assert [call.name for call in final.tool_calls] == ["web_fetch"]


@pytest.mark.asyncio
async def test_chat_stream_complete_stream_is_still_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "full answer"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    final = collected[-1]
    assert isinstance(final, Done)
    assert final.content == "full answer"


@pytest.mark.asyncio
async def test_chat_stream_end_turn_after_thinking_with_no_answer_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "I'll handle the research myself using web_fetch.",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "abc"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 229},
        },
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    final = collected[-1]
    assert isinstance(final, Error)
    assert "stop_reason=end_turn" in final.message
    assert "without answering or calling a tool" in final.message


@pytest.mark.asyncio
async def test_chat_stream_end_turn_with_no_thinking_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 4},
        },
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    assert isinstance(collected[-1], Error)


@pytest.mark.asyncio
async def test_chat_stream_refusal_with_text_is_still_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "I can't help with that."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "refusal"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]
    _install_fake(monkeypatch, _FakeResponse(status=200, sse_lines=_sse_lines(events)))
    provider = AnthropicProvider(api_key="k")
    collected = [
        event
        async for event in provider.chat_stream(
            [Message(role="user", content="hi")], model="claude-opus-4-8"
        )
    ]
    final = collected[-1]
    assert isinstance(final, Done)
    assert final.content == "I can't help with that."


# ---------------------------------------------------------------------------
# J. Parser edge cases, driven directly (no transport)
# ---------------------------------------------------------------------------


def test_parser_ignores_malformed_blocks_and_deltas() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    assert list(parser.feed({"type": "content_block_start", "index": 0}, acc)) == []
    assert (
        list(
            parser.feed(
                {"type": "content_block_start", "index": 0, "content_block": "x"}, acc
            )
        )
        == []
    )
    assert list(parser.feed({"type": "content_block_delta", "index": 0}, acc)) == []
    assert (
        list(
            parser.feed({"type": "content_block_delta", "index": 0, "delta": "x"}, acc)
        )
        == []
    )
    assert list(parser.feed({"type": "content_block_stop", "index": "bad"}, acc)) == []


def test_parser_empty_text_and_thinking_deltas_yield_nothing() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": ""},
                },
                acc,
            )
        )
        == []
    )
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "thinking_delta", "thinking": ""},
                },
                acc,
            )
        )
        == []
    )
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "signature_delta", "signature": ""},
                },
                acc,
            )
        )
        == []
    )


def test_parser_thinking_and_signature_deltas_create_state() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    out = list(
        parser.feed(
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "thinking_delta", "thinking": "t"},
            },
            acc,
        )
    )
    assert len(out) == 1 and isinstance(out[0], ReasoningDelta)
    out2 = list(
        parser.feed(
            {
                "type": "content_block_delta",
                "index": 3,
                "delta": {"type": "signature_delta", "signature": "s"},
            },
            acc,
        )
    )
    assert out2 == []
    assert parser._thinking_state[3]["signature"] == "s"


def test_parser_tool_use_without_name_stays_silent() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use"},
                },
                acc,
            )
        )
        == []
    )
    assert parser._tool_calls[0]["id"] == "call_0"
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": ""},
                },
                acc,
            )
        )
        == []
    )
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"a":1}'},
                },
                acc,
            )
        )
        == []
    )
    assert list(parser.feed({"type": "content_block_stop", "index": 0}, acc)) == []
    assert list(parser.on_eof(acc)) != []


def test_parser_tool_use_input_object_initial_input() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "t",
                        "name": "n",
                        "input": {"a": 1},
                    },
                },
                acc,
            )
        )
        != []
    )
    assert parser._tool_calls[0]["initial_input"] == '{"a": 1}'


def test_parser_tool_use_input_non_dict_ignored() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    list(
        parser.feed(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "t",
                    "name": "n",
                    "input": "nope",
                },
            },
            acc,
        )
    )
    assert "initial_input" not in parser._tool_calls[0]


def test_parser_tool_use_unserialisable_input_ignored() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    list(
        parser.feed(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "t",
                    "name": "n",
                    "input": {"bad": object()},
                },
            },
            acc,
        )
    )
    assert "initial_input" not in parser._tool_calls[0]


def test_parser_input_json_delta_creates_implicit_entry() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_delta",
                    "index": 5,
                    "delta": {"type": "input_json_delta", "partial_json": '{"a":1}'},
                },
                acc,
            )
        )
        == []
    )
    assert parser._tool_calls[5]["id"] == "call_5"
    assert parser._tool_calls[5]["arguments"] == '{"a":1}'


def test_parser_on_eof_when_finished_delegates_to_build_done() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    parser.finished = True
    terminal = list(parser.on_eof(acc))
    assert len(terminal) == 1 and isinstance(terminal[0], Error)
    assert "stop_reason=end_turn" in terminal[0].message


def test_parser_tool_call_cap() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    for index in range(64):
        list(
            parser.feed(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": f"t{index}"},
                },
                acc,
            )
        )
    assert len(parser._tool_calls) == 64
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_start",
                    "index": 64,
                    "content_block": {"type": "tool_use", "id": "overflow"},
                },
                acc,
            )
        )
        == []
    )
    assert len(parser._tool_calls) == 64
    assert (
        list(
            parser.feed(
                {
                    "type": "content_block_delta",
                    "index": 99,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                },
                acc,
            )
        )
        == []
    )
    assert len(parser._tool_calls) == 64


def test_parser_repeated_stop_yields_tool_once() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    list(
        parser.feed(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "t", "name": "n"},
            },
            acc,
        )
    )
    first = list(parser.feed({"type": "content_block_stop", "index": 0}, acc))
    assert [type(event).__name__ for event in first] == ["ToolInputEnd", "ToolCall"]
    assert list(parser.feed({"type": "content_block_stop", "index": 0}, acc)) == []


def test_parser_on_eof_excludes_invalid_tool_calls() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    list(
        parser.feed(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "bad", "name": "n"},
            },
            acc,
        )
    )
    list(
        parser.feed(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{oops"},
            },
            acc,
        )
    )
    list(parser.feed({"type": "content_block_stop", "index": 0}, acc))
    assert parser._input_error is True
    terminal = list(parser.on_eof(acc))
    assert len(terminal) == 1 and isinstance(terminal[0], Error)
    assert terminal[0].tool_calls == []


def test_parser_build_done_max_tokens_keeps_provider_meta() -> None:
    parser = _AnthropicStreamParser()
    acc = _acc()
    list(
        parser.feed(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "r",
                    "signature": "SIG",
                },
            },
            acc,
        )
    )
    list(parser.feed({"type": "content_block_stop", "index": 0}, acc))
    list(
        parser.feed(
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}, acc
        )
    )
    parser.finished = True
    terminal = parser.build_done(acc)
    assert isinstance(terminal, Error)
    assert "stop_reason=max_tokens" in terminal.message
    assert terminal.provider_meta == {"thinking_signature": "SIG"}


# ---------------------------------------------------------------------------
# K. Interface conformance and token counting
# ---------------------------------------------------------------------------


def test_provider_is_llm_provider() -> None:
    provider = AnthropicProvider()
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_count_tokens_cjk_denser() -> None:
    provider = AnthropicProvider()
    ascii_count = await provider.count_tokens(
        "hello world hello world", model="claude-3-opus-20240229"
    )
    cjk_count = await provider.count_tokens(
        "你好世界你好世界你好世界", model="claude-3-opus-20240229"
    )
    assert ascii_count > 0
    assert cjk_count > 0
    assert cjk_count > ascii_count


@pytest.mark.asyncio
async def test_count_tokens_empty_is_zero() -> None:
    provider = AnthropicProvider()
    assert await provider.count_tokens("", model="claude-3-opus-20240229") == 0
