"""Offline coverage for the Ollama provider.

Everything here runs without a network: the stream parser and the request
shaping helpers are exercised directly, and ``chat``/``chat_stream`` run
against fake sessions that replay canned NDJSON lines.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from novakit.llm.accumulator import StreamAccumulator
from novakit.llm.framing import NDJSONFramer
from novakit.llm.provider import (
    STREAM_IDLE_TIMEOUT_SECONDS,
    Done,
    Error,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolInputEnd,
    ToolInputStart,
)
from novakit.llm.providers.ollama import OllamaProvider

_AIOHTTP_DEFAULT_LINE_LIMIT = 131072


def _acc() -> StreamAccumulator:
    return StreamAccumulator("llama3", None, None)


def _provider(**kwargs: Any) -> OllamaProvider:
    return OllamaProvider(base_url="http://localhost:11434", **kwargs)


def _line(payload: dict[str, Any]) -> bytes:
    return f"{json.dumps(payload)}\n".encode()


class _FakeConnector:
    """Stands in for ``aiohttp.TCPConnector`` without touching sockets."""

    def __init__(self) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        self._closed = True


class _FakeContent:
    """Replays canned lines, honouring the caller's line limit."""

    def __init__(
        self,
        lines: list[bytes],
        error: BaseException | None = None,
    ) -> None:
        self._lines = list(lines)
        self._error = error

    async def readline(self, max_line_length: int | None = None) -> bytes:
        if self._error is not None:
            raise self._error
        if not self._lines:
            return b""
        line = self._lines.pop(0)
        if max_line_length is not None and len(line) > max_line_length:
            raise ValueError("line too long")
        return line


class _FakeResponse:
    """Stands in for ``aiohttp.ClientResponse`` for chat and streams."""

    def __init__(
        self,
        status: int = 200,
        lines: list[bytes] | None = None,
        payload: dict[str, Any] | None = None,
        text_body: str = "",
        read_error: BaseException | None = None,
    ) -> None:
        self.status = status
        self.headers: dict[str, str] = {}
        self.content = _FakeContent(lines or [], read_error)
        self._payload = payload or {}
        self._text_body = text_body
        self.content_type = "application/json"
        self.closed = False

    async def text(self) -> str:
        return self._text_body

    async def json(self) -> dict[str, Any]:
        return self._payload

    def close(self) -> None:
        self.closed = True

    async def release(self) -> None:
        return None

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False


class _FakeSession:
    """Records posts and always answers with the canned response."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def post(
        self,
        url: str,
        timeout: Any = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> _FakeResponse:
        self.calls.append(
            {"url": url, "timeout": timeout, "json": json, "headers": headers}
        )
        return self._response

    async def close(self) -> None:
        self.closed = True


def _install(
    provider: OllamaProvider,
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeResponse,
) -> _FakeSession:
    session = _FakeSession(response)
    connector = _FakeConnector()
    monkeypatch.setattr(provider, "_open_session", lambda: (connector, session))
    return session


async def _collect(provider: OllamaProvider, *args: Any, **kwargs: Any) -> list[Any]:
    return [event async for event in provider.chat_stream(*args, **kwargs)]


# -- stream parser: text and reasoning ----------------------------------------


def test_feed_plain_text_yields_delta_and_accumulates() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()

    events = list(parser.feed({"message": {"content": "hi"}, "done": False}, acc))

    assert events == [TextDelta(content="hi")]
    assert acc.content == "hi"


def test_feed_reasoning_content_yields_reasoning_delta() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()

    events = list(
        parser.feed({"message": {"reasoning_content": "plan"}, "done": False}, acc)
    )

    assert events == [ReasoningDelta(content="plan")]
    assert acc.content == ""


def test_feed_thinking_falls_back_when_reasoning_content_missing() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()

    events = list(parser.feed({"message": {"thinking": "hmm"}, "done": False}, acc))

    assert events == [ReasoningDelta(content="hmm")]


def test_feed_empty_message_yields_nothing() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()

    assert list(parser.feed({"message": {}}, acc)) == []


def test_feed_done_without_tools_or_text_yields_nothing() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()

    assert list(parser.feed({"done": True}, acc)) == []


# -- stream parser: tool calls -------------------------------------------------


def test_feed_tool_call_start_and_call() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()
    event = {
        "message": {
            "content": "",
            "tool_calls": [
                {"index": 0, "function": {"name": "read", "arguments": '{"p": 1}'}}
            ],
        },
        "done": False,
    }

    events = list(parser.feed(event, acc))

    assert events == [
        ToolInputStart(id="call_0", name="read"),
        ToolCall(id="call_0", name="read", arguments='{"p": 1}'),
    ]


def test_feed_tool_call_dict_arguments_are_dumped() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()
    event = {
        "message": {
            "tool_calls": [
                {"index": 0, "function": {"name": "f", "arguments": {"p": 1}}}
            ]
        },
        "done": False,
    }

    events = list(parser.feed(event, acc))

    call = next(e for e in events if isinstance(e, ToolCall))
    assert json.loads(call.arguments) == {"p": 1}


def test_feed_tool_call_invalid_json_arguments_kept_raw() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()
    event = {
        "message": {
            "tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{bad"}}]
        },
        "done": False,
    }

    events = list(parser.feed(event, acc))

    call = next(e for e in events if isinstance(e, ToolCall))
    assert call.arguments == "{bad"


def test_feed_tool_call_non_object_arguments_replace_state() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()
    first = {
        "message": {
            "tool_calls": [
                {"index": 0, "function": {"name": "f", "arguments": '{"a": 1}'}}
            ]
        },
        "done": False,
    }
    second = {
        "message": {"tool_calls": [{"index": 0, "function": {"arguments": 42}}]},
        "done": False,
    }

    list(parser.feed(first, acc))
    events = list(parser.feed(second, acc))

    assert [type(e) for e in events] == [ToolCall]
    assert events[0].arguments == "42"


@pytest.mark.xfail(
    reason="Ollama parser references args before assignment after corrupt state",
    strict=True,
)
def test_feed_tool_call_recovers_after_corrupt_stored_arguments() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()
    first = {
        "message": {
            "tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{bad"}}]
        },
        "done": False,
    }
    second = {
        "message": {
            "tool_calls": [{"index": 0, "function": {"arguments": '{"a": 1}'}}]
        },
        "done": False,
    }

    list(parser.feed(first, acc))
    events = list(parser.feed(second, acc))

    assert [type(event) for event in events] == [ToolCall]
    assert json.loads(events[0].arguments) == {"a": 1}


def test_feed_same_index_merges_arguments_and_starts_once() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()
    first = {
        "message": {
            "tool_calls": [
                {"index": 0, "function": {"name": "f", "arguments": '{"a": 1}'}}
            ]
        },
        "done": False,
    }
    second = {
        "message": {
            "tool_calls": [{"index": 0, "function": {"arguments": '{"b": 2}'}}]
        },
        "done": False,
    }

    first_events = list(parser.feed(first, acc))
    second_events = list(parser.feed(second, acc))

    assert sum(isinstance(e, ToolInputStart) for e in first_events) == 1
    assert not any(isinstance(e, ToolInputStart) for e in second_events)
    call = next(e for e in second_events if isinstance(e, ToolCall))
    assert json.loads(call.arguments) == {"a": 1, "b": 2}


def test_feed_tool_call_without_name_yields_nothing() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()
    event = {
        "message": {"tool_calls": [{"index": 2, "function": {}}]},
        "done": False,
    }

    assert list(parser.feed(event, acc)) == []
    assert parser.build_done(acc).tool_calls == []


def test_feed_done_ends_each_started_tool_once() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()
    tool_chunk = {
        "message": {
            "tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{}"}}]
        },
        "done": False,
    }

    list(parser.feed(tool_chunk, acc))
    first = list(parser.feed({"message": {"content": ""}, "done": True}, acc))
    second = list(parser.feed({"done": True}, acc))

    assert first == [ToolInputEnd(id="call_0", name="f")]
    assert second == []


def test_build_done_sorts_calls_and_skips_unnamed() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()
    chunks = [
        {
            "message": {
                "content": "x",
                "tool_calls": [{"index": 1, "function": {"name": "b"}}],
            }
        },
        {"message": {"tool_calls": [{"index": 0, "function": {"name": "a"}}]}},
        {"message": {"tool_calls": [{"index": 2, "function": {}}]}},
    ]
    for chunk in chunks:
        list(parser.feed(chunk, acc))

    done = parser.build_done(acc)

    assert isinstance(done, Done)
    assert done.content == "x"
    assert [(c.id, c.name) for c in (done.tool_calls or [])] == [
        ("call_0", "a"),
        ("call_1", "b"),
    ]


def test_build_done_without_tools_carries_content() -> None:
    parser = _provider()._new_parser("llama3")
    acc = _acc()

    list(parser.feed({"message": {"content": "hi"}, "done": False}, acc))
    done = parser.build_done(acc)

    assert done == Done(content="hi", tool_calls=[])


# -- request body ---------------------------------------------------------------


def test_build_body_defaults_to_stream_flag_and_no_tools() -> None:
    provider = _provider()

    body = provider._build_body(messages=[], model="m", stream=False, tools=None)

    assert body == {"model": "m", "messages": [], "stream": False}


def test_build_body_attaches_tools_when_given() -> None:
    provider = _provider()
    tools = [{"type": "function", "function": {"name": "f"}}]

    body = provider._build_body(messages=[], model="m", stream=True, tools=tools)

    assert body["tools"] == tools


def test_build_body_config_tools_false_forces_empty_list() -> None:
    provider = _provider(request_options={"tools": False})
    tools = [{"type": "function", "function": {"name": "f"}}]

    body = provider._build_body(messages=[], model="m", stream=False, tools=tools)

    assert body["tools"] == []


def test_build_body_merges_extra_options() -> None:
    provider = _provider(request_options={"temperature": 0.5})

    body = provider._build_body(messages=[], model="m", stream=False, tools=None)

    assert body["temperature"] == 0.5
    assert "tools" not in body


# -- message formatting ----------------------------------------------------------


def test_format_messages_user_and_assistant_reasoning() -> None:
    provider = _provider()

    formatted = provider._format_messages(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="ok", reasoning_content="why"),
            Message(role="assistant", content="plain"),
        ]
    )

    assert formatted[0] == {"role": "user", "content": "hi"}
    assert formatted[1]["reasoning_content"] == "why"
    assert "reasoning_content" not in formatted[2]


def test_format_messages_images_force_user_role() -> None:
    provider = _provider()

    formatted = provider._format_messages(
        [Message(role="assistant", content="see", images=["img"])]
    )

    assert formatted[0]["role"] == "user"
    assert formatted[0]["images"] == ["img"]


def test_format_messages_keeps_tool_call_id() -> None:
    provider = _provider()

    formatted = provider._format_messages(
        [Message(role="tool", content="out", tool_call_id="call_0")]
    )

    assert formatted[0]["tool_call_id"] == "call_0"


def test_format_messages_dict_tool_call_parses_json_arguments() -> None:
    provider = _provider()
    msg = Message(
        role="assistant",
        content="",
        tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": '{"p": 1}'}}],
    )

    formatted = provider._format_messages([msg])

    assert formatted[0]["tool_calls"] == [
        {"id": "c1", "function": {"name": "f", "arguments": {"p": 1}}}
    ]


def test_format_messages_dict_tool_call_keeps_bad_json_arguments() -> None:
    provider = _provider()
    msg = Message(
        role="assistant",
        content="",
        tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": "{bad"}}],
    )

    formatted = provider._format_messages([msg])

    assert formatted[0]["tool_calls"][0]["function"]["arguments"] == "{bad"


def test_format_messages_dict_tool_call_keeps_object_arguments() -> None:
    provider = _provider()
    msg = Message(
        role="assistant",
        content="",
        tool_calls=[{"id": "c1", "function": {"name": "f", "arguments": {"p": 1}}}],
    )

    formatted = provider._format_messages([msg])

    assert formatted[0]["tool_calls"][0]["function"]["arguments"] == {"p": 1}


def test_format_messages_model_dump_tool_call() -> None:
    provider = _provider()
    msg = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(id="c1", name="f", arguments='{"p": 1}'),
            ToolCall(id="c2", name="g", arguments="{bad"),
            "raw",
        ],
    )

    formatted = provider._format_messages([msg])

    assert formatted[0]["tool_calls"][0] == {
        "id": "c1",
        "function": {"name": "f", "arguments": {"p": 1}},
    }
    assert formatted[0]["tool_calls"][1]["function"]["arguments"] == "{bad"
    assert formatted[0]["tool_calls"][2] == "raw"


def test_format_messages_dict_passthrough_and_unknown_skipped() -> None:
    provider = _provider()
    raw: dict[str, Any] = {"role": "user", "content": "hi"}

    formatted = provider._format_messages([raw, "oops"])  # type: ignore[list-item]

    assert formatted == [raw]


# -- request preparation and response parsing ------------------------------------


def test_prepare_request_strips_slash_and_sets_agent_header() -> None:
    provider = OllamaProvider(base_url="http://localhost:11434/")

    url, headers, body = provider._prepare_request(
        [{"role": "user", "content": "hi"}], "llama3", False, None, None, None
    )

    assert url == "http://localhost:11434/api/chat"
    assert headers == {"User-Agent": provider._user_agent}
    assert provider._user_agent.startswith("novakit/")
    assert body["model"] == "llama3"
    assert body["stream"] is False


def test_init_defaults_and_timeout() -> None:
    provider = OllamaProvider()

    assert provider.base_url == ""
    assert provider.request_options == {}
    assert provider.timeout_seconds == 120
    assert provider.trace_label == "ollama"
    assert isinstance(provider.framer, NDJSONFramer)
    assert provider.stream_policy.line_limit == 16 * 1024 * 1024
    assert provider.transport_policy.use_retry is False


def test_parse_response_text_only() -> None:
    provider = _provider()

    done = provider._parse_response({"message": {"content": "hi"}})

    assert done == Done(content="hi", tool_calls=[])


def test_parse_response_tool_calls_with_str_and_dict_arguments() -> None:
    provider = _provider()

    done = provider._parse_response(
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "abc",
                        "function": {"name": "f", "arguments": '{"p": 1}'},
                    },
                    {"function": {"name": "g", "arguments": {"q": 2}}},
                ],
            }
        }
    )

    assert done.tool_calls is not None
    assert done.tool_calls[0] == ToolCall(id="abc", name="f", arguments='{"p": 1}')
    assert done.tool_calls[1].id == "call_1"
    assert json.loads(done.tool_calls[1].arguments) == {"q": 2}


def test_parse_response_tool_call_defaults() -> None:
    provider = _provider()

    done = provider._parse_response({"message": {"content": "", "tool_calls": [{}]}})

    assert done.tool_calls == [ToolCall(id="call_0", name="", arguments="{}")]


@pytest.mark.asyncio
async def test_count_tokens_quarters_character_length() -> None:
    provider = _provider()

    assert await provider.count_tokens("abcdefgh") == 2
    assert await provider.count_tokens("", model="llama3") == 0


# -- non-streaming chat ------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_returns_done_and_posts_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(timeout_seconds=5)
    response = _FakeResponse(payload={"message": {"content": "hi", "tool_calls": []}})
    session = _install(provider, monkeypatch, response)

    result = await provider.chat([Message(role="user", content="hi")], model="llama3")

    assert result == Done(content="hi", tool_calls=[])
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "http://localhost:11434/api/chat"
    assert call["headers"] == {"User-Agent": provider._user_agent}
    assert call["timeout"].total == 5
    assert call["json"]["model"] == "llama3"
    assert call["json"]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_chat_http_error_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    _install(provider, monkeypatch, _FakeResponse(status=500, text_body="down"))

    result = await provider.chat([Message(role="user", content="hi")])

    assert isinstance(result, Error)
    assert "HTTP 500" in result.message


@pytest.mark.asyncio
async def test_chat_transport_exception_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    session = _install(provider, monkeypatch, _FakeResponse())

    async def _boom(*args: Any, **kwargs: Any) -> _FakeResponse:
        raise RuntimeError("unreachable")

    monkeypatch.setattr(session, "post", _boom)

    result = await provider.chat([Message(role="user", content="hi")])

    assert isinstance(result, Error)
    assert "unreachable" in result.message


@pytest.mark.asyncio
async def test_chat_aborted_before_response_returns_aborted_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    connector = _FakeConnector()
    session = _FakeSession(_FakeResponse())
    monkeypatch.setattr(provider, "_open_session", lambda: (connector, session))

    async def _no_response(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        yield None

    monkeypatch.setattr(provider, "_post_with_retry", _no_response)

    result = await provider.chat([Message(role="user", content="hi")])

    assert result == Done(content="", tool_calls=[], aborted=True)


# -- streaming chat ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_text_then_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    lines = [
        _line({"message": {"content": "hi"}, "done": False}),
        _line({"message": {"content": " there"}, "done": True}),
    ]
    session = _install(provider, monkeypatch, _FakeResponse(lines=lines))

    collected = await _collect(
        provider,
        [Message(role="user", content="hi")],
        model="llama3",
        total_timeout_seconds=10,
    )

    assert not any(isinstance(e, Error) for e in collected)
    assert [e.content for e in collected if isinstance(e, TextDelta)] == [
        "hi",
        " there",
    ]
    assert collected[-1] == Done(content="hi there", tool_calls=[])
    assert session.calls[0]["timeout"].total == 10
    assert session.calls[0]["timeout"].sock_read == STREAM_IDLE_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_chat_stream_reasoning_tool_calls_and_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    lines = [
        _line({"message": {"reasoning_content": "plan"}, "done": False}),
        _line(
            {
                "message": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"name": "read", "arguments": '{"p": 1}'},
                        }
                    ]
                },
                "done": False,
            }
        ),
        _line({"message": {"content": ""}, "done": True}),
    ]
    _install(provider, monkeypatch, _FakeResponse(lines=lines))

    collected = await _collect(provider, [Message(role="user", content="hi")])

    assert any(isinstance(e, ReasoningDelta) for e in collected)
    assert ToolInputStart(id="call_0", name="read") in collected
    assert ToolInputEnd(id="call_0", name="read") in collected
    done = collected[-1]
    assert isinstance(done, Done)
    assert done.tool_calls == [ToolCall(id="call_0", name="read", arguments='{"p": 1}')]


@pytest.mark.asyncio
async def test_chat_stream_skips_blank_garbage_and_non_dict_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    lines = [
        b"\n",
        b"not json\n",
        b"[1, 2]\n",
        _line({"message": {"content": "ok"}, "done": True}),
    ]
    _install(provider, monkeypatch, _FakeResponse(lines=lines))

    collected = await _collect(provider, [Message(role="user", content="hi")])

    assert collected[-1] == Done(content="ok", tool_calls=[])


@pytest.mark.asyncio
async def test_chat_stream_large_line_within_provider_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    big = "x" * 200_000
    lines = [
        _line({"message": {"content": "hi"}, "done": False}),
        _line({"message": {"content": " there"}, "done": False, "pad": big}),
        _line({"message": {"content": ""}, "done": True}),
    ]
    assert len(lines[1]) > _AIOHTTP_DEFAULT_LINE_LIMIT
    _install(provider, monkeypatch, _FakeResponse(lines=lines))

    collected = await _collect(provider, [Message(role="user", content="hi")])

    assert not any(isinstance(e, Error) for e in collected)
    assert collected[-1] == Done(content="hi there", tool_calls=[])


@pytest.mark.asyncio
async def test_chat_stream_ends_on_eof_without_trailing_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    lines = [b'{"message":{"content":"ok"},"done":true}']
    _install(provider, monkeypatch, _FakeResponse(lines=lines))

    collected = await _collect(provider, [Message(role="user", content="hi")])

    assert isinstance(collected[-1], Done)
    assert collected[-1].content == "ok"


@pytest.mark.asyncio
async def test_chat_stream_http_error_yields_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    _install(provider, monkeypatch, _FakeResponse(status=500, text_body="down"))

    collected = await _collect(provider, [Message(role="user", content="hi")])

    assert len(collected) == 1
    assert isinstance(collected[0], Error)
    assert "HTTP 500" in collected[0].message


@pytest.mark.asyncio
async def test_chat_stream_aborted_before_response_yields_aborted_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    connector = _FakeConnector()
    session = _FakeSession(_FakeResponse())
    monkeypatch.setattr(provider, "_open_session", lambda: (connector, session))

    async def _no_response(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        yield None

    monkeypatch.setattr(provider, "_post_with_retry", _no_response)

    collected = await _collect(provider, [Message(role="user", content="hi")])

    assert collected == [Done(content="", tool_calls=[], aborted=True)]


@pytest.mark.asyncio
async def test_chat_stream_read_error_yields_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    response = _FakeResponse(read_error=RuntimeError("boom"))
    _install(provider, monkeypatch, response)

    collected = await _collect(provider, [Message(role="user", content="hi")])

    assert len(collected) == 1
    assert isinstance(collected[0], Error)
    assert "boom" in collected[0].message


def test_stream_timeout_leaves_total_unset_by_default() -> None:
    provider = _provider()

    timeout = provider._stream_timeout(None)

    assert timeout.total is None
    assert timeout.sock_read == STREAM_IDLE_TIMEOUT_SECONDS
