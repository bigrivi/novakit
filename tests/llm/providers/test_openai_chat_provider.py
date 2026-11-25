"""Offline tests for the OpenAI Chat Completions provider (novakit).

Pure offline: aiohttp is faked at the session/connector boundary, hooks are
monkeypatched, and no test touches the network or the ``nova`` package.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

import novakit.llm.providers.openai_chat as openai_chat_mod
from novakit.llm.accumulator import StreamAccumulator
from novakit.llm.provider import (
    Done,
    Error,
    Message,
    ReasoningDelta,
    RetryNotice,
    ToolCall,
    ToolInputDelta,
    ToolInputError,
    ToolInputStart,
)
from novakit.llm.providers.openai_chat import OpenAIProvider, _OpenAIChatStreamParser


class _FakeConnector:
    """Records nothing; only models open/close lifecycle."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        self._closed = True


class _FakeStreamContent:
    """Byte-line iterator standing in for ``resp.content`` (no line limit)."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def __aiter__(self) -> _FakeStreamContent:
        return self

    async def __anext__(self) -> bytes:
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class _FakeResponse:
    """Minimal aiohttp response double for chat and chat_stream paths."""

    def __init__(
        self,
        *,
        status: int = 200,
        json_data: dict[str, Any] | None = None,
        lines: list[bytes] | None = None,
        text_data: str = "",
        headers: dict[str, str] | None = None,
        raise_json: bool = False,
    ) -> None:
        self.status = status
        self.content_type = "application/json"
        self.headers = dict(headers or {})
        self._json_data = dict(json_data or {})
        self._text_data = text_data
        self._raise_json = raise_json
        self.content = _FakeStreamContent(lines or [])

    async def text(self) -> str:
        return self._text_data

    async def json(self) -> dict[str, Any]:
        if self._raise_json:
            raise ValueError("not json")
        return self._json_data

    def close(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _FakeSession:
    """Replays queued responses; records every ``post`` for shape assertions."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    async def close(self) -> None:
        pass


@dataclass
class _Installed:
    session: _FakeSession
    captured: dict[str, Any] = field(default_factory=dict)


def _install(
    monkeypatch: pytest.MonkeyPatch, responses: list[_FakeResponse]
) -> _Installed:
    """Arrange: fake aiohttp transport, capturing session/connector kwargs."""
    installed = _Installed(session=_FakeSession(responses))

    def _fake_session(*args: Any, **kwargs: Any) -> _FakeSession:
        installed.captured.update(kwargs)
        return installed.session

    def _fake_connector(*args: Any, **kwargs: Any) -> _FakeConnector:
        installed.captured["connector_kwargs"] = kwargs
        return _FakeConnector()

    # Arrange
    monkeypatch.setattr(
        "novakit.llm.providers.openai_chat.aiohttp.ClientSession", _fake_session
    )
    monkeypatch.setattr(
        "novakit.llm.providers.openai_chat.aiohttp.TCPConnector", _fake_connector
    )
    return installed


def _sse(payload: dict[str, Any]) -> bytes:
    """Arrange: one SSE data line with the required ``data: `` prefix."""
    return b"data: " + json.dumps(payload).encode() + b"\n"


async def _no_sleep(seconds: float, abort_event: asyncio.Event | None = None) -> bool:
    return False


async def _collect(provider: OpenAIProvider, **kwargs: Any) -> list[Any]:
    """Act: drain one streaming turn."""
    return [
        event
        async for event in provider.chat_stream(
            [{"role": "user", "content": "hi"}], model="m", **kwargs
        )
    ]


def _kinds(events: list[Any]) -> list[str]:
    return [type(event).__name__ for event in events]


# -- _cached_tokens_from_usage -------------------------------------------------


def test_cached_tokens_rejects_non_dict_usage() -> None:
    assert openai_chat_mod._cached_tokens_from_usage(None) is None
    assert openai_chat_mod._cached_tokens_from_usage("usage") is None


def test_cached_tokens_rejects_non_dict_details() -> None:
    assert (
        openai_chat_mod._cached_tokens_from_usage({"prompt_tokens_details": "x"})
        is None
    )


def test_cached_tokens_absent_is_none() -> None:
    assert openai_chat_mod._cached_tokens_from_usage({}) is None
    assert (
        openai_chat_mod._cached_tokens_from_usage({"prompt_tokens_details": {}}) is None
    )


def test_cached_tokens_value() -> None:
    assert (
        openai_chat_mod._cached_tokens_from_usage(
            {"prompt_tokens_details": {"cached_tokens": 70}}
        )
        == 70
    )


# -- headers --------------------------------------------------------------------


def test_headers_minimal_omit_auth() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    headers = provider._build_headers()

    # Assert
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers
    assert headers["User-Agent"].startswith("novakit/")


def test_headers_full_merge_with_user_agent_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(
        openai_chat_mod,
        "run_session_hook",
        lambda path, sid: {"X-Sess": "1", "User-Agent": "evil"},
    )
    monkeypatch.setattr(
        openai_chat_mod, "run_request_hook", lambda path, sid: {"X-Req": "2"}
    )
    provider = OpenAIProvider(
        api_key="k",
        extra_headers={"X-Extra": "e"},
        request_hook="req.sh",
        request_session_hook="sess.sh",
        user_agent="Custom/1.0",
    )

    # Act
    headers = provider._build_headers(session_id="s")

    # Assert
    assert headers["Authorization"] == "Bearer k"
    assert headers["X-Extra"] == "e"
    assert headers["X-Sess"] == "1"
    assert headers["X-Req"] == "2"
    assert headers["User-Agent"] == "Custom/1.0"


# -- request body -----------------------------------------------------------------


def test_body_defaults_omit_optionals() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    body = provider._build_body(messages=[], model="", stream=False)

    # Assert
    assert body == {"messages": []}


def test_body_full_passes_tools_and_cache_key() -> None:
    # Arrange
    tools = [{"type": "function", "function": {"name": "f"}}]
    provider = OpenAIProvider(request_options={"temperature": 0.5})

    # Act
    body = provider._build_body(
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        stream=True,
        tools=tools,
        session_id="s1",
    )

    # Assert
    assert body["model"] == "m"
    assert body["stream"] is True
    assert body["tools"] == tools
    assert body["temperature"] == 0.5
    assert body["prompt_cache_key"] == "s1"


def test_body_tools_disabled_forces_empty() -> None:
    # Arrange
    tools = [{"type": "function", "function": {"name": "f"}}]
    provider = OpenAIProvider(request_options={"tools": False})

    # Act
    body = provider._build_body(messages=[], model="m", stream=False, tools=tools)

    # Assert
    assert body["tools"] == []


def test_body_prompt_caching_disabled() -> None:
    # Arrange
    provider = OpenAIProvider(request_options={"prompt_caching": False})

    # Act
    body = provider._build_body(messages=[], model="m", stream=True, session_id="s1")

    # Assert
    assert "prompt_cache_key" not in body


def test_prepare_request_effort_per_turn_and_default() -> None:
    # Arrange
    provider = OpenAIProvider(api_key="sk", base_url="https://example.com/v1/")
    defaulted = OpenAIProvider(api_key="sk", default_reasoning_effort="low")

    # Act
    _, _, picked = provider._prepare_request([], "m", True, None, None, "high")
    _, _, fallback = defaulted._prepare_request([], "m", True, None, None, None)
    url, headers, _ = provider._prepare_request([], "m", True, None, None, None)

    # Assert
    assert picked["reasoning_effort"] == "high"
    assert fallback["reasoning_effort"] == "low"
    assert url == "https://example.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk"


# -- tool-call normalization ------------------------------------------------------


def test_normalize_tool_call_variants() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    passthrough = provider._normalize_tool_call("junk")  # type: ignore[arg-type]
    shaped = provider._normalize_tool_call(
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    )
    flat = provider._normalize_tool_call({"id": "c2", "name": "g", "arguments": "{}"})

    # Assert
    assert passthrough == "junk"
    assert shaped == {
        "id": "c1",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }
    assert flat == {
        "id": "c2",
        "type": "function",
        "function": {"name": "g", "arguments": "{}"},
    }


# -- message formatting -------------------------------------------------------------


def test_format_messages_skips_missing_role_and_keeps_name() -> None:
    # Arrange
    provider = OpenAIProvider()
    messages: list[Any] = [
        Message(role=None, content="gone"),  # type: ignore[arg-type]
        Message(role="assistant", content="a", name="n"),
    ]

    # Act
    formatted = provider._format_messages(messages)

    # Assert
    assert len(formatted) == 1
    assert formatted[0]["name"] == "n"
    assert formatted[0]["reasoning_content"] == ""


def test_format_messages_assistant_reasoning_object_and_dict() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    formatted = provider._format_messages(
        [
            Message(role="assistant", content="a", reasoning_content="deep"),
            {"role": "assistant", "content": "b", "reasoning_content": "shallow"},
        ]
    )

    # Assert
    assert [m["reasoning_content"] for m in formatted] == ["deep", "shallow"]


def test_format_messages_user_images_single_turn() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    formatted = provider._format_messages(
        [{"role": "user", "content": "look", "images": ["QUJD"]}]
    )

    # Assert
    assert len(formatted) == 1
    assert [part["type"] for part in formatted[0]["content"]] == ["text", "image_url"]


def test_format_messages_tool_images_keep_role() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    formatted = provider._format_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_image", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Image loaded: shot.png",
                "tool_call_id": "call_1",
                "images": ["QUJD"],
            },
        ]
    )

    # Assert
    assert [m["role"] for m in formatted] == ["assistant", "tool", "user"]
    assert formatted[1]["tool_call_id"] == "call_1"
    assert formatted[1]["name"] == "read_image"
    assert formatted[2]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    ]
    assert all(not m.get("tool_call_id") for m in formatted if m["role"] == "user")


def test_format_messages_tool_calls_normalize_and_name_fill() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    formatted = provider._format_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                    "junk",
                    {"id": "", "name": "flat", "arguments": "{}"},
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "c1"},
            {"role": "tool", "content": "other", "tool_call_id": "missing"},
            {
                "role": "tool",
                "content": "named",
                "name": "already",
                "tool_call_id": "c1",
            },
        ]
    )

    # Assert
    assert [m["role"] for m in formatted] == ["assistant", "tool", "tool", "tool"]
    assert formatted[0]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "", "type": "function", "function": {"name": "flat", "arguments": "{}"}},
    ]
    assert formatted[1]["name"] == "f"
    assert "name" not in formatted[2]
    assert formatted[3]["name"] == "already"


# -- non-streaming chat ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_text_usage_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    payload = {
        "choices": [
            {
                "message": {
                    "content": "hi",
                    "tool_calls": [
                        {"id": "t1", "function": {"name": "f", "arguments": "{}"}}
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    }
    installed = _install(monkeypatch, [_FakeResponse(json_data=payload)])
    provider = OpenAIProvider(api_key="k")

    # Act
    result = await provider.chat([{"role": "user", "content": "hi"}], model="m")

    # Assert
    assert isinstance(result, Done)
    assert result.content == "hi"
    assert result.tokens_input == 100
    assert result.tokens_output == 5
    assert result.cache_read_tokens == 80
    assert result.tool_calls == [ToolCall(id="t1", name="f", arguments="{}")]
    assert installed.session.calls[0]["json"]["model"] == "m"


def test_parse_response_empty_choices_and_missing_usage() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    empty_list = provider._parse_response({"choices": []})
    non_list = provider._parse_response({"choices": {}})
    bare = provider._parse_response({"choices": [{"message": {}}]})

    # Assert
    assert empty_list == Done(content="", tool_calls=[])
    assert non_list == Done(content="", tool_calls=[])
    assert bare.content == ""
    assert bare.tokens_input is None
    assert bare.cache_read_tokens is None


def test_parse_response_tool_call_defaults() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    result = provider._parse_response(
        {"choices": [{"message": {"content": "hi", "tool_calls": [{"id": "t2"}]}}]}
    )

    # Assert
    assert result.tool_calls == [ToolCall(id="t2", name="", arguments="")]


@pytest.mark.asyncio
async def test_chat_http_error_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    _install(monkeypatch, [_FakeResponse(status=400, text_data="bad request")])
    provider = OpenAIProvider(api_key="k")

    # Act
    result = await provider.chat([{"role": "user", "content": "hi"}], model="m")

    # Assert
    assert isinstance(result, Error)
    assert "HTTP 400" in result.message


@pytest.mark.asyncio
async def test_chat_invalid_json_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    _install(monkeypatch, [_FakeResponse(raise_json=True, text_data="<html>")])
    provider = OpenAIProvider(api_key="k")

    # Act
    result = await provider.chat([{"role": "user", "content": "hi"}], model="m")

    # Assert
    assert isinstance(result, Error)
    assert result.message == "unexpected response from API"


# -- streaming: text / reasoning / usage -----------------------------------------------


@pytest.mark.asyncio
async def test_stream_text_reasoning_usage_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [
        _sse(
            {"choices": [{"delta": {"content": "hi", "reasoning_content": "thinking"}}]}
        ),
        _sse(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 70},
                },
            }
        ),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == ["ReasoningDelta", "TextDelta", "Done"]
    done = collected[-1]
    assert isinstance(done, Done)
    assert (
        done.content,
        done.tokens_input,
        done.tokens_output,
        done.cache_read_tokens,
    ) == ("hi", 100, 5, 70)


@pytest.mark.asyncio
async def test_stream_custom_reasoning_field(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    lines = [
        _sse({"choices": [{"delta": {"reasoning": "r", "content": "hi"}}]}),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k", reasoning_field="reasoning")

    # Act
    collected = await _collect(provider)

    # Assert
    assert isinstance(collected[0], ReasoningDelta)
    assert collected[0].content == "r"
    assert isinstance(collected[-1], Done)


@pytest.mark.asyncio
async def test_stream_stop_only_chunk_ends_empty_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [_sse({"choices": [{"finish_reason": "stop"}]}), b"data: [DONE]\n"]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == ["Done"]
    assert collected[0].content == ""


# -- streaming: tool calls -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_tool_call_before_arguments_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "write",
                                        "arguments": '{"filePath": "/tmp/x",',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ' "content": "hello"}'},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == [
        "ToolInputStart",
        "ToolInputDelta",
        "ToolInputDelta",
        "ToolInputEnd",
        "ToolCall",
        "Done",
    ]
    start = collected[0]
    call = collected[4]
    assert isinstance(start, ToolInputStart) and isinstance(call, ToolCall)
    assert (start.id, start.name) == ("call_1", "write")
    assert start.id == call.id
    assert (
        "".join(e.delta for e in collected if isinstance(e, ToolInputDelta))
        == call.arguments
    )


@pytest.mark.asyncio
async def test_stream_late_id_starts_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"name": "write"}}]
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "write"},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"a":1}'}}
                            ]
                        }
                    }
                ]
            }
        ),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == [
        "ToolInputStart",
        "ToolInputDelta",
        "ToolInputEnd",
        "ToolCall",
        "Done",
    ]
    assert isinstance(collected[0], ToolInputStart) and collected[0].id == "call_1"
    assert isinstance(collected[3], ToolCall) and collected[3].arguments == '{"a":1}'


@pytest.mark.asyncio
async def test_stream_pre_start_arguments_emit_no_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [
        _sse(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"name": "f"}}]}}
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"a":'}}
                            ]
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "c1", "function": {"name": "f"}}
                            ]
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": "1}"}}
                            ]
                        }
                    }
                ]
            }
        ),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == [
        "ToolInputStart",
        "ToolInputDelta",
        "ToolInputEnd",
        "ToolCall",
        "Done",
    ]
    call = collected[3]
    assert isinstance(call, ToolCall) and call.arguments == '{"a":1}'
    assert "".join(e.delta for e in collected if isinstance(e, ToolInputDelta)) == "1}"


@pytest.mark.asyncio
async def test_stream_extra_tool_calls_skipped_over_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(openai_chat_mod, "_MAX_TOOL_CALLS", 1)
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_0",
                                    "function": {"name": "f", "arguments": '{"a":1}'},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_1",
                                    "function": {"name": "g", "arguments": '{"b":2}'},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert "call_1" not in [getattr(e, "id", "") for e in collected]
    done = collected[-1]
    assert isinstance(done, Done)
    assert [c.id for c in done.tool_calls or []] == ["call_0"]


@pytest.mark.asyncio
async def test_stream_malformed_arguments_report_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_bad",
                                    "function": {
                                        "name": "write",
                                        "arguments": '{"filePath": ',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == [
        "ToolInputStart",
        "ToolInputDelta",
        "ToolInputEnd",
        "ToolInputError",
        "Done",
    ]
    err = collected[3]
    assert isinstance(err, ToolInputError) and err.id == "call_bad"
    assert collected[-1].tool_calls == []


@pytest.mark.asyncio
async def test_stream_finished_call_not_repeated_at_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "f", "arguments": '{"a":'},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": "1}"}}
                            ]
                        }
                    }
                ]
            }
        ),
        _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == [
        "ToolInputStart",
        "ToolInputDelta",
        "ToolInputDelta",
        "ToolInputEnd",
        "ToolCall",
        "Done",
    ]
    done = collected[-1]
    assert isinstance(done, Done)
    assert [c.id for c in done.tool_calls or []] == ["c1"]


@pytest.mark.asyncio
async def test_stream_unstarted_entry_skipped_at_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 3, "function": {"arguments": "xx"}}
                            ]
                        }
                    }
                ]
            }
        ),
        _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == ["Done"]
    assert collected[0].tool_calls == []


# -- streaming: terminal reasons / errors / retry -----------------------------------------------


@pytest.mark.asyncio
async def test_stream_length_without_answer_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [
        _sse({"choices": [{"delta": {"reasoning_content": "thinking hard"}}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert isinstance(collected[-1], Error)
    assert "finish_reason=length" in collected[-1].message


@pytest.mark.asyncio
async def test_stream_length_with_partial_answer_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [
        _sse({"choices": [{"delta": {"content": "partial"}}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        b"data: [DONE]\n",
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert isinstance(collected[-1], Done)
    assert collected[-1].content == "partial"


@pytest.mark.asyncio
async def test_stream_runaway_tool_args_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    big = "x" * 1_000_001
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "f", "arguments": big},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    ]
    _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert isinstance(collected[0], ToolInputStart)
    assert isinstance(collected[-1], Error)
    assert "runaway" in collected[-1].message


@pytest.mark.asyncio
async def test_stream_http_error_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    _install(monkeypatch, [_FakeResponse(status=400, text_data="bad request")])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == ["Error"]
    assert "HTTP 400" in collected[0].message


@pytest.mark.asyncio
async def test_stream_retry_then_done(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setattr("novakit.llm.http_provider.sleep_unless_aborted", _no_sleep)
    ok = _FakeResponse(
        lines=[_sse({"choices": [{"delta": {"content": "hi"}}]}), b"data: [DONE]\n"]
    )
    _install(monkeypatch, [_FakeResponse(status=503, text_data="busy"), ok])
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert isinstance(collected[0], RetryNotice)
    assert isinstance(collected[-1], Done)
    assert collected[-1].content == "hi"


@pytest.mark.asyncio
async def test_stream_terminal_429_is_error_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _install(
        monkeypatch,
        [_FakeResponse(status=429, text_data="insufficient_quota exceeded")],
    )
    provider = OpenAIProvider(api_key="k")

    # Act
    collected = await _collect(provider)

    # Assert
    assert _kinds(collected) == ["Error"]
    assert not any(isinstance(e, RetryNotice) for e in collected)


# -- request shape: headers / timeouts / transport --------------------------------------------------


@pytest.mark.asyncio
async def test_stream_request_shape_timeout_headers_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [_sse({"choices": [{"delta": {"content": "hi"}}]}), b"data: [DONE]\n"]
    installed = _install(monkeypatch, [_FakeResponse(lines=lines)])
    provider = OpenAIProvider(api_key="k", base_url="https://example.com/v1")

    # Act
    collected = await _collect(provider, session_id="s1", total_timeout_seconds=7)

    # Assert
    assert isinstance(collected[-1], Done)
    call = installed.session.calls[0]
    assert call["url"] == "https://example.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer k"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["json"]["stream"] is True
    assert call["json"]["prompt_cache_key"] == "s1"
    assert call["timeout"].total == 7
    assert call["timeout"].sock_read == 300
    assert installed.captured["connector_kwargs"] == {
        "limit": 10,
        "limit_per_host": 5,
        "ttl_dns_cache": 300,
    }
    assert installed.captured["trust_env"] is True


@pytest.mark.asyncio
async def test_chat_request_shape_timeout_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    tools = [{"type": "function", "function": {"name": "f"}}]
    installed = _install(
        monkeypatch,
        [_FakeResponse(json_data={"choices": [{"message": {"content": "hi"}}]})],
    )
    provider = OpenAIProvider(timeout_seconds=5)

    # Act
    result = await provider.chat(
        [{"role": "user", "content": "hi"}], model="m", tools=tools
    )

    # Assert
    assert isinstance(result, Done)
    call = installed.session.calls[0]
    assert call["url"] == "/chat/completions"
    assert "Authorization" not in call["headers"]
    assert call["json"]["tools"] == tools
    assert "prompt_cache_key" not in call["json"]
    assert call["timeout"].total == 5


# -- misc direct parser / token estimate ---------------------------------------------------------------


def test_feed_ignores_non_dict_usage_and_non_list_choices() -> None:
    # Arrange
    parser = _OpenAIChatStreamParser("reasoning_content")
    acc = StreamAccumulator("m", 10, 10)

    # Act
    events = list(parser.feed({"usage": "nope", "choices": {"delta": {}}}, acc))

    # Assert
    assert events == []


@pytest.mark.asyncio
async def test_count_tokens_ascii_and_chinese() -> None:
    # Arrange
    provider = OpenAIProvider()

    # Act
    ascii_count = await provider.count_tokens("hello world")
    chinese_count = await provider.count_tokens("你好")
    mixed_count = await provider.count_tokens("hi你好", model="m")

    # Assert
    assert ascii_count == int(11 / 4)
    assert chinese_count == int(2 / 2)
    assert mixed_count == int(2 / 2 + 2 / 4)
