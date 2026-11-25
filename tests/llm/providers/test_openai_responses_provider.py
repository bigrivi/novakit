"""Pure-offline coverage for the OpenAI Responses provider.

Every test here runs without the network: the parser is fed decoded event
dicts directly, the pure helpers are called directly, and the two
end-to-end paths (``chat`` / ``chat_stream``) run against a fake
connector/session/stream. Nothing imports ``nova`` or touches the network.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from novakit.llm.accumulator import StreamAccumulator
from novakit.llm.provider import (
    STREAM_IDLE_TIMEOUT_SECONDS,
    Done,
    Error,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolInputDelta,
    ToolInputEnd,
    ToolInputError,
    ToolInputStart,
)
from novakit.llm.providers.openai_responses import (
    OpenAIResponsesProvider,
    _ResponsesStreamParser,
)

# -- Fakes -----------------------------------------------------------------


class _FakeConnector:
    """Stands in for ``aiohttp.TCPConnector``; only open/closed matters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        self._closed = True


class _FakeLines:
    """Mimics ``aiohttp.StreamReader`` for one fixed list of SSE lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)
        self._index = 0

    async def readline(self, *, max_line_length: int | None = None) -> bytes:
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line


class _FakeResp:
    """Mimics ``aiohttp.ClientResponse`` for one canned HTTP reply."""

    def __init__(
        self,
        *,
        status: int = 200,
        lines: list[bytes] | None = None,
        text_data: str = "",
        json_data: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.headers: dict[str, str] = {}
        self.content = _FakeLines(lines or [])
        self.content_type = "application/json"
        self._text_data = text_data
        self._json_data = json_data if json_data is not None else {}

    async def text(self) -> str:
        return self._text_data

    async def json(self) -> dict[str, Any]:
        return self._json_data

    def close(self) -> None:
        pass

    async def release(self) -> None:
        pass

    async def __aenter__(self) -> _FakeResp:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _FakeSession:
    """Records POSTs and answers with one canned response (or raises)."""

    def __init__(self, response: _FakeResp | BaseException) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def post(self, url: str, **kwargs: Any) -> _FakeResp:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response

    async def close(self) -> None:
        self.closed = True


def _install_fake(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResp | BaseException
) -> _FakeSession:
    """Route the provider's aiohttp construction at the fake transport."""
    session = _FakeSession(response)
    connector = _FakeConnector()
    monkeypatch.setattr(
        "novakit.llm.providers.openai_responses.aiohttp.ClientSession",
        lambda *args, **kwargs: session,
    )
    monkeypatch.setattr(
        "novakit.llm.providers.openai_responses.aiohttp.TCPConnector",
        lambda *args, **kwargs: connector,
    )
    return session


# -- Small builders ----------------------------------------------------------


def _provider(**kwargs: Any) -> OpenAIResponsesProvider:
    """Arrange: a provider with a key and a dummy base URL unless told otherwise."""
    kwargs.setdefault("api_key", "k")
    kwargs.setdefault("base_url", "https://api.example.com/v1/")
    return OpenAIResponsesProvider(**kwargs)


def _acc(
    model: str = "m",
    content_max: int | None = 2_000_000,
    tool_max: int | None = 1_000_000,
) -> StreamAccumulator:
    """Arrange: a fresh accumulator with generous ceilings by default."""
    return StreamAccumulator(model, content_max, tool_max)


def _with_parser(
    provider: OpenAIResponsesProvider,
) -> tuple[_ResponsesStreamParser, StreamAccumulator]:
    """Arrange: a fresh parser/accumulator pair for one streaming turn."""
    return _ResponsesStreamParser(provider), _acc()


def _feed(
    parser: _ResponsesStreamParser,
    acc: StreamAccumulator,
    event: dict[str, Any],
) -> list[Any]:
    """Act: run one decoded event through the parser, collecting its output."""
    return list(parser.feed(event, acc))


def _sse(payload: dict[str, Any]) -> bytes:
    """Arrange: encode one event the way the wire carries it."""
    return f"data: {json.dumps(payload)}\n".encode()


def _completed_line(text: str, usage: dict[str, Any]) -> bytes:
    """Arrange: a terminal ``response.completed`` line carrying one message."""
    return _sse(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "usage": usage,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    },
                ],
            },
        }
    )


async def _collect(provider: OpenAIResponsesProvider, **kwargs: Any) -> list[Any]:
    """Act: drain one ``chat_stream`` turn into a list."""
    kwargs.setdefault("messages", [Message(role="user", content="hi")])
    kwargs.setdefault("model", "gpt-5")
    return [event async for event in provider.chat_stream(**kwargs)]


# -- _collect_tool_calls -----------------------------------------------------


def test_collect_prefers_already_emitted_calls() -> None:
    """Emitted calls are the ones the turn acted on, so they win."""
    # Arrange
    by_index = {
        0: {"id": "c0", "name": "first", "arguments": '{"a":1}', "emitted": True},
        1: {"id": "c1", "name": "second", "arguments": '{"b":2}'},
    }
    # Act
    from novakit.llm.providers.openai_responses import _collect_tool_calls

    calls = _collect_tool_calls(by_index)
    # Assert
    assert [(c.id, c.name, c.arguments) for c in calls] == [("c0", "first", '{"a":1}')]


def test_collect_falls_back_to_unemitted_named_calls() -> None:
    """Gateways that deliver a call whole never emit, but still count."""
    # Arrange
    from novakit.llm.providers.openai_responses import _collect_tool_calls

    by_index = {0: {"id": "c0", "name": "lookup", "arguments": '{"q":1}'}}
    # Act
    calls = _collect_tool_calls(by_index)
    # Assert
    assert [(c.id, c.name) for c in calls] == [("c0", "lookup")]


def test_collect_skips_invalid_and_unnamed_and_defaults_args() -> None:
    # Arrange
    from novakit.llm.providers.openai_responses import _collect_tool_calls

    by_index = {
        0: {"id": "bad", "name": "x", "arguments": "{oops", "invalid": True},
        1: {"id": "noname", "name": "", "arguments": "{}"},
        2: {"id": "noargs", "name": "ok"},
    }
    # Act
    calls = _collect_tool_calls(by_index)
    # Assert
    assert [(c.id, c.name, c.arguments) for c in calls] == [("noargs", "ok", "{}")]


def test_collect_emitted_invalid_falls_back_to_valid() -> None:
    # Arrange: the only emitted call is broken, so the emitted list is empty.
    from novakit.llm.providers.openai_responses import _collect_tool_calls

    by_index = {
        0: {
            "id": "bad",
            "name": "x",
            "arguments": "{oops",
            "emitted": True,
            "invalid": True,
        },
        1: {"id": "good", "name": "fine", "arguments": "{}"},
    }
    # Act
    calls = _collect_tool_calls(by_index)
    # Assert
    assert [c.id for c in calls] == ["good"]


def test_collect_empty_when_nothing_named() -> None:
    from novakit.llm.providers.openai_responses import _collect_tool_calls

    assert _collect_tool_calls({}) == []
    assert _collect_tool_calls({0: {"id": "x"}}) == []


# -- response.completed --------------------------------------------------------


def test_completed_records_usage_and_cache() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    events = _feed(
        parser,
        acc,
        {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "input_tokens_details": {"cached_tokens": 6},
                }
            },
        },
    )
    # Assert
    assert events == []
    assert parser.finished is True
    assert (acc.tokens_input, acc.tokens_output, acc.cache_read_tokens) == (10, 4, 6)


def test_completed_without_response_object_keeps_tokens_unset() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    events = _feed(parser, acc, {"type": "response.completed"})
    # Assert
    assert events == []
    assert acc.tokens_input is None and acc.tokens_output is None
    assert acc.cache_read_tokens is None


def test_completed_with_non_dict_response_and_usage() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    events = _feed(
        parser, acc, {"type": "response.completed", "response": {"usage": [1, 2]}}
    )
    # Assert
    assert events == []
    assert acc.tokens_input is None and acc.cache_read_tokens is None


def test_completed_with_non_dict_details_and_missing_cache() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    _feed(
        parser,
        acc,
        {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 1,
                    "input_tokens_details": "x",
                }
            },
        },
    )
    parser2, acc2 = _with_parser(_provider())
    _feed(parser2, acc2, {"type": "response.completed", "response": "oops"})
    # Assert
    assert acc.cache_read_tokens is None
    assert acc2.tokens_input is None and parser2.finished is True


# -- server failures inside the stream -----------------------------------------


def test_failed_event_becomes_partial_error() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    acc.add_text("half")
    # Act
    events = _feed(
        parser,
        acc,
        {
            "type": "response.failed",
            "response": {"error": {"code": "server_error", "message": "boom"}},
        },
    )
    # Assert
    assert len(events) == 1 and isinstance(events[0], Error)
    assert "response.failed" in events[0].message
    assert "server_error" in events[0].message and "boom" in events[0].message
    assert events[0].content == "half"
    assert parser.stopped is True


def test_incomplete_and_error_events_become_errors() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    incomplete = _feed(parser, acc, {"type": "response.incomplete", "response": {}})
    error = _feed(
        parser,
        acc,
        {"type": "error", "response": {"error": {"type": "t", "message": "m"}}},
    )
    # Assert
    assert isinstance(incomplete[0], Error) and isinstance(error[0], Error)
    assert "(no detail from server)" in incomplete[0].message


def test_failure_carries_tool_calls_seen_so_far() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "f",
                "arguments": "{}",
            },
        },
    )
    # Act
    events = _feed(parser, acc, {"type": "response.failed", "response": {}})
    # Assert
    assert isinstance(events[0], Error)
    assert [c.id for c in events[0].tool_calls] == ["c1"]


# -- text and reasoning deltas --------------------------------------------------


def test_output_text_delta_yields_and_empty_delta_skipped() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    got = _feed(parser, acc, {"type": "response.output_text.delta", "delta": "hi"})
    skipped = _feed(parser, acc, {"type": "response.output_text.delta", "delta": ""})
    missing = _feed(parser, acc, {"type": "response.output_text.delta"})
    # Assert
    assert got == [TextDelta(content="hi")] and acc.content == "hi"
    assert skipped == [] and missing == []


def test_reasoning_deltas_yield_and_empty_skipped() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    a = _feed(parser, acc, {"type": "response.reasoning_text.delta", "delta": "r"})
    b = _feed(
        parser, acc, {"type": "response.reasoning_summary_text.delta", "delta": "s"}
    )
    empty_a = _feed(parser, acc, {"type": "response.reasoning_text.delta"})
    empty_b = _feed(
        parser, acc, {"type": "response.reasoning_summary_text.delta", "delta": ""}
    )
    # Assert
    assert a == [ReasoningDelta(content="r")]
    assert b == [ReasoningDelta(content="s")]
    assert empty_a == [] and empty_b == []


def test_unknown_event_is_ignored() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act / Assert
    assert _feed(parser, acc, {"type": "response.created"}) == []
    assert _feed(parser, acc, {}) == []


# -- function_call_arguments.delta -----------------------------------------------


def test_first_delta_creates_entry_without_name_yields_nothing() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act: no output_item.added came first, so the name is still unknown.
    events = _feed(
        parser,
        acc,
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 2,
            "delta": '{"a":',
        },
    )
    # Assert
    assert events == []
    assert parser._tool_calls[2]["id"] == "call_2"


def test_first_delta_uses_item_id_when_present() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    _feed(
        parser,
        acc,
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "item_id": "call_xyz",
            "delta": "{}",
        },
    )
    # Assert
    assert parser._tool_calls[0]["id"] == "call_xyz"


def test_delta_with_known_name_yields_input_delta() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "lookup",
                "arguments": "",
            },
        },
    )
    # Act
    events = _feed(
        parser,
        acc,
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": '{"q":1}',
        },
    )
    # Assert
    assert events == [ToolInputDelta(id="c1", name="lookup", delta='{"q":1}')]


def test_empty_delta_appends_nothing() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "f",
                "arguments": "",
            },
        },
    )
    # Act
    events = _feed(
        parser,
        acc,
        {"type": "response.function_call_arguments.delta", "output_index": 0},
    )
    # Assert
    assert events == []


# -- output_item.added ------------------------------------------------------------


def test_added_non_function_item_is_ignored() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    events = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message"},
        },
    )
    # Assert
    assert events == [] and parser._tool_calls == {}


def test_added_without_item_is_ignored() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    events = _feed(
        parser, acc, {"type": "response.output_item.added", "output_index": 0}
    )
    # Assert
    assert events == [] and parser._tool_calls == {}


def test_added_emits_start_and_falls_back_for_ids() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    with_id = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "call_i",
                "name": "f",
                "arguments": "",
            },
        },
    )
    fallback = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 3,
            "item": {"type": "function_call", "name": "g"},
        },
    )
    # Assert
    assert with_id == [ToolInputStart(id="call_i", name="f")]
    assert fallback == [ToolInputStart(id="call_3", name="g")]
    assert parser._tool_calls[3]["arguments"] == ""


def test_added_without_name_registers_silently() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    events = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "function_call", "call_id": "c9", "arguments": ""},
        },
    )
    # Assert
    assert events == []
    assert parser._tool_calls[1]["name"] == ""


# -- output_item.done ---------------------------------------------------------------


def test_done_non_function_item_is_ignored() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act
    events = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "message"},
        },
    )
    # Assert
    assert events == []


def test_done_completes_call_with_end_and_call() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "lookup",
                "arguments": "",
            },
        },
    )
    # Act
    events = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "name": "lookup", "arguments": '{"q":1}'},
        },
    )
    # Assert
    assert events == [
        ToolInputEnd(id="c1", name="lookup"),
        ToolCall(id="c1", name="lookup", arguments='{"q":1}'),
    ]


@pytest.mark.xfail(
    reason="Done-only function calls lack the id populated by added events",
    strict=True,
)
def test_done_without_added_event_uses_item_id() -> None:
    parser, acc = _with_parser(_provider())

    events = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "lookup",
                "arguments": "{}",
            },
        },
    )

    assert events == [
        ToolInputEnd(id="c1", name="lookup"),
        ToolCall(id="c1", name="lookup", arguments="{}"),
    ]


def test_done_missing_item_keeps_existing_state() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "f",
                "arguments": "{}",
            },
        },
    )
    # Act: done carries a bare function_call marker; stored state wins.
    events = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call"},
        },
    )
    # Assert
    assert [type(e).__name__ for e in events] == ["ToolInputEnd", "ToolCall"]


def test_done_twice_emits_once() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "f",
                "arguments": "{}",
            },
        },
    )
    first = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call"},
        },
    )
    # Act
    second = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call"},
        },
    )
    # Assert
    assert len(first) == 2 and second == []


def test_done_without_name_emits_nothing() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "c1", "arguments": ""},
        },
    )
    # Act
    events = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call"},
        },
    )
    # Assert
    assert events == []


def test_done_with_bad_arguments_reports_input_error() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "cb",
                "name": "write",
                "arguments": "",
            },
        },
    )
    # Act
    events = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "arguments": '{"q":'},
        },
    )
    # Assert
    assert [type(e).__name__ for e in events] == ["ToolInputEnd", "ToolInputError"]
    assert isinstance(events[1], ToolInputError) and events[1].raw == '{"q":'
    assert parser._tool_calls[0]["invalid"] is True


def test_done_for_unknown_index_without_name_is_ignored() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act: no stored state and no name/arguments on the item, so nothing finishes.
    events = _feed(
        parser,
        acc,
        {
            "type": "response.output_item.done",
            "output_index": 7,
            "item": {"type": "function_call"},
        },
    )
    # Assert
    assert events == []


# -- function_call_arguments.done ------------------------------------------------------


def test_arguments_done_finishes_pending_call() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "lookup",
                "arguments": "",
            },
        },
    )
    _feed(
        parser,
        acc,
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": '{"q":1}',
        },
    )
    # Act
    events = _feed(
        parser,
        acc,
        {"type": "response.function_call_arguments.done", "output_index": 0},
    )
    # Assert
    assert [type(e).__name__ for e in events] == ["ToolInputEnd", "ToolCall"]


def test_arguments_done_missing_pending_is_ignored() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    # Act / Assert
    assert (
        _feed(
            parser,
            acc,
            {"type": "response.function_call_arguments.done", "output_index": 9},
        )
        == []
    )


def test_arguments_done_without_name_or_when_emitted_is_ignored() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(
        parser,
        acc,
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{}",
        },
    )
    # Act: entry exists but has no name yet.
    nameless = _feed(
        parser,
        acc,
        {"type": "response.function_call_arguments.done", "output_index": 0},
    )
    # Arrange: give it a name and finish it once.
    parser._tool_calls[0]["name"] = "f"
    parser._tool_calls[0]["id"] = "c1"
    first = _feed(
        parser,
        acc,
        {"type": "response.function_call_arguments.done", "output_index": 0},
    )
    # Act: a second done arrives after emission.
    second = _feed(
        parser,
        acc,
        {"type": "response.function_call_arguments.done", "output_index": 0},
    )
    # Assert
    assert nameless == []
    assert len(first) == 2 and second == []


# -- _finish_tool_input -----------------------------------------------------------------


def test_finish_skips_end_when_already_ended() -> None:
    # Arrange
    parser, _ = _with_parser(_provider())
    # Act
    events = parser._finish_tool_input(
        {"id": "c1", "name": "f", "arguments": "{}", "ended": True}
    )
    # Assert
    assert events == [ToolCall(id="c1", name="f", arguments="{}")]


def test_finish_defaults_missing_arguments_to_empty_object() -> None:
    # Arrange
    parser, _ = _with_parser(_provider())
    # Act
    events = parser._finish_tool_input({"id": "c1", "name": "f", "ended": False})
    # Assert
    assert events[0] == ToolInputEnd(id="c1", name="f")
    assert events[1] == ToolCall(id="c1", name="f", arguments="{}")


# -- build_done / on_eof / on_exception ------------------------------------------------------


def test_build_done_carries_text_tools_and_usage() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    acc.add_text("hello")
    parser._tool_calls[0] = {"id": "c1", "name": "f", "arguments": "{}"}
    acc.tokens_input = 5
    acc.tokens_output = 6
    acc.cache_read_tokens = 2
    # Act
    done = parser.build_done(acc)
    # Assert
    assert isinstance(done, Done)
    assert (
        done.content,
        done.tokens_input,
        done.tokens_output,
        done.cache_read_tokens,
    ) == ("hello", 5, 6, 2)
    assert [(c.id, c.name) for c in done.tool_calls or []] == [("c1", "f")]


def test_on_eof_without_completed_is_truncation_error() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    acc.add_text("ok")
    # Act
    events = list(parser.on_eof(acc))
    # Assert
    assert len(events) == 1 and isinstance(events[0], Error)
    assert "response.completed" in events[0].message
    assert events[0].content == "ok"


def test_on_eof_after_completed_builds_done() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    _feed(parser, acc, {"type": "response.completed", "response": {}})
    # Act
    events = list(parser.on_eof(acc))
    # Assert
    assert len(events) == 1 and isinstance(events[0], Done)


def test_on_exception_reports_partial_error() -> None:
    # Arrange
    parser, acc = _with_parser(_provider())
    acc.add_text("part")
    # Act
    error = parser.on_exception(RuntimeError("net down"), acc)
    # Assert
    assert isinstance(error, Error)
    assert "net down" in error.message and error.content == "part"


def test_new_parser_returns_responses_parser() -> None:
    # Arrange
    provider = _provider()
    # Act
    parser = provider._new_parser("gpt-5")
    # Assert
    assert isinstance(parser, _ResponsesStreamParser)


# -- _partial_error -------------------------------------------------------------------------------


def test_partial_error_reports_size_with_content() -> None:
    # Arrange / Act
    error = _provider()._partial_error("quiet", "half an answer", {})
    # Assert
    assert error.content == "half an answer"
    assert "14 characters had already arrived" in error.message


def test_partial_error_without_content_is_just_reason() -> None:
    # Arrange / Act
    error = _provider()._partial_error("quiet", "", {})
    # Assert
    assert error.message == "quiet"


def test_partial_error_with_empty_reason_names_exception_type() -> None:
    # Arrange / Act
    error = _provider()._partial_error(ValueError(""), "abc", {})
    # Assert
    assert error.message.startswith("ValueError")
    assert "3 characters had already arrived" in error.message


# -- _server_failure --------------------------------------------------------------------------------


def test_server_failure_variants() -> None:
    # Arrange
    provider = _provider()
    # Act / Assert: code + detail from the response error object.
    assert (
        provider._server_failure(
            "response.failed",
            {"response": {"error": {"code": "server_error", "message": "boom"}}},
        )
        == "response.failed [server_error] boom"
    )
    # `type` doubles as the code; error object may hang off the event itself.
    assert (
        provider._server_failure(
            "error", {"error": {"type": "rate_limit", "message": "slow"}}
        )
        == "error [rate_limit] slow"
    )
    # Top-level code/message when there is no error object.
    assert (
        provider._server_failure("error", {"code": "c", "message": "m"})
        == "error [c] m"
    )
    # A bare message with no code.
    assert (
        provider._server_failure("response.failed", {"message": "m"})
        == "response.failed m"
    )
    # Incomplete details supply the reason.
    assert (
        provider._server_failure(
            "response.incomplete",
            {"response": {"incomplete_details": {"reason": "max_output_tokens"}}},
        )
        == "response.incomplete incomplete: max_output_tokens"
    )
    # Nothing usable from the server.
    assert provider._server_failure("response.failed", {"response": {}}) == (
        "response.failed (no detail from server)"
    )
    # Non-dict response/error shapes never crash.
    assert provider._server_failure("error", {"response": "x", "error": "y"}) == (
        "error (no detail from server)"
    )


# -- count_tokens --------------------------------------------------------------------------------------


def test_count_tokens_scales_ascii_and_chinese() -> None:
    # Arrange
    provider = _provider()
    # Act / Assert
    assert asyncio.run(provider.count_tokens("")) == 0
    assert asyncio.run(provider.count_tokens("hello world")) == 2
    assert asyncio.run(provider.count_tokens("你好")) == 1
    assert asyncio.run(provider.count_tokens("hi你好", model="gpt-5")) == 1


# -- headers --------------------------------------------------------------------------------------------


def test_default_headers_name_novakit_without_key() -> None:
    # Arrange
    provider = OpenAIResponsesProvider(base_url="https://x.example")
    # Act
    headers = provider._build_headers()
    # Assert
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers
    assert headers["User-Agent"].startswith("novakit/")


def test_headers_carry_bearer_key_extra_and_hooks() -> None:
    # Arrange
    provider = _provider(extra_headers={"X-E": "1", "User-Agent": "spoof"})
    # Act
    headers = provider._build_headers()
    # Assert: config headers merge in, but User-Agent is always written last.
    assert headers["Authorization"] == "Bearer k"
    assert headers["X-E"] == "1"
    assert headers["User-Agent"].startswith("novakit/")


def test_headers_run_session_and_request_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    provider = _provider(request_hook="r.py", request_session_hook="s.py")
    monkeypatch.setattr(
        "novakit.llm.providers.openai_responses.run_session_hook",
        lambda path, session_id: {"X-S": f"{path}:{session_id}"},
    )
    monkeypatch.setattr(
        "novakit.llm.providers.openai_responses.run_request_hook",
        lambda path, session_id: {"X-R": f"{path}:{session_id}"},
    )
    # Act
    headers = provider._build_headers(session_id="ses_1")
    # Assert
    assert headers["X-S"] == "s.py:ses_1"
    assert headers["X-R"] == "r.py:ses_1"


def test_custom_user_agent_is_used_and_blank_falls_back() -> None:
    # Arrange / Act / Assert
    assert (
        _provider(user_agent="custom/1.0")._build_headers()["User-Agent"]
        == "custom/1.0"
    )
    assert (
        _provider(user_agent="  ")._build_headers()["User-Agent"].startswith("novakit/")
    )


def test_init_normalises_base_url_and_defaults() -> None:
    # Arrange / Act
    provider = _provider()
    # Assert
    assert provider.base_url == "https://api.example.com/v1"
    assert provider.api_key == "k"
    assert provider.timeout_seconds == 120
    assert provider.request_options == {}


# -- _format_input ------------------------------------------------------------------------------------------


def test_format_skips_messages_without_role() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input(
        [{"content": "hi"}, {"role": "user", "content": "ok"}]
    )
    # Assert
    assert result == [{"role": "user", "content": "ok"}]


def test_format_tool_result_without_images_stays_string() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input(
        [{"role": "tool", "content": "done", "tool_call_id": "c1"}]
    )
    # Assert
    assert result == [
        {"type": "function_call_output", "call_id": "c1", "output": "done"}
    ]


def test_format_tool_result_with_images_keeps_images() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input(
        [
            {
                "role": "tool",
                "content": "saw it",
                "tool_call_id": "c1",
                "images": ["QUJD"],
            }
        ]
    )
    # Assert
    assert result == [
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": [
                {"type": "input_text", "text": "saw it"},
                {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
            ],
        }
    ]


def test_format_tool_result_with_empty_content_and_images() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input(
        [{"role": "tool", "content": "", "tool_call_id": "c1", "images": ["QUJD"]}]
    )
    # Assert
    output = result[0]["output"]
    assert isinstance(output, list) and output[0] == {"type": "input_text", "text": ""}


def test_format_tool_result_without_content_defaults_empty() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input([{"role": "tool", "tool_call_id": "c1"}])
    # Assert
    assert result == [{"type": "function_call_output", "call_id": "c1", "output": ""}]


def test_format_assistant_tool_calls_nested_and_flat() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input(
        [
            {
                "role": "assistant",
                "content": "working",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "f", "arguments": '{"a":1}'}},
                    {"id": "c2", "name": "g", "arguments": {"b": 2}},
                    "not-a-dict",
                    {"id": "c3", "function": {}, "name": "h", "arguments": "{}"},
                    {"id": "c4"},
                ],
            }
        ]
    )
    # Assert
    assert result[0] == {"role": "assistant", "content": "working"}
    assert result[1] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "f",
        "arguments": '{"a":1}',
    }
    assert result[2]["arguments"] == json.dumps({"b": 2}, ensure_ascii=False)
    assert result[3]["name"] == "h"
    assert result[4] == {
        "type": "function_call",
        "call_id": "c4",
        "name": "",
        "arguments": "",
    }


def test_format_assistant_tool_calls_without_text() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input(
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "name": "f", "arguments": "{}"}],
            }
        ]
    )
    # Assert
    assert result == [
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
    ]


def test_format_regular_message_with_images_uses_parts() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input(
        [{"role": "user", "content": "look", "images": ["QUJD"]}]
    )
    # Assert
    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
            ],
        }
    ]


def test_format_plain_message_defaults_empty_content() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input([{"role": "system"}])
    # Assert
    assert result == [{"role": "system", "content": ""}]


def test_format_accepts_message_objects() -> None:
    # Arrange
    provider = _provider()
    messages = [
        Message(role="user", content="hi", images=["QUJD"]),
        Message(role="assistant", content="ok", tool_calls=[{"id": "c1", "name": "f"}]),
        Message(role="tool", content="done", tool_call_id="c1"),
    ]
    # Act
    result = provider._format_input(messages)
    # Assert
    assert result[0]["role"] == "user" and isinstance(result[0]["content"], list)
    assert result[1] == {"role": "assistant", "content": "ok"}
    assert result[2]["type"] == "function_call"
    assert result[3] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "done",
    }


def test_format_tool_role_without_call_id_is_regular_message() -> None:
    # Arrange
    provider = _provider()
    # Act
    result = provider._format_input([{"role": "tool", "content": "orphan"}])
    # Assert
    assert result == [{"role": "tool", "content": "orphan"}]


# -- _build_body / _prepare_request -------------------------------------------------------------------------------


def test_build_body_stream_session_options_and_tools() -> None:
    # Arrange
    provider = _provider(request_options={"temperature": 0.5, "tools": True})
    tools = [
        {
            "function": {
                "name": "lookup",
                "description": "d",
                "parameters": {"type": "object"},
            }
        },
        {"name": "flat", "description": "f"},
        {"function": {"description": "nameless"}},
    ]
    # Act
    body = provider._build_body(
        [{"role": "user", "content": "hi"}],
        model="gpt-5",
        stream=True,
        tools=tools,
        session_id="ses_1",
        reasoning_effort="high",
    )
    # Assert
    assert body["stream"] is True
    assert body["prompt_cache_key"] == "ses_1"
    assert body["temperature"] == 0.5
    assert provider.request_options == {"temperature": 0.5, "tools": True}
    assert [t["name"] for t in body["tools"]] == ["lookup", "flat"]
    assert body["tools"][1]["parameters"] == {"type": "object", "properties": {}}
    assert body["reasoning"] == {"effort": "high"}


def test_build_body_minimal_omits_stream_key_tools_and_reasoning() -> None:
    # Arrange
    provider = _provider()
    # Act
    body = provider._build_body([{"role": "user", "content": "hi"}], model="gpt-5")
    # Assert
    assert "stream" not in body and "prompt_cache_key" not in body
    assert "tools" not in body and "reasoning" not in body


def test_build_body_uses_default_effort_and_keeps_existing_reasoning() -> None:
    # Arrange
    with_default = _provider(default_reasoning_effort="low")
    # Act
    body = with_default._build_body([{"role": "user", "content": "hi"}], model="gpt-5")
    against_explicit = _provider(request_options={"reasoning": {"effort": "low"}})
    kept = against_explicit._build_body(
        [{"role": "user", "content": "hi"}], model="gpt-5", reasoning_effort="high"
    )
    # Assert
    assert body["reasoning"] == {"effort": "low"}
    assert kept["reasoning"] == {"effort": "low"}


def test_build_body_empty_tools_list_sends_no_tools() -> None:
    # Arrange
    provider = _provider()
    # Act
    body = provider._build_body(
        [{"role": "user", "content": "hi"}], model="gpt-5", tools=[]
    )
    # Assert
    assert "tools" not in body


def test_prepare_request_targets_responses_and_sets_accept_for_stream() -> None:
    # Arrange
    provider = _provider()
    # Act
    url, headers, body = provider._prepare_request(
        [{"role": "user", "content": "hi"}], "gpt-5", True, None, "ses_1", None
    )
    url2, headers2, _ = provider._prepare_request(
        [{"role": "user", "content": "hi"}], "gpt-5", False, None, None, None
    )
    # Assert
    assert url == "https://api.example.com/v1/responses"
    assert headers["Accept"] == "text/event-stream"
    assert "Accept" not in headers2
    assert body["model"] == "gpt-5" and url2 == url


# -- _parse_output_to_done --------------------------------------------------------------------------------------------


def test_parse_output_reads_text_tools_usage_and_cache() -> None:
    # Arrange
    provider = _provider()
    # Act
    done = provider._parse_output_to_done(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "he"},
                        {"type": "output_text", "text": "llo"},
                        {"type": "refusal", "text": "no"},
                        "not-a-dict",
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "f",
                    "arguments": '{"a":1}',
                },
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 60},
            },
        }
    )
    # Assert
    assert isinstance(done, Done)
    assert done.content == "hello"
    assert [(c.id, c.name) for c in done.tool_calls or []] == [("c1", "f")]
    assert (done.tokens_input, done.tokens_output, done.cache_read_tokens) == (
        100,
        5,
        60,
    )


def test_parse_output_tolerates_shapes_and_missing_cache() -> None:
    # Arrange
    provider = _provider()
    # Act
    non_dict = provider._parse_output_to_done(["nope"])
    odd_items = provider._parse_output_to_done(
        {
            "output": [
                "nope",
                {"type": "reasoning", "content": "encrypted"},
                {"type": "message"},
                {
                    "type": "function_call",
                    "id": "only-id",
                    "name": "f",
                    "arguments": {"a": 1},
                },
                {"type": "function_call"},
                {"type": "other"},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
    )
    # Assert
    assert non_dict == Done(content="", tool_calls=[])
    assert odd_items.content == ""
    assert odd_items.cache_read_tokens is None
    first, second = odd_items.tool_calls or []
    assert (first.id, first.name) == ("only-id", "f")
    assert json.loads(first.arguments) == {"a": 1}
    assert (second.id, second.name) == ("", "")


@pytest.mark.xfail(
    reason="Non-stream function calls without arguments serialize as null",
    strict=True,
)
def test_parse_output_defaults_missing_function_arguments() -> None:
    provider = _provider()

    done = provider._parse_output_to_done({"output": [{"type": "function_call"}]})

    assert done.tool_calls is not None
    assert done.tool_calls[0].arguments == "{}"


def test_parse_output_non_dict_details_means_no_cache() -> None:
    # Arrange
    provider = _provider()
    # Act
    done = provider._parse_output_to_done(
        {"output": [], "usage": {"input_tokens_details": "x"}}
    )
    # Assert
    assert done.cache_read_tokens is None
    assert done.tokens_input is None and done.tokens_output is None


def test_parse_response_delegates_to_output_parser() -> None:
    # Arrange
    provider = _provider()
    body = {"output": [], "usage": {"input_tokens": 1, "output_tokens": 1}}
    # Act
    done = provider._parse_response(body)
    # Assert
    assert isinstance(done, Done) and done.tokens_input == 1


# -- end-to-end through the shared transport -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_returns_done(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    _install_fake(
        monkeypatch,
        _FakeResp(
            json_data={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hi"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "f",
                        "arguments": "{}",
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }
        ),
    )
    # Act
    result = await _provider().chat([Message(role="user", content="hi")], model="gpt-5")
    # Assert
    assert isinstance(result, Done)
    assert result.content == "hi" and [c.id for c in result.tool_calls or []] == ["c1"]


@pytest.mark.asyncio
async def test_chat_http_error_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    _install_fake(monkeypatch, _FakeResp(status=400, text_data="bad key"))
    # Act
    result = await _provider().chat([Message(role="user", content="hi")], model="gpt-5")
    # Assert
    assert isinstance(result, Error) and "HTTP 400" in result.message


@pytest.mark.asyncio
async def test_chat_unexpected_exception_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _install_fake(monkeypatch, RuntimeError("boom"))
    # Act
    result = await _provider().chat([Message(role="user", content="hi")], model="gpt-5")
    # Assert
    assert isinstance(result, Error) and "boom" in result.message


@pytest.mark.asyncio
async def test_stream_end_to_end_with_tools_usage_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    usage = {
        "input_tokens": 100,
        "output_tokens": 5,
        "input_tokens_details": {"cached_tokens": 60},
    }
    lines = [
        _sse({"type": "response.output_text.delta", "delta": "hi"}),
        _sse(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "lookup",
                    "arguments": "",
                },
            }
        ),
        _sse(
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "delta": '{"q":1}',
            }
        ),
        _sse({"type": "response.function_call_arguments.done", "output_index": 0}),
        _completed_line("hi", usage),
    ]
    session = _install_fake(monkeypatch, _FakeResp(lines=lines))
    # Act
    collected = await _collect(_provider(), session_id="ses_1")
    # Assert
    kinds = [type(event).__name__ for event in collected]
    assert kinds == [
        "TextDelta",
        "ToolInputStart",
        "ToolInputDelta",
        "ToolInputEnd",
        "ToolCall",
        "Done",
    ]
    done = collected[-1]
    assert isinstance(done, Done)
    assert (done.tokens_input, done.tokens_output, done.cache_read_tokens) == (
        100,
        5,
        60,
    )
    sent = session.calls[0]
    assert sent["url"].endswith("/responses")
    assert sent["headers"]["Accept"] == "text/event-stream"
    assert sent["json"]["prompt_cache_key"] == "ses_1"
    assert sent["timeout"].sock_read == STREAM_IDLE_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_stream_non_200_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    _install_fake(monkeypatch, _FakeResp(status=500, text_data="down"))
    # Act
    collected = await _collect(_provider())
    # Assert
    assert isinstance(collected[-1], Error) and "HTTP 500" in collected[-1].message


@pytest.mark.asyncio
async def test_stream_post_exception_yields_partial_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _install_fake(monkeypatch, RuntimeError("net down"))
    # Act
    collected = await _collect(_provider())
    # Assert
    assert isinstance(collected[-1], Error) and "net down" in collected[-1].message


@pytest.mark.asyncio
async def test_stream_failed_event_stops_at_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    lines = [
        _sse({"type": "response.output_text.delta", "delta": "half"}),
        _sse(
            {
                "type": "response.failed",
                "response": {"error": {"code": "server_error", "message": "boom"}},
            }
        ),
    ]
    _install_fake(monkeypatch, _FakeResp(lines=lines))
    # Act
    collected = await _collect(_provider())
    # Assert
    assert isinstance(collected[-1], Error)
    assert collected[-1].content == "half" and "boom" in collected[-1].message


@pytest.mark.asyncio
async def test_stream_truncated_without_completed_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _install_fake(
        monkeypatch,
        _FakeResp(lines=[_sse({"type": "response.output_text.delta", "delta": "ok"})]),
    )
    # Act
    collected = await _collect(_provider())
    # Assert
    assert isinstance(collected[-1], Error)
    assert "response.completed" in collected[-1].message
