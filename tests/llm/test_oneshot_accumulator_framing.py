"""Offline coverage for oneshot drains, the accumulator guards, and framers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from novakit.llm.accumulator import RunawayOutput, StreamAccumulator
from novakit.llm.framing import NDJSONFramer, SSEFramer
from novakit.llm.oneshot import stream_text_once
from novakit.llm.provider import Done, Error, Message, ReasoningDelta, TextDelta


class _StubProvider:
    """Yields canned events, raises, or hangs to drive the one-shot loop."""

    def __init__(
        self,
        events: list[Any] | None = None,
        error: Exception | None = None,
        delay: float = 0,
    ) -> None:
        self._events = events if events is not None else []
        self._error = error
        self._delay = delay
        self.kwargs: list[dict[str, Any]] = []

    async def chat_stream(
        self, messages: list[Any], model: str = "m", **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        self.kwargs.append({"messages": messages, "model": model, **kwargs})
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_accumulates_deltas_and_reports_each_chunk() -> None:
    provider = _StubProvider(
        [TextDelta(content="Hello"), TextDelta(content=" world"), Done(content="x")]
    )
    seen: list[str] = []

    text = await stream_text_once(
        provider,  # type: ignore[arg-type]
        messages=[Message(role="user", content="hi")],
        model="m",
        on_delta=seen.append,
    )

    assert text == "Hello world"
    assert seen == ["Hello", " world"]


@pytest.mark.asyncio
async def test_accumulates_without_callback() -> None:
    provider = _StubProvider([TextDelta(content="a"), Done(content="")])

    text = await stream_text_once(
        provider,  # type: ignore[arg-type]
        messages=[Message(role="user", content="hi")],
        model="m",
    )

    assert text == "a"


@pytest.mark.asyncio
async def test_ignores_non_text_frames_but_keeps_done_fallback() -> None:
    provider = _StubProvider(
        [ReasoningDelta(content="plan"), Done(content="only on done")]
    )

    text = await stream_text_once(
        provider,  # type: ignore[arg-type]
        messages=[Message(role="user", content="hi")],
        model="m",
    )

    assert text == "only on done"


@pytest.mark.asyncio
async def test_empty_done_yields_empty_string() -> None:
    provider = _StubProvider([Done(content="")])

    text = await stream_text_once(
        provider,  # type: ignore[arg-type]
        messages=[Message(role="user", content="hi")],
        model="m",
    )

    assert text == ""


@pytest.mark.asyncio
async def test_error_event_returns_none_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _StubProvider([Error(message="HTTP 403 from gateway")])

    with caplog.at_level(logging.WARNING):
        result = await stream_text_once(
            provider,  # type: ignore[arg-type]
            messages=[Message(role="user", content="hi")],
            model="m",
            label="unit test call",
        )

    assert result is None
    assert "HTTP 403 from gateway" in caplog.text


@pytest.mark.asyncio
async def test_provider_raise_returns_none() -> None:
    provider = _StubProvider(error=RuntimeError("boom"))

    result = await stream_text_once(
        provider,  # type: ignore[arg-type]
        messages=[Message(role="user", content="hi")],
        model="m",
    )

    assert result is None


@pytest.mark.asyncio
async def test_timeout_returns_none() -> None:
    provider = _StubProvider([Done(content="too late")], delay=5)

    result = await stream_text_once(
        provider,  # type: ignore[arg-type]
        messages=[Message(role="user", content="hi")],
        model="m",
        timeout=0.01,
    )

    assert result is None


@pytest.mark.asyncio
async def test_forwards_tools_session_and_timeout() -> None:
    schema = [{"type": "function", "function": {"name": "read_file"}}]
    provider = _StubProvider([Done(content="ok")])

    text = await stream_text_once(
        provider,  # type: ignore[arg-type]
        messages=[Message(role="user", content="hi")],
        model="m",
        tools=schema,
        session_id="sess-1",
        timeout=5.0,
    )

    assert text == "ok"
    assert provider.kwargs[0]["tools"] == schema
    assert provider.kwargs[0]["session_id"] == "sess-1"


def test_runaway_output_carries_message() -> None:
    error = RunawayOutput("too big")

    assert str(error) == "too big"
    assert error.message == "too big"


def test_accumulator_appends_text_under_ceiling() -> None:
    acc = StreamAccumulator(model="m", max_content_chars=10, max_tool_arg_chars=10)

    acc.add_text("abc")
    acc.add_text("def")

    assert acc.content == "abcdef"
    assert acc.tokens_input is None
    assert acc.provider_meta is None


def test_accumulator_allows_exact_ceiling_but_rejects_overflow() -> None:
    acc = StreamAccumulator(model="m", max_content_chars=3, max_tool_arg_chars=3)
    acc.add_text("abc")

    with pytest.raises(RunawayOutput, match="runaway/unbounded output"):
        acc.add_text("d")

    assert acc.content == "abc"


def test_accumulator_without_ceiling_never_trips() -> None:
    acc = StreamAccumulator(model="m", max_content_chars=None, max_tool_arg_chars=None)

    acc.add_text("x" * 1000)
    acc.guard_tool_args(10**9)

    assert len(acc.content) == 1000


def test_guard_tool_args_boundary_and_overflow() -> None:
    acc = StreamAccumulator(model="m", max_content_chars=10, max_tool_arg_chars=4)
    acc.guard_tool_args(4)

    with pytest.raises(RunawayOutput, match="runaway/unbounded tool arguments"):
        acc.guard_tool_args(5)


def test_sse_framer_decodes_data_prefixed_json() -> None:
    framer = SSEFramer()

    assert framer.frame('data: {"a": 1}') == {"a": 1}
    assert framer.frame(b'data: {"a": 1}') == {"a": 1}
    assert framer.frame(bytearray(b'data: {"a": 1}')) == {"a": 1}
    assert framer.frame('{"a": 1}') == {"a": 1}


def test_sse_framer_skips_noise_and_sentinels() -> None:
    framer = SSEFramer()

    assert framer.frame("") is None
    assert framer.frame("   ") is None
    assert framer.frame("event: message") is None
    assert framer.frame("data:") is None
    assert framer.frame("data: [DONE]") is None
    assert framer.frame("[DONE]") is None
    assert framer.frame("not json {") is None
    assert framer.frame("[1, 2]") is None


def test_sse_framer_variants_cover_wire_quirks() -> None:
    chat = SSEFramer(require_prefix_space=True, skip_event_lines=False)
    assert chat.frame('data: {"a": 1}') == {"a": 1}
    assert chat.frame("event: message") is None
    assert chat.frame('data:{"a": 1}') is None

    responses = SSEFramer(decode_errors="replace", done_before_unwrap=True)
    assert responses.frame("data: [DONE]") is None
    assert responses.frame(b"\xff data") is None


def test_sse_framer_strict_decode_propagates() -> None:
    with pytest.raises(UnicodeDecodeError):
        SSEFramer(decode_errors="strict").frame(b"\xff\xfe")


def test_sse_framer_non_string_raw_falls_back_to_str() -> None:
    assert SSEFramer().frame(123) is None  # type: ignore[arg-type]


def test_ndjson_framer_decodes_each_line() -> None:
    framer = NDJSONFramer()

    assert framer.frame('{"a": 1}') == {"a": 1}
    assert framer.frame(b'{"a": 1}') == {"a": 1}
    assert framer.frame(bytearray(b'{"a": 1}')) == {"a": 1}
    assert framer.frame("") is None
    assert framer.frame("oops") is None
    assert framer.frame("[1]") is None


def test_ndjson_framer_strict_decode_propagates() -> None:
    with pytest.raises(UnicodeDecodeError):
        NDJSONFramer(decode_errors="strict").frame(b"\xff\xfe")

    assert NDJSONFramer(decode_errors="replace").frame(b"\xff") is None
