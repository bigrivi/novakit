"""Offline coverage for novakit.llm.provider event types and retry helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, get_args

import pytest

from novakit.llm.provider import (
    ChatStreamEvent,
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
    ToolResult,
    is_terminal_429,
)


class _StubProvider(LLMProvider):
    """Minimal concrete provider so the abstract bases can be exercised."""

    async def chat(
        self,
        messages: list[Any],
        model: str = "m",
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        *,
        abort_event: asyncio.Event | None = None,
        session_id: str | None = None,
        reasoning_effort: str | None = None,
        **kwargs: object,
    ) -> Done | Error:
        return Done(content="ok")

    def chat_stream(
        self,
        messages: list[Any],
        model: str = "m",
        tools: list[dict[str, Any]] | None = None,
        abort_event: asyncio.Event | None = None,
        total_timeout_seconds: int | None = None,
        session_id: str | None = None,
        reasoning_effort: str | None = None,
        **kwargs: object,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        return self._gen()

    async def _gen(self) -> AsyncGenerator[ChatStreamEvent, None]:
        yield Done(content="ok")
        return

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return 0


def test_terminal_429_needs_quota_or_billing_marker() -> None:
    assert is_terminal_429(500, "insufficient_quota") is False
    assert is_terminal_429(429, "") is False
    assert is_terminal_429(429, None) is False  # type: ignore[arg-type]
    assert is_terminal_429(429, "Rate limited, slow down") is False
    assert is_terminal_429(429, "BILLING problem with account") is True
    assert is_terminal_429(429, "rate limited but quota exceeded") is True


def test_terminal_429_recognizes_every_quota_marker() -> None:
    for marker in (
        "insufficient_quota",
        "insufficient quota",
        "quota exceeded",
        "current quota",
        "exceeded your quota",
        "usage limit reached",
        "billing",
        "credit balance",
    ):
        assert is_terminal_429(429, f"wrap {marker} wrap") is True


def test_message_and_tool_result_defaults() -> None:
    message = Message(role="user", content="hi")
    result = ToolResult()

    assert message.name is None
    assert message.tool_calls is None
    assert message.images is None
    assert message.provider_meta is None
    assert message.model is None
    assert result.success is True
    assert result.content == ""
    assert result.requires_input is False
    assert result.denied is False

    full = Message(
        role="assistant",
        content="t",
        name="n",
        tool_calls=[],
        tool_call_id="c1",
        images=["img"],
        reasoning_content="r",
        provider_meta={"k": "v"},
        model="m",
    )
    assert full.images == ["img"]
    assert ToolResult(success=False, error="boom", denied=True).denied is True


def test_text_and_reasoning_deltas() -> None:
    assert TextDelta().type == "text_delta"
    assert TextDelta(content="a").content == "a"
    assert ReasoningDelta().type == "reasoning_delta"


def test_tool_input_frames_model_dump() -> None:
    assert ToolInputStart(id="c1", name="read").model_dump() == {
        "type": "tool_input_start",
        "id": "c1",
        "name": "read",
    }
    assert ToolInputDelta(id="c1", name="read", delta="{").model_dump() == {
        "type": "tool_input_delta",
        "id": "c1",
        "name": "read",
        "delta": "{",
    }
    assert ToolInputEnd(id="c1", name="read").model_dump() == {
        "type": "tool_input_end",
        "id": "c1",
        "name": "read",
    }
    assert ToolInputError(id="c1", name="read", raw="}").model_dump() == {
        "type": "tool_input_error",
        "id": "c1",
        "name": "read",
        "raw": "}",
    }
    assert ToolCall(id="c1", name="read", arguments="{}").model_dump() == {
        "type": "tool_call",
        "id": "c1",
        "name": "read",
        "arguments": "{}",
    }


def test_done_post_init_defaults_tool_calls_to_empty() -> None:
    assert Done().tool_calls == []
    assert Done(content="t").content == "t"

    calls = [ToolCall(id="c1", name="read", arguments="{}")]
    assert Done(tool_calls=calls).tool_calls == calls


def test_error_defaults_isolate_mutable_state() -> None:
    first = Error(message="a")
    second = Error(message="b")

    assert first.tool_calls == [] and second.tool_calls == []
    assert first.tool_calls is not second.tool_calls
    assert first.content == "" and first.provider_meta is None


def test_retry_notice_defaults() -> None:
    notice = RetryNotice()

    assert notice.type == "retry"
    assert (notice.attempt, notice.total, notice.wait_seconds, notice.reason) == (
        0,
        0,
        0.0,
        "",
    )


def test_chat_stream_event_union_covers_every_frame() -> None:
    members = set(get_args(ChatStreamEvent))

    assert members == {
        TextDelta,
        ReasoningDelta,
        ToolInputStart,
        ToolInputDelta,
        ToolInputEnd,
        ToolInputError,
        ToolCall,
        Done,
        Error,
        RetryNotice,
    }


@pytest.mark.asyncio
async def test_abstract_bases_execute_their_bodies() -> None:
    stub = _StubProvider()

    assert await LLMProvider.chat(stub, []) is None
    assert LLMProvider.chat_stream(stub, []) is None
    assert await LLMProvider.count_tokens(stub, "x") is None
    assert isinstance(stub, LLMProvider)

    assert await stub.chat([]) == Done(content="ok")
    assert [e async for e in stub.chat_stream([])] == [Done(content="ok")]
    assert await stub.count_tokens("hi") == 0
