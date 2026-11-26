"""Gap coverage for compaction accessors, snip edges, and compactor paths."""

from __future__ import annotations

import copy
import json
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from novakit.compaction.compactor import (
    DefaultCompactor,
    compact,
    run_compaction_plan,
)
from novakit.compaction.config import CompactionConfig
from novakit.compaction.messages import (
    compacted_at_of,
    content_of,
    created_at_of,
    id_of,
    is_summary,
    new_message_id,
    role_of,
    session_id_of,
    set_compacted_at,
    tokens_input_of,
    tokens_output_of,
    tool_call_id_of,
    tool_call_ids_of,
    tool_calls_of,
)
from novakit.compaction.policy import CompactionPlan, snip_old_tool_results
from novakit.compaction.store import MemoryCompactionStore
from novakit.events import AgentEvent
from novakit.session.memory import InMemorySession
from novakit.session.models import Message, SessionContext


class StubResolver:
    """Deterministic token source: 1 token per 4 chars, fixed window."""

    def __init__(self, limit: int = 1000) -> None:
        self.limit = limit

    def estimate_messages(self, messages: list, model: str) -> int:
        total = 0
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content", "") or ""
            else:
                content = getattr(message, "content", "") or ""
            total += max(1, len(content) // 4)
        return total

    def context_limit(self, model: str, provider: str) -> int:
        return self.limit


def test_new_message_id_is_a_unique_uuid() -> None:
    first, second = new_message_id(), new_message_id()
    assert first != second
    uuid.UUID(first)
    uuid.UUID(second)


def test_tool_call_ids_from_json_string_dict() -> None:
    message = {"tool_calls": json.dumps([{"id": "a"}, {"id": "b"}, {"name": "x"}])}
    assert tool_call_ids_of(message) == ["a", "b"]


def test_tool_call_ids_from_json_string_object() -> None:
    message = SimpleNamespace(tool_calls=json.dumps([{"id": "c"}]))
    assert tool_call_ids_of(message) == ["c"]


def test_tool_call_ids_rejects_corrupt_and_non_list() -> None:
    assert tool_call_ids_of({"tool_calls": '{"id":'}) == []
    assert tool_call_ids_of({"tool_calls": '"just a string"'}) == []
    assert tool_call_ids_of({"tool_calls": '{"id": "a"}'}) == []
    assert tool_call_ids_of({"tool_calls": "5"}) == []
    assert tool_call_ids_of({"tool_calls": None}) == []
    assert tool_call_ids_of({}) == []


def test_tool_call_id_dict_and_object() -> None:
    assert tool_call_id_of({"tool_call_id": "tc-1"}) == "tc-1"
    assert tool_call_id_of({}) == ""
    assert tool_call_id_of({"tool_call_id": ""}) == ""
    assert tool_call_id_of(SimpleNamespace(tool_call_id="tc-2")) == "tc-2"


def test_session_id_dict_none_and_object() -> None:
    assert session_id_of({"id": "s-1"}) == "s-1"
    assert session_id_of(None) is None
    assert session_id_of(SimpleNamespace(id="s-2")) == "s-2"


def test_compacted_at_dict_and_setters() -> None:
    assert compacted_at_of({"compacted_at": 42}) == 42
    assert compacted_at_of({}) is None
    assert compacted_at_of(None) is None

    session: dict[str, Any] = {}
    set_compacted_at(session, 7)
    assert session["compacted_at"] == 7
    assert set_compacted_at(None, 7) is None

    context = SessionContext.create("test-agent")
    assert compacted_at_of(context) is None
    set_compacted_at(context, 9)
    assert context.compacted_at == 9
    assert compacted_at_of(context) == 9


def test_object_branches_of_scalar_accessors() -> None:
    message = SimpleNamespace(
        summary=0,
        content="hello",
        tool_calls=[{"id": "t"}],
        role="assistant",
        id="m-1",
        time_created=123,
        tokens_input=4,
        tokens_output=7,
    )
    assert is_summary(message) is False
    assert content_of(message) == "hello"
    assert tool_calls_of(message) == [{"id": "t"}]
    assert role_of(message) == "assistant"
    assert id_of(message) == "m-1"
    assert created_at_of(message) == 123
    assert tokens_input_of(message) == 4
    assert tokens_output_of(message) == 7


def test_dict_tokens_output() -> None:
    assert tokens_output_of({"tokens_output": 9}) == 9
    assert tokens_output_of({}) == 0


def test_snip_skips_non_string_and_empty_tool_content() -> None:
    list_content = {"role": "tool", "content": ["block"], "id": "t1"}
    empty_content = {"role": "tool", "content": "", "id": "t2"}
    none_object = SimpleNamespace(role="tool", content=None, id="t3")
    messages: list = [list_content, empty_content, none_object]
    assert snip_old_tool_results(messages, resolver=StubResolver()) is messages
    assert list_content["content"] == ["block"]
    assert empty_content["content"] == ""
    assert none_object.content is None


def test_short_tool_content_spends_budget_but_stays_verbatim() -> None:
    short = {"role": "tool", "content": "s" * 400, "id": "short"}
    big = {"role": "tool", "content": "v" * 3000, "id": "big"}
    messages = [dict(big), dict(short)]
    snip_old_tool_results(
        messages,
        max_chars=200,
        preserve_last_n_messages=10,
        tool_output_token_budget=600,
        resolver=StubResolver(),
    )
    # Newest short message is inside the verbatim window and budget: untouched.
    assert messages[1]["content"] == "s" * 400
    # Its 100 tokens consumed the budget, so the older message is trimmed.
    assert "chars snipped" in messages[0]["content"]


def test_generous_budget_keeps_both_tool_messages() -> None:
    messages = [
        {"role": "tool", "content": "v" * 3000, "id": "t1"},
        {"role": "tool", "content": "s" * 400, "id": "t2"},
    ]
    snip_old_tool_results(
        messages,
        max_chars=200,
        preserve_last_n_messages=10,
        tool_output_token_budget=100_000,
        resolver=StubResolver(),
    )
    assert "chars snipped" not in messages[0]["content"]
    assert "chars snipped" not in messages[1]["content"]


@pytest.mark.asyncio
async def test_run_compaction_plan_short_circuits_when_not_needed() -> None:
    store = MemoryCompactionStore(InMemorySession())
    plan = CompactionPlan(
        session_id="s",
        model_max_tokens=100,
        token_count=1,
        message_count=1,
        needs_compaction=False,
    )
    assert (
        await run_compaction_plan(plan, store, None)  # type: ignore[arg-type]
        is False
    )


@pytest.mark.asyncio
async def test_compact_empty_history_returns_false() -> None:
    store = MemoryCompactionStore(InMemorySession())
    assert (
        await compact("s", store, None, messages=[])  # type: ignore[arg-type]
        is False
    )


@pytest.mark.asyncio
async def test_compact_non_positive_split_returns_false() -> None:
    store = MemoryCompactionStore(InMemorySession())
    messages = [{"role": "user", "content": "hi", "id": "u1"}]
    assert (
        await compact(
            "s",
            store,
            None,  # type: ignore[arg-type]
            messages=messages,
            split_index=0,
        )
        is False
    )


class FlakyStore:
    """Store whose history read fails; writes are recorded in memory."""

    def __init__(self) -> None:
        self.updated: dict[str, str] = {}
        self.summaries: list[Message] = []
        self.marked: list[str] = []
        self.stamps: dict[str, int] = {}

    async def update_message_content(self, message_id: str, content: str) -> None:
        self.updated[message_id] = content

    async def add_summary(
        self, session_id: str, content: str, time_created: int
    ) -> Message:
        message = Message(
            id="sum-1",
            session_id=session_id,
            role="assistant",
            content=content,
            summary=1,
            time_created=time_created,
        )
        self.summaries.append(message)
        return message

    async def mark_messages_compacted_by_ids(
        self, session_id: str, message_ids: list[str]
    ) -> None:
        self.marked.extend(message_ids)

    async def update_session_compacted_at(
        self, session_id: str, timestamp: int
    ) -> None:
        self.stamps[session_id] = timestamp

    async def get_messages(self, session_id: str) -> list:
        raise RuntimeError("store gone")


@pytest.mark.asyncio
async def test_post_compaction_context_failure_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream(llm: Any, **kwargs: Any) -> str | None:
        on_delta = kwargs.get("on_delta")
        if on_delta is not None:
            on_delta("folded")
        return "folded"

    monkeypatch.setattr("novakit.compaction.summary.stream_text_once", fake_stream)
    config = CompactionConfig(
        output_reserve_tokens=20,
        summary_reserve_tokens=10,
        summary_keep_tokens=50,
        max_consecutive_failures=3,
    )
    compactor = DefaultCompactor(
        "stub-model",
        "stub-provider",
        store=FlakyStore(),  # type: ignore[arg-type]
        config=config,
        resolver=StubResolver(limit=400),
    )
    messages = [
        {"role": "user", "content": "x" * 400, "id": f"u{i}", "time_created": i}
        for i in range(10)
    ]
    session: dict[str, Any] = {"id": "s"}
    emitted: list = []
    yielded: list = []

    async def emit(event: Any, data: Any) -> None:
        emitted.append((event, data))

    async for event, data in compactor.maybe_compact(
        messages,
        session,
        None,
        emit,  # type: ignore[arg-type]
    ):
        yielded.append((event, data))

    kinds = [event for event, _ in yielded]
    assert kinds == [
        AgentEvent.CONTEXT_UPDATE,
        AgentEvent.COMPACTION_START,
        AgentEvent.COMPACTION_DELTA,
        AgentEvent.COMPACTION_END,
    ]
    assert emitted == [
        frame for frame in yielded if frame[0] is not AgentEvent.COMPACTION_DELTA
    ]
    assert compactor.compacted is True


@pytest.mark.asyncio
async def test_compacted_at_reaches_detached_current_with_same_id() -> None:
    sessions = InMemorySession()
    session = await sessions.create_session(first_message="hi")
    detached = copy.copy(session)
    sessions.set_current_session(detached)

    assert await sessions.update_session_compacted_at(session.id, 1234) is True

    assert session.compacted_at == 1234
    assert detached.compacted_at == 1234
