"""Offline coverage for session models and in-memory session semantics."""

from __future__ import annotations

import json

import pytest

from novakit.session.memory import InMemorySession
from novakit.session.models import (
    DISPLAY_TEXT_KEY,
    Message,
    MessageFilter,
    NewMessage,
    Session,
    SessionContext,
    decode_display_text,
    encode_display_text,
)


def test_session_context_create_defaults_and_custom_key() -> None:
    defaulted = SessionContext.create()
    assert defaulted.id
    assert defaulted.agent_key == "main"
    assert defaulted.metadata == {}
    assert defaulted.created_at == defaulted.updated_at > 0

    custom = SessionContext.create(agent_key="dev")
    assert custom.agent_key == "dev"
    assert custom.id != defaulted.id


def test_dataclass_defaults_cover_factories() -> None:
    first = SessionContext(id="s1")
    second = SessionContext(id="s2")
    first.metadata["k"] = "v"
    assert second.metadata == {}
    assert first.title is None

    message = Message(id="m1", session_id="s1", role="user", content="hi")
    assert message.time_created > 0
    assert message.summary == 0
    assert message.compacted == 0

    session = Session(id="s1")
    assert session.created_at > 0 and session.updated_at > 0
    assert session.pinned is False
    assert session.metadata is None

    assert MessageFilter().limit is None
    draft = NewMessage(session_id="s1", role="user", content="hi")
    assert draft.summary is False and draft.time_created is None


def test_message_full_fields_round_trip() -> None:
    message = Message(
        id="m1",
        session_id="s1",
        role="tool",
        content="model text",
        model="m",
        format="text",
        variant="v",
        summary=1,
        compacted=0,
        finish="stop",
        error=None,
        cost=0.5,
        tokens_input=10,
        tokens_output=20,
        cache_read_tokens=2,
        reasoning_effort="low",
        time_created=123,
        tool_calls=[{"name": "t"}],
        tool_call_id="call-1",
        data=json.dumps({DISPLAY_TEXT_KEY: "seen"}),
        images=["img"],
        reasoning_content="why",
        group_id="g",
        reasoning_elapsed_ms=7,
        provider_meta={"sig": "abc"},
    )
    assert message.tool_calls == [{"name": "t"}]
    assert decode_display_text(message.data) == "seen"
    assert message.provider_meta == {"sig": "abc"}


def test_session_and_filter_custom_values() -> None:
    session = Session(
        id="s1",
        agent_key="dev",
        title="t",
        parent_id="p",
        workspace_dir="/tmp/ws",
        pinned=True,
        summary_goal="g",
        summary_accomplished="a",
        summary_remaining="r",
        compacted_at=9,
        message_count=3,
        turn_count=2,
        metadata={"k": "v"},
        project_id="proj",
        provider="prov",
        model="m",
        reasoning_effort="low",
    )
    assert session.pinned is True and session.message_count == 3

    only = MessageFilter(
        include_compacted=True, exclude_tool_role=True, only_non_summary=True, limit=5
    )
    assert (only.include_compacted, only.limit) == (True, 5)


def test_encode_display_text() -> None:
    assert encode_display_text(None, "hello") is None
    assert encode_display_text("hello", "hello") is None
    packed = encode_display_text("seen 你好", "model text")
    assert packed is not None
    assert json.loads(packed) == {DISPLAY_TEXT_KEY: "seen 你好"}


def test_decode_display_text_matrix() -> None:
    assert decode_display_text(None) is None
    assert decode_display_text("") is None
    assert decode_display_text("not json{{{") is None
    assert decode_display_text("[1, 2]") is None
    assert decode_display_text("42") is None
    assert decode_display_text('"just a string"') is None
    assert decode_display_text(json.dumps({"other": "x"})) is None
    assert decode_display_text(json.dumps({DISPLAY_TEXT_KEY: 7})) is None
    assert decode_display_text(json.dumps({DISPLAY_TEXT_KEY: "seen"})) == "seen"
    assert decode_display_text(123) is None  # type: ignore[arg-type]


def test_current_session_set_get_and_clear() -> None:
    store = InMemorySession()
    assert store.get_current_session() is None
    session = SessionContext(id="s1")
    store.set_current_session(session)
    assert store.get_current_session() is session
    store.set_current_session(None)
    assert store.get_current_session() is None


@pytest.mark.asyncio
async def test_create_title_persist_and_metadata_defaults() -> None:
    store = InMemorySession()
    assert (await store.create_session(first_message="  hi  ")).title == "hi"
    assert (await store.create_session(first_message="")).title == "New Session"
    assert (await store.create_session(first_message="   ")).title == "New Session"
    assert (await store.create_session(first_message=None)).title == "New Session"
    session = await store.create_session(
        persist=False,
        first_message="x",
        parent_id="p",
        workspace_dir="/w",
        project_id="proj",
    )
    assert session.metadata == {}
    assert session.parent_id == "p"


@pytest.mark.asyncio
async def test_get_messages_empty_limit_last_n_edges() -> None:
    assert await InMemorySession().get_messages() == []
    assert await InMemorySession().get_messages(session_id="missing") == []

    store = InMemorySession()
    session = await store.create_session()
    for index in range(3):
        await store.add_message(role="user", content=f"m{index}")
    assert await store.get_messages(session_id="missing") == []
    assert await store.get_messages(last_n=0) == []
    assert await store.get_messages(limit=0) == []
    assert [m.content for m in await store.get_messages(session_id=session.id)] == [
        "m0",
        "m1",
        "m2",
    ]


@pytest.mark.asyncio
async def test_get_messages_sorts_by_time_created() -> None:
    store = InMemorySession()
    await store.create_session()
    messages = [
        await store.add_message(role="user", content=f"m{index}") for index in range(3)
    ]
    messages[2].time_created = messages[0].time_created - 10
    assert [m.content for m in await store.get_messages()] == ["m2", "m0", "m1"]


@pytest.mark.asyncio
async def test_apply_generated_title_current_branches() -> None:
    store = InMemorySession()
    assert await store.apply_generated_title("missing", "t", "e") is False

    first = await store.create_session(first_message="draft")
    store.set_current_session(None)
    assert await store.apply_generated_title(first.id, "solo", "draft") is True
    assert first.title == "solo"

    second = await store.create_session(first_message="other")
    assert await store.apply_generated_title(first.id, "fresh", "solo") is True
    assert first.title == "fresh"
    assert second.title == "other"


@pytest.mark.asyncio
async def test_rollback_edges_session_missing_and_clamp() -> None:
    assert await InMemorySession().rollback_messages(["m1"]) == 0

    store = InMemorySession()
    session = await store.create_session()
    assert await store.rollback_messages([]) == 0
    assert await store.rollback_messages(["missing"]) == 0
    assert await store.rollback_messages(["missing"], session_id="missing") == 0

    first = await store.add_message(role="user", content="one")
    second = await store.add_message(role="user", content="two")
    session.message_count = 0
    session.turn_count = 0
    assert await store.rollback_messages([first.id, second.id]) == 2
    assert (session.message_count, session.turn_count) == (0, 0)

    third = await store.add_message(role="user", content="three")
    store._sessions.clear()
    assert await store.rollback_messages([third.id]) == 1


@pytest.mark.asyncio
async def test_add_message_full_params_and_detached_current() -> None:
    store = InMemorySession()
    session = await store.create_session()
    detached = SessionContext(id=session.id, agent_key=session.agent_key)
    store.set_current_session(detached)
    message = await store.add_message(
        role="assistant",
        content="done",
        tool_calls=[{"name": "t"}],
        tool_call_id="call-1",
        images=["img"],
        reasoning_content="why",
        group_id="g",
        reasoning_elapsed_ms=3,
        error=None,
        tokens_input=1,
        tokens_output=2,
        cache_read_tokens=0,
        cost=0.1,
        provider_meta={"sig": "s"},
        model="m",
        reasoning_effort="low",
        variant="v",
        data="d",
    )
    assert message.session_id == session.id
    assert message.tool_call_id == "call-1"
    assert store._sessions[session.id].message_count == 1
    assert detached.message_count == 1

    assert await store.rollback_messages([message.id]) == 1
    assert store._sessions[session.id].message_count == 0
    assert detached.message_count == 0
