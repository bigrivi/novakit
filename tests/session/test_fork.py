"""Direct tests for the portable session-fork algorithm."""

from __future__ import annotations

from typing import Any

import pytest

from novakit.session.fork import FORK_SUFFIX, ForkPointNotFound, fork_session
from novakit.session.models import Message, NewMessage, SessionContext


class MemoryForkStore:
    """Small in-memory implementation of the fork storage seam."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.messages: dict[str, list[Message]] = {}
        self.routes: dict[str, tuple[str | None, str | None, str | None]] = {}

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        session = self.sessions.get(session_id)
        return dict(session) if session is not None else None

    async def get_messages(self, session_id: str) -> list[Message]:
        return list(self.messages.get(session_id, []))

    async def save_session(self, session: SessionContext) -> None:
        self.sessions[session.id] = {
            "id": session.id,
            "agent_key": session.agent_key,
            "title": session.title,
            "workspace_dir": session.workspace_dir,
            "project_id": session.project_id,
        }

    async def set_session_route(
        self,
        session_id: str,
        provider: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> bool:
        self.routes[session_id] = (provider, model, reasoning_effort)
        return True

    async def add_message(self, message: NewMessage) -> Message:
        record = Message(
            id=f"new-{len(self.messages.get(message.session_id, []))}",
            session_id=message.session_id,
            role=message.role,
            content=message.content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            summary=1 if message.summary else 0,
            images=message.images,
            reasoning_content=message.reasoning_content,
            group_id=message.group_id,
            reasoning_elapsed_ms=message.reasoning_elapsed_ms,
            tokens_input=message.tokens_input,
            tokens_output=message.tokens_output,
            cache_read_tokens=message.cache_read_tokens,
            provider_meta=message.provider_meta,
            model=message.model,
            reasoning_effort=message.reasoning_effort,
            error=message.error,
            variant=message.variant,
            time_created=message.time_created or 0,
            data=message.data,
        )
        self.messages.setdefault(message.session_id, []).append(record)
        return record


def _message(message_id: str, role: str, content: str, **extra: object) -> Message:
    return Message(
        id=message_id,
        session_id="source",
        role=role,
        content=content,
        **extra,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_fork_copies_whole_history_metadata_and_route() -> None:
    store = MemoryForkStore()
    store.sessions["source"] = {
        "id": "source",
        "agent_key": "dev",
        "title": "Origin",
        "workspace_dir": "/repo",
        "project_id": "p1",
        "provider": "openai",
        "model": "gpt-test",
        "reasoning_effort": "high",
    }
    store.messages["source"] = [
        _message("u1", "user", "question"),
        _message(
            "a1",
            "assistant",
            "",
            tool_calls=[{"id": "call-1", "name": "read"}],
            tokens_input=10,
        ),
        _message("t1", "tool", "answer", tool_call_id="call-1"),
    ]

    result = await fork_session(store, "source")

    forked_id = result["session_id"]
    assert result["title"] == f"Origin{FORK_SUFFIX}"
    assert result["messages"] == 3
    assert result["fork_point_message_id"] == "t1"
    assert store.sessions[forked_id]["agent_key"] == "dev"
    assert store.sessions[forked_id]["workspace_dir"] == "/repo"
    assert store.routes[forked_id] == ("openai", "gpt-test", "high")
    copied = store.messages[forked_id]
    assert [(message.role, message.content) for message in copied] == [
        ("user", "question"),
        ("assistant", ""),
        ("tool", "answer"),
    ]
    assert copied[1].tool_calls == [{"id": "call-1", "name": "read"}]
    assert copied[2].tool_call_id == "call-1"
    assert len(store.messages["source"]) == 3


@pytest.mark.asyncio
async def test_fork_prefix_is_inclusive_and_title_is_explicit() -> None:
    store = MemoryForkStore()
    store.sessions["source"] = {"id": "source", "title": "Origin"}
    store.messages["source"] = [
        _message("u1", "user", "one"),
        _message("a1", "assistant", "two"),
        _message("u2", "user", "three"),
    ]

    result = await fork_session(store, "source", message_id="a1", title="Branch")

    assert result["messages"] == 2
    assert result["title"] == "Branch"
    assert result["fork_point_message_id"] == "a1"
    assert [message.content for message in store.messages[result["session_id"]]] == [
        "one",
        "two",
    ]


@pytest.mark.asyncio
async def test_fork_missing_session_and_missing_point() -> None:
    store = MemoryForkStore()
    with pytest.raises(FileNotFoundError, match="not found"):
        await fork_session(store, "missing")
    store.sessions["source"] = {"id": "source"}
    store.messages["source"] = [_message("u1", "user", "one")]
    with pytest.raises(ForkPointNotFound, match="not in session"):
        await fork_session(store, "source", message_id="missing")


@pytest.mark.asyncio
async def test_fork_empty_history_uses_defaults() -> None:
    store = MemoryForkStore()
    store.sessions["source"] = {"id": "source"}

    result = await fork_session(store, "source", default_agent_key="fallback")

    assert result["messages"] == 0
    assert result["fork_point_message_id"] is None
    assert result["title"] == f"Session{FORK_SUFFIX}"
    assert store.sessions[result["session_id"]]["agent_key"] == "fallback"
