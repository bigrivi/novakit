"""Novakit session fork: copy a session prefix into a new session.

What you'll learn:
    - How to adapt an existing store to the five-method SessionForkStore.
    - How fork_session copies a whole history with metadata and model route.
    - How a message_id fork point copies just the inclusive prefix.
    - What happens with an unknown fork point (ForkPointNotFound).

Run:
    python examples/06_session_fork.py

Everything stays in memory inside a temp workspace; the source session is
never modified by forking.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import replace
from typing import Any

from novakit.session import InMemorySession
from novakit.session.fork import FORK_SUFFIX, ForkPointNotFound, fork_session
from novakit.session.models import Message, NewMessage, SessionContext


class InMemoryForkStore:
    """Teach-by-example fork adapter over a single InMemorySession.

    Copy this pattern to plug forking into your own storage: forked sessions
    are recorded as rows here while their messages live in the wrapped
    InMemorySession; source sessions are read straight from it.
    """

    def __init__(self) -> None:
        self.memory = InMemorySession()
        self._rows: dict[str, dict[str, Any]] = {}
        self._routes: dict[str, tuple[str | None, str | None, str | None]] = {}
        self._forked_ids: dict[str, str] = {}

    def _target(self, session_id: str) -> str:
        """Translate a forked id to the inner session holding its messages."""
        return self._forked_ids.get(session_id, session_id)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Return the session row merged with its model route, if known."""
        if session_id in self._rows:
            row = dict(self._rows[session_id])
        else:
            stored = await self.memory.load_session(session_id)
            if stored is None:
                return None
            row = {
                "id": stored.id,
                "agent_key": stored.agent_key,
                "title": stored.title,
                "workspace_dir": stored.workspace_dir,
                "project_id": stored.project_id,
            }
        provider, model, effort = self._routes.get(session_id, (None, None, None))
        return {
            **row,
            "provider": provider,
            "model": model,
            "reasoning_effort": effort,
        }

    async def get_messages(self, session_id: str) -> list[Message]:
        """Return the visible messages of one session, oldest first."""
        if await self.memory.load_session(self._target(session_id)) is None:
            return []
        return await self.memory.get_messages()

    async def save_session(self, session: SessionContext) -> None:
        """Record the forked row and back it with an inner session."""
        created = await self.memory.create_session(
            agent_key=session.agent_key, metadata=dict(session.metadata)
        )
        created.title = session.title
        created.workspace_dir = session.workspace_dir
        created.project_id = session.project_id
        self._forked_ids[session.id] = created.id
        self._rows[session.id] = {
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
        """Remember the model route for one session."""
        self._routes[session_id] = (provider, model, reasoning_effort)
        return True

    async def add_message(self, message: NewMessage) -> Message:
        """Append one copied message to the forked session."""
        await self.memory.load_session(self._target(message.session_id))
        stored = await self.memory.add_message(
            message.role,
            message.content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            summary=message.summary,
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
            time_created=message.time_created,
            data=message.data,
        )
        return replace(stored, session_id=message.session_id)


async def main() -> None:
    """Fork a full history, a prefix, then a bad point, in order."""
    print("06 session fork")
    with tempfile.TemporaryDirectory() as workspace:
        store = InMemoryForkStore()
        source_id = await build_source_session(store, workspace)
        await show_full_fork(store, source_id)
        await show_point_fork(store, source_id)
        await show_bad_fork_point(store, source_id)
    print("06 session fork: done")


async def build_source_session(store: InMemoryForkStore, workspace: str) -> str:
    """Build a source session with a user/tool-call/tool-result exchange."""
    # Configure: one titled session rooted at the temp workspace.
    source = await store.memory.create_session(
        first_message="Plan the release",
        metadata={"team": "core"},
        workspace_dir=workspace,
        project_id="p1",
    )
    # Execute: a realistic trio — prompt, tool call, tool result.
    # Why a tool pair: forks must keep call id/arguments linked to the result.
    await store.memory.add_message(role="user", content="Please read file README.md")
    await store.memory.add_message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "name": "read",
                "arguments": '{"filePath": "README.md"}',
            }
        ],
    )
    await store.memory.add_message(
        role="tool", content="file bytes", tool_call_id="call-1"
    )
    await store.set_session_route(source.id, "test-provider", "test-model", "low")
    print(f"  source {source.id} holds 3 messages (user/assistant/tool)")
    return source.id


async def show_full_fork(store: InMemoryForkStore, source_id: str) -> None:
    """Fork without a point: copy every message, metadata, and route."""
    # Configure + execute: no message_id means "copy the whole history".
    result = await fork_session(store, source_id)
    forked_id = str(result["session_id"])
    copied = await store.get_messages(forked_id)

    # Observe: messages, tool linkage, metadata, and route all carried over.
    print(f"  full fork -> {forked_id} ({result['messages']} messages)")
    print(f"  title: {result['title']!r} (fork marker: {FORK_SUFFIX!r})")
    print(f"  copied roles: {[(m.role, m.content) for m in copied]}")
    print(f"  tool call kept: {copied[1].tool_calls}")
    print(f"  tool result linked via call id: {copied[2].tool_call_id!r}")
    row = await store.get_session(forked_id)
    source_row = await store.get_session(source_id)
    print(
        f"  workspace carried over: {row['workspace_dir'] == source_row['workspace_dir']}"
    )
    print(f"  project: {row['project_id']!r}")
    print(
        f"  route: provider={row['provider']!r} "
        f"model={row['model']!r} effort={row['reasoning_effort']!r}"
    )


async def show_point_fork(store: InMemoryForkStore, source_id: str) -> None:
    """Fork at a message id: copy just the inclusive prefix under a title."""
    # Configure: pick the first message as the branch point.
    source_messages = await store.get_messages(source_id)

    # Execute: everything up to and including that message is copied.
    result = await fork_session(
        store, source_id, message_id=source_messages[0].id, title="Branch"
    )

    # Observe: a one-message branch with our chosen title.
    branched = await store.get_messages(str(result["session_id"]))
    print(f"  point fork -> {result['session_id']} ({result['messages']} message)")
    print(f"  title: {result['title']!r}")
    print(f"  fork point: {result['fork_point_message_id']}")
    print(f"  branched messages: {[m.content for m in branched]}")


async def show_bad_fork_point(store: InMemoryForkStore, source_id: str) -> None:
    """Fork with an unknown id: the source is untouched, an error surfaces."""
    # Execute + observe: bad points raise instead of copying half a history.
    # Why raise here: silently branching from nowhere would corrupt the tree,
    # so the caller learns about it through ForkPointNotFound.
    try:
        await fork_session(store, source_id, message_id="missing")
    except ForkPointNotFound as exc:
        print(f"  unknown point raised ForkPointNotFound: {exc}")
    print(f"  source still holds {len(await store.get_messages(source_id))} messages")


if __name__ == "__main__":
    asyncio.run(main())
