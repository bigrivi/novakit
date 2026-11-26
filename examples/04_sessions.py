"""Novakit sessions: InMemorySession CRUD plus a SessionProtocol injection.

What you'll learn:
    - How to create, read, update, and roll back messages in InMemorySession.
    - How load_session behaves for known and unknown ids.
    - How to wrap a session store behind the SessionProtocol interface.
    - How that wrapper plugs into an Agent as host-owned persistence.

Run:
    python examples/04_sessions.py

Everything stays in memory inside a temp workspace; no disk or network.
"""

from __future__ import annotations

import asyncio
import tempfile

from novakit import Agent, AgentConfig, AgentOptions
from novakit.constants import DEFAULT_AGENT_KEY
from novakit.llm import FakerLLMProvider
from novakit.session import InMemorySession, SessionProtocol
from novakit.session.models import Message, SessionContext


async def main() -> None:
    """Show session CRUD, then the persistence-wrapper pattern."""
    print("04 sessions")
    with tempfile.TemporaryDirectory() as tmpdir:
        await show_session_crud(tmpdir)
    await show_protocol_injection()
    print("04 sessions: done")


async def show_session_crud(tmpdir: str) -> None:
    """Create a session, edit its messages, and print each change."""
    # Configure: a fresh in-memory store rooted at the temp workspace.
    session = InMemorySession()
    created = await session.create_session(
        first_message="hello there",
        metadata={"team": "core"},
        workspace_dir=tmpdir,
        project_id="demo",
    )
    print(f"  created session {created.id} title={created.title!r}")

    # Execute: append three messages, then read them back.
    await session.add_message(role="user", content="first")
    assistant = await session.add_message(role="assistant", content="one")
    await session.add_message(role="user", content="second")
    all_messages = await session.get_messages()
    print(f"  messages: {[m.content for m in all_messages]}")
    print(
        f"  limit=1 gives: {[m.content for m in await session.get_messages(limit=1)]}"
    )
    print(
        f"  last_n=1 gives: {[m.content for m in await session.get_messages(last_n=1)]}"
    )

    # Execute + observe: fix the middle message in place.
    # Why update in place: it corrects stored content without shifting ids.
    print(
        f"  update existing message: {await session.update_message_content(assistant.id, 'uno')}"
    )
    print(
        f"  update missing message: {await session.update_message_content('missing', 'x')}"
    )
    print(f"  after update: {[m.content for m in await session.get_messages()]}")

    # Execute + observe: rolling back to one id drops everything after it.
    removed = await session.rollback_messages([assistant.id])
    print(f"  rollback removed {removed} message(s)")
    print(f"  roles after rollback: {[m.role for m in await session.get_messages()]}")

    # Observe: loading a known id returns the session, unknown gives None.
    loaded = await session.load_session(created.id)
    print(f"  load known id: {loaded.id if loaded is not None else None}")
    print(f"  load missing id: {await session.load_session('missing')}")


class LoggingSession:
    """Teach-by-example SessionProtocol: count writes, delegate the rest.

    This is the pattern to copy when wiring real persistence: implement the
    SessionProtocol surface and forward each call to your own store. Here the
    inner store is InMemorySession, so behavior stays identical.
    """

    def __init__(self, inner: InMemorySession) -> None:
        self.inner = inner
        self.add_calls = 0

    def get_current_session(self) -> SessionContext | None:
        """Return the session running in this task."""
        return self.inner.get_current_session()

    async def create_session(
        self,
        *,
        persist: bool = True,
        first_message: str | None = None,
        agent_key: str = DEFAULT_AGENT_KEY,
        metadata: dict | None = None,
        parent_id: str | None = None,
        workspace_dir: str | None = None,
        project_id: str | None = None,
    ) -> SessionContext:
        """Create a session through the inner store."""
        return await self.inner.create_session(
            persist=persist,
            first_message=first_message,
            agent_key=agent_key,
            metadata=metadata,
            parent_id=parent_id,
            workspace_dir=workspace_dir,
            project_id=project_id,
        )

    async def load_session(self, session_id: str) -> SessionContext | None:
        """Load a session through the inner store."""
        return await self.inner.load_session(session_id)

    async def apply_generated_title(
        self, session_id: str, title: str, expected_title: str
    ) -> bool:
        """Apply a generated title through the inner store."""
        return await self.inner.apply_generated_title(session_id, title, expected_title)

    async def get_messages(
        self,
        session_id: str | None = None,
        limit: int | None = None,
        last_n: int | None = None,
    ) -> list[Message]:
        """Read messages through the inner store."""
        return await self.inner.get_messages(
            session_id=session_id, limit=limit, last_n=last_n
        )

    async def rollback_messages(
        self, message_ids: list[str], session_id: str | None = None
    ) -> int:
        """Delete messages through the inner store."""
        return await self.inner.rollback_messages(message_ids, session_id)

    async def add_message(
        self,
        role: str,
        content: str,
        *,
        tool_calls: list | None = None,
        tool_call_id: str | None = None,
        images: list[str] | None = None,
        reasoning_content: str | None = None,
        group_id: str | None = None,
        reasoning_elapsed_ms: int | None = None,
        error: str | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        cache_read_tokens: int | None = None,
        cost: float | None = None,
        provider_meta: dict | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        variant: str | None = None,
        data: str | None = None,
    ) -> Message:
        """Count writes, then append through the inner store."""
        self.add_calls += 1
        return await self.inner.add_message(
            role,
            content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            images=images,
            reasoning_content=reasoning_content,
            group_id=group_id,
            reasoning_elapsed_ms=reasoning_elapsed_ms,
            error=error,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_read_tokens=cache_read_tokens,
            cost=cost,
            provider_meta=provider_meta,
            model=model,
            reasoning_effort=reasoning_effort,
            variant=variant,
            data=data,
        )


async def show_protocol_injection() -> None:
    """Drive a real agent turn through the wrapper session."""
    # Configure: the agent talks only to the SessionProtocol surface.
    store = LoggingSession(InMemorySession())
    agent = Agent(
        AgentConfig(model="demo"),
        FakerLLMProvider(seed=1, reasoning_probability=0.0),
        options=AgentOptions(session=store),
    )

    # Execute: one offline turn persists user + assistant messages.
    frames = [frame async for frame in agent.chat_stream("hello")]

    # Observe: the turn completes and every write went through the wrapper.
    print(f"  turn ended with: {frames[-1][0].name}")
    print(f"  wrapper stored {store.add_calls} message(s)")
    print(
        f"  agent.session honors SessionProtocol: {isinstance(agent.session, SessionProtocol)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
