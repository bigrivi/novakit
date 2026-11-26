"""Portable session-fork algorithm behind a narrow storage seam."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from novakit.constants import DEFAULT_AGENT_KEY
from novakit.session.models import Message, NewMessage, SessionContext

log = logging.getLogger(__name__)

FORK_SUFFIX = " (fork)"


class ForkPointNotFound(ValueError):
    """Raised when the message to fork from is not in the source session."""

    def __init__(self, session_id: str, message_id: str) -> None:
        super().__init__(f"Message {message_id} is not in session {session_id}")


class SessionForkStore(Protocol):
    """Storage operations required to copy one session into another."""

    async def get_session(self, session_id: str) -> dict[str, Any] | None: ...

    async def get_messages(self, session_id: str) -> list[Message]: ...

    async def save_session(self, session: SessionContext) -> None: ...

    async def set_session_route(
        self,
        session_id: str,
        provider: str | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> bool: ...

    async def add_message(self, message: NewMessage) -> Message: ...


async def fork_session(
    store: SessionForkStore,
    session_id: str,
    *,
    message_id: str | None = None,
    title: str | None = None,
    default_agent_key: str = DEFAULT_AGENT_KEY,
) -> dict[str, Any]:
    """Copy a session and its whole-message prefix into a new session.

    Args:
        store: Host persistence implementation.
        session_id: Source session to copy.
        message_id: Last message to include; omitted copies all visible history.
        title: Title for the copy, or ``<origin title> (fork)`` when omitted.
        default_agent_key: Fallback owner when the source row has no agent key.

    Returns:
        Metadata describing the new session and copied message count.

    Raises:
        FileNotFoundError: When the source session does not exist.
        ForkPointNotFound: When *message_id* is not in the source history.
    """
    source = await store.get_session(session_id)
    if source is None:
        raise FileNotFoundError(f"Session {session_id} not found")

    messages = await store.get_messages(session_id)
    copied = _messages_up_to(messages, session_id, message_id)
    fork_point = copied[-1].id if copied else None

    forked = SessionContext.create(
        agent_key=source.get("agent_key") or default_agent_key
    )
    forked.title = title or f"{source.get('title') or 'Session'}{FORK_SUFFIX}"
    forked.workspace_dir = source.get("workspace_dir")
    forked.project_id = source.get("project_id")
    forked.forked_from = session_id
    forked.fork_point_message_id = fork_point
    forked.turn_count = sum(1 for message in copied if message.role == "user")
    await store.save_session(forked)
    await store.set_session_route(
        forked.id,
        source.get("provider"),
        source.get("model"),
        source.get("reasoning_effort"),
    )

    for message in copied:
        await store.add_message(
            NewMessage(
                session_id=forked.id,
                role=message.role,
                content=message.content,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
                summary=bool(message.summary),
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
        )

    log.info(
        "Forked session %s into %s (%d messages, point=%s)",
        session_id,
        forked.id,
        len(copied),
        fork_point,
    )
    return {
        "session_id": forked.id,
        "title": forked.title,
        "messages": len(copied),
        "forked_from": session_id,
        "fork_point_message_id": fork_point,
    }


def _messages_up_to(
    messages: list[Message], session_id: str, message_id: str | None
) -> list[Message]:
    """Return the whole-message prefix through *message_id*, inclusive."""
    if message_id is None:
        return list(messages)
    for index, message in enumerate(messages):
        if message.id == message_id:
            return list(messages[: index + 1])
    raise ForkPointNotFound(session_id, message_id)


__all__ = [
    "FORK_SUFFIX",
    "ForkPointNotFound",
    "SessionForkStore",
    "fork_session",
]
