"""Persistence boundary for compaction and its in-memory implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from novakit.session.models import Message

if TYPE_CHECKING:
    from novakit.session.memory import InMemorySession


class CompactionStore(Protocol):
    """Async persistence the compactor needs from its host.

    The host owns message storage; the compactor only calls these methods and
    never touches the underlying database directly.
    """

    async def update_message_content(self, message_id: str, content: str) -> None:
        """Replace a snipped message's stored content.

        Args:
            message_id: Message to update.
            content: Trimmed content to persist.
        """
        ...

    async def add_summary(
        self, session_id: str, content: str, time_created: int
    ) -> Message:
        """Persist the summary message replacing compacted history.

        Args:
            session_id: Session the summary belongs to.
            content: Summary text to store.
            time_created: Backdated stamp placing it at the fold boundary.

        Returns:
            The persisted summary message.
        """
        ...

    async def mark_messages_compacted_by_ids(
        self, session_id: str, message_ids: list[str]
    ) -> None:
        """Hide compacted messages from later history reads.

        Args:
            session_id: Session owning the messages.
            message_ids: Messages to hide.
        """
        ...

    async def update_session_compacted_at(
        self, session_id: str, timestamp: int
    ) -> None:
        """Record when the last successful compaction ran.

        Args:
            session_id: Session to stamp.
            timestamp: Successful-compaction time in milliseconds.
        """
        ...

    async def get_messages(self, session_id: str) -> list:
        """Return the session's visible messages oldest-first.

        Args:
            session_id: Session to read.

        Returns:
            Visible messages in chronological order.
        """
        ...


class MemoryCompactionStore:
    """A :class:`CompactionStore` backed by an :class:`InMemorySession`.

    Keeps no state of its own; every call delegates to the wrapped session
    store.
    """

    def __init__(self, sessions: InMemorySession) -> None:
        """Wrap *sessions*; the store keeps no state of its own."""
        self._sessions = sessions

    async def update_message_content(self, message_id: str, content: str) -> None:
        """Persist snipped tool output through the wrapped session store.

        Args:
            message_id: Message to update.
            content: Trimmed content to persist.
        """
        await self._sessions.update_message_content(message_id, content)

    async def add_summary(
        self, session_id: str, content: str, time_created: int
    ) -> Message:
        """Add the summary to *session_id*, restoring the current session after.

        The wrapped store tracks one current session per task, so the target
        session is loaded, written, and then the previous current session is
        restored even when the write fails.

        Args:
            session_id: Session the summary belongs to.
            content: Summary text to store.
            time_created: Backdated stamp placing it at the fold boundary.

        Returns:
            The persisted summary message.

        Raises:
            KeyError: When *session_id* is unknown to the wrapped store.
        """
        previous = self._sessions.get_current_session()
        try:
            loaded = await self._sessions.load_session(session_id)
            if loaded is None:
                raise KeyError(f"unknown session: {session_id}")
            return await self._sessions.add_message(
                role="assistant",
                content=content,
                summary=True,
                time_created=time_created,
            )
        finally:
            self._sessions.set_current_session(previous)

    async def mark_messages_compacted_by_ids(
        self, session_id: str, message_ids: list[str]
    ) -> None:
        """Hide the named messages of *session_id* from later reads.

        Args:
            session_id: Session owning the messages.
            message_ids: Messages to hide.
        """
        await self._sessions.mark_messages_compacted_by_ids(session_id, message_ids)

    async def update_session_compacted_at(
        self, session_id: str, timestamp: int
    ) -> None:
        """Stamp the last successful compaction time on *session_id*.

        Args:
            session_id: Session to stamp.
            timestamp: Successful-compaction time in milliseconds.
        """
        await self._sessions.update_session_compacted_at(session_id, timestamp)

    async def get_messages(self, session_id: str) -> list:
        """Return the session's visible messages oldest-first.

        Args:
            session_id: Session to read.

        Returns:
            Visible messages in chronological order.
        """
        return await self._sessions.get_messages(session_id)


__all__ = ["CompactionStore", "MemoryCompactionStore"]
