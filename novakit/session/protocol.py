from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from novakit.session.models import Message

if TYPE_CHECKING:
    from novakit.session.models import SessionContext


@runtime_checkable
class SessionProtocol(Protocol):
    """Host persistence boundary for sessions and messages.

    Hosts implement storage; the agent only calls these methods and never
    touches user directories or a database directly.
    """

    def get_current_session(self) -> SessionContext | None:
        """Return the session running in this task, if any."""
        ...

    async def create_session(
        self,
        *,
        persist: bool = True,
        first_message: str | None = None,
        agent_key: str = ...,
        metadata: dict | None = None,
        parent_id: str | None = None,
        workspace_dir: str | None = None,
        project_id: str | None = None,
    ) -> SessionContext:
        """Create a session and make it current.

        Args:
            persist: Whether the session is kept by the store.
            first_message: First user text used to derive the title.
            agent_key: Agent owning the session.
            metadata: Optional session metadata.
            parent_id: Parent session for forks, if any.
            workspace_dir: Workspace associated with the session.
            project_id: Project associated with the session.

        Returns:
            The new current session context.
        """
        ...

    async def load_session(self, session_id: str) -> SessionContext | None:
        """Load a session and make it current.

        Args:
            session_id: Session identifier to load.

        Returns:
            The session context, or None when unknown.
        """
        ...

    async def apply_generated_title(
        self, session_id: str, title: str, expected_title: str
    ) -> bool:
        """Apply a background-generated title unless already renamed.

        Args:
            session_id: Session to rename.
            title: Generated title to store.
            expected_title: Title the session must still carry to be replaced.

        Returns:
            True when the title was applied.
        """
        ...

    async def get_messages(
        self,
        session_id: str | None = None,
        limit: int | None = None,
        last_n: int | None = None,
    ) -> list[Message]:
        """Return session messages oldest-first.

        Args:
            session_id: Session to read, defaulting to the current one.
            limit: Maximum messages from the start, if set.
            last_n: Maximum messages from the end, if set.

        Returns:
            Visible messages in chronological order.
        """
        ...

    async def rollback_messages(
        self, message_ids: list[str], session_id: str | None = None
    ) -> int:
        """Delete the named messages.

        Args:
            message_ids: Messages to delete.
            session_id: Session to modify, defaulting to the current one.

        Returns:
            Number of messages deleted.
        """
        ...

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
        """Append a message to the current session.

        Args:
            role: Message role such as user, assistant, or tool.
            content: Stored message text.
            tool_calls: Tool calls attached to an assistant message.
            tool_call_id: Tool call this message answers, for tool messages.
            images: Image references attached to the message.
            reasoning_content: Model reasoning text, if any.
            group_id: Tool-call group shared by a turn and its results.
            reasoning_elapsed_ms: Reasoning time in milliseconds, if known.
            error: Failure detail recorded on the message, if any.
            tokens_input: Input token count, if known.
            tokens_output: Output token count, if known.
            cache_read_tokens: Cached input tokens, if known.
            cost: Turn cost, if known.
            provider_meta: Opaque per-vendor state to replay verbatim.
            model: Model that produced the message, if known.
            reasoning_effort: Reasoning level the message ran with.
            variant: Variant tag for the message.
            data: Extra encoded payload such as display text.

        Returns:
            The persisted message.
        """
        ...
