"""Narrow contracts for consumers that only need part of an Agent."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol

from novakit.events import AgentEventFrame


class InterruptibleAgent(Protocol):
    """The cancellation hook required by the per-session request registry."""

    def interrupt(self) -> None:
        """Signal the running request to stop at the next safe point."""
        ...


class ChatAgent(InterruptibleAgent, Protocol):
    """The streaming and cancellation surface used by chat execution."""

    def chat_stream(
        self,
        user_input: str,
        session_id: str | None = None,
        attachments: list[dict] | None = None,
        workspace_dir: str | None = None,
        project_id: str | None = None,
        message_variant: str | None = None,
        display_text: str | None = None,
        regenerate: bool = False,
    ) -> AsyncGenerator[AgentEventFrame, None]:
        """Stream one user turn as agent event frames.

        Args:
            user_input: Model-facing text for this turn.
            session_id: Existing session to continue, or None to create one.
            attachments: Optional attachment payloads for the user message.
            workspace_dir: Workspace for a newly created session.
            project_id: Project for a newly created session.
            message_variant: Variant tag stored on the user message.
            display_text: UI-facing text when it differs from user_input.
            regenerate: Whether this turn replaces the previous assistant reply.

        Yields:
            Agent event frames in stream order.
        """
        ...
