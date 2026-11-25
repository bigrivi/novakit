"""Agent event protocol and dispatch.

Kept apart from the agent implementation so that consumers which only need the
event vocabulary - the server's stream adapters, for instance - do not have to
import the agent and everything it depends on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from novakit.llm import (
        ToolCall,
        ToolInputDelta,
        ToolInputEnd,
        ToolInputError,
        ToolInputStart,
    )

log = logging.getLogger(__name__)


class AgentEvent(Enum):
    """Event kinds emitted by an agent run."""

    # Session lifecycle
    SESSION = "session"
    START = "start"
    DONE = "done"
    ERROR = "error"

    # Turn lifecycle
    TURN_START = "turn_start"
    TURN_END = "turn_end"

    # Transport retry: the provider is about to re-send the request and says
    # why. Carries attempt, total, wait_seconds and the upstream's reason.
    RETRY = "retry"

    # Reasoning stream
    REASONING_START = "reasoning_start"
    REASONING_DELTA = "reasoning_delta"
    REASONING_END = "reasoning_end"

    # Text stream
    TEXT_START = "text_start"
    TEXT_DELTA = "text_delta"
    TEXT_END = "text_end"

    # Tool execution
    TOOL_INPUT_START = "tool_input_start"
    TOOL_INPUT_DELTA = "tool_input_delta"
    TOOL_INPUT_END = "tool_input_end"
    TOOL_INPUT_ERROR = "tool_input_error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # Context compaction
    COMPACTION_START = "compaction_start"
    COMPACTION_DELTA = "compaction_delta"
    COMPACTION_END = "compaction_end"
    CONTEXT_UPDATE = "context_update"

    # Danger command approval (desktop / TUI)
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_HEARTBEAT = "approval_heartbeat"
    APPROVAL_RESULT = "approval_result"


# One item in an agent event stream: the event kind and its payload.
AgentEventFrame = tuple[AgentEvent, Any]


def done_payload(reason: str, content: str | None = None) -> dict[str, Any]:
    """Build the payload for a terminal done event.

    Args:
        reason: Short machine-readable stop reason.
        content: Optional text to carry with the event.

    Returns:
        Event payload dict with reason and content.
    """
    return {"reason": reason, "content": content}


def error_payload(reason: str, message: str) -> dict[str, Any]:
    """Build the payload for a terminal error event.

    Args:
        reason: Short machine-readable failure reason.
        message: Human-readable failure detail.

    Returns:
        Event payload dict with reason and message.
    """
    return {"reason": reason, "message": message}


class EventBus:
    """Fan-out for agent events to in-process subscribers.

    Subscribers are observers: a failing handler must not derail the run that
    produced the event, so exceptions are swallowed here.
    """

    def __init__(self) -> None:
        self._handlers: dict[AgentEvent, list[Callable]] = {}

    def on(self, event: AgentEvent, handler: Callable) -> None:
        """Subscribe a handler to one event kind.

        Args:
            event: Event kind to observe.
            handler: Sync or async callable receiving (event, data).
        """
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: AgentEvent, handler: Callable) -> None:
        """Remove a previously subscribed handler.

        Args:
            event: Event kind the handler was subscribed to.
            handler: Handler to remove, if present.
        """
        if event in self._handlers:
            self._handlers[event].remove(handler)

    async def emit(
        self,
        event: AgentEvent,
        data: (
            dict[str, Any]
            | str
            | ToolCall
            | ToolInputStart
            | ToolInputDelta
            | ToolInputEnd
            | ToolInputError
            | None
        ) = None,
    ) -> None:
        """Deliver an event to subscribers without failing the run.

        Args:
            event: Event kind to deliver.
            data: Event payload passed through to each handler.
        """
        for handler in self._handlers.get(event, []):
            try:
                if iscoroutinefunction(handler):
                    await handler(event, data)
                else:
                    handler(event, data)
            except Exception:
                log.warning("event handler failed for %s", event, exc_info=True)
