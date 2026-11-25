"""Host-provided capabilities used by the core agent loop."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from novakit.events import AgentEventFrame
from novakit.llm import LLMProvider
from novakit.session.models import SessionContext
from novakit.tools.approval import ApprovalManager

BeginTurn = Callable[..., object]


class Compactor(Protocol):
    """Compaction behavior required by the core turn loop.

    Hosts inject this to shorten history without touching the database;
    the compactor never receives host database access.
    """

    @property
    def compacted(self) -> bool:
        """Whether any compaction has run for the active session."""
        ...

    force_next: bool

    def maybe_compact(
        self,
        messages: list,
        session: SessionContext | dict | None,
        llm: LLMProvider,
        emit: Callable[..., Awaitable[None]],
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[AgentEventFrame, None]:
        """Compact history when needed, yielding compaction events.

        Args:
            messages: Current history under consideration.
            session: Active session context, if any.
            llm: Provider used for summarization.
            emit: Callback used to fan out compaction events.
            tools: Tool schemas available to the summarization call.

        Yields:
            Compaction and context-update event frames.
        """
        ...


@dataclass(frozen=True)
class AgentPorts:
    """Optional host capabilities for an otherwise in-memory agent.

    Attributes:
        compactor: History compactor, if the host provides one.
        approval: Approval manager for dangerous tool calls, if any.
        begin_turn: Hook invoked when a turn begins, if any.
    """

    compactor: Compactor | None = None
    approval: ApprovalManager | None = None
    begin_turn: BeginTurn | None = None


__all__ = ["AgentPorts", "BeginTurn", "Compactor"]
