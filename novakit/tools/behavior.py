"""Generic tool execution hooks, independent of application-specific policies."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from novakit.tools.approval import ApprovalManager


@dataclass
class PreExecutionCheck:
    """Result of a before_execute hook."""

    allowed: bool = True
    reject_reason: str | None = None
    approval_request: dict | None = None


@dataclass
class TurnContext:
    """Context passed to a tool behavior for one invocation."""

    approval_manager: ApprovalManager | None = None
    event_emitter: Callable[[dict], Awaitable[None]] | None = None
    session_id: str = ""


class ToolBehavior(Protocol):
    """Hooks for normalizing, authorizing and post-processing a tool call."""

    def normalize_input(self, args: dict) -> dict:
        """Normalize raw tool arguments before execution.

        Args:
            args: Raw argument dict from the model.

        Returns:
            Normalized argument dict.
        """
        ...

    async def before_execute(self, args: dict, ctx: TurnContext) -> PreExecutionCheck:
        """Decide whether a call may run and whether approval is needed.

        Args:
            args: Normalized argument dict.
            ctx: Invocation context with approval and session state.

        Returns:
            The allow/reject/approval decision.
        """
        ...

    def postprocess(self, raw_content: str) -> tuple[str, list | None]:
        """Post-process tool output for model context.

        Args:
            raw_content: Raw text produced by the tool.

        Returns:
            Tuple of displayable content and optional image references.
        """
        ...

    def on_success(self, ctx: TurnContext) -> None:
        """Record post-execution state after a successful call.

        Args:
            ctx: Invocation context with approval and session state.
        """
        ...


class DefaultToolBehavior:
    """Safe no-op behavior for tools without special lifecycle hooks."""

    def normalize_input(self, args: dict) -> dict:
        """Return arguments unchanged.

        Args:
            args: Raw argument dict from the model.

        Returns:
            The same dict.
        """
        return args

    async def before_execute(self, args: dict, ctx: TurnContext) -> PreExecutionCheck:
        """Allow execution without requesting approval.

        Args:
            args: Normalized argument dict.
            ctx: Invocation context with approval and session state.

        Returns:
            A default allowing check.
        """
        return PreExecutionCheck()

    def postprocess(self, raw_content: str) -> tuple[str, list | None]:
        """Return tool output unchanged with no images.

        Args:
            raw_content: Raw text produced by the tool.

        Returns:
            Tuple of the raw content and None.
        """
        return raw_content, None

    def on_success(self, ctx: TurnContext) -> None:
        """Record nothing after a successful call.

        Args:
            ctx: Invocation context with approval and session state.
        """
        return None
