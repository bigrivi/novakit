"""Task-local sub-agent spawn-depth tracking."""

from __future__ import annotations

from contextvars import ContextVar

MAX_SPAWN_DEPTH: int = 3
"""Root agent is depth 0; a child at ``MAX_SPAWN_DEPTH`` cannot spawn again."""

SPAWN_DEPTH: ContextVar[int] = ContextVar("novakit_spawn_depth", default=0)
"""Depth of the agent whose turn is currently executing in this task."""

__all__ = ["MAX_SPAWN_DEPTH", "SPAWN_DEPTH"]
