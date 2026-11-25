"""Core agent configuration plus host compatibility superset.

The runtime depends only on :class:`CoreConfig`. :class:`AgentConfig` keeps
the original field order for host compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class CoreConfig(Protocol):
    """Configuration fields read by the shared agent turn loop."""

    model: str
    provider: str
    max_iterations: int
    reasoning_effort: str | None


@dataclass
class AgentConfig:
    """Tunable inputs for one agent run.

    Only ``model``, ``provider``, ``max_iterations``, and ``reasoning_effort``
    (the :class:`CoreConfig` fields) are read by the standalone core; the rest
    are host compatibility fields the core ignores.

    Field order is frozen in 0.x: do not add new host fields here, and
    removing or reordering fields is a major change.
    """

    # Model key as defined in config.json (e.g., "my-gemma")
    model: str = "gpt-4o"
    provider: str = "ollama"
    max_iterations: int = 100
    # Host compatibility field; the core ignores it.
    max_tokens: int = 8192
    # Host compatibility field; the core ignores it.
    temperature: float = 0.7
    # Host compatibility field; the core ignores it.
    tools: list | None = None
    # Host compatibility field; the core ignores it.
    memory_review_interval: int = 10
    # Reasoning level for this run. None means "whatever the model config
    # declares", which is also what a provider that has no such concept gets.
    reasoning_effort: str | None = None
    # Host compatibility field; the core ignores it. Ask a model to review
    # flagged actions instead of interrupting the user for every one.
    shell_review: bool = False
    # Host compatibility field; the core ignores it. ``ask`` waits for a
    # human; ``dontAsk`` refuses instead, for runs with nobody at keyboard.
    permission_mode: str = "ask"
    # Host compatibility field; the core ignores it. When True, the shell
    # tool refuses anything it cannot prove only reads.
    shell_read_only: bool = False


__all__ = ["AgentConfig", "CoreConfig"]
