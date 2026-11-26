"""In-memory parent/child links shared by embeddable agent runtimes."""

from __future__ import annotations

from typing import Protocol


class AgentNode(Protocol):
    """Minimal agent surface needed to maintain an in-memory hierarchy."""

    parent_agent: AgentNode | None
    is_sub_agent: bool


class InMemoryHierarchy:
    """Track one agent's live children without persistence or global lookups."""

    def __init__(
        self,
        agent_key: str,
        parent_agent: AgentNode | None = None,
    ) -> None:
        """Create a hierarchy root.

        Args:
            agent_key: Stable identity of the owning agent.
            parent_agent: Optional live parent agent.
        """
        self.agent_key = agent_key
        self.parent_agent = parent_agent
        self._sub_agents: list[AgentNode] = []

    def add_sub_agent(self, owner: AgentNode, sub_agent: AgentNode) -> None:
        """Attach a live child to its owner.

        Args:
            owner: Agent that owns the child.
            sub_agent: Child agent to attach.
        """
        sub_agent.parent_agent = owner
        sub_agent.is_sub_agent = True
        self._sub_agents.append(sub_agent)

    def sub_agents(self) -> list[AgentNode]:
        """Return a snapshot of the currently attached children."""
        return self._sub_agents.copy()


__all__ = ["AgentNode", "InMemoryHierarchy"]
