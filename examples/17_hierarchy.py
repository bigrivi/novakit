"""Agent hierarchy: parent/child links, options, and spawn-depth isolation.

What you'll learn:
  - How parent/child agent links are tracked in a hierarchy.
  - Which AgentOptions fields carry hierarchy info into real agents.
  - Why SPAWN_DEPTH is task-local and where the depth limit sits.

Run:
    python examples/17_hierarchy.py
"""

from __future__ import annotations

import asyncio

from novakit import Agent, AgentConfig, AgentOptions
from novakit.agent.hierarchy import AgentNode, InMemoryHierarchy
from novakit.agent.spawn import MAX_SPAWN_DEPTH, SPAWN_DEPTH
from novakit.llm.faker import FakerLLMProvider


class _Node:
    def __init__(self) -> None:
        self.parent_agent: AgentNode | None = None
        self.is_sub_agent: bool = False


async def _collect_text(agent: Agent, message: str) -> str:
    chunks: list[str] = []
    async for _, data in agent.chat_stream(message):
        if isinstance(data, dict) and data.get("content"):
            chunks.append(str(data["content"]))
    return "".join(chunks)


async def _amain() -> None:
    # Prepare data: plain nodes show the hierarchy shape without running agents.
    owner = _Node()
    child_node = _Node()
    hierarchy = InMemoryHierarchy("owner", owner)

    # Call SDK: attach the child, then observe the parent link it records.
    hierarchy.add_sub_agent(owner, child_node)
    print(f"hierarchy key: {hierarchy.agent_key}")
    print(f"child parent is owner: {child_node.parent_agent is owner}")
    print(f"child is_sub_agent: {child_node.is_sub_agent}")
    snapshot = hierarchy.sub_agents()
    print(f"sub_agents snapshot size: {len(snapshot)}")
    snapshot.clear()  # snapshots are copies; mutating one leaves the hierarchy intact
    print(f"sub_agents after clearing snapshot: {len(hierarchy.sub_agents())}")

    # Prepare data: Faker agents run offline with a fixed seed, so parent
    # and child replies are deterministic and safe for tutorials.
    parent = Agent(
        AgentConfig(model="demo", max_iterations=2),
        FakerLLMProvider(seed=7, reasoning_probability=0.0),
        options=AgentOptions(agent_key="parent"),
    )
    sub = Agent(
        AgentConfig(model="demo", max_iterations=2),
        FakerLLMProvider(seed=7, reasoning_probability=0.0),
        options=AgentOptions(
            agent_key="child",
            parent_agent=parent,
            is_sub_agent=True,
            depth=1,
            allowed_tools=frozenset({"read"}),
        ),
    )

    # Observe output: AgentOptions fields surface directly on the agent.
    print(f"parent key/depth: {parent.agent_key}/{parent.depth}")
    print(f"child key/depth: {sub.agent_key}/{sub.depth}")
    print(f"child parent is parent agent: {sub.parent_agent is parent}")
    print(f"child is_sub_agent: {sub.is_sub_agent}")
    print(f"child allowed_tools: {sorted(sub.allowed_tools)}")

    # Call SDK: same seed + same prompt gives the same reply every run.
    first = await _collect_text(parent, "hello")
    second = await _collect_text(parent, "hello")
    child_reply = await _collect_text(sub, "hello")
    print(f"parent reply deterministic: {first == second} ({len(first)} chars)")
    print(f"parent reply preview: {first[:80]!r}")
    print(f"child reply preview: {child_reply[:80]!r}")

    # Call SDK: SPAWN_DEPTH is a ContextVar, so a child asyncio task sees
    # its own value while the parent task keeps depth 0.
    print(f"SPAWN_DEPTH in main task: {SPAWN_DEPTH.get()}")
    started = asyncio.Event()
    release = asyncio.Event()

    async def _child_task() -> None:
        token = SPAWN_DEPTH.set(2)
        try:
            started.set()
            await release.wait()
            print(f"SPAWN_DEPTH in child task: {SPAWN_DEPTH.get()}")
        finally:
            SPAWN_DEPTH.reset(token)

    task = asyncio.create_task(_child_task())
    await started.wait()
    print(f"SPAWN_DEPTH in parent task while child runs: {SPAWN_DEPTH.get()}")
    release.set()
    await task
    print(f"SPAWN_DEPTH after child task: {SPAWN_DEPTH.get()}")

    # Observe output: the depth limit is explanatory — an agent already at
    # MAX_SPAWN_DEPTH should stop spawning, which keeps fan-out bounded.
    print(f"MAX_SPAWN_DEPTH: {MAX_SPAWN_DEPTH}")
    print("depth 0 agent: may spawn (below the limit)")
    print(f"depth {MAX_SPAWN_DEPTH} agent: at the limit, SDK blocks further spawns")
    print(f"depth {MAX_SPAWN_DEPTH + 1} agent: above the limit, also blocked")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
