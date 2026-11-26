"""Direct tests for portable hierarchy links and spawn depth."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from novakit.agent.hierarchy import InMemoryHierarchy
from novakit.agent.spawn import MAX_SPAWN_DEPTH, SPAWN_DEPTH


@dataclass
class Node:
    parent_agent: object | None = None
    is_sub_agent: bool = False


def test_hierarchy_attaches_children_and_returns_snapshot() -> None:
    owner = Node()
    child = Node()
    hierarchy = InMemoryHierarchy("owner", owner)

    hierarchy.add_sub_agent(owner, child)

    assert hierarchy.agent_key == "owner"
    assert hierarchy.parent_agent is owner
    assert child.parent_agent is owner
    assert child.is_sub_agent is True
    snapshot = hierarchy.sub_agents()
    assert snapshot == [child]
    snapshot.clear()
    assert hierarchy.sub_agents() == [child]


def test_spawn_depth_default_set_and_reset() -> None:
    assert SPAWN_DEPTH.get() == 0
    token = SPAWN_DEPTH.set(MAX_SPAWN_DEPTH)
    try:
        assert SPAWN_DEPTH.get() == MAX_SPAWN_DEPTH
    finally:
        SPAWN_DEPTH.reset(token)
    assert SPAWN_DEPTH.get() == 0


@pytest.mark.asyncio
async def test_spawn_depth_is_task_local() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def child() -> None:
        token = SPAWN_DEPTH.set(2)
        try:
            started.set()
            await release.wait()
            assert SPAWN_DEPTH.get() == 2
        finally:
            SPAWN_DEPTH.reset(token)

    task = asyncio.create_task(child())
    await started.wait()
    assert SPAWN_DEPTH.get() == 0
    release.set()
    await task
    assert SPAWN_DEPTH.get() == 0
