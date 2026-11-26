# Hierarchy

`novakit/agent/hierarchy.py` tracks live parent/child agent links in memory.
It performs no persistence and no global lookups — it only records which
child agents belong to which owner while they are alive.

## Linking agents

```python
from novakit.agent.hierarchy import InMemoryHierarchy

hierarchy = InMemoryHierarchy(agent_key="root")
hierarchy.add_sub_agent(owner, sub_agent)
print(hierarchy.sub_agents())
```

- `InMemoryHierarchy(agent_key, parent_agent=None)` creates a tracker for one
  owning agent. Any object satisfying the `AgentNode` protocol (`parent_agent`
  / `is_sub_agent` attributes) can participate.
- `add_sub_agent(owner, sub_agent)` sets `sub_agent.parent_agent = owner`,
  marks `sub_agent.is_sub_agent = True`, and records the child.
- `sub_agents()` returns a snapshot copy of the currently attached children.

`Agent` exposes these live links so hosts can walk the running tree
(e.g. for cancellation or display) without consulting a database.

## Spawn depth

`novakit/agent/spawn.py` provides task-local delegation-depth tracking:

```python
from novakit.agent.spawn import MAX_SPAWN_DEPTH, SPAWN_DEPTH

depth = SPAWN_DEPTH.get()  # 0 for a root agent
token = SPAWN_DEPTH.set(depth + 1)
try:
    ...
finally:
    SPAWN_DEPTH.reset(token)
```

- `SPAWN_DEPTH` is a `ContextVar[int]` defaulting to `0`; each task sees its
  own value, so concurrent turns do not interfere.
- `MAX_SPAWN_DEPTH = 3`: the root agent is depth 0, and a child at depth 3
  cannot spawn again. Hosts check the var before delegating and refuse (or
  flatten) beyond the cap.

## API reference

::: novakit.agent.hierarchy.InMemoryHierarchy
::: novakit.agent.spawn.SPAWN_DEPTH
::: novakit.agent.spawn.MAX_SPAWN_DEPTH
