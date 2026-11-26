# 层级

`novakit/agent/hierarchy.py` 在内存中跟踪存活的父子智能体链接。
它不做持久化，也不做全局查找——只记录存活期间
哪些子智能体归属于哪个所有者。

## 链接智能体

```python
from novakit.agent.hierarchy import InMemoryHierarchy

hierarchy = InMemoryHierarchy(agent_key="root")
hierarchy.add_sub_agent(owner, sub_agent)
print(hierarchy.sub_agents())
```

- `InMemoryHierarchy(agent_key, parent_agent=None)` 为单个
  持有智能体创建跟踪器。任何满足 `AgentNode` 协议（`parent_agent`
  / `is_sub_agent` 属性）的对象都可参与。
- `add_sub_agent(owner, sub_agent)` 设置 `sub_agent.parent_agent = owner`，
  标记 `sub_agent.is_sub_agent = True`，并记录该子智能体。
- `sub_agents()` 返回当前已关联子智能体的快照副本。

`Agent` 暴露这些存活链接，宿主可据此遍历运行中的树
（例如用于取消或展示），而无需查数据库。

## 生成深度

`novakit/agent/spawn.py` 提供任务本地的委派深度跟踪：

```python
from novakit.agent.spawn import MAX_SPAWN_DEPTH, SPAWN_DEPTH

depth = SPAWN_DEPTH.get()  # 0 for a root agent
token = SPAWN_DEPTH.set(depth + 1)
try:
    ...
finally:
    SPAWN_DEPTH.reset(token)
```

- `SPAWN_DEPTH` 是默认值为 `0` 的 `ContextVar[int]`；每个任务看到
  自己的值，因此并发轮次互不干扰。
- `MAX_SPAWN_DEPTH = 3`：根智能体深度为 0，深度 3 的子智能体
  不能再生成。宿主在委派前检查该变量，超限时拒绝（或压平）。

## API 参考

::: novakit.agent.hierarchy.InMemoryHierarchy
::: novakit.agent.spawn.SPAWN_DEPTH
::: novakit.agent.spawn.MAX_SPAWN_DEPTH
