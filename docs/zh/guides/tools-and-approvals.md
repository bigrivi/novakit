# 工具与审批

工具是注册在 `ToolRegistry` 中的普通函数。行为（Behavior）定制
执行过程；审批对其设置门禁。

## 注册工具

使用 `@tool` 装饰器，然后将函数注册到注册表中：

```python
from novakit import ToolRegistry, tool


@tool(name="add", description="Add two numbers.")
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


registry = ToolRegistry()
registry.register(add)
```

`Tool` 携带 `name`、`description`、`func`、`params_schema`，
以及 `read_only`、`concurrent_safe` 和 `from_mcp` 标志。
`ToolRegistry` 还可以复制另一个注册表（`ToolRegistry(source=...)`），
并按名称解析 MCP-backed 工具（`from_mcp`），因为针对内置工具编写的
按参数宿主规则不适用于任意服务器的 schema。

## ToolBehavior

用 `set_behavior` 为每个工具绑定横切逻辑：

```python
from novakit.tools import DefaultToolBehavior

registry.set_behavior("add", DefaultToolBehavior())
print(registry.behavior_for("add"))
```

`ToolBehavior` 是一个 `Protocol`，提供钩子用于规范化输入、鉴权
执行、后处理结果和运行成功钩子。`behavior_for(name)`
返回绑定的行为或空操作的 `DefaultToolBehavior`；需要区分“已绑定”
与“默认”时，请使用 `has_behavior(name)`。

## 审批

默认值是进程本地的 `InMemoryApprovalManager`，支持心跳、
中止和会话级授权。订阅 `APPROVAL_REQUIRED` 并显式处理：

```python
from novakit.events import AgentEvent

async def on_event(event, payload):
    if event is AgentEvent.APPROVAL_REQUIRED:
        agent.resolve_approval(payload["request_id"], True)
```

宿主使用的调用形状：

```python
agent.resolve_approval(request_id, approved)
```

当宿主自己持有审批时，通过端口接缝注入不同的策略：

```python
from novakit import AgentOptions
from novakit.agent import AgentPorts

options = AgentOptions(ports=AgentPorts(approval=my_approval_manager))
```

`ApprovalManager` 与 `InMemoryApprovalManager` 位于
`novakit/tools/approval.py`；`AgentPorts` 位于 `novakit/agent/ports.py`。

## 终态配对与防护栏

每个向调用方宣告的工具调用都会收到终态结果、错误、
拒绝或取消结果——在 `ToolInvoker` 内部保证。循环防护栏默认使用
历史宿主只读工具名以保持兼容。
请显式传入自己的集合；一个方便的来源是注册表的
`Tool.read_only` 标志：

```python
from novakit.agent import ToolGuardrails

guardrails = ToolGuardrails(read_tools=frozenset({"read", "search"}))
```

默认集合在 `0.x` 中冻结；修改它属于行为变更，而非补丁修复。

## API 参考

::: novakit.tools.ToolRegistry
::: novakit.tools.tool
::: novakit.tools.ToolBehavior
::: novakit.tools.DefaultToolBehavior
::: novakit.tools.ApprovalManager
::: novakit.tools.InMemoryApprovalManager
