# Tools and approvals

Tools are plain functions registered in a `ToolRegistry`. Behaviors customize
execution; approvals gate it.

## Registering tools

Use the `@tool` decorator, then register the function into a registry:

```python
from novakit import ToolRegistry, tool


@tool(name="add", description="Add two numbers.")
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


registry = ToolRegistry()
registry.register(add)
```

`Tool` carries `name`, `description`, `func`, `params_schema`, plus
`read_only`, `concurrent_safe`, and `from_mcp` flags. `ToolRegistry` can also
copy another registry (`ToolRegistry(source=...)`) and resolve MCP-backed
tools by name (`from_mcp`), since per-argument host rules written for
built-in tools do not transfer to an arbitrary server's schema.

## ToolBehavior

Bind cross-cutting logic per tool with `set_behavior`:

```python
from novakit.tools import DefaultToolBehavior

registry.set_behavior("add", DefaultToolBehavior())
print(registry.behavior_for("add"))
```

`ToolBehavior` is a `Protocol` with hooks to normalize input, authorize
execution, post-process results, and run success hooks. `behavior_for(name)`
returns the bound behavior or a no-op `DefaultToolBehavior`; use
`has_behavior(name)` when you need to distinguish "bound" from "default".

## Approvals

The default is a process-local `InMemoryApprovalManager` with heartbeat,
abort, and session-scoped grants. Subscribe to `APPROVAL_REQUIRED` and resolve
it explicitly:

```python
from novakit.events import AgentEvent

async def on_event(event, payload):
    if event is AgentEvent.APPROVAL_REQUIRED:
        agent.resolve_approval(payload["request_id"], True)
```

Call shape used by hosts:

```python
agent.resolve_approval(request_id, approved)
```

Inject a different policy through the ports seam when the host owns approvals:

```python
from novakit import AgentOptions
from novakit.agent import AgentPorts

options = AgentOptions(ports=AgentPorts(approval=my_approval_manager))
```

`ApprovalManager` and `InMemoryApprovalManager` live in
`novakit/tools/approval.py`; `AgentPorts` in `novakit/agent/ports.py`.

## Terminal pairing and guardrails

Every tool call announced to the caller receives a terminal result, error,
rejection, or cancellation result — guaranteed inside `ToolInvoker`. Loop
guardrails default to the historical host read-tool names for compatibility.
Pass your own set explicitly; a convenient source is the registry's
`Tool.read_only` flags:

```python
from novakit.agent import ToolGuardrails

guardrails = ToolGuardrails(read_tools=frozenset({"read", "search"}))
```

The default set is frozen in `0.x`; changing it is a behavior change, not a
patch fix.

## API reference

::: novakit.tools.ToolRegistry
::: novakit.tools.tool
::: novakit.tools.ToolBehavior
::: novakit.tools.DefaultToolBehavior
::: novakit.tools.ApprovalManager
::: novakit.tools.InMemoryApprovalManager
