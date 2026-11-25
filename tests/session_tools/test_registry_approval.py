"""Offline coverage for tool registry metadata/calls and approval grants."""

from __future__ import annotations

import asyncio

import pytest

from novakit.tools import InMemoryApprovalManager
from novakit.tools.behavior import DefaultToolBehavior
from novakit.tools.registry import (
    Tool,
    ToolRegistry,
    get_registry,
)
from novakit.tools.registry import (
    tool as tool_decorator,
)


@tool_decorator()
def _plain_tool() -> str:
    """Plain docstring."""
    return "ok"


@tool_decorator(
    name="custom",
    description="custom desc",
    parameters={"properties": {"x": {}}, "required": ["x"]},
)
def _impl() -> str:
    return "ok"


@tool_decorator()
def _nodoc() -> str:
    return "ok"


def _bare(x: int) -> int:
    return x + 1


async def _async_double(x: int) -> int:
    return x * 2


def test_tool_decorator_metadata_variants() -> None:
    from novakit.tools import registry as registry_module

    assert registry_module._tool_metadata["_plain_tool"]["description"] == (
        "Plain docstring."
    )
    assert registry_module._tool_metadata["custom"]["parameters"] == {
        "properties": {"x": {}},
        "required": ["x"],
    }
    assert registry_module._tool_metadata["_nodoc"]["description"] == ""


def test_register_uses_metadata_or_plain_defaults() -> None:
    registry = ToolRegistry()
    registry.register(_plain_tool)
    assert registry.get("_plain_tool") is not None
    assert registry.get("_plain_tool").description == "Plain docstring."

    registry.register(_bare)
    assert registry.get("_bare").description == ""
    assert registry.get("_bare").params_schema == {}


def test_copy_constructor_is_independent() -> None:
    source = ToolRegistry()
    source.register(_plain_tool)
    source.set_behavior("_plain_tool", DefaultToolBehavior())
    copied = ToolRegistry(source)
    assert set(copied.tools) == {"_plain_tool"}
    assert copied.has_behavior("_plain_tool") is True

    copied.register(_bare)
    copied.unregister("_plain_tool")
    assert source.get("_bare") is None
    assert source.get("_plain_tool") is not None


def test_behavior_binding_and_default() -> None:
    registry = ToolRegistry()
    assert registry.has_behavior("ghost") is False
    assert isinstance(registry.behavior_for("ghost"), DefaultToolBehavior)
    behavior = DefaultToolBehavior()
    registry.set_behavior("ghost", behavior)
    assert registry.has_behavior("ghost") is True
    assert registry.behavior_for("ghost") is behavior


def test_register_by_metadata_branches() -> None:
    from novakit.tools import registry as registry_module

    registry = ToolRegistry()
    assert registry.register_by_metadata("definitely-missing") is False

    registry_module._tool_metadata["broken-entry"] = {"name": "broken-entry"}
    try:
        assert registry.register_by_metadata("broken-entry") is False
    finally:
        registry_module._tool_metadata.pop("broken-entry", None)

    assert registry.register_by_metadata("custom") is True
    assert registry.get("custom") is not None


def test_register_direct_unregister_get_list_and_global() -> None:
    registry = ToolRegistry()
    registry.register_direct("sync", "s", _bare, {"properties": {}})
    registry.register_direct("mcp", "m", _bare, {}, from_mcp=True)
    assert registry.get("mcp").from_mcp is True
    assert {t.name for t in registry.list_tools()} == {"sync", "mcp"}

    registry.unregister("sync")
    assert registry.get("sync") is None
    registry.unregister("sync")

    assert get_registry() is get_registry()


def test_tool_defaults() -> None:
    tool = Tool(name="t", description="d", func=_bare)
    assert (tool.read_only, tool.concurrent_safe, tool.from_mcp) == (True, True, False)


def test_get_schema_and_convert_edges() -> None:
    assert ToolRegistry().get_schema() == []

    registry = ToolRegistry()
    registry.register_direct("bare", "b", _bare, {})
    registry.register_direct(
        "rich", "r", _bare, {"properties": {"x": {}}, "required": ["x"]}
    )
    registry.register_direct("odd", "o", _bare, {"unexpected": 1})
    schemas = {s["function"]["name"]: s["function"] for s in registry.get_schema()}
    assert schemas["bare"]["parameters"] == {"type": "object", "properties": {}}
    assert schemas["rich"]["parameters"] == {
        "type": "object",
        "properties": {"x": {}},
        "required": ["x"],
    }
    assert schemas["odd"]["parameters"]["required"] == []


def _boom() -> int:
    raise RuntimeError("kaput")


async def _async_boom() -> int:
    raise RuntimeError("async kaput")


@pytest.mark.asyncio
async def test_call_matrix() -> None:
    registry = ToolRegistry()
    registry.register_direct("missing-check", "m", _bare, {})
    registry.unregister("missing-check")
    assert await registry.call("missing-check") == {
        "success": False,
        "error": "Tool missing-check not found",
    }

    registry.register_direct("adds", "a", _bare, {})
    assert await registry.call("adds", x=1) == {"success": True, "result": 2}

    registry.register_direct("async-adds", "a", _async_double, {})
    assert await registry.call("async-adds", x=3) == {"success": True, "result": 6}

    registry.register_direct("wrapped", "w", lambda: {"a": 1}, {})
    assert await registry.call("wrapped") == {"success": True, "result": {"a": 1}}

    passthrough = {"success": False, "error": "nope"}
    registry.register_direct("pass", "p", lambda: passthrough, {})
    assert await registry.call("pass") == passthrough

    registry.register_direct("boom", "b", _boom, {})
    assert await registry.call("boom") == {"success": False, "error": "kaput"}
    registry.register_direct("async-boom", "b", _async_boom, {})
    assert await registry.call("async-boom") == {
        "success": False,
        "error": "async kaput",
    }


def test_pre_request_without_rule_always_pends() -> None:
    manager = InMemoryApprovalManager()
    assert manager.pre_request("run", session_id="s1") != ""
    assert manager.is_granted("anything", "f", "d", "nobody") is False


def test_is_granted_matrix() -> None:
    manager = InMemoryApprovalManager()
    req = manager.pre_request(
        "git push", session_id="s1", rule="r", family="f", digest="d"
    )
    assert manager.resolve(req, True, remember=True) is True

    assert manager.is_granted("r", "f", "d", "s1") is True
    assert manager.is_granted("other", "f", "d", "s1") is False
    assert manager.is_granted("r", "other", "d", "s1") is False
    assert manager.is_granted("r", "f", "other", "s1") is False
    assert manager.is_granted("r", "f", "d", "s2") is False
    assert manager.is_granted("r", "", "d", "s1") is True

    bare = manager.pre_request("run", session_id="s2", rule="r2")
    assert manager.resolve(bare, True, remember=True) is True
    assert manager.is_granted("r2", "anything", "anydigest", "s2") is True
    assert manager.pre_request("run", session_id="s2", rule="r2") == ""


def test_resolve_variants_store_no_grant() -> None:
    manager = InMemoryApprovalManager()
    assert manager.resolve("missing", True) is False

    assert manager.resolve(manager.pre_request("a", session_id="s1"), True) is True
    assert manager.is_granted("r", "f", "d", "s1") is False

    denied = manager.pre_request("b", session_id="s1", rule="r", family="f")
    assert manager.resolve(denied, False, remember=True) is True
    assert manager.is_granted("r", "f", "", "s1") is False

    no_rule = manager.pre_request("c", session_id="s1")
    assert manager.resolve(no_rule, True, remember=True) is True

    no_session = manager.pre_request("d", rule="r")
    assert manager.resolve(no_session, True, remember=True) is True
    assert manager.is_granted("r", "", "", "") is False


def test_resolve_without_event_entry_still_resolves() -> None:
    manager = InMemoryApprovalManager()
    req = manager.pre_request("deploy", session_id="s1")
    del manager._events[req]
    assert manager.resolve(req, True) is True


@pytest.mark.asyncio
async def test_wait_heartbeat_ticks_then_verdict() -> None:
    manager = InMemoryApprovalManager()
    req = manager.pre_request("deploy", session_id="s1")
    waiter = manager.wait_with_heartbeat(req, heartbeat_interval=0)

    assert await asyncio.wait_for(anext(waiter), timeout=1) is None
    assert manager.resolve(req, True) is True
    assert await asyncio.wait_for(anext(waiter), timeout=1) is True
    with pytest.raises(StopAsyncIteration):
        await anext(waiter)


@pytest.mark.asyncio
async def test_wait_spurious_wakeup_continues_until_resolve() -> None:
    manager = InMemoryApprovalManager()
    req = manager.pre_request("deploy", session_id="s1")
    manager._events[req].set()
    waiter = manager.wait_with_heartbeat(req, heartbeat_interval=60)
    pending = asyncio.create_task(anext(waiter))

    await asyncio.sleep(0)
    assert manager.resolve(req, False) is True
    assert await asyncio.wait_for(pending, timeout=1) is False
    with pytest.raises(StopAsyncIteration):
        await anext(waiter)
