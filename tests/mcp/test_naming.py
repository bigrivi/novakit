"""Direct tests for portable MCP tool naming."""

from __future__ import annotations

import pytest

from novakit.mcp import naming
from novakit.mcp.naming import (
    MAX_TOOL_NAME_LENGTH,
    MCP_TOOL_PREFIX,
    McpNamingError,
    allocate,
    qualify,
    sanitize,
)


def test_sanitize_and_qualify() -> None:
    assert sanitize("@modelcontextprotocol/server-filesystem", "server") == (
        "modelcontextprotocol_server-filesystem"
    )
    assert sanitize("list_services.v2", "tool") == "list_services_v2"
    assert sanitize("///", "server") == "server"
    assert sanitize("", "tool") == "tool"
    assert qualify("github", "list_issues") == "mcp__github__list_issues"
    assert qualify("my_server", "query") == "mcp__my_server__query"
    assert qualify("github", "list_issues", prefix=False) == "list_issues"


def test_long_names_are_bounded_and_keep_readable_prefix() -> None:
    name = qualify("s" * 60, "t" * 60)
    assert len(name) <= MAX_TOOL_NAME_LENGTH
    assert name.startswith(MCP_TOOL_PREFIX + "s" * 20)


def test_allocate_is_deterministic_and_collision_safe() -> None:
    forward = allocate([("s", "a"), ("s", "b")], reserved=["edit"])
    backward = allocate([("s", "b"), ("s", "a")], reserved=["edit"])
    assert forward == backward
    assert "edit" not in forward.values()

    colliding = allocate([("a.b", "query"), ("a/b", "query")])
    assert len(set(colliding.values())) == 2
    assert all(
        "_" in value.rsplit(MCP_TOOL_PREFIX, 1)[-1] for value in colliding.values()
    )


def test_allocate_respects_unprefixed_servers() -> None:
    names = allocate(
        [("plain", "read"), ("other", "read")],
        reserved=["taken"],
        unprefixed=["plain"],
    )
    assert names[("plain", "read")] == "read"
    assert names[("other", "read")].startswith(MCP_TOOL_PREFIX)


def test_allocate_hashes_when_base_is_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(naming, "_MAX_DISAMBIGUATIONS", 1)
    monkeypatch.setattr(naming, "_shorten", lambda *_args, **_kwargs: "taken")
    with pytest.raises(McpNamingError, match="could not find a free name"):
        allocate([("server", "tool")], reserved=["taken"])
