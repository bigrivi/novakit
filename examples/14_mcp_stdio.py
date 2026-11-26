"""Offline MCP stdio tour: a minimal JSON-RPC server in a subprocess.

What you'll learn:
    * How to build a stdio transport with ``create_transport`` and
      ``StdioTransport``.
    * How ``McpClient`` runs ``initialize`` / ``list_tools`` / ``call_tool``,
      and what ``client_name`` / ``client_version`` the server sees.
    * How ``allocate`` / ``qualify`` / ``sanitize`` resolve tool-name
      collisions before registering server tools in a ``ToolRegistry``.

Run: ``python examples/14_mcp_stdio.py``
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from _support.mock_mcp_servers import write_stdio_server_script

from novakit.mcp import (
    McpClient,
    McpError,
    StdioTransport,
    allocate,
    create_transport,
    qualify,
    sanitize,
)
from novakit.tools import ToolRegistry


async def main() -> None:
    """Spawn the server, run the client flow, naming, and registry demos."""
    print("--- stdio transport + client ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        server_path = write_stdio_server_script(Path(tmpdir) / "demo_mcp_server.py")
        seen_path = Path(tmpdir) / "client_info.json"
        print(f"server script: {server_path.name} (local teaching stub)")

        transport = create_transport(
            {"command": sys.executable, "args": [str(server_path), str(seen_path)]}
        )
        print("transport type:", type(transport).__name__)
        print("is StdioTransport:", isinstance(transport, StdioTransport))
        client = McpClient(
            "demo", transport, client_name="demo-host", client_version="0.0.1"
        )
        print("client: name=demo client_name=demo-host client_version=0.0.1")
        try:
            await transport.connect()
            print("transport connected")
            info = await client.initialize()
            print("server info:", info["serverInfo"])
            print("is_initialized:", client.is_initialized)
            print(
                "server saw clientInfo:",
                json.loads(seen_path.read_text(encoding="utf-8")),
            )

            tools = await client.list_tools()
            print("tools:", [t["name"] for t in tools])
            print("call echo:", await client.call_tool("echo", {"text": "hello"}))
            print(
                "call list_services.v2:",
                await client.call_tool("list_services.v2", {}),
            )
            try:
                await client.call_tool("missing", {})
            except McpError as exc:
                print(f"unknown tool error: code={exc.code} message={exc.message}")
            try:
                await transport.send_request("bogus/method")
            except McpError as exc:
                print(f"unknown method error: code={exc.code} message={exc.message}")

            print("--- naming ---")
            print("sanitize dotted tool:", sanitize("list_services.v2", "tool"))
            print("sanitize slashes:", sanitize("///", "server"))
            print("qualify:", qualify("github", "list_issues"))
            names = allocate(
                [
                    ("demo", "echo"),
                    ("demo2", "echo"),
                    ("a.b", "query"),
                    ("a/b", "query"),
                ],
                reserved=["edit"],
            )
            # allocate() de-duplicates clashing pairs and skips reserved names.
            print("allocated names:", names)

            print("--- registry ---")
            registry = ToolRegistry()
            for (server, tool), public in sorted(
                names.items(), key=lambda item: item[1]
            ):

                async def _call(text: str = "", _tool: str = tool) -> str:
                    return await client.call_tool(_tool, {"text": text})

                registry.register_direct(
                    public,
                    f"MCP tool {server}/{tool}",
                    _call,
                    {"type": "object", "properties": {"text": {"type": "string"}}},
                    from_mcp=True,
                )
            echo_name = names[("demo", "echo")]
            print("echo public name:", echo_name)
            entry = registry.get(echo_name)
            if entry is None:
                print("registry entry missing:", echo_name)
            else:
                print("from_mcp:", entry.from_mcp)
                outcome = await registry.call(echo_name, text="via registry")
                print("registry call result:", outcome["result"])
        finally:
            await client.close()
            print("closed, is_initialized:", client.is_initialized)
    print("mcp stdio demo finished")


if __name__ == "__main__":
    asyncio.run(main())
