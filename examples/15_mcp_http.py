"""Offline MCP HTTP tour: an in-process JSON-RPC server plus HttpTransport.

What you'll learn:
    * How to connect ``HttpTransport`` via ``create_transport`` with a URL.
    * How ``McpClient`` runs ``initialize`` (and what ``client_name`` /
      ``client_version`` the server sees), ``list_tools``, ``call_tool``,
      error responses, and ``close``.
    * How ``allocate`` / ``qualify`` / ``sanitize`` turn server/tool pairs
      into collision-free names.

Run: ``python examples/15_mcp_http.py``
"""

from __future__ import annotations

import asyncio

from _support.mock_mcp_servers import start_mock_mcp_http_server

from novakit.mcp import (
    McpClient,
    McpError,
    allocate,
    create_transport,
    qualify,
    sanitize,
)


async def main() -> None:
    """Serve MCP over local HTTP, then run the client and naming demos."""
    print("--- http transport + client ---")
    runner, url, seen = await start_mock_mcp_http_server()
    try:
        print(f"server url: {url} (local teaching stub)")
        transport = create_transport(
            {"url": url, "headers": {"X-Demo": "1"}, "timeout": 30}
        )
        print("transport type:", type(transport).__name__)
        client = McpClient(
            "demo-http", transport, client_name="demo-host", client_version="0.0.1"
        )
        print("client: name=demo-http client_name=demo-host client_version=0.0.1")
        try:
            await transport.connect()
            print("transport connected")
            info = await client.initialize()
            print("server info:", info["serverInfo"])
            print("is_initialized:", client.is_initialized)
            print("server saw clientInfo:", seen.get("client_info"))

            tools = await client.list_tools()
            print("tools:", [t["name"] for t in tools])
            print("call echo:", await client.call_tool("echo", {"text": "over http"}))
            print("call add:", await client.call_tool("add", {"a": 2, "b": 3}))
            try:
                await client.call_tool("missing", {})
            except McpError as exc:
                print(f"unknown tool error: code={exc.code} message={exc.message}")
            try:
                await transport.send_request("bogus/method")
            except McpError as exc:
                print(f"unknown method error: code={exc.code} message={exc.message}")
            await transport.send_notification("notifications/initialized")
            print("sent notification: notifications/initialized")
        finally:
            await client.close()
            print("closed, is_initialized:", client.is_initialized)

        print("--- naming ---")
        print(
            "sanitize:",
            sanitize("@modelcontextprotocol/server-filesystem", "server"),
        )
        print("qualify:", qualify("shop", "list_orders"))
        names = allocate(
            [("shop", "list_orders"), ("shop", "get_order"), ("blog", "list_orders")],
            reserved=["edit"],
            unprefixed=["blog"],
        )
        print("allocated names:", names)
        print("blog keeps short name:", names[("blog", "list_orders")])
        print("shop prefixed:", names[("shop", "list_orders")])
    finally:
        await runner.cleanup()
    print("mcp http demo finished")


if __name__ == "__main__":
    asyncio.run(main())
