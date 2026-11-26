"""Local mock MCP servers for examples 14 and 15.

Teaching stubs only: a minimal stdio JSON-RPC server script plus an
in-process HTTP JSON-RPC handler. They run locally (subprocess or
127.0.0.1 ephemeral port) with no external network so the numbered
examples can focus on ``create_transport`` / ``McpClient`` usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiohttp import web

HOST = "127.0.0.1"

STDIO_SERVER_SCRIPT = """\
import json
import sys

seen_path = sys.argv[1] if len(sys.argv) > 1 else None


def send(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "id" not in msg:
        continue  # notification, e.g. notifications/initialized
    rid, method = msg["id"], msg.get("method")
    params = msg.get("params", {})
    if method == "initialize":
        if seen_path:
            with open(seen_path, "w", encoding="utf-8") as fh:
                json.dump(params.get("clientInfo", {}), fh)
        send({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "serverInfo": {"name": "demo-stdio", "version": "1.0"},
        }})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": rid, "result": {"tools": [
            {"name": "echo", "description": "Echo text back",
             "inputSchema": {"type": "object",
                             "properties": {"text": {"type": "string"}}}},
            {"name": "list_services.v2", "description": "Dotted name",
             "inputSchema": {"type": "object", "properties": {}}},
        ]}})
    elif method == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        if name == "echo":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": args.get("text", "")}]}})
        elif name == "list_services.v2":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "content": [{"type": "text", "text": "svc-a,svc-b"}]}})
        else:
            send({"jsonrpc": "2.0", "id": rid,
                  "error": {"code": -32602, "message": f"unknown tool {name}"}})
    else:
        send({"jsonrpc": "2.0", "id": rid,
              "error": {"code": -32601, "message": f"unknown method {method}"}})
"""


def write_stdio_server_script(path: Path) -> Path:
    """Write the demo stdio server script to *path* and return it."""
    path.write_text(STDIO_SERVER_SCRIPT, encoding="utf-8")
    return path


def _result(rid: Any, result: dict[str, Any]) -> web.Response:
    return web.json_response({"jsonrpc": "2.0", "id": rid, "result": result})


def _error(rid: Any, code: int, message: str) -> web.Response:
    return web.json_response(
        {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}
    )


async def start_mock_mcp_http_server() -> tuple[web.AppRunner, str, dict[str, Any]]:
    """Serve the demo MCP surface; return (runner, url, seen)."""
    seen: dict[str, Any] = {}

    async def handle_mcp(request: web.Request) -> web.Response:
        payload: dict[str, Any] = await request.json()
        if "id" not in payload:
            return web.json_response({"jsonrpc": "2.0", "result": {}})
        rid, method = payload["id"], payload.get("method")
        params = payload.get("params", {})
        if method == "initialize":
            seen["client_info"] = params.get("clientInfo")
            return _result(
                rid,
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "serverInfo": {"name": "demo-http", "version": "1.0"},
                },
            )
        if method == "tools/list":
            return _result(
                rid,
                {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo text back",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                            },
                        },
                        {
                            "name": "add",
                            "description": "Add two numbers",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                    ]
                },
            )
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments", {})
            if name == "echo":
                return _result(
                    rid, {"content": [{"type": "text", "text": args.get("text", "")}]}
                )
            if name == "add":
                total = args.get("a", 0) + args.get("b", 0)
                return _result(rid, {"content": [{"type": "text", "text": str(total)}]})
            return _error(rid, -32602, f"unknown tool {name}")
        return _error(rid, -32601, f"unknown method {method}")

    app = web.Application()
    app.router.add_post("/mcp", handle_mcp)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, 0)
    await site.start()
    url = f"http://{HOST}:{runner.addresses[0][1]}/mcp"
    return runner, url, seen
