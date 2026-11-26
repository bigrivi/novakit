"""Direct offline tests for the portable MCP client and transports."""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from novakit.mcp.client import McpClient
from novakit.mcp.transport import (
    HttpTransport,
    McpError,
    StdioTransport,
    create_transport,
)


@pytest.fixture
def mcp_server_path(tmp_path: Path) -> Path:
    script = tmp_path / "mcp_server.py"
    script.write_text(
        textwrap.dedent(
            """\
            import json, sys
            for line in sys.stdin:
                msg = json.loads(line)
                rid = msg.get("id")
                method = msg.get("method")
                params = msg.get("params", {})
                if method == "initialize":
                    result = {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {"name": "test-server"}}
                elif method == "tools/list":
                    result = {"tools": [{"name": "echo", "inputSchema": {}}]}
                elif method == "tools/call" and params.get("name") == "echo":
                    result = {"content": [{"type": "text", "text": params.get("arguments", {}).get("text", "")}]}
                else:
                    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Unknown"}}) + "\\n")
                    sys.stdout.flush()
                    continue
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}) + "\\n")
                sys.stdout.flush()
            """
        ),
        encoding="utf-8",
    )
    return script


@pytest.mark.asyncio
async def test_stdio_round_trip_and_client_flow(mcp_server_path: Path) -> None:
    transport = StdioTransport(sys.executable, [str(mcp_server_path)], {"EXTRA": "1"})
    client = McpClient(
        "server",
        transport,
        client_name="host",
        client_version="9.0",
    )
    try:
        await transport.connect()
        result = await client.initialize()
        assert result["serverInfo"]["name"] == "test-server"
        assert client.is_initialized is True
        assert (await client.list_tools())[0]["name"] == "echo"
        assert await client.call_tool("echo", {"text": "hello"}) == "hello"
        with pytest.raises(McpError) as error:
            await transport.send_request("unknown")
        assert error.value.code == -32601
        await transport.send_notification("notifications/initialized")
    finally:
        await client.close()
        await client.close()
    assert client.is_initialized is False


def test_stdio_requires_connection() -> None:
    transport = StdioTransport("missing")
    with pytest.raises(RuntimeError, match="not connected"):
        transport._stdin()
    with pytest.raises(RuntimeError, match="not connected"):
        transport._stdout()


@pytest.mark.asyncio
async def test_stdio_reader_skips_bad_lines_and_unknown_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = StdioTransport("missing")
    pending: asyncio.Future = asyncio.get_running_loop().create_future()

    class Lines:
        def __init__(self) -> None:
            self.items = iter(
                [
                    b"not json\n",
                    json.dumps({"id": "1", "result": {"ok": True}}).encode() + b"\n",
                    json.dumps({"id": "unknown", "result": {}}).encode() + b"\n",
                    json.dumps({"method": "notifications/progress"}).encode() + b"\n",
                    b"",
                ]
            )

        async def readline(self) -> bytes:
            return next(self.items)

    transport._pending["1"] = pending
    monkeypatch.setattr(transport, "_stdout", lambda: Lines())
    await transport._read_stdout()
    assert await pending == {"id": "1", "result": {"ok": True}}


@pytest.mark.asyncio
async def test_stdio_stderr_reader_drains_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = StdioTransport("missing")

    class Stderr:
        def __init__(self) -> None:
            self.items = iter([b"warning\n", b""])

        async def readline(self) -> bytes:
            return next(self.items)

    process = SimpleNamespace(stderr=Stderr())
    monkeypatch.setattr(transport, "_process", process)
    transport._start_stderr_reader()
    assert transport._stderr_task is not None
    await transport._stderr_task


@pytest.mark.asyncio
async def test_stdio_close_kills_process_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = StdioTransport("missing")

    class Process:
        returncode = None
        terminated = False
        killed = False

        def terminate(self) -> None:
            self.terminated = True

        async def wait(self) -> int:
            if not self.killed:
                raise TimeoutError
            self.returncode = 0
            return 0

        def kill(self) -> None:
            self.killed = True

    process = Process()
    monkeypatch.setattr(transport, "_process", process)
    await transport.close()
    assert process.terminated is True
    assert process.killed is True


@pytest.mark.asyncio
async def test_http_transport_requests_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = AsyncMock()
    response.__aenter__.return_value.json = AsyncMock(
        return_value={"id": "1", "result": {"ok": True}}
    )
    response.__aenter__.return_value.read = AsyncMock(return_value=b"")
    session = MagicMock()
    session.post.return_value = response
    session.close = AsyncMock()
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **_kwargs: session)

    transport = HttpTransport("https://example.test/mcp", {"X-Test": "1"}, timeout=5)
    with pytest.raises(RuntimeError, match="not connected"):
        transport._client()
    await transport.connect()
    assert await transport.send_request("ping", {"a": 1}) == {"ok": True}
    await transport.send_notification("notifications/initialized")
    await transport.close()
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_transport_error_response() -> None:
    response = AsyncMock()
    response.__aenter__.return_value.json = AsyncMock(
        return_value={"error": {"code": -1, "message": "bad"}}
    )
    session = MagicMock()
    session.post.return_value = response
    transport = HttpTransport("https://example.test/mcp")
    transport._session = session
    with pytest.raises(McpError, match="bad"):
        await transport.send_request("ping")


def test_create_transport_variants() -> None:
    assert isinstance(create_transport({"command": "python"}), StdioTransport)
    assert isinstance(create_transport({"url": "https://example.test"}), HttpTransport)
    with pytest.raises(ValueError, match="either 'command' or 'url'"):
        create_transport({})


@pytest.fixture
def mock_transport() -> AsyncMock:
    transport = AsyncMock()
    transport.send_request = AsyncMock()
    transport.send_notification = AsyncMock()
    transport.close = AsyncMock()
    return transport


@pytest.mark.asyncio
async def test_client_initialize_uses_injected_identity(
    mock_transport: AsyncMock,
) -> None:
    mock_transport.send_request.return_value = {
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "server"},
    }
    client = McpClient(
        "name",
        mock_transport,
        client_name="product",
        client_version="1.2.3",
    )
    await client.initialize()
    mock_transport.send_request.assert_awaited_once_with(
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "product", "version": "1.2.3"},
        },
    )
    mock_transport.send_notification.assert_awaited_once_with(
        "notifications/initialized"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ([{"type": "text", "text": "hello"}], "hello"),
        ([{"type": "resource", "resource": {"text": "file"}}], "file"),
        (
            [{"type": "resource", "resource": {"uri": "img", "mimeType": "image/png"}}],
            "[Image: img]",
        ),
        ([{"type": "image", "mimeType": "image/png"}], "[Image: image/png data]"),
        ([{"type": "unknown"}], "{'type': 'unknown'}"),
    ],
)
async def test_client_formats_content(
    mock_transport: AsyncMock, content: list[dict], expected: str
) -> None:
    mock_transport.send_request.return_value = {"content": content}
    client = McpClient("name", mock_transport)
    assert await client.call_tool("tool", {}) == expected
