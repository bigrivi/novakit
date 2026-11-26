# MCP

Install the optional module with `pip install novakit[mcp]`. It provides a
JSON-RPC client, stdio/HTTP transports, and deterministic tool-name allocation
— without choosing servers, settings, or permissions for the host.

## Connecting to a server

```python
from novakit.mcp import McpClient, create_transport

transport = create_transport({"command": "my-mcp-server"})
await transport.connect()
client = McpClient(
    "my-server",
    transport,
    client_name="my-product",
    client_version="1.0",
)
await client.initialize()
tools = await client.list_tools()
```

- `create_transport(config)` (`novakit/mcp/transport.py`) builds a
  `StdioTransport` from `{"command": ...}` or an `HttpTransport` from a URL
  config; both implement the `McpTransport` interface, and failures raise
  `McpError`.
- `McpClient(server_name, transport, client_name=..., client_version=...)`
  sends `initialize` with the injected `clientInfo` identity, then exposes
  `list_tools()` and `call_tool(name, arguments)`.
- Client identity is always caller-supplied: `novakit` never invents a
  product name or version.

## Tool-name allocation

MCP servers can export colliding or over-long tool names. `novakit.mcp.naming`
sanitizes, qualifies, and deterministically allocates registry-safe names:

```python
from novakit.mcp import allocate, qualify, sanitize

print(sanitize("my tool!"))
print(qualify("server", "tool", prefix=True))
taken: set[str] = set()
print(allocate("server", "tool", taken, prefix=True))
```

- `sanitize(part, fallback)` normalizes one name fragment.
- `qualify(server, tool, prefix=True)` builds the `server__tool` base name.
- `allocate(server, tool, taken, prefix=...)` returns a unique name, appending
  a short hash when the base collides or exceeds `MAX_TOOL_NAME_LENGTH`;
  repeated calls with the same inputs are deterministic. Collisions that
  cannot be resolved raise `McpNamingError`.

`ToolRegistry` marks tools arriving from MCP with `from_mcp=True`, which
changes how the call is gated (resolved by name alone) without changing
whether it runs.

## API reference

::: novakit.mcp.McpClient
::: novakit.mcp.McpTransport
::: novakit.mcp.McpTransport
::: novakit.mcp.StdioTransport
::: novakit.mcp.HttpTransport
::: novakit.mcp.create_transport
::: novakit.mcp.allocate
