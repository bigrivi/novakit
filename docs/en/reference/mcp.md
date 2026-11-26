# MCP reference

Optional portable MCP client, transports, and tool naming. Requires
`pip install novakit[mcp]`. Import from `novakit.mcp`.

## Client and transports

::: novakit.mcp.McpClient
::: novakit.mcp.McpTransport
::: novakit.mcp.StdioTransport
::: novakit.mcp.HttpTransport
::: novakit.mcp.create_transport
::: novakit.mcp.McpError

`create_transport(config)` builds a `StdioTransport` (command config) or
`HttpTransport` (URL config). `McpClient(server_name, transport,
client_name=..., client_version=...)` sends the injected `clientInfo`
identity on `initialize()`, then offers `list_tools()` and
`call_tool(name, arguments)`. Transport failures raise `McpError`.

## Tool naming

::: novakit.mcp.allocate
::: novakit.mcp.qualify
::: novakit.mcp.sanitize
::: novakit.mcp.McpNamingError

`sanitize` normalizes one fragment, `qualify` builds the `server__tool` base
name, and `allocate` returns a unique registry-safe name (short hash on
collision or over `MAX_TOOL_NAME_LENGTH`), deterministically. Unresolvable
collisions raise `McpNamingError`.
