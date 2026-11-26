# MCP

使用 `pip install novakit[mcp]` 安装可选模块。它提供
JSON-RPC 客户端、stdio/HTTP 传输和确定性工具名分配——
不替宿主选择服务器、设置或权限。

## 连接服务器

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

- `create_transport(config)`（`novakit/mcp/transport.py`）从 `{"command": ...}`
  构建 `StdioTransport`，或从 URL 配置构建 `HttpTransport`；
  两者都实现 `McpTransport` 接口，失败时抛出 `McpError`。
- `McpClient(server_name, transport, client_name=..., client_version=...)`
  用注入的 `clientInfo` 身份发送 `initialize`，然后提供
  `list_tools()` 和 `call_tool(name, arguments)`。
- 客户端身份始终由调用方提供：`novakit` 从不虚构产品名或版本。

## 工具名分配

MCP 服务器可能导出冲突或过长的工具名。`novakit.mcp.naming`
对注册表安全的名称做清洗、限定和确定性分配：

```python
from novakit.mcp import allocate, qualify, sanitize

print(sanitize("my tool!"))
print(qualify("server", "tool", prefix=True))
taken: set[str] = set()
print(allocate("server", "tool", taken, prefix=True))
```

- `sanitize(part, fallback)` 规范化单个名称片段。
- `qualify(server, tool, prefix=True)` 构建 `server__tool` 基名。
- `allocate(server, tool, taken, prefix=...)` 返回唯一名称，当基名冲突
  或超过 `MAX_TOOL_NAME_length` 时追加短哈希；
  相同输入的重复调用是确定性的。无法解决的冲突抛出 `McpNamingError`。

`ToolRegistry` 用 `from_mcp=True` 标记来自 MCP 的工具，
这改变调用的门禁方式（仅按名称解析），不改变是否运行。

## API 参考

::: novakit.mcp.McpClient
::: novakit.mcp.McpTransport
::: novakit.mcp.McpTransport
::: novakit.mcp.StdioTransport
::: novakit.mcp.HttpTransport
::: novakit.mcp.create_transport
::: novakit.mcp.allocate
