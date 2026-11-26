# MCP 参考

可选的可移植 MCP 客户端、传输层和工具命名。需要先执行
`pip install novakit[mcp]`。从 `novakit.mcp` 导入。

## 客户端与传输层

::: novakit.mcp.McpClient
::: novakit.mcp.McpTransport
::: novakit.mcp.StdioTransport
::: novakit.mcp.HttpTransport
::: novakit.mcp.create_transport
::: novakit.mcp.McpError

`create_transport(config)` 构建 `StdioTransport`（命令配置）或
`HttpTransport`（URL 配置）。`McpClient(server_name, transport,
client_name=..., client_version=...)` 在 `initialize()` 时发送注入的
`clientInfo` 身份，之后提供 `list_tools()` 和
`call_tool(name, arguments)`。传输层失败时抛出 `McpError`。

## 工具命名

::: novakit.mcp.allocate
::: novakit.mcp.qualify
::: novakit.mcp.sanitize
::: novakit.mcp.McpNamingError

`sanitize` 规范化单个片段，`qualify` 构建 `server__tool` 基础名，
`allocate` 返回唯一的注册表安全名称（冲突或超过
`MAX_TOOL_NAME_LENGTH` 时附加短哈希），结果是确定性的。无法解决的
冲突抛出 `McpNamingError`。
