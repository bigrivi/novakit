# Agent 参考

公开的 Agent 运行时符号。从 `novakit` 导入核心接口，
从 `novakit.agent` 导入端口、护栏和辅助函数。

## 核心运行时

::: novakit.agent.Agent
::: novakit.agent.AgentOptions
::: novakit.agent.AgentRuntimeDependencies
::: novakit.agent.CoreConfig
::: novakit.agent.AgentConfig

`Agent(config, llm, options=None)` 拥有共享的轮次循环。`AgentOptions`
携带可注入的接缝（`session`、`tools`、`events`、`ports`）；
`AgentRuntimeDependencies` 是在一次运行内部使用的解析后形态。

## 宿主端口

::: novakit.agent.AgentPorts
::: novakit.agent.ports.Compactor

`AgentPorts(compactor=..., approval=..., begin_turn=...)` 注入宿主
行为：`Compactor` 用于上下文压缩，`ApprovalManager` 用于工具
审批策略，`begin_turn` 用于宿主检查点记录。

## 护栏与调用

::: novakit.agent.ToolGuardrails
::: novakit.agent.ToolInvoker

`ToolGuardrails(read_tools=...)` 控制循环行为；请显式传入只读工具
（例如从 `Tool.read_only` 派生）。`ToolInvoker` 执行工具调用，
并保证每一次已宣告的调用都有终态结果、错误、拒绝或取消。

## 会话标题

::: novakit.agent.generate_session_title
::: novakit.agent.build_title_messages
::: novakit.agent.clean_title

`generate_session_title(llm, first_message=..., model=..., session_id=...)`
构建非受信消息提示词，清理 markdown/引号包裹，强制执行
`MAX_TITLE_LENGTH`，并应用一次性超时。失败时返回 `None`，
且从不调度任务或写入标题——宿主决定何时运行以及如何持久化结果。
