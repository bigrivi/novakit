# 快速上手

本页运行一个完整的内存智能体，无需网络、文件和数据库。
它使用 `novakit.llm` 中的 `FakerLLMProvider`，为测试和示例回放
确定性的脚本化轮次。

## 最小示例

```python
import asyncio

from novakit import Agent, AgentConfig
from novakit.llm import FakerLLMProvider


async def main() -> None:
    agent = Agent(
        AgentConfig(model="fake", max_iterations=4),
        FakerLLMProvider(seed=1),
    )
    async for event, payload in agent.chat_stream("Hello"):
        print(event, payload)


asyncio.run(main())
```

调用 `chat_stream("Hello")` 时会发生什么：

1. 智能体解析或创建会话（默认：归属于当前 `Agent` 实例的 `InMemorySession`）。
2. 持久化用户消息。
3. 构建提供商消息并消费模型流。
4. 持久化助手消息。
5. 执行请求的工具，然后重复，直到完成、需要输入、执行停止，
   或达到 `max_iterations`。

## 值得了解的默认值

- 默认会话仅存在于内存中。不会向 `HOME`、`NOVA_HOME` 或 `~/.nova` 写入任何内容，
  也不会打开数据库连接。
- 每个向调用方宣告的工具调用都会收到终态结果、错误、拒绝或取消——
  `novakit` 在其 `ToolInvoker` 内部保证终态配对。
- `AgentConfig(model="fake", max_iterations=4)` 只用到四个
  `CoreConfig` 字段（`model`、`provider`、`max_iterations`、
  `reasoning_effort`）；其余 `AgentConfig` 字段是核心忽略的宿主
  兼容字段。见[配置](../guides/configuration.md)。

## 接下来去哪里

- [配置](../guides/configuration.md)——`CoreConfig` / `AgentConfig` 划分
- [会话](../guides/sessions.md)——持久化与消息模型
- [工具与审批](../guides/tools-and-approvals.md)——注册工具并设置门禁
