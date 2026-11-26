<div align="center">
  <h1>novakit</h1>
</div>
<p align="center" markdown="1">
  <i>可嵌入、内存优先的 LLM 智能体 Python 运行时。</i><br/>
  <sub>提供商、流式事件、工具执行、会话与审批——无需服务器、用户目录或数据库。</sub>
</p>
<p align="center" markdown="1">
<a href="https://github.com/bigrivi/novakit/actions/workflows/ci.yml" target="_blank">
  <img src="https://github.com/bigrivi/novakit/actions/workflows/ci.yml/badge.svg" alt="CI"/>
</a>
<a href="https://pypi.org/project/novakit/" target="_blank">
  <img src="https://img.shields.io/pypi/v/novakit.svg?color=%2334D058&label=pypi%20package" alt="PyPI version"/>
</a>
<a href="https://pypi.org/project/novakit/" target="_blank">
  <img src="https://img.shields.io/pypi/pyversions/novakit.svg?color=%2334D058" alt="Supported Python Versions"/>
</a>
<a href="https://github.com/bigrivi/novakit/releases/latest" target="_blank">
  <img src="https://img.shields.io/github/v/release/bigrivi/novakit.svg?color=%2334D058" alt="GitHub release"/>
</a>
<a href="https://github.com/bigrivi/novakit/blob/main/LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache 2.0 license"/>
</a>
</p>

---

**文档**：本页面（仓库中的 `docs/`）

**源代码**：<a href="https://github.com/bigrivi/novakit" target="_blank">https://github.com/bigrivi/novakit</a>

**PyPI**：<a href="https://pypi.org/project/novakit/" target="_blank">https://pypi.org/project/novakit/</a>

**变更日志**：[changelog.md](changelog.md)

---

`novakit` 是一个独立、可嵌入的 Python 智能体 SDK。它拥有唯一的智能体轮次循环
（解析会话 → 持久化用户消息 → 构建提供商消息 → 消费模型流 →
持久化助手消息 → 执行工具 → 重复），并暴露宿主所需的接缝：
会话、工具、审批、上下文压缩、后台任务、MCP、提示词辅助函数和计费计算。

它默认完全基于内存。不会向 `HOME`、`NOVA_HOME` 或 `~/.nova` 写入任何内容，
也不会打开数据库连接，除非宿主通过 `SessionProtocol` 等端口注入。

## 特性

- 单一 `Agent` 轮次循环，流式输出 `AgentEvent` 帧
- `CoreConfig` 协议（4 个字段）外加冻结、向后兼容的 `AgentConfig`
- 默认 `InMemorySession`；通过 `SessionProtocol` 自带持久化
- `ToolRegistry` + `@tool` 装饰器、`ToolBehavior` 钩子、终态工具配对
- `InMemoryApprovalManager`，支持心跳、中止和会话级授权
- 可移植的上下文压缩（`DefaultCompactor`、`MemoryCompactionStore`），可注入
  令牌解析器、工具输出接收器和存储
- 后台任务，支持前台/后台线程池、配额、超时和取消
- 可选的 `novakit[mcp]` 客户端（JSON-RPC、stdio/HTTP、确定性工具命名）
- 提示词辅助函数：项目指令发现、片段组合、技能格式化
- 纯计费计算（`novakit.pricing`）与工具函数（`novakit.skills`、`novakit.checkpoints`）

## 快速开始

按照[快速上手](getting-started/quickstart.md)指南运行示例：

```python
import asyncio

from novakit import Agent, AgentConfig
from novakit.llm import FakerLLMProvider


async def main() -> None:
    agent = Agent(AgentConfig(model="fake", max_iterations=4), FakerLLMProvider(seed=1))
    async for event, payload in agent.chat_stream("Hello"):
        print(event, payload)


asyncio.run(main())
```

默认会话仅存在于 `Agent` 实例内部。Python 版本与扩展见
[安装](getting-started/installation.md)，核心/宿主字段划分见
[配置](guides/configuration.md)。

## 许可证

[Apache-2.0](https://github.com/bigrivi/novakit/blob/main/LICENSE)
