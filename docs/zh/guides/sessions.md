# 会话

`InMemorySession`（位于 `novakit/session/memory.py`）是默认持久化。
每个实例拥有自己的会话、消息和当前会话 `ContextVar`。
不创建文件，也不打开数据库。

## 默认用法

```python
from novakit import Agent, AgentConfig
from novakit.llm import FakerLLMProvider
from novakit.session import InMemorySession

session = InMemorySession()
agent = Agent(
    AgentConfig(model="fake", max_iterations=4),
    FakerLLMProvider(seed=1),
)
```

需要显式持有实例时，通过 `AgentOptions(session=...)` 传入会话；
否则 `Agent` 会为你创建一个。两种方式下数据都保留在进程内存中。

## 注入自己的持久化

实现 `SessionProtocol`（`novakit/session/protocol.py`）并注入：

```python
from novakit import AgentOptions

options = AgentOptions(session=my_session)
agent = Agent(config, llm, options=options)
```

`SessionProtocol` 是轮次循环所面向的接缝：会话查找
与创建、消息持久化、标题处理和路由元数据。
数据库宿主提供薄适配器；核心从不假设 SQL、文件或用户目录。

## 消息与会话模型

`novakit/session/models.py` 定义流经该接缝的纯数据类型：

- `SessionContext`——会话行（id、标题、工作区、路由、分叉血缘、
  轮次计数）；`SessionContext.create(...)` 用于新建。
- `NewMessage`——待持久化的消息（角色、内容、工具调用、令牌
  计数、提供商元数据、时间戳）。
- `Message`——已分配 id 的持久化消息。
- `Session`——轻量会话视图。
- `MessageFilter`——列出消息时的查询过滤器。
- `encode_display_text` / `decode_display_text`——显示文本辅助函数。

示例：通过分叉存储构建待持久化的消息（与协议使用相同的 `NewMessage`
形状）：

```python
from novakit.session.models import NewMessage

pending = NewMessage(session_id=session_id, role="user", content="Hello")
```

需要分支的宿主应使用带 `SessionForkStore` 的 `fork_session`，
而不是手工复制行——见[会话分叉](session-fork.md)。

## API 参考

::: novakit.session.SessionProtocol
::: novakit.session.InMemorySession
::: novakit.session.Message
::: novakit.session.NewMessage
::: novakit.session.SessionContext
