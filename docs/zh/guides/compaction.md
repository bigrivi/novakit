# 上下文压缩

`novakit.compaction.DefaultCompactor` 提供可移植的上下文管理：
历史规划、安全切分、工具结果裁剪、带重试的摘要生成和熔断——
同时满足已有的 `Compactor` 端口（`novakit/agent/ports.py`）。
默认内存适配器可与 `InMemorySession` 配合使用。

## 接入默认压缩器

```python
from novakit import AgentOptions
from novakit.agent import AgentPorts
from novakit.compaction import (
    CompactionConfig,
    DefaultCompactor,
    MemoryCompactionStore,
)
from novakit.session import InMemorySession

session = InMemorySession()
compactor = DefaultCompactor(
    model="my-model",
    provider="my-provider",
    store=MemoryCompactionStore(session),
    config=CompactionConfig(summary_keep_tokens=20_000),
)
options = AgentOptions(
    session=session,
    ports=AgentPorts(compactor=compactor),
)
```

`CompactionConfig`（`novakit/compaction/config.py`）保存配额：

- `output_reserve_tokens`（默认值 `16000`）——模型回复的预留空间
- `summary_reserve_tokens`（默认值 `8000`）——摘要请求的预留空间
- `snip_max_chars`（默认值 `2000`）——每条被裁剪工具结果保留的字符数
- `snip_tool_output_token_budget`（默认值 `50000`）——反向令牌预算
- `snip_preserve_last_n_messages`（默认值 `12`）——免于裁剪的最新消息数
- `summary_keep_tokens`（默认值 `20000`）——逐字保留的最近历史
- `max_consecutive_failures`（默认值 `3`）——熔断上限；`<= 0` 表示无上限

## 注入的宿主部件

宿主注入依赖其环境的部分：

- `TokenResolver`（`novakit/compaction/policy.py`，`Protocol`）提供令牌
  估算和上下文上限。默认 `NovakitTokenResolver` 使用内置模型表；
  拥有自己目录的宿主注入自定义解析器。
- `ToolOutputSink`（同一模块，`Protocol`）在工具输出被裁剪前可选地存储
  完整输出。`novakit` 调用该接收器，但从不选择或创建转存目录。
- `CompactionStore`（`novakit/compaction/store.py`，`Protocol`）持久化
  内容更新、摘要、压缩标记和时间戳。
  `MemoryCompactionStore` 是独立默认值；数据库宿主提供薄适配器。

示例：自定义配额加自定义接收器：

```python
from novakit.compaction import CompactionConfig, MemoryCompactionStore

store = MemoryCompactionStore(session)
config = CompactionConfig(snip_max_chars=1000, max_consecutive_failures=5)
```

## 安全行为

`find_split_point` / `first_safe_boundary` 从不在工具调用
对内部切分；`snip_old_tool_results` 用 `SNIP_MARKER` 标记被裁剪的输出；
`generate_summary` 重试瞬时失败，并在 `max_consecutive_failures` 后熔断；
压缩会发出 `COMPACTION_START` / `COMPACTION_DELTA` / `COMPACTION_END`
以及 `CONTEXT_UPDATE` 事件。除非宿主注入的代码会这样做，否则任何
压缩路径都不会读取 `HOME`/`NOVA_HOME`、创建用户目录或执行数据库 I/O。

## API 参考

::: novakit.compaction.CompactionConfig
::: novakit.compaction.DefaultCompactor
::: novakit.compaction.CompactionStore
::: novakit.compaction.MemoryCompactionStore
::: novakit.compaction.TokenResolver
::: novakit.compaction.ToolOutputSink
