# 配置

运行时依赖于 `novakit/agent/config.py` 中定义的窄 `CoreConfig` 协议。
`AgentConfig` 是一个具体的、向后兼容的超集，在 `0.x` 中保持原有字段顺序冻结。

## 四个核心字段

```python
from novakit import CoreConfig
```

`CoreConfig` 是一个运行时可检查的 `Protocol`，恰好包含四个属性：

| 字段 | 类型 | 含义 |
| ----- | ---- | ------- |
| `model` | `str` | 本次运行的模型键 |
| `provider` | `str` | 提供商键（例如 `"ollama"`、`"openai"`） |
| `max_iterations` | `int` | 轮次循环迭代上限 |
| `reasoning_effort` | `str \| None` | 推理强度；`None` 表示“采用模型配置声明的值” |

独立核心只读取这四个字段。

## AgentConfig：核心加冻结的宿主字段

```python
from novakit import AgentConfig

config = AgentConfig(model="my-model", provider="my-provider", max_iterations=10)
```

`AgentConfig` 是一个字段顺序在 `0.x` 中**冻结**的数据类。删除
或重排字段属于重大变更，且不得向其中添加新的宿主字段：

| 字段 | 归属 | 含义 |
| ----- | ----- | ------- |
| `model` / `provider` / `max_iterations` / `reasoning_effort` | 核心 | 由轮次循环读取 |
| `max_tokens` / `temperature` / `tools` / `memory_review_interval` / `shell_review` / `permission_mode` / `shell_read_only` | 宿主 | 为兼容保留；核心忽略 |

回归测试锁定了 `AgentConfig` 的位置参数顺序，
另有运行时测试验证核心忽略宿主字段。

## 宿主自定义配置

新的宿主应依赖 `CoreConfig` 形状的对象，而不是扩展
`AgentConfig`。任何具有这四个属性的对象都满足协议，
可直接传给 `Agent`：

```python
from dataclasses import dataclass


@dataclass
class MyConfig:
    model: str = "my-model"
    provider: str = "my-provider"
    max_iterations: int = 10
    reasoning_effort: str | None = None
```

宿主特有设置应放在自己的配置类型中，而不是加入 `AgentConfig`。

## API 参考

::: novakit.agent.AgentConfig
::: novakit.agent.CoreConfig
