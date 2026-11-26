# Configuration

The runtime depends on the narrow `CoreConfig` protocol defined in
`novakit/agent/config.py`. `AgentConfig` is a concrete, backward-compatible
superset that keeps its original field order frozen in `0.x`.

## The four core fields

```python
from novakit import CoreConfig
```

`CoreConfig` is a runtime-checkable `Protocol` with exactly four attributes:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `model` | `str` | Model key for this run |
| `provider` | `str` | Provider key (e.g. `"ollama"`, `"openai"`) |
| `max_iterations` | `int` | Turn-loop iteration cap |
| `reasoning_effort` | `str \| None` | Reasoning level; `None` means "whatever the model config declares" |

Only these four fields are read by the standalone core.

## AgentConfig: core plus frozen host fields

```python
from novakit import AgentConfig

config = AgentConfig(model="my-model", provider="my-provider", max_iterations=10)
```

`AgentConfig` is a dataclass whose field order is **frozen in `0.x`**. Removing
or reordering fields is a major change, and new host fields must not be added
to it:

| Field | Owner | Meaning |
| ----- | ----- | ------- |
| `model` / `provider` / `max_iterations` / `reasoning_effort` | core | Read by the turn loop |
| `max_tokens` / `temperature` / `tools` / `memory_review_interval` / `shell_review` / `permission_mode` / `shell_read_only` | host | Kept for compatibility; ignored by core |

A regression test locks the positional `AgentConfig` order, plus a
core-ignores-host-fields runtime test.

## Host-defined config

New hosts should depend on a `CoreConfig`-shaped object instead of extending
`AgentConfig`. Any object with the four attributes satisfies the protocol and
can be passed directly to `Agent`:

```python
from dataclasses import dataclass


@dataclass
class MyConfig:
    model: str = "my-model"
    provider: str = "my-provider"
    max_iterations: int = 10
    reasoning_effort: str | None = None
```

Keep host-specific settings in your own configuration type rather than adding
them to `AgentConfig`.

## API reference

::: novakit.agent.AgentConfig
::: novakit.agent.CoreConfig
