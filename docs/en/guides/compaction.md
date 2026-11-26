# Compaction

`novakit.compaction.DefaultCompactor` provides portable context management:
history planning, safe splitting, tool-result snipping, summary generation
with retry, and circuit breaking — while satisfying the existing `Compactor`
port (`novakit/agent/ports.py`). The default in-memory adapter works with
`InMemorySession`.

## Wiring the default compactor

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

`CompactionConfig` (`novakit/compaction/config.py`) holds the budgets:

- `output_reserve_tokens` (default `16000`) — headroom for the model reply
- `summary_reserve_tokens` (default `8000`) — headroom for the summary request
- `snip_max_chars` (default `2000`) — characters kept per trimmed tool result
- `snip_tool_output_token_budget` (default `50000`) — reverse token budget
- `snip_preserve_last_n_messages` (default `12`) — newest messages exempt
- `summary_keep_tokens` (default `20000`) — recent history kept verbatim
- `max_consecutive_failures` (default `3`) — breaker limit; `<= 0` means no limit

## Injected host pieces

Hosts inject the parts that depend on their environment:

- `TokenResolver` (`novakit/compaction/policy.py`, `Protocol`) supplies token
  estimates and context limits. The default `NovakitTokenResolver` uses the
  built-in model table; hosts with their own catalogue inject a custom
  resolver.
- `ToolOutputSink` (same module, `Protocol`) optionally stores full tool
  output before it is snipped. `novakit` calls the sink but never chooses or
  creates an offload directory.
- `CompactionStore` (`novakit/compaction/store.py`, `Protocol`) persists
  content updates, summaries, compaction markers, and timestamps.
  `MemoryCompactionStore` is the standalone default; database hosts provide a
  thin adapter.

Example: custom budgets plus a custom sink:

```python
from novakit.compaction import CompactionConfig, MemoryCompactionStore

store = MemoryCompactionStore(session)
config = CompactionConfig(snip_max_chars=1000, max_consecutive_failures=5)
```

## Safety behavior

`find_split_point` / `first_safe_boundary` never split inside a tool-call
pair; `snip_old_tool_results` marks snipped output with `SNIP_MARKER`;
`generate_summary` retries transient failures and trips the breaker after
`max_consecutive_failures`; compaction emits `COMPACTION_START` /
`COMPACTION_DELTA` / `COMPACTION_END` plus `CONTEXT_UPDATE` events. No
compaction path reads `HOME`/`NOVA_HOME`, creates user directories, or
performs database I/O unless the host injects code that does.

## API reference

::: novakit.compaction.CompactionConfig
::: novakit.compaction.DefaultCompactor
::: novakit.compaction.CompactionStore
::: novakit.compaction.MemoryCompactionStore
::: novakit.compaction.TokenResolver
::: novakit.compaction.ToolOutputSink
