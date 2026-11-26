# Compaction reference

Portable context-compaction planning, summarization, and persistence. Import
from `novakit.compaction`.

## Compactor

::: novakit.compaction.DefaultCompactor

`DefaultCompactor(model, provider, store, config=...)` satisfies the
`Compactor` port with history planning, safe splitting, tool-result snipping,
summary generation with retry, and circuit breaking. Module helpers
`prepare_compaction`, `run_compaction_plan`, `compact`, and
`snip_tool_results_in_store` expose the individual stages.

## Configuration and policy

::: novakit.compaction.CompactionConfig
::: novakit.compaction.TokenResolver
::: novakit.compaction.NovakitTokenResolver
::: novakit.compaction.ToolOutputSink

`CompactionConfig` holds the token budgets and breaker limit (see
[Compaction](../guides/compaction.md)). `TokenResolver` supplies token
estimates and context limits (`NovakitTokenResolver` is the built-in model
table default); `ToolOutputSink` optionally stores full tool output before
snipping.

## Persistence

::: novakit.compaction.CompactionStore
::: novakit.compaction.MemoryCompactionStore

`CompactionStore` persists content updates, summaries, compaction markers,
and timestamps. `MemoryCompactionStore(session)` is the standalone adapter
over `InMemorySession`.

## Summary and message helpers

::: novakit.compaction.generate_summary
::: novakit.compaction.format_for_summary
::: novakit.compaction.find_split_point
::: novakit.compaction.first_safe_boundary
::: novakit.compaction.should_compact

`generate_summary` renders history via `format_for_summary` and retries
transient failures; `find_split_point` / `first_safe_boundary` locate a split
that never breaks a tool-call pair; `should_compact` / `evaluate_compaction`
decide when a fold is needed.
