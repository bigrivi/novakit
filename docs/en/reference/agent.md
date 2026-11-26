# Agent reference

Public agent runtime symbols. Import from `novakit` for the core surface and
from `novakit.agent` for ports, guardrails, and helpers.

## Core runtime

::: novakit.agent.Agent
::: novakit.agent.AgentOptions
::: novakit.agent.AgentRuntimeDependencies
::: novakit.agent.CoreConfig
::: novakit.agent.AgentConfig

`Agent(config, llm, options=None)` owns the shared turn loop. `AgentOptions`
carries the injectable seams (`session`, `tools`, `events`, `ports`);
`AgentRuntimeDependencies` is the resolved form used inside a run.

## Host ports

::: novakit.agent.AgentPorts
::: novakit.agent.ports.Compactor

`AgentPorts(compactor=..., approval=..., begin_turn=...)` injects host
behavior: `Compactor` for context compaction, `ApprovalManager` for tool
approval policy, and `begin_turn` for host checkpoint recording.

## Guardrails and invocation

::: novakit.agent.ToolGuardrails
::: novakit.agent.ToolInvoker

`ToolGuardrails(read_tools=...)` gates loop behavior; pass read tools
explicitly (e.g. derived from `Tool.read_only`). `ToolInvoker` executes tool
calls and guarantees a terminal result, error, rejection, or cancellation for
every announced call.

## Session titles

::: novakit.agent.generate_session_title
::: novakit.agent.build_title_messages
::: novakit.agent.clean_title

`generate_session_title(llm, first_message=..., model=..., session_id=...)`
builds the untrusted-message prompt, cleans markdown/quote wrappers, enforces
`MAX_TITLE_LENGTH`, and applies a one-shot timeout. It returns `None` on
failure and never schedules work or writes a title — the host decides when to
run it and how to persist the result.
