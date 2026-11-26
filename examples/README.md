# novakit Examples

Runnable, offline-first examples covering every public feature area of the SDK.

## Requirements

- Python >= 3.12.
- `02`–`20` run offline and deterministically: no network calls, no API keys,
  no user-directory access, no database.
- `01_quickstart.py` is the only exception: it is a real-provider quickstart
  that calls your configured provider over the network and requires provider
  credentials (see below). Offline smoke only checks that it compiles.
- Provider and MCP examples in `02`–`20` talk to a local in-process mock server,
  never to a real endpoint.
- Filesystem examples (instructions, skills, checkpoints, processes) only use
  temporary directories.
- The image demo in `13_llm_utilities.py` additionally requires the `images` extra:

```bash
uv sync --locked --extra dev --extra images
```

### `01_quickstart.py` (real provider, online)

`NOVAKIT_PROVIDER` selects the provider: `openai` (default),
`openai-responses`, `anthropic`, or `ollama`. `NOVAKIT_MODEL` overrides the
default model, and `*_BASE_URL` (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`,
`OLLAMA_BASE_URL`) overrides the gateway endpoint.

OpenAI:

```bash
export NOVAKIT_PROVIDER=openai
export OPENAI_API_KEY=sk-...
# optional: export NOVAKIT_MODEL=gpt-4o-mini
# optional: export OPENAI_BASE_URL=https://...
uv run python examples/01_quickstart.py
```

Ollama (local daemon, no API key):

```bash
export NOVAKIT_PROVIDER=ollama
# optional: export NOVAKIT_MODEL=llama3.1
# optional: export OLLAMA_BASE_URL=http://localhost:11434
uv run python examples/01_quickstart.py
```

## Tutorial style

The numbered files are tutorials, not test suites. Read each one from
configuration to execution to printed output, then copy the relevant pattern
into your own application. Assertions and automated pass/fail checks live in
`tests/test_examples.py`, not in the examples.

`examples/_support/` contains small local mock LLM and MCP servers used by the
offline Provider/MCP tutorials. It is teaching infrastructure, not SDK API.

## Run

Run a single offline example from the repository root:

```bash
uv run python examples/02_events_and_streaming.py
```

Every offline example exits with status `0` on success and prints its own
progress. `tests/test_examples.py` executes each `02`–`20` file in a subprocess
with a 45s timeout as CI smoke coverage, and only compile-checks
`01_quickstart.py`.

## Index

`01_quickstart.py` is a real-provider quickstart (online, requires credentials
as described above). It builds one minimal `Agent`, registers a `current_time`
tool, streams the first turn, and reuses the session for a follow-up. Start
there when you have a key or a local Ollama daemon; start at `02` for the
offline tour.

| File | Topic | Covered API |
| --- | --- | --- |
| `01_quickstart.py` | Real-provider quickstart (online, needs credentials) | `Agent`, `AgentConfig`, `AgentOptions`, `ToolRegistry`/`tool`, `chat_stream`, provider selected by `NOVAKIT_PROVIDER` |
| `02_events_and_streaming.py` | Events and streaming | `Agent.chat_stream`, `AgentEvent`, `AgentEventFrame`, `EventBus`, `FakerLLMProvider.chat`/`chat_stream`, `TextDelta`, `ReasoningDelta`, `ToolCall`, `Done` |
| `03_configuration_and_ports.py` | Configuration and host ports | `CoreConfig`, `AgentConfig`, `AgentOptions`, `AgentPorts`, `begin_turn` |
| `04_sessions.py` | Sessions and persistence | `InMemorySession`, `SessionProtocol`, `AgentOptions(session=...)` |
| `05_session_titles.py` | Session titles | `generate_session_title`, `build_title_messages`, `clean_title` |
| `06_session_fork.py` | Session fork | `fork_session`, `SessionForkStore` |
| `07_tools_and_behaviors.py` | Tools and behaviors | `ToolRegistry`, `tool`, `ToolBehavior`, `TurnContext`, `ToolContext` |
| `08_approvals_and_guardrails.py` | Approvals and guardrails | `InMemoryApprovalManager`, `APPROVAL_REQUIRED`, `resolve_approval`, `ToolGuardrails`, `AgentPorts(approval=...)` |
| `09_compaction.py` | Context compaction | `DefaultCompactor`, `CompactionConfig`, `MemoryCompactionStore`, `Compactor` port |
| `10_background_tasks.py` | Background tasks | `BackgroundTaskManager`, `TaskExecutor`, `TaskRecord`, task statuses, listeners, quotas, retention |
| `11_process_tasks.py` | Process tasks (temp dir only) | `run_process`, `kill_process_tree` |
| `12_providers.py` | Providers via local mock server | `OpenAIProvider`, `OpenAIResponsesProvider`, `AnthropicProvider`, `OllamaProvider`, retry policy, User-Agent |
| `13_llm_utilities.py` | LLM utilities | tokenizer, `novakit.llm.images`, reasoning metadata, `RetryBudget`, request hooks, User-Agent, `stream_text_once` |
| `14_mcp_stdio.py` | MCP stdio (local mock server) | `McpClient`, `StdioTransport`, `create_transport`, `qualify`, `sanitize`, `allocate` |
| `15_mcp_http.py` | MCP HTTP (local mock server) | `McpClient`, `HttpTransport`, `create_transport`, `qualify`, `sanitize`, `allocate` |
| `16_project_instructions.py` | Project instructions and prompt composition | `discover_project_instructions`, `ProjectInstructionConfig`, `has_threats`, `scan_text`, `join_sections`, `PromptEnvironment`, `format_subagent_roster`, `format_available_skills` |
| `17_hierarchy.py` | Hierarchy and delegation | `InMemoryHierarchy`, `SPAWN_DEPTH`, `MAX_SPAWN_DEPTH` |
| `18_skills.py` | Skills (temp dir only) | `SkillCatalog`, `scan_skills_dir`, `load_skill_document` |
| `19_checkpoints.py` | Checkpoints (explicit root only) | `CheckpointStore`, checkpoint summaries, restore reports, `AgentPorts(begin_turn=...)` |
| `20_pricing.py` | Pricing math | `estimate_cost_from_prices`, `usage_cost`, `cheapest_prices`, `numeric_prices` |
