# novakit

`novakit` is an embeddable, in-memory-first Python runtime for LLM agents. It
provides providers, streaming events, tool execution, sessions, and approvals
without coupling applications to Nova's server, desktop, user directory, or
database.

## Installation

Python 3.12 or newer is required.

```bash
pip install novakit
```

For local development:

```bash
git clone git@github.com:bigrivi/novakit.git
cd novakit
uv sync --extra dev
```

## Quickstart

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

The default session lives only in the `Agent` instance. Nothing is written to
`HOME`, `NOVA_HOME`, or `~/.nova`, and no database connection is opened.

## Core concepts

### Agent runtime

`Agent` owns the shared turn loop:

1. Resolve or create a session.
2. Persist the user message.
3. Build provider messages.
4. Consume the model stream.
5. Persist the assistant message.
6. Execute requested tools.
7. Repeat until completion, input is required, execution stops, or the iteration
   limit is reached.

Every tool call announced to the caller receives a terminal result, error,
rejection, or cancellation result.

### Configuration

The runtime depends on the narrow `CoreConfig` protocol:

- `model`
- `provider`
- `max_iterations`
- `reasoning_effort`

`AgentConfig` remains a concrete, backward-compatible superset with Nova's
original field order. Only the four `CoreConfig` fields affect the standalone
core; the rest are host compatibility fields the core ignores:

| Field | Owner | Meaning |
| --- | --- | --- |
| `model` / `provider` / `max_iterations` / `reasoning_effort` | core | Read by the turn loop |
| `max_tokens` / `temperature` / `tools` / `memory_review_interval` / `shell_review` / `permission_mode` / `shell_read_only` | host | Kept for compatibility; ignored by core |

Field order is frozen in `0.x`: do not add host fields to `AgentConfig`, and
removing or reordering fields is a major change. New hosts can depend on a
`CoreConfig`-shaped object instead of inheriting compatibility fields:

```python
from dataclasses import dataclass


@dataclass
class MyConfig:
    model: str = "my-model"
    provider: str = "my-provider"
    max_iterations: int = 10
    reasoning_effort: str | None = None
```

This object satisfies `CoreConfig` and can be passed directly to `Agent`. Keep
host-specific settings in your own configuration type rather than adding them to
`AgentConfig`.

### Sessions and persistence

`InMemorySession` is the default. Each instance owns its sessions, messages, and
current-session `ContextVar`.

Implement `SessionProtocol` and inject it through `AgentOptions(session=...)` to
use a database or another persistence backend.

### Tools and approvals

Register tools with `ToolRegistry` and the `@tool` decorator. Tool behaviors can
normalize input, authorize execution, post-process results, and run success
hooks.

Approvals use a process-local `InMemoryApprovalManager` by default. Subscribe to
`APPROVAL_REQUIRED` and call:

```python
agent.resolve_approval(request_id, approved)
```

Inject a different manager through `AgentPorts(approval=...)` when the host has
its own approval policy.

Loop guardrails default to the historical host read-tool names for
compatibility. Pass your own set explicitly:

```python
from novakit.agent import ToolGuardrails

guardrails = ToolGuardrails(read_tools=frozenset({"read", "search"}))
```

A convenient source is the registry's `Tool.read_only` flags. The default
set is frozen in `0.x`; changing it is a behavior change, not a patch fix.

### Host ports

```python
from novakit import AgentOptions
from novakit.agent import AgentPorts

options = AgentOptions(
    session=my_session,
    tools=my_tools,
    events=my_events,
    ports=AgentPorts(
        compactor=my_compactor,
        approval=my_approval_manager,
        begin_turn=my_checkpoint_recorder,
    ),
)
```

- `SessionProtocol` controls persistence.
- `Compactor` controls context compaction and never receives a Nova database type.
- `ApprovalManager` controls tool approval policy.
- `begin_turn` records host-specific checkpoints.

Broader stream-end cleanup remains the host server or UI's responsibility;
`novakit` guarantees terminal pairing inside its tool invoker.

## Providers

Included providers:

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic
- Ollama

The package also includes streaming framing, retry policy, image handling,
tokenization, reasoning metadata, and configurable User-Agent support.

## Development

```bash
uv sync --locked --extra dev
uv run pytest -q --cov=novakit --cov-report=term-missing
uv run ruff check novakit tests
uv run ruff format novakit tests
uv run mypy
uv build --wheel --out-dir dist
bash scripts/smoke_wheel.sh
```

The test suite enforces at least 99% line coverage for the `novakit` package.
Coverage runs offline; provider transports, filesystem access, and external
services are replaced with test doubles.

The wheel smoke test installs the built wheel into an isolated directory and
verifies that a default agent can complete an in-memory conversation without
loading Nova.

## Documentation

- [Agent handoff and roadmap](AGENT_HANDOFF.md)
- [Changelog](CHANGELOG.md)
- [Issues](https://github.com/bigrivi/novakit/issues)

## License

Apache-2.0. See [LICENSE](LICENSE).
