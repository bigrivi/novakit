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

Hosts can use `InMemoryHierarchy` for live parent/child links and `SPAWN_DEPTH`
for task-local delegation depth; neither performs persistence or global lookups.

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

Hosts that need branching can use `novakit.session.fork.fork_session` with a
`SessionForkStore`; it copies whole messages, preserves tool pairing and route
metadata, and never mutates the source session.

### Session titles

`novakit.agent.title.generate_session_title` builds the untrusted-message prompt,
cleans markdown/quote wrappers, enforces the length limit, and applies a one-shot
timeout. It returns `None` on failure and never schedules work or writes a title;
the host decides when to run it and how to persist the result.

```python
from novakit.agent.title import generate_session_title

title = await generate_session_title(
    llm,
    first_message="Investigate the flaky tests",
    model="my-model",
    session_id=session.id,
)
if title is not None:
    await session.apply_generated_title(session.id, title, expected_title)
```

### Project instructions

`novakit.prompt.discover_project_instructions` discovers host-supplied global
instructions and project files from the git root to the workspace, applies
filename precedence, screens project files for high-confidence injection shapes,
and truncates from the broadest end. The host supplies the global directory and
policy through `ProjectInstructionConfig`; novakit never reads `HOME` or
`NOVA_HOME` itself.

```python
from novakit.prompt import ProjectInstructionConfig, discover_project_instructions

instructions = discover_project_instructions(
    workspace,
    ProjectInstructionConfig(global_dir=host_instruction_dir),
)
```

`novakit.prompt` also exposes product-neutral section helpers for ordered
composition, environment rendering, sub-agent rosters, and abbreviated skill
lists. Product persona, delegation policy, and final section order remain with
the host.

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

### Compaction

`novakit.compaction.DefaultCompactor` provides portable history planning, tool
result snipping, summary generation, retry, and circuit-breaking while satisfying
the existing `Compactor` port. The default in-memory persistence adapter works
with `InMemorySession`:

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

Hosts inject the pieces that depend on their environment:

- `TokenResolver` supplies token estimates and context limits. The default uses
  novakit's built-in model table; hosts with a model catalogue can inject their
  own resolver.
- `ToolOutputSink` optionally stores full tool output before it is snipped.
  novakit calls the sink but never chooses or creates an offload directory.
- `CompactionStore` persists content updates, summaries, compaction markers, and
  timestamps. `MemoryCompactionStore` is the standalone default; database hosts
  provide a thin adapter.

No compaction path reads `HOME`/`NOVA_HOME`, creates user directories, or performs
database I/O unless the host injects code that does.

### Background tasks

`novakit.tasks.BackgroundTaskManager` provides executor registration, separate
foreground/background concurrency pools, per-session quotas, timeouts,
cancellation, progress, listeners, retention, and bounded output. Hosts supply
executors; the manager has no process-global singleton or built-in registry.

```python
from novakit.tasks import BackgroundTaskManager

manager = BackgroundTaskManager(max_concurrent=2)
# Register a host executor that implements kind and execute(...).
record = manager.submit(
    "host_kind",
    {"argument": "value"},
    session_id=session.id,
    label="Host work",
    timeout_seconds=30,
    background=True,
)
completed = await manager.wait(record.task_id, session.id, timeout=5)
```

## Optional runtime utilities

- `novakit.tasks.process.run_process` executes an explicit argv/cwd/environment,
  streams merged output, and kills the process tree on cancellation.
- `novakit.skills` parses and scans `SKILL.md` files under a caller-supplied root
  and provides a case-insensitive catalog.
- `novakit.checkpoints.CheckpointStore` captures and restores file pre-images
  under an explicit root path or root provider. It never chooses a home directory.

## Providers

Included providers:

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic
- Ollama

The package also includes streaming framing, retry policy, image handling,
tokenization, reasoning metadata, and configurable User-Agent support.

`novakit.pricing` provides pure per-million-token cost math and unknown-price
handling. Catalogue loading and model discovery remain host responsibilities.

### Optional MCP support

Install the optional namespace with `pip install novakit[mcp]`. It provides a
JSON-RPC client, stdio/HTTP transports, and deterministic tool-name allocation
without choosing servers, settings, or permissions for the host.

```python
from novakit.mcp import McpClient, create_transport

transport = create_transport({"command": "my-mcp-server"})
await transport.connect()
client = McpClient(
    "my-server",
    transport,
    client_name="my-product",
    client_version="1.0",
)
await client.initialize()
tools = await client.list_tools()
```

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

- [Changelog](CHANGELOG.md)
- [Issues](https://github.com/bigrivi/novakit/issues)

## Release process

Tagging `vX.Y.Z` starts `.github/workflows/publish.yml`. It verifies the tag
against both `pyproject.toml` and `novakit/version.py`, builds sdist/wheel, then
publishes the same artifacts to TestPyPI before PyPI using OIDC Trusted
Publishing.

Repository administrators must create protected GitHub environments named
`testpypi` and `pypi`, enable required reviewers for manual approval, and register
both environments as trusted publishers on their respective PyPI indexes. The
workflow deliberately has no long-lived API tokens.

## License

Apache-2.0. See [LICENSE](LICENSE).
