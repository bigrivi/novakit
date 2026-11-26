# Quickstart

This page runs a complete in-memory agent with no network, no files, and no
database. It uses `FakerLLMProvider` from `novakit.llm`, which replays
deterministic scripted turns for tests and examples.

## Minimal example

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

What happens on `chat_stream("Hello")`:

1. The agent resolves or creates a session (default: `InMemorySession` owned by
   this `Agent` instance).
2. It persists the user message.
3. It builds provider messages and consumes the model stream.
4. It persists the assistant message.
5. It executes any requested tools, then repeats until completion, input is
   required, execution stops, or `max_iterations` is reached.

## Defaults worth knowing

- The default session lives only in memory. Nothing is written to `HOME`,
  `NOVA_HOME`, or `~/.nova`, and no database connection is opened.
- Every tool call announced to the caller receives a terminal result, error,
  rejection, or cancellation — `novakit` guarantees terminal pairing inside
  its `ToolInvoker`.
- `AgentConfig(model="fake", max_iterations=4)` only exercises the four
  `CoreConfig` fields (`model`, `provider`, `max_iterations`,
  `reasoning_effort`); the remaining `AgentConfig` fields are host
  compatibility fields the core ignores. See
  [Configuration](../guides/configuration.md).

## Where to go next

- [Configuration](../guides/configuration.md) — the `CoreConfig` / `AgentConfig` split
- [Sessions](../guides/sessions.md) — persistence and message models
- [Tools and approvals](../guides/tools-and-approvals.md) — registering and gating tools
