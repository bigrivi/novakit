<div align="center">
  <h1>novakit</h1>
</div>
<p align="center" markdown="1">
  <i>Embeddable, in-memory-first Python runtime for LLM agents.</i><br/>
  <sub>Providers, streaming events, tool execution, sessions, and approvals — without a server, user directory, or database.</sub>
</p>
<p align="center" markdown="1">
<a href="https://github.com/bigrivi/novakit/actions/workflows/ci.yml" target="_blank">
  <img src="https://github.com/bigrivi/novakit/actions/workflows/ci.yml/badge.svg" alt="CI"/>
</a>
<a href="https://pypi.org/project/novakit/" target="_blank">
  <img src="https://img.shields.io/pypi/v/novakit.svg?color=%2334D058&label=pypi%20package" alt="PyPI version"/>
</a>
<a href="https://pypi.org/project/novakit/" target="_blank">
  <img src="https://img.shields.io/pypi/pyversions/novakit.svg?color=%2334D058" alt="Supported Python Versions"/>
</a>
<a href="https://github.com/bigrivi/novakit/releases/latest" target="_blank">
  <img src="https://img.shields.io/github/v/release/bigrivi/novakit.svg?color=%2334D058" alt="GitHub release"/>
</a>
<a href="https://github.com/bigrivi/novakit/blob/main/LICENSE" target="_blank">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache 2.0 license"/>
</a>
</p>

---

**Documentation**: these pages (`docs/` in the repository)

**Source Code**: <a href="https://github.com/bigrivi/novakit" target="_blank">https://github.com/bigrivi/novakit</a>

**PyPI**: <a href="https://pypi.org/project/novakit/" target="_blank">https://pypi.org/project/novakit/</a>

**Changelog**: [changelog.md](changelog.md)

---

`novakit` is a standalone, embeddable Python Agent SDK. It owns one agent turn loop
(resolve session → persist user message → build provider messages → consume the model
stream → persist the assistant message → execute tools → repeat) and exposes the seams
a host needs: sessions, tools, approvals, compaction, background tasks, MCP, prompt
helpers, and pricing math.

It is in-memory by default. Nothing is written to `HOME`, `NOVA_HOME`, or `~/.nova`,
and no database connection is opened unless the host injects one through a port such
as `SessionProtocol`.

## Features

- Single `Agent` turn loop with streaming `AgentEvent` frames
- `CoreConfig` protocol (4 fields) plus a frozen, backward-compatible `AgentConfig`
- `InMemorySession` default; bring-your-own persistence via `SessionProtocol`
- `ToolRegistry` + `@tool` decorator, `ToolBehavior` hooks, terminal tool pairing
- `InMemoryApprovalManager` with heartbeat, abort, and session-scoped grants
- Portable compaction (`DefaultCompactor`, `MemoryCompactionStore`) with injectable
  token resolver, tool-output sink, and store
- Background tasks with foreground/background pools, quotas, timeouts, and cancellation
- Optional `novakit[mcp]` client (JSON-RPC, stdio/HTTP, deterministic tool naming)
- Prompt helpers: project-instruction discovery, section composition, skill formatting
- Pure pricing math (`novakit.pricing`) and utilities (`novakit.skills`, `novakit.checkpoints`)

## Quick Start

Follow the [Quickstart](getting-started/quickstart.md) guide for a runnable example:

```python
import asyncio

from novakit import Agent, AgentConfig
from novakit.llm import FakerLLMProvider


async def main() -> None:
    agent = Agent(AgentConfig(model="fake", max_iterations=4), FakerLLMProvider(seed=1))
    async for event, payload in agent.chat_stream("Hello"):
        print(event, payload)


asyncio.run(main())
```

The default session lives only inside the `Agent` instance. See
[Installation](getting-started/installation.md) for Python version and extras, and
[Configuration](guides/configuration.md) for the core/host field split.

For more complete scenarios, browse the
[runnable examples](https://github.com/bigrivi/novakit/tree/main/examples),
including providers, tools, approvals, compaction, tasks, MCP, and utilities.

## License

[Apache-2.0](https://github.com/bigrivi/novakit/blob/main/LICENSE)
