"""Novakit quickstart: real provider, one tool, multi-turn.

This is a tutorial, not a test: it makes real network calls, so it needs
credentials. Set them up first, then run::

    export NOVAKIT_PROVIDER=openai          # or openai-responses|anthropic|ollama
    export OPENAI_API_KEY=sk-...            # cloud providers need a key
    # export OPENAI_BASE_URL=https://...   # optional, for gateways/proxies
    # export NOVAKIT_MODEL=gpt-4o-mini      # optional, sensible default per provider
    python examples/01_quickstart.py

Ollama needs no key, just a running server::

    export NOVAKIT_PROVIDER=ollama
    # export OLLAMA_BASE_URL=http://localhost:11434
    python examples/01_quickstart.py
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

from novakit import Agent, AgentConfig, AgentOptions
from novakit.events import AgentEvent
from novakit.llm import (
    AnthropicProvider,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenAIResponsesProvider,
    ToolResult,
)
from novakit.tools import ToolRegistry, tool


@tool(name="current_time", description="Return the current UTC time in ISO 8601.")
async def current_time() -> ToolResult:
    """Return the current UTC time."""
    return ToolResult(content=datetime.now(UTC).isoformat())


def _require_env(value: str, name: str) -> str:
    """Return a nonempty env value or explain which variable to set.

    Args:
        value: Raw environment variable value.
        name: Variable name, used in the error message.

    Returns:
        The stripped value.

    Raises:
        RuntimeError: When the variable is missing or empty.
    """
    if not value:
        raise RuntimeError(f"Set {name} to run this quickstart.")
    return value


def build_provider() -> tuple[LLMProvider, str, str]:
    """Build a provider from environment variables.

    Reads ``NOVAKIT_PROVIDER`` (``openai`` | ``openai-responses`` |
    ``anthropic`` | ``ollama``), the matching ``*_API_KEY`` / ``*_BASE_URL``
    variables, and ``NOVAKIT_MODEL`` (falls back to a default per provider).

    Returns:
        Tuple of (provider, provider name, model).

    Raises:
        RuntimeError: For an unknown provider or a missing cloud API key.
    """
    kind = os.getenv("NOVAKIT_PROVIDER", "openai").strip().lower()
    model = os.getenv("NOVAKIT_MODEL", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    if kind == "openai":
        api_key = _require_env(
            os.getenv("OPENAI_API_KEY", "").strip(), "OPENAI_API_KEY"
        )
        return (
            OpenAIProvider(api_key=api_key, base_url=base_url),
            kind,
            model or "gpt-4o-mini",
        )
    if kind == "openai-responses":
        api_key = _require_env(
            os.getenv("OPENAI_API_KEY", "").strip(), "OPENAI_API_KEY"
        )
        return (
            OpenAIResponsesProvider(api_key=api_key, base_url=base_url),
            kind,
            model or "gpt-4o-mini",
        )
    if kind == "anthropic":
        api_key = _require_env(
            os.getenv("ANTHROPIC_API_KEY", "").strip(), "ANTHROPIC_API_KEY"
        )
        base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
        return (
            AnthropicProvider(api_key=api_key, base_url=base_url),
            kind,
            model or "claude-sonnet-4-5",
        )
    if kind == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "").strip() or None
        return OllamaProvider(base_url=base_url), kind, model or "llama3.1"
    raise RuntimeError(
        f"Unknown NOVAKIT_PROVIDER={kind!r}; use openai|openai-responses|anthropic|ollama."
    )


async def run_turn(
    agent: Agent, text: str, session_id: str | None = None
) -> str | None:
    """Stream one turn: print text live, report tool use, return the session id.

    Args:
        agent: Configured agent to chat with.
        text: User message for this turn.
        session_id: Reused session, or None to start a new one.

    Returns:
        The session id for the next turn.

    Raises:
        RuntimeError: When the run ends with an error event.
    """
    stream = (
        agent.chat_stream(text, session_id=session_id)
        if session_id
        else agent.chat_stream(text)
    )
    async for event, data in stream:
        if event is AgentEvent.SESSION:
            session_id = data
        elif event is AgentEvent.TEXT_DELTA:
            print(data, end="", flush=True)
        elif event is AgentEvent.TOOL_CALL:
            print(f"\n[tool call] {data.name}({data.arguments})")
        elif event is AgentEvent.TOOL_RESULT:
            print(f"[tool result] {data['result'].content}")
        elif event is AgentEvent.ERROR:
            raise RuntimeError(f"{data.get('reason')}: {data.get('message')}")
        elif event is AgentEvent.DONE:
            print(f"\n[DONE reason={data.get('reason')}]")
    return session_id


async def main() -> None:
    """Chat with a real provider: tool call first, then a follow-up turn."""
    provider, provider_name, model = build_provider()
    registry = ToolRegistry()
    registry.register(current_time)
    agent = Agent(
        AgentConfig(model=model, provider=provider_name, max_iterations=4),
        provider,
        options=AgentOptions(
            system_prompt="You are a concise assistant.",
            tools=registry,
        ),
    )
    session_id = await run_turn(
        agent, "What time is it? Call the current_time tool to find out."
    )
    await run_turn(
        agent, "Thanks. In which timezone was that time?", session_id=session_id
    )


if __name__ == "__main__":
    asyncio.run(main())
