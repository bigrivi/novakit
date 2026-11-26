"""Offline tour of every built-in LLM provider against a local mock server.

What you'll learn:
    * How to construct each provider (``OpenAIProvider``,
      ``OpenAIResponsesProvider``, ``AnthropicProvider``, ``OllamaProvider``)
      with API key, base URL, request options, reasoning effort, and headers.
    * How ``chat`` returns a ``Done`` result with text, token usage, and
      cache-read counters, and what request fields the SDK sends.
    * How ``chat_stream`` yields ``TextDelta`` / ``ReasoningDelta`` /
      tool-call / ``Done`` frames, plus the ``Error`` result shape.

Run: ``python examples/12_providers.py``
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from _support.mock_llm_server import BAD_MODEL, start_mock_llm_server

from novakit.llm import (
    AnthropicProvider,
    Done,
    Error,
    OllamaProvider,
    OpenAIProvider,
    OpenAIResponsesProvider,
    ToolCall,
)

SESSION_ID = "sess-12"


async def _collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


async def show_openai_chat(base_url: str, seen: dict[str, Any]) -> None:
    print("--- OpenAI Chat Completions ---")
    provider = OpenAIProvider(
        api_key="test-key",
        base_url=base_url,
        request_options={"temperature": 0},
        default_reasoning_effort="low",
        extra_headers={"X-Demo": "1"},
    )
    print("construct: temperature=0, reasoning_effort=low, extra header X-Demo=1")
    result = await provider.chat(
        [{"role": "user", "content": "hi"}], model="m", session_id=SESSION_ID
    )
    if isinstance(result, Done):
        print(f"chat result: content={result.content!r}")
        print(
            f"usage: input={result.tokens_input} output={result.tokens_output} "
            f"cache_read={result.cache_read_tokens}"
        )
    body = seen["chat"]["body"]
    print(
        f"request body: prompt_cache_key={body.get('prompt_cache_key')!r} "
        f"reasoning_effort={body.get('reasoning_effort')!r} "
        f"auth={seen['chat']['auth']!r}"
    )

    events = await _collect(
        provider.chat_stream([{"role": "user", "content": "hi"}], model="m")
    )
    print("stream frames:", [type(e).__name__ for e in events])
    if isinstance(events[-1], Done):
        print(
            f"stream done: content={events[-1].content!r} "
            f"input={events[-1].tokens_input} cache_read={events[-1].cache_read_tokens}"
        )

    tool_events = await _collect(
        provider.chat_stream([{"role": "user", "content": "hi"}], model="tool-model")
    )
    print("tool stream frames:", [type(e).__name__ for e in tool_events])
    for call in [e for e in tool_events if isinstance(e, ToolCall)]:
        print(
            f"tool call: id={call.id} name={call.name} "
            f"arguments={json.loads(call.arguments)}"
        )

    think_events = await _collect(
        provider.chat_stream([{"role": "user", "content": "hi"}], model="think-model")
    )
    print("reasoning stream frames:", [type(e).__name__ for e in think_events])

    bad = await provider.chat([{"role": "user", "content": "hi"}], model=BAD_MODEL)
    if isinstance(bad, Error):
        print(f"chat error: {bad.message[:60]}")
    bad_stream = await _collect(
        provider.chat_stream([{"role": "user", "content": "hi"}], model=BAD_MODEL)
    )
    print(
        "stream error:",
        [type(e).__name__ for e in bad_stream],
        getattr(bad_stream[0], "message", "")[:60],
    )


async def show_openai_responses(base_url: str, seen: dict[str, Any]) -> None:
    print("--- OpenAI Responses ---")
    provider = OpenAIResponsesProvider(
        api_key="test-key",
        base_url=base_url,
        default_reasoning_effort="low",
        user_agent="demo-agent/1.0",
    )
    print("construct: reasoning_effort=low, user_agent=demo-agent/1.0")
    result = await provider.chat([{"role": "user", "content": "hi"}], model="m")
    if isinstance(result, Done):
        print(f"chat result: content={result.content!r}")
        print(
            f"usage: input={result.tokens_input} output={result.tokens_output} "
            f"cache_read={result.cache_read_tokens}"
        )
    print(
        f"request body: reasoning={seen['responses']['body'].get('reasoning')} "
        f"user_agent={seen['responses']['user_agent']!r}"
    )

    events = await _collect(
        provider.chat_stream([{"role": "user", "content": "hi"}], model="m")
    )
    print("stream frames:", [type(e).__name__ for e in events])
    if isinstance(events[-1], Done):
        print(f"stream done: content={events[-1].content!r}")

    bad = await provider.chat([{"role": "user", "content": "hi"}], model=BAD_MODEL)
    if isinstance(bad, Error):
        print(f"chat error: {bad.message[:60]}")


async def show_anthropic(base_url: str, seen: dict[str, Any]) -> None:
    print("--- Anthropic Messages ---")
    provider = AnthropicProvider(
        api_key="test-key",
        base_url=base_url,
        max_output_tokens=64,
        request_options={"temperature": 0},
    )
    print("construct: max_output_tokens=64, temperature=0")
    result = await provider.chat([{"role": "user", "content": "hi"}], model="m")
    if isinstance(result, Done):
        print(f"chat result: content={result.content!r}")
        print(
            f"usage: input={result.tokens_input} cache_read={result.cache_read_tokens}"
        )
    print(f"request body: max_tokens={seen['anthropic']['body'].get('max_tokens')}")

    events = await _collect(
        provider.chat_stream([{"role": "user", "content": "hi"}], model="m")
    )
    print("stream frames:", [type(e).__name__ for e in events])
    if isinstance(events[-1], Done):
        print(
            f"stream done: content={events[-1].content!r} "
            f"input={events[-1].tokens_input} output={events[-1].tokens_output}"
        )

    bad = await provider.chat([{"role": "user", "content": "hi"}], model=BAD_MODEL)
    if isinstance(bad, Error):
        print(f"chat error: {bad.message[:60]}")


async def show_ollama(base_url: str, seen: dict[str, Any]) -> None:
    print("--- Ollama ---")
    provider = OllamaProvider(base_url=base_url, request_options={"temperature": 0.2})
    print("construct: temperature=0.2")
    result = await provider.chat([{"role": "user", "content": "hi"}], model="m")
    if isinstance(result, Done):
        print(f"chat result: content={result.content!r}")
    print(f"request body: temperature={seen['ollama']['body'].get('temperature')}")

    events = await _collect(
        provider.chat_stream([{"role": "user", "content": "hi"}], model="m")
    )
    print("stream frames:", [type(e).__name__ for e in events])
    if isinstance(events[-1], Done):
        print(f"stream done: content={events[-1].content!r}")

    bad = await provider.chat([{"role": "user", "content": "hi"}], model=BAD_MODEL)
    if isinstance(bad, Error):
        print(f"chat error: {bad.message[:60]}")


async def main() -> None:
    """Start the mock server, drive all four providers, then shut down."""
    runner, base_url, seen = await start_mock_llm_server()
    try:
        await show_openai_chat(base_url, seen)
        await show_openai_responses(base_url, seen)
        await show_anthropic(base_url, seen)
        await show_ollama(base_url, seen)
    finally:
        await runner.cleanup()
    print("provider tour complete")


if __name__ == "__main__":
    asyncio.run(main())
