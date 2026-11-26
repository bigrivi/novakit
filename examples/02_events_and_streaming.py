"""Novakit events and streaming: EventBus, agent frames, and Faker streams.

What you'll learn:
    - How to subscribe to EventBus with sync and async handlers.
    - How an agent turn looks as a SESSION -> ... -> DONE frame sequence.
    - How FakerLLMProvider.chat compares with chat_stream.
    - What ReasoningDelta and ToolCall stream events look like.
    - How abort_event stops a provider stream early.

Run:
    python examples/02_events_and_streaming.py

Everything here is offline and deterministic: FakerLLMProvider with a fixed
seed stands in for a real model so the event shapes stay stable.
"""

from __future__ import annotations

import asyncio

from novakit import Agent, AgentConfig, AgentOptions
from novakit.events import AgentEvent, AgentEventFrame, EventBus
from novakit.llm import (
    ChatStreamEvent,
    Done,
    FakerLLMProvider,
    Message,
    TextDelta,
    ToolCall,
)


async def main() -> None:
    """Run each streaming scene in order."""
    print("02 events and streaming")
    await show_event_bus()
    await show_agent_frames()
    await show_chat_vs_stream()
    await show_reasoning_and_tool_call()
    await show_abort()
    await show_shared_bus()
    print("02 events and streaming: done")


async def show_event_bus() -> None:
    """Subscribe sync + async handlers, then emit and unsubscribe."""
    # Configure: one bus, two subscribers watching the same event.
    bus = EventBus()
    seen: list[tuple[AgentEvent, object]] = []

    def sync_handler(event: AgentEvent, data: object = None) -> None:
        seen.append((event, data))

    async def async_handler(event: AgentEvent, data: object = None) -> None:
        seen.append((event, data))

    # Execute: emit once with both handlers, remove one, emit again.
    bus.on(AgentEvent.TEXT_DELTA, sync_handler)
    bus.on(AgentEvent.TEXT_DELTA, async_handler)
    await bus.emit(AgentEvent.TEXT_DELTA, "hi")
    bus.off(AgentEvent.TEXT_DELTA, sync_handler)
    await bus.emit(AgentEvent.TEXT_DELTA, "again")

    # Observe: first emit reached both handlers, second only the remaining one.
    print(f"  bus deliveries: {len(seen)} (2 for 'hi', 1 for 'again')")


async def show_agent_frames() -> None:
    """Stream one agent turn and print its frame sequence."""
    # Configure: a minimal offline agent.
    agent = Agent(
        AgentConfig(model="demo"),
        FakerLLMProvider(seed=7, reasoning_probability=0.0),
    )
    started: list[tuple[AgentEvent, object]] = []
    agent.on(AgentEvent.START, lambda event, data=None: started.append((event, data)))

    # Execute: collect every frame of one turn.
    frames: list[AgentEventFrame] = [f async for f in agent.chat_stream("hello")]

    # Observe: turns open with SESSION and close with DONE.
    sequence = " -> ".join(event.name for event, _ in frames)
    print(f"  agent frames ({len(frames)}): {sequence}")
    print(f"  START handler fired x{len(started)}")


async def show_chat_vs_stream() -> None:
    """Compare the one-shot chat reply with the streamed deltas."""
    # Configure: same seed so both calls draw the same reply.
    provider = FakerLLMProvider(seed=7, reasoning_probability=0.0)
    messages = [Message(role="user", content="hello")]

    # Execute: one-shot reply, then the full streamed event list.
    reply = await provider.chat(messages, model="demo")
    streamed: list[ChatStreamEvent] = [
        event async for event in provider.chat_stream(messages, model="demo")
    ]

    # Observe: deltas join into the final Done text, matching chat().
    chunks = [e.content for e in streamed if isinstance(e, TextDelta)]
    tail = streamed[-1]
    print(f"  chat() reply: {reply.content!r}")
    print(f"  chat_stream() deltas ({len(chunks)}): {chunks}")
    print(f"  chat_stream() final: {type(tail).__name__} content={tail.content!r}")


async def show_reasoning_and_tool_call() -> None:
    """Print what reasoning deltas and tool calls look like in a stream."""
    # Configure: force reasoning in one provider, a tool call in another.
    reasoning_provider = FakerLLMProvider(seed=7, reasoning_probability=1.0)
    tool_provider = FakerLLMProvider(
        seed=7, reasoning_probability=0.0, tool_call_probability=1.0
    )
    schemas = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look things up.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]

    # Execute: stream from both providers.
    reasoning_events = [
        e
        async for e in reasoning_provider.chat_stream(
            [Message(role="user", content="hello")], model="demo"
        )
    ]
    tool_events = [
        e
        async for e in tool_provider.chat_stream(
            [Message(role="user", content="Summarize this idea briefly.")],
            model="demo",
            tools=schemas,
        )
    ]

    # Observe: each stream mixes its own delta type before the closing Done.
    reasoning_kinds = sorted({type(e).__name__ for e in reasoning_events})
    tool_kinds = sorted({type(e).__name__ for e in tool_events})
    calls = [e for e in tool_events if isinstance(e, ToolCall)]
    print(f"  reasoning stream event kinds: {reasoning_kinds}")
    print(f"  tool stream event kinds: {tool_kinds}")
    for call in calls:
        print(f"  tool call: {call.name}({call.arguments})")
    tail = tool_events[-1]
    if isinstance(tail, Done):
        print(f"  tool stream Done carries {len(tail.tool_calls or [])} tool call(s)")


async def show_abort() -> None:
    """Stop a provider stream with an already-set abort event."""
    # Configure: Faker provider plus a pre-set abort flag.
    provider = FakerLLMProvider(seed=7, reasoning_probability=0.0)
    abort = asyncio.Event()
    abort.set()

    # Execute: the stream sees the flag immediately.
    events = [
        e
        async for e in provider.chat_stream(
            [Message(role="user", content="hello")],
            model="demo",
            abort_event=abort,
        )
    ]

    # Observe: a single aborted Done instead of text deltas.
    first = events[0] if events else None
    print(f"  aborted stream events: {len(events)}")
    print(
        f"  first event: {type(first).__name__} aborted={getattr(first, 'aborted', '?')}"
    )


async def show_shared_bus() -> None:
    """Inject one EventBus and watch the agent publish into it."""
    # Configure: a bus the host owns, handed to the agent via options.
    bus = EventBus()
    seen: list[AgentEvent] = []
    bus.on(AgentEvent.DONE, lambda event, data=None: seen.append(event))
    agent = Agent(
        AgentConfig(model="demo"),
        FakerLLMProvider(seed=7, reasoning_probability=0.0),
        options=AgentOptions(events=bus),
    )

    # Execute: run one turn; the bus observes it from the side.
    frames = [f async for f in agent.chat_stream("hello")]

    # Observe: the stream and the side-channel bus tell the same story.
    print(f"  stream ended with: {frames[-1][0].name}")
    print(f"  injected bus observed DONE x{len(seen)}")


if __name__ == "__main__":
    asyncio.run(main())
