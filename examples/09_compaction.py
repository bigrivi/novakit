"""09 Context compaction: plan, summarize, persist, and inject into an agent.

What you'll learn:
- How ``CompactionConfig`` sets budgets and how a token estimator
  decides the threshold
- How layer-1 snipping offloads large tool outputs via ``ToolOutputSink``
- How ``prepare_compaction`` picks a safe split point and
  ``run_compaction_plan`` folds history
- The ``DefaultCompactor`` event order and ``MemoryCompactionStore`` state
- How to inject the compactor through ``AgentPorts`` and trigger it
  with ``compact_now``

Fixed-window note: ``CharacterTokenResolver`` estimates roughly 1 token
per 4 characters with a fixed window, just so this tutorial can demo the
compaction flow offline and deterministically. Plug in the real resolver
for your model in production use.

Run:
    python examples/09_compaction.py
"""

from __future__ import annotations

import asyncio

from novakit.agent import Agent, AgentConfig, AgentOptions, AgentPorts
from novakit.compaction import (
    CompactionConfig,
    DefaultCompactor,
    MemoryCompactionStore,
    find_split_point,
    first_safe_boundary,
    prepare_compaction,
    run_compaction_plan,
)
from novakit.events import AgentEvent
from novakit.llm import FakerLLMProvider
from novakit.session import InMemorySession

BIG = "x" * 400


class CharacterTokenResolver:
    """Fixed-window token source for experiments: 1 token per 4 chars."""

    def __init__(self, limit: int = 1000) -> None:
        self.limit = limit

    def estimate_messages(self, messages: list, model: str) -> int:
        """Estimate tokens at roughly four characters each."""
        total = 0
        for message in messages:
            content = getattr(message, "content", "") or ""
            total += max(1, len(content) // 4)
        return total

    def context_limit(self, model: str, provider: str) -> int:
        """Return the fixed context window used for planning."""
        return self.limit


def compaction_config() -> CompactionConfig:
    """Small budgets so a handful of messages already exceed the threshold."""
    return CompactionConfig(
        output_reserve_tokens=200,
        summary_reserve_tokens=100,
        summary_keep_tokens=100,
        snip_max_chars=200,
        snip_tool_output_token_budget=50,
        snip_preserve_last_n_messages=2,
        max_consecutive_failures=3,
    )


async def seed_history(count: int = 10) -> tuple[InMemorySession, object, list]:
    """Create a session with alternating turns plus one giant tool result."""
    sessions = InMemorySession()
    session = await sessions.create_session(first_message="demo")
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        await sessions.add_message(
            role=role,
            content=f"m{index}-" + BIG,
            time_created=1000 + index * 1000,
        )
    await sessions.add_message(
        role="tool", content="v" * 5000, tool_call_id="tc-old", time_created=99999
    )
    messages = await sessions.get_messages(session.id)
    return sessions, session, messages


async def show_plan_and_run() -> None:
    """Plan directly, run the fold, and print the persisted summary."""
    print("- plan: estimate threshold, pick a split, offload the giant output")
    sessions, session, messages = await seed_history()
    store = MemoryCompactionStore(sessions)
    offloaded: list[tuple[str, str]] = []

    def sink(message_id: str, content: str) -> str | None:
        offloaded.append((message_id, content))
        return f"mem://{message_id}"

    resolver = CharacterTokenResolver()
    plan = await prepare_compaction(
        session_id=session.id,
        messages=messages,
        last_compacted_at=None,
        store=store,
        model="stub-model",
        provider="stub-provider",
        config=compaction_config(),
        resolver=resolver,
        sink=sink,
    )
    print(
        f"  over_threshold={plan.over_threshold} needs_compaction={plan.needs_compaction}"
    )
    print(
        f"  split_index={plan.split_index} role_at_split={messages[plan.split_index].role}"
    )
    print(f"  offloaded tool outputs via sink: {len(offloaded)}")

    deltas: list[str] = []
    done = await run_compaction_plan(
        plan,
        store,
        FakerLLMProvider(seed=11),
        model="stub-model",
        provider="stub-provider",
        messages=messages,
        on_delta=deltas.append,
        config=compaction_config(),
        resolver=resolver,
    )
    visible = await sessions.get_messages(session.id)
    print(f"  run done={done} deltas={len(deltas)}")
    print(f"  before={len(messages)} after={len(visible)}")
    print(f"  summary head: {visible[0].content[:60]!r}")
    print(f"  summary flag={visible[0].summary} compacted_at={session.compacted_at}")


async def show_compactor_frames() -> None:
    """Drive ``DefaultCompactor`` directly and print frame order and state."""
    print(
        "- compactor frames: CONTEXT_UPDATE ... COMPACTION_START/END ... CONTEXT_UPDATE"
    )
    sessions, session, messages = await seed_history()
    compactor = DefaultCompactor(
        "stub-model",
        "stub-provider",
        store=MemoryCompactionStore(sessions),
        config=compaction_config(),
        resolver=CharacterTokenResolver(),
    )
    emitted: list = []
    yielded: list = []

    async def emit(event: AgentEvent, data: object) -> None:
        emitted.append((event, data))

    async for event, data in compactor.maybe_compact(
        messages, session, FakerLLMProvider(seed=11), emit
    ):
        yielded.append((event, data))
    kinds = [event for event, _ in yielded]
    print(f"  yielded: {[k.name for k in kinds]}")
    print(f"  compactor.compacted={compactor.compacted}")
    print(f"  emitted (excl. deltas)={len(emitted)}")
    visible = await sessions.get_messages(session.id)
    print(f"  summaries kept: {sum(1 for m in visible if m.summary == 1)}")


async def show_agent_ports() -> None:
    """Inject the compactor into an agent and force one compaction pass."""
    print("- agent ports: compact_now triggers the injected compactor")
    sessions = InMemorySession()
    session = await sessions.create_session(first_message="demo")
    for index in range(10):
        role = "user" if index % 2 == 0 else "assistant"
        await sessions.add_message(
            role=role, content=f"a{index}-" + BIG, time_created=1000 + index * 1000
        )
    compactor = DefaultCompactor(
        "stub-model",
        "stub-provider",
        store=MemoryCompactionStore(sessions),
        config=compaction_config(),
        resolver=CharacterTokenResolver(),
    )
    agent = Agent(
        AgentConfig(model="demo", provider="demo", max_iterations=5),
        FakerLLMProvider(seed=11),
        options=AgentOptions(session=sessions, ports=AgentPorts(compactor=compactor)),
    )
    outcome = await agent.compact_now(session.id)
    print(f"  compact_now outcome={outcome} compactor.compacted={compactor.compacted}")


def show_safe_boundaries() -> None:
    """Show split points never strand a tool result from its assistant turn."""
    print("- split: safe boundary never leaves a tool message dangling")
    messages = [
        {"role": "user", "content": "u" + BIG},
        {"role": "assistant", "content": "calling"},
        {"role": "tool", "content": "t" + BIG},
        {"role": "user", "content": "next-" + BIG},
    ]
    resolver = CharacterTokenResolver()
    split = find_split_point(messages, keep_tokens=10, force=True, resolver=resolver)
    print(f"  split_index={split} role_at_split={messages[split]['role']}")
    print(f"  first_safe_boundary(messages, 2)={first_safe_boundary(messages, 2)}")


async def main() -> None:
    """Demo compaction in three stages: configure, run, then observe."""
    print("== 09 compaction ==")
    # Each show_* builds its own budgets and estimator from
    # compaction_config + CharacterTokenResolver.
    show_safe_boundaries()
    await show_plan_and_run()
    await show_compactor_frames()
    await show_agent_ports()
    print("09 compaction: done")


if __name__ == "__main__":
    asyncio.run(main())
