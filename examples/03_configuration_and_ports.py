"""Novakit configuration and ports: CoreConfig, AgentConfig, options, ports.

What you'll learn:
    - Which four fields make up CoreConfig (any matching dataclass works).
    - That extra AgentConfig host fields are accepted but ignored by the core.
    - How AgentOptions injects session, tools, events, and system prompt.
    - How the AgentPorts begin_turn hook reports each new turn.
    - How to inject a host approval manager through ports.

Run:
    python examples/03_configuration_and_ports.py

Everything runs offline with FakerLLMProvider; the focus here is wiring,
not model output.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from novakit import Agent, AgentConfig, AgentOptions, EventBus
from novakit.agent import AgentPorts
from novakit.events import AgentEventFrame
from novakit.llm import FakerLLMProvider
from novakit.session import InMemorySession
from novakit.tools import InMemoryApprovalManager, ToolRegistry


@dataclass
class CustomConfig:
    """Host-owned config carrying exactly the CoreConfig fields."""

    model: str = "demo"
    provider: str = "demo"
    max_iterations: int = 3
    reasoning_effort: str | None = None


async def main() -> None:
    """Walk through config, options, and ports in order."""
    print("03 configuration and ports")
    await show_core_config()
    await show_compat_fields()
    await show_options()
    await show_begin_turn()
    await show_approval_port()
    print("03 configuration and ports: done")


async def run_turn(agent: Agent, text: str) -> list[AgentEventFrame]:
    """Collect every event frame of one chat request."""
    return [frame async for frame in agent.chat_stream(text)]


async def show_core_config() -> None:
    """Run an agent from a plain host dataclass instead of AgentConfig."""
    # Configure: only the four CoreConfig fields matter to the core.
    config = CustomConfig(reasoning_effort="low")
    agent = Agent(config, FakerLLMProvider(seed=1, reasoning_probability=0.0))

    # Execute: one offline turn.
    frames = await run_turn(agent, "hello")

    # Observe: the turn completes and the host config is kept as-is.
    print(f"  turn ended with: {frames[-1][0].name}")
    print(
        "  CoreConfig fields: "
        f"model={agent.config.model!r} provider={agent.config.provider!r} "
        f"max_iterations={agent.config.max_iterations} "
        f"reasoning_effort={agent.config.reasoning_effort!r}"
    )


async def show_compat_fields() -> None:
    """Pass host-only AgentConfig fields; the core ignores them safely."""
    # Configure: fields a host application understands but novakit core does not.
    # They exist so shared configs keep working when embedded elsewhere.
    config = AgentConfig(
        model="demo",
        max_iterations=2,
        max_tokens=-1,
        temperature=999.0,
        tools=["no-such-tool"],
        memory_review_interval=-1,
        shell_read_only=True,
        shell_review=True,
        permission_mode="bogus-mode",
    )
    agent = Agent(config, FakerLLMProvider(seed=2, reasoning_probability=0.0))

    # Execute: the turn runs with core settings only.
    frames = await run_turn(agent, "hello")

    # Observe: bogus host values never break the turn.
    print(f"  turn ended with: {frames[-1][0].name}")
    print("  host-only fields (tools, temperature, permission_mode, ...) ignored")


async def show_options() -> None:
    """Wire session, tools, events, and system prompt through AgentOptions."""
    # Configure: the four most common host injections in one place.
    session = InMemorySession()
    events = EventBus()
    options = AgentOptions(
        session=session,
        tools=ToolRegistry(),
        events=events,
        system_prompt="Be concise.",
    )
    agent = Agent(
        AgentConfig(model="demo"),
        FakerLLMProvider(seed=3, reasoning_probability=0.0),
        options=options,
    )

    # Execute: one turn against the injected session.
    frames = await run_turn(agent, "hello")

    # Observe: the agent kept our session object and finished the turn.
    print(f"  turn ended with: {frames[-1][0].name}")
    print(f"  session kept: {agent.session is session}")
    print("  options wired: session/tools/events/system_prompt")


async def show_begin_turn() -> None:
    """Print what the begin_turn port hook receives for each turn."""
    # Configure: record whatever the hook is called with.
    calls: list[dict[str, object]] = []

    def record_turn(
        session_id: str,
        checkpoint_id: str,
        label: str,
        workspace: str | None,
    ) -> None:
        calls.append(
            {
                "session_id": session_id,
                "checkpoint_id": checkpoint_id,
                "label": label,
                "workspace": workspace,
            }
        )

    agent = Agent(
        AgentConfig(model="demo"),
        FakerLLMProvider(seed=4, reasoning_probability=0.0),
        options=AgentOptions(ports=AgentPorts(begin_turn=record_turn)),
    )

    # Execute: one turn opens one checkpoint, firing the hook once.
    frames = await run_turn(agent, "hello")

    # Observe: the hook sees routing info (session/checkpoint/label/workspace).
    print(f"  turn ended with: {frames[-1][0].name}")
    print(f"  hook fired x{len(calls)}; stream opened session {frames[0][1]!r}")
    for call in calls:
        print(f"  begin_turn got: {call}")


async def show_approval_port() -> None:
    """Inject a host approval manager through AgentPorts."""
    # Configure: a host-owned approval policy plugged into the agent.
    manager = InMemoryApprovalManager()
    agent = Agent(
        AgentConfig(model="demo"),
        FakerLLMProvider(seed=5, reasoning_probability=0.0),
        options=AgentOptions(ports=AgentPorts(approval=manager)),
    )

    # Execute: a plain turn needs no approvals but still carries the port.
    frames = await run_turn(agent, "hello")

    # Observe: the turn completes with the host manager attached.
    print(f"  turn ended with: {frames[-1][0].name}")
    print("  approval manager port injected")


if __name__ == "__main__":
    asyncio.run(main())
