"""08 Approvals and guardrails: grant, deny, remember, abort, no manager.

What you'll learn:
- How ``InMemoryApprovalManager`` grants approvals and remembers
  per-session grants
- The ``Agent`` + ``AgentPorts`` approval flow: the ``APPROVAL_REQUIRED``
  event and ``resolve_approval``
- How a denial skips the tool while the call/result pair still closes
- How ``wait_with_heartbeat`` plus an abort event cancels a pending wait
- How a guarded tool is refused before running when there is no manager

Run:
    python examples/08_approvals_and_guardrails.py
"""

from __future__ import annotations

import asyncio
import json

from novakit.agent import Agent, AgentConfig, AgentOptions, AgentPorts, ToolInvoker
from novakit.events import AgentEvent
from novakit.llm import FakerLLMProvider, ToolCall, ToolResult
from novakit.session import InMemorySession
from novakit.tools import (
    DefaultToolBehavior,
    InMemoryApprovalManager,
    PreExecutionCheck,
    ToolRegistry,
    TurnContext,
    tool,
)

ran: list[str] = []


@tool(
    name="shell",
    description="Run a shell command (demo: echoed, never executed).",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)
async def shell(command: str = "") -> ToolResult:
    """Echo the command instead of executing it."""
    ran.append(command)
    return ToolResult(success=True, content=f"ran: {command}")


class ShellApproval(DefaultToolBehavior):
    """Request approval through the turn's manager, honoring remembered grants."""

    async def before_execute(self, args: dict, ctx: TurnContext) -> PreExecutionCheck:
        """Create a pending request, or pass when already granted or unmanaged."""
        manager = ctx.approval_manager
        if manager is None:
            return PreExecutionCheck(approval_request={"id": "shell-no-manager"})
        request_id = manager.pre_request(
            command=f"run {args.get('command', '')}",
            description="demo shell approval",
            session_id=ctx.session_id,
            rule="demo-shell",
            family="shell",
        )
        if not request_id:
            return PreExecutionCheck()
        return PreExecutionCheck(
            approval_request={"id": request_id, "command": args.get("command", "")}
        )


def build_agent(manager: InMemoryApprovalManager) -> Agent:
    """Build an agent whose ``shell`` tool always asks the given manager."""
    registry = ToolRegistry()
    registry.register(shell, "shell")
    registry.set_behavior("shell", ShellApproval())
    return Agent(
        AgentConfig(model="demo", provider="demo", max_iterations=5),
        FakerLLMProvider(seed=3),
        options=AgentOptions(
            session=InMemorySession(),
            tools=registry,
            ports=AgentPorts(approval=manager),
        ),
    )


async def run_prompt(
    agent: Agent, approve: bool, session_id: str | None = None
) -> list:
    """Stream one prompt, resolving the approval as told when it appears."""
    frames: list = []
    stream = agent.chat_stream("please run command line check", session_id=session_id)
    async for event, data in stream:
        if event is AgentEvent.APPROVAL_REQUIRED:
            # In this tutorial you play the approver: answer on the spot
            # using the approve flag.
            resolved = agent.resolve_approval(data["id"], approve, remember=approve)
            print(f"  approval {data['id']}: approve={approve} resolved={resolved}")
            frames.append((event, data))
        else:
            frames.append((event, data))
    return frames


def show_tool_results(frames: list, label: str) -> None:
    """Print the success/denial state and content of this turn's tool results."""
    for event, data in frames:
        if event is AgentEvent.TOOL_RESULT:
            result = data["result"]
            print(
                f"  [{label}] success={result.success} "
                f"denied={result.denied} content={result.content!r}"
            )


async def show_remembered_grant() -> None:
    """Approve once with remember, then reuse the session without a prompt."""
    print("- remember: first turn asks, second turn reuses the grant")
    manager = InMemoryApprovalManager()
    agent = build_agent(manager)
    first = await run_prompt(agent, True)
    asked_first = any(e is AgentEvent.APPROVAL_REQUIRED for e, _ in first)
    print(f"  first turn asked for approval: {asked_first}")
    show_tool_results(first, "first")
    session_id = next(d for e, d in first if e is AgentEvent.SESSION)
    second = await run_prompt(agent, True, session_id=session_id)
    asked_second = any(e is AgentEvent.APPROVAL_REQUIRED for e, _ in second)
    print(
        f"  second turn asked for approval: {asked_second} (remembered grant skips it)"
    )
    show_tool_results(second, "second")
    print(f"  reused session: {session_id}")


async def show_denial() -> None:
    """Deny approval: the tool never runs and the pairing still terminates."""
    print("- deny: tool is skipped, but the call/result pair still closes")
    ran.clear()
    manager = InMemoryApprovalManager()
    agent = build_agent(manager)
    frames = await run_prompt(agent, False)
    asked = any(e is AgentEvent.APPROVAL_REQUIRED for e, _ in frames)
    print(f"  asked for approval: {asked}")
    show_tool_results(frames, "denied")
    print(f"  tool actually ran: {ran} (empty means the denial took effect)")
    print(f"  resolve unknown id returns: {agent.resolve_approval('unknown-id', True)}")


async def show_abort() -> None:
    """Abort a pending wait: the waiter ends empty and the request is gone."""
    print("- abort: waiting on approval can be cancelled")
    manager = InMemoryApprovalManager()
    request_id = manager.pre_request("deploy demo", session_id="s-x")
    abort = asyncio.Event()

    async def waiter() -> list:
        return [
            verdict
            async for verdict in manager.wait_with_heartbeat(
                request_id, heartbeat_interval=60, abort_event=abort
            )
        ]

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    abort.set()
    verdicts = await task
    print(f"  waiter verdicts after abort: {verdicts}")
    print(f"  resolve after abort returns: {manager.resolve(request_id, True)}")


async def show_no_manager() -> None:
    """Without a manager a guarded tool is refused before it can run."""
    print("- no manager: guarded tool is refused, never executed")
    ran.clear()
    session = InMemorySession()
    await session.create_session()
    registry = ToolRegistry()
    registry.register(shell, "shell")
    registry.set_behavior("shell", ShellApproval())
    emitted: list = []

    async def emit(event: AgentEvent, data: object = None) -> None:
        emitted.append((event, data))

    async def stop() -> dict | None:
        return None

    # Drive ToolInvoker directly: without an agent you can still watch
    # the "no approval manager means refusal" behavior.
    invoker = ToolInvoker(
        registry=registry,
        session=session,
        approval=None,
        guardrails=None,
        llm=FakerLLMProvider(seed=1),
        model="demo",
        provider="demo",
        abort_event=asyncio.Event(),
        emit=emit,
        emit_approval=emit,
        stop_if_aborted=stop,
    )
    frames = [
        frame
        async for frame in invoker.run(
            [ToolCall(id="c1", name="shell", arguments=json.dumps({"command": "x"}))]
        )
    ]
    print(f"  event order: {[event.name for event, _ in frames]}")
    print("  (TOOL_CALL -> APPROVAL_REQUIRED -> APPROVAL_RESULT -> TOOL_RESULT)")
    print(f"  denied={frames[-1][1]['result'].denied} ran={ran}")


async def main() -> None:
    """Demo the full approval lifecycle: configure, run, then observe."""
    print("== 08 approvals and guardrails ==")
    # Each show_* builds its own manager/agent/invoker; main only runs
    # them in order and prints what to watch for.
    await show_remembered_grant()
    await show_denial()
    await show_abort()
    await show_no_manager()
    print("08 approvals and guardrails: done")


if __name__ == "__main__":
    asyncio.run(main())
