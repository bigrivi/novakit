"""07 Tools and behaviors: register tools, add behaviors, then run one agent turn.

What you'll learn:
- Register tools with the ``@tool`` decorator and ``register_direct``,
  and mark read vs. write tools with ``read_only``
- Use ``DefaultToolBehavior`` for input normalization, result
  post-processing, and the ``on_success`` hook
- How ``ToolContext`` is injected into tool functions and how
  ``TurnContext`` is seen by behaviors
- How ``TOOL_CALL``/``TOOL_RESULT`` frames pair up in one agent tool turn
- How ``ToolGuardrails`` halts repeated calls

Run:
    python examples/07_tools_and_behaviors.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from novakit.agent import (
    Agent,
    AgentConfig,
    AgentOptions,
    GuardrailAction,
    ToolGuardrails,
)
from novakit.events import AgentEvent
from novakit.llm import FakerLLMProvider, ToolResult
from novakit.session import InMemorySession
from novakit.tools import (
    DefaultToolBehavior,
    ToolContext,
    ToolRegistry,
    TurnContext,
    tool,
)

seen_context: dict[str, bool] = {}


@tool(
    name="read",
    description="Read a UTF-8 text file.",
    parameters={
        "type": "object",
        "properties": {"filePath": {"type": "string"}},
        "required": ["filePath"],
    },
)
async def read(
    filePath: str = "", turn_context: ToolContext | None = None
) -> ToolResult:
    """Read one text file and return its content."""
    # ToolContext is injected by the caller; record what arrived here,
    # and let the main flow print the observations.
    seen_context["injected"] = isinstance(turn_context, ToolContext)
    seen_context["model_named"] = bool(
        turn_context is not None and isinstance(turn_context.model, str)
    )
    try:
        text = Path(filePath).read_text(encoding="utf-8")
    except OSError as exc:
        return ToolResult(success=False, content=f"cannot read: {exc}", error=str(exc))
    return ToolResult(success=True, content=text)


async def grep(pattern: str = "", path: str = "") -> ToolResult:
    """Return lines of one file that contain the pattern."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return ToolResult(success=False, content=f"cannot read: {exc}", error=str(exc))
    return ToolResult(
        success=True,
        content="\n".join(line for line in lines if pattern in line),
    )


def _append_line(path: str, text: str) -> None:
    """Append one line to a scratch file."""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


async def write_note(path: str = "", text: str = "") -> ToolResult:
    """Append one line to a scratch file (a write tool)."""
    try:
        await asyncio.to_thread(_append_line, path, text)
    except OSError as exc:
        return ToolResult(success=False, content=f"cannot write: {exc}", error=str(exc))
    return ToolResult(success=True, content="noted")


class TrimBehavior(DefaultToolBehavior):
    """Normalize input, tag output, and count successful runs."""

    def __init__(self) -> None:
        self.success_count = 0

    def normalize_input(self, args: dict) -> dict:
        """Strip stray whitespace the model may add around arguments."""
        return {k: (v.strip() if isinstance(v, str) else v) for k, v in args.items()}

    def postprocess(self, raw_content: str) -> tuple[str, list | None]:
        """Append a tag so the post-processing stays visible in stored content."""
        return raw_content.rstrip("\n") + "\n[verified]", None

    def on_success(self, ctx: TurnContext) -> None:
        """Record that the turn finished this tool successfully."""
        self.success_count += 1


def build_registry() -> tuple[ToolRegistry, TrimBehavior]:
    """Register decorator and direct tools and mark their read/write nature."""
    registry = ToolRegistry()
    registry.register(read, "read")
    registry.register_direct(
        name="grep",
        description="Search one file for a pattern.",
        func=grep,
        params_schema={
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern", "path"],
        },
    )
    registry.register_direct(
        name="write_note",
        description="Append a line to a scratch file.",
        func=write_note,
        params_schema={
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"],
        },
    )
    behavior = TrimBehavior()
    registry.set_behavior("read", behavior)
    # Decorator-registered tools are read-only by default;
    # the write tool is explicitly marked as writable.
    write_tool = registry.get("write_note")
    if write_tool is not None:
        write_tool.read_only = False
    return registry, behavior


def show_registry(registry: ToolRegistry) -> None:
    """Print each registered tool's schema and read/write flag."""
    for entry in registry.list_tools():
        print(f"- tool={entry.name} read_only={entry.read_only}")
        print(f"  schema={json.dumps(entry.params_schema, ensure_ascii=False)}")


def show_pairing(frames: list) -> tuple[list, list]:
    """Print how each announced call pairs with its terminal result."""
    calls = [data for event, data in frames if event is AgentEvent.TOOL_CALL]
    results = [data for event, data in frames if event is AgentEvent.TOOL_RESULT]
    print(f"- announced calls: {len(calls)}, terminal results: {len(results)}")
    for call, result in zip(calls, results, strict=False):
        payload = result["result"]
        print(f"  call id={call.id} name={call.name} args={call.arguments}")
        print(
            f"  result for {result['tool_call_id']}: "
            f"success={payload.success} content={payload.content[:40]!r}"
        )
    return calls, results


async def run_agent_turn(registry: ToolRegistry, demo_file: Path) -> tuple:
    """Run one Faker-driven turn that calls the ``read`` tool."""
    agent = Agent(
        AgentConfig(model="demo", provider="demo", max_iterations=5),
        FakerLLMProvider(seed=7),
        options=AgentOptions(session=InMemorySession(), tools=registry),
    )
    frames = [
        frame async for frame in agent.chat_stream(f"please read the file {demo_file}")
    ]
    return agent, frames


async def show_guardrail(demo_file: Path) -> None:
    """Observe the same read call three times, from allow to halt."""
    guards = ToolGuardrails(max_same_call=2, read_tools=frozenset({"read", "grep"}))
    action = GuardrailAction.ALLOW
    for index in range(3):
        action = guards.observe("read", {"filePath": str(demo_file)}, True)
        print(f"- observe #{index + 1}: {action.value}")


async def main() -> None:
    """Demo the tool stack: configure, run, then observe the results."""
    print("== 07 tools and behaviors ==")
    with tempfile.TemporaryDirectory() as tmp:
        demo_file = Path(tmp) / "demo.txt"
        demo_file.write_text("The quick brown fox.\nsecond line\n", encoding="utf-8")

        # 1) Configure: register tools and behaviors, then show the registry.
        registry, behavior = build_registry()
        print("registered tools:")
        show_registry(registry)

        # 2) Run: one Faker-driven agent turn calling the read tool.
        agent, frames = await run_agent_turn(registry, demo_file)

        # 3) Observe: call pairing, ToolContext, post-processing, guardrails.
        print("tool call pairing:")
        show_pairing(frames)
        print(f"ToolContext injected: {seen_context.get('injected')}")
        print(f"ToolContext carries model name: {seen_context.get('model_named')}")
        stored = [m for m in await agent.session.get_messages() if m.role == "tool"]
        if stored:
            print(f"stored tool message tail: {stored[-1].content[-20:]!r}")
            print("  (the tail shows postprocess appended '[verified]')")
        print(f"on_success hook runs: {behavior.success_count}")
        print("guardrail on repeated reads:")
        await show_guardrail(demo_file)
    print("07 tools and behaviors: done")


if __name__ == "__main__":
    asyncio.run(main())
