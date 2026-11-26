"""Agent end-to-end: compaction fires mid-dialogue without breaking tools."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from novakit import Agent, AgentConfig, AgentOptions
from novakit.agent.ports import AgentPorts
from novakit.compaction.compactor import DefaultCompactor
from novakit.compaction.config import CompactionConfig
from novakit.compaction.store import MemoryCompactionStore
from novakit.events import AgentEvent, AgentEventFrame
from novakit.llm import (
    ChatStreamEvent,
    Done,
    LLMProvider,
    TextDelta,
    ToolCall,
    ToolResult,
)
from novakit.session import InMemorySession
from novakit.tools import ToolRegistry, tool


class StubResolver:
    """Deterministic token source: 1 token per 4 chars, fixed window."""

    def __init__(self, limit: int = 4000) -> None:
        self.limit = limit

    def estimate_messages(self, messages: list, model: str) -> int:
        total = 0
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content", "") or ""
            else:
                content = getattr(message, "content", "") or ""
            total += max(1, len(content) // 4)
        return total

    def context_limit(self, model: str, provider: str) -> int:
        return self.limit


class ScriptedProvider(LLMProvider):
    """Serve one script per chat_stream call: summary, tool call, then text."""

    def __init__(self, scripts: list[list[object]]) -> None:
        self._scripts = scripts
        self._index = 0
        self.seen_messages: list[list] = []

    async def chat(
        self,
        messages: list,
        model: str = "m",
        stream: bool = False,
        tools: list[dict] | None = None,
        **kwargs: object,
    ) -> Done:
        return Done(content="unused")

    def chat_stream(
        self,
        messages: list,
        model: str = "m",
        tools: list[dict] | None = None,
        **kwargs: object,
    ) -> AsyncGenerator[ChatStreamEvent, None]:
        return self._drain(messages)

    async def _drain(self, messages: list) -> AsyncGenerator[ChatStreamEvent, None]:
        self.seen_messages.append(messages)
        script = self._scripts[min(self._index, len(self._scripts) - 1)]
        self._index += 1
        for item in script:
            await asyncio.sleep(0)
            yield item  # type: ignore[misc]

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text)


@pytest.mark.asyncio
async def test_agent_compacts_mid_dialogue_and_keeps_tool_pairs() -> None:
    sessions = InMemorySession()
    session = await sessions.create_session(first_message="big task")
    await sessions.add_message(
        role="user", content="u0-" + "x" * 2000, time_created=1000
    )
    await sessions.add_message(
        role="assistant",
        content="calling",
        tool_calls=[{"id": "call-0", "name": "echo", "arguments": "{}"}],
        time_created=2000,
    )
    await sessions.add_message(
        role="tool",
        content="old-result",
        tool_call_id="call-0",
        time_created=3000,
    )
    for index in range(3, 10):
        role = "user" if index % 2 else "assistant"
        await sessions.add_message(
            role=role,
            content=f"f{index}-" + "y" * 2000,
            time_created=1000 + index * 1000,
        )
    old_ids = {m.id for m in await sessions.get_messages(session.id)}
    recent_id = max(
        await sessions.get_messages(session.id), key=lambda m: m.time_created
    ).id

    provider = ScriptedProvider(
        [
            [TextDelta(content="folded history")],
            [ToolCall(id="call-9", name="echo", arguments='{"text":"hi"}')],
            [TextDelta(content="done")],
        ]
    )
    registry = ToolRegistry()

    @tool(
        name="echo",
        parameters={
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    async def echo(text: str) -> ToolResult:
        return ToolResult(content=f"echo:{text}")

    registry.register(echo, name="echo")
    compactor = DefaultCompactor(
        "test",
        "test",
        store=MemoryCompactionStore(sessions),
        config=CompactionConfig(
            output_reserve_tokens=200,
            summary_reserve_tokens=100,
            summary_keep_tokens=500,
            max_consecutive_failures=3,
        ),
        resolver=StubResolver(),
    )
    agent = Agent(
        AgentConfig(model="test", max_iterations=3),
        provider,
        options=AgentOptions(
            session=sessions,
            tools=registry,
            ports=AgentPorts(compactor=compactor),
        ),
    )

    frames: list[AgentEventFrame] = [
        frame async for frame in agent.chat_stream("go on", session_id=session.id)
    ]

    kinds = [event for event, _ in frames]
    assert kinds[0] is AgentEvent.SESSION
    assert kinds[-1] is AgentEvent.DONE
    assert frames[-1][1]["content"] == "done"
    start_at = kinds.index(AgentEvent.COMPACTION_START)
    end_at = kinds.index(AgentEvent.COMPACTION_END)
    first_turn = kinds.index(AgentEvent.TURN_START)
    assert kinds[start_at - 1] is AgentEvent.CONTEXT_UPDATE
    assert frames[start_at][1]["reason"] == "threshold"
    assert AgentEvent.COMPACTION_DELTA in kinds[start_at:end_at]
    assert kinds[end_at + 1] is AgentEvent.CONTEXT_UPDATE
    assert end_at < first_turn
    assert kinds.index(AgentEvent.TOOL_CALL) < kinds.index(AgentEvent.TOOL_RESULT)
    assert compactor.consecutive_failures == 0

    visible = await sessions.get_messages(session.id)
    assert visible[0].summary == 1
    assert visible[0].content.startswith("[Previous conversation summary]")
    assert "folded history" in visible[0].content
    visible_ids = {m.id for m in visible if not m.summary}
    assert visible_ids & old_ids == {recent_id}
    assert recent_id in visible_ids

    declared = {
        call["id"]
        for message in visible
        if message.role == "assistant"
        for call in (message.tool_calls or [])
        if isinstance(call, dict) and call.get("id")
    }
    for message in visible:
        if message.role == "tool":
            assert message.tool_call_id in declared
    assert "call-9" in declared

    turn_request = provider.seen_messages[1]
    assert any("folded history" in m.content for m in turn_request)
    seen_declared: set[str] = set()
    for m in turn_request:
        for call in m.tool_calls or []:
            cid = (
                call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            )
            if cid:
                seen_declared.add(cid)
    for m in turn_request:
        if m.role == "tool":
            assert m.tool_call_id in seen_declared
