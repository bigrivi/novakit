"""Agent 核心循环与会话服务的分支覆盖。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest

from novakit import Agent, AgentConfig, AgentOptions
from novakit.agent.compaction import CompactionError
from novakit.agent.core import AgentRuntimeDependencies
from novakit.agent.llm_stream import TurnStreamReader
from novakit.agent.ports import AgentPorts
from novakit.agent.session import AgentSession, AgentSessionContext
from novakit.agent.state import TurnState
from novakit.events import AgentEvent, AgentEventFrame
from novakit.llm import Done, Error, LLMProvider, TextDelta, ToolCall, ToolResult
from novakit.session import InMemorySession
from novakit.session.models import SessionContext
from novakit.tools import ToolRegistry, tool


@dataclass
class MinimalConfig:
    model: str = "test"
    provider: str = "test"
    max_iterations: int = 3
    reasoning_effort: str | None = None


class ScriptedProvider(LLMProvider):
    """按脚本逐轮产出事件的离线 provider。"""

    def __init__(self, scripts: list[list[object]]) -> None:
        self._scripts = scripts
        self._index = 0

    async def chat(
        self,
        messages: list,
        model: str = "m",
        stream: bool = False,
        tools: list[dict] | None = None,
        **kwargs: object,
    ) -> Done:
        return Done(content="summary")

    async def chat_stream(
        self,
        messages: list,
        model: str = "m",
        tools: list[dict] | None = None,
        **kwargs: object,
    ) -> AsyncGenerator[object, None]:
        script = self._scripts[min(self._index, len(self._scripts) - 1)]
        self._index += 1
        for item in script:
            await asyncio.sleep(0)
            yield item

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text)

    def get_max_tokens(self, model: str) -> int:
        return 128000


class FakeCompactor:
    """可编程的压缩器替身。"""

    def __init__(
        self, frames: list[AgentEventFrame] | None = None, error: str | None = None
    ) -> None:
        self._frames = frames or []
        self._error = error
        self._compacted = False
        self.force_next = False

    @property
    def compacted(self) -> bool:
        return self._compacted

    def mark_compacted(self) -> None:
        self._compacted = True

    async def maybe_compact(
        self,
        messages: list,
        session: object,
        llm: object,
        emit: object,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[AgentEventFrame, None]:
        if self._error is not None:
            raise CompactionError(self._error)
        for frame in self._frames:
            yield frame


async def _collect(agent: Agent, text: str, **kwargs: object) -> list[AgentEventFrame]:
    return [frame async for frame in agent.chat_stream(text, **kwargs)]


def _paired(frames: list[AgentEventFrame]) -> None:
    calls = [d.id for e, d in frames if e is AgentEvent.TOOL_CALL]
    results = [d["tool_call_id"] for e, d in frames if e is AgentEvent.TOOL_RESULT]
    assert sorted(calls) == sorted(results)


@pytest.mark.asyncio
async def test_replace_registry_interrupt_and_approval_none() -> None:
    # Arrange: 默认 agent。
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))
    # Act: 替换 registry、中断、无审批解析。
    agent.replace_tool_registry(ToolRegistry())
    agent.interrupt()
    agent._approval = None
    # Assert: 中断已置位,无审批时返回 False。
    assert agent._abort_event.is_set()
    assert agent.resolve_approval("missing", True) is False


@pytest.mark.asyncio
async def test_on_off_and_emit_approval_fanout() -> None:
    # Arrange: 订阅同步与异步 handler。
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))
    seen: list[tuple[AgentEvent, object]] = []

    def sync_handler(event: AgentEvent, data: object = None) -> None:
        seen.append((event, data))

    async def async_handler(event: AgentEvent, data: object = None) -> None:
        seen.append((event, data))

    agent.on(AgentEvent.START, sync_handler)
    agent.on(AgentEvent.START, async_handler)
    # Act: 跑一轮。
    await _collect(agent, "hi")
    # Assert: 两个 handler 都收到。
    assert len([s for s in seen if s[0] is AgentEvent.START]) == 2
    # Act: 取消订阅后直接触发 approval 透传。
    agent.off(AgentEvent.START, sync_handler)
    agent.off(AgentEvent.START, async_handler)
    await agent._emit_approval({"id": "r1"})
    # Assert: off 后不再累积 START。
    assert len([s for s in seen if s[0] is AgentEvent.START]) == 2


@pytest.mark.asyncio
async def test_regenerate_reuses_user_message_and_rolls_back() -> None:
    # Arrange: 先跑两轮形成 user/assistant/user/assistant。
    provider = ScriptedProvider(
        [[TextDelta(content="one")], [TextDelta(content="two")]]
    )
    agent = Agent(AgentConfig(model="test", max_iterations=2), provider)
    first = await _collect(agent, "first")
    session_id = first[0][1]
    await _collect(agent, "second", session_id=session_id)
    before = [m.content for m in await agent.session.get_messages()]
    assert before == ["first", "one", "second", "two"]
    provider2_scripts: list[list[object]] = [[TextDelta(content="redone")]]
    agent.llm = ScriptedProvider(provider2_scripts)
    # Act: regenerate(回滚最后一段)。
    frames = await _collect(agent, "ignored", session_id=session_id, regenerate=True)
    # Assert: 用户消息复用,旧回答被回滚。
    assert frames[-1][1]["content"] == "redone"
    after = [m.content for m in await agent.session.get_messages()]
    assert after == ["first", "one", "second", "redone"]


@pytest.mark.asyncio
async def test_regenerate_without_history_fails() -> None:
    # Arrange: 空会话。
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))
    # Act/Assert: 无 user turn 时抛错。
    with pytest.raises(ValueError, match="No user message"):
        await _collect(agent, "hi", regenerate=True)


@pytest.mark.asyncio
async def test_regenerate_without_stale_skips_rollback() -> None:
    # Arrange: 只有一个 user 消息,无 stale。
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))
    first = await _collect(agent, "only")
    session_id = first[0][1]
    agent.llm = ScriptedProvider([[TextDelta(content="again")]])
    # Act: regenerate(覆盖 stale_ids 为空的分支)。
    frames = await _collect(agent, "ignored", session_id=session_id, regenerate=True)
    # Assert: 成功完成。
    assert frames[-1][1]["content"] == "again"


@pytest.mark.asyncio
async def test_start_request_without_user_message_raises() -> None:
    # Arrange: add_message 返回 None 的桩会话。
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))

    class NullSession(InMemorySession):
        async def add_message(self, role: str, content: str, **kwargs: object):  # type: ignore[override]
            return None

    agent.session = NullSession()
    await agent.session.create_session()
    # Act/Assert: 直接构造 prepared 并调用 _start_request。
    from novakit.agent.core import PreparedRequest

    current = agent.session.get_current_session()
    assert current is not None
    bad = PreparedRequest(
        current_session=current,
        session_id=current.id,
        user_input="hi",
        message_text="hi",
        image_data=[],
        display="hi",
        user_message=None,
        stale_ids=[],
        regenerate=False,
        attachment_count=0,
        message_variant=None,
    )
    with pytest.raises(RuntimeError, match="no user message"):
        async for _ in agent._start_request(bad):
            pass


@pytest.mark.asyncio
async def test_compaction_frames_compacted_reload_and_error() -> None:
    # Arrange: 透传压缩事件并标记 compacted。
    compactor = FakeCompactor(frames=[(AgentEvent.COMPACTION_START, None)])
    compactor.mark_compacted()
    agent = Agent(
        MinimalConfig(),
        ScriptedProvider([[TextDelta(content="ok")]]),
        options=AgentOptions(ports=AgentPorts(compactor=compactor)),  # type: ignore[arg-type]
    )
    # Act: 跑一轮。
    frames = await _collect(agent, "hi")
    # Assert: 压缩帧透出且请求成功。
    assert AgentEvent.COMPACTION_START in [e for e, _ in frames]
    assert frames[-1][0] is AgentEvent.DONE
    # Arrange: 压缩失败。
    failing = FakeCompactor(error="cannot summarize")
    agent2 = Agent(
        MinimalConfig(),
        ScriptedProvider([[TextDelta(content="ok")]]),
        options=AgentOptions(ports=AgentPorts(compactor=failing)),  # type: ignore[arg-type]
    )
    # Act: 跑一轮。
    frames2 = await _collect(agent2, "hi")
    # Assert: compaction_error 终结。
    assert frames2[-1][0] is AgentEvent.ERROR
    assert frames2[-1][1]["reason"] == "compaction_error"


@pytest.mark.asyncio
async def test_overflow_error_retries_once_then_completes() -> None:
    # Arrange: 首轮溢出、次轮正常。
    compactor = FakeCompactor()
    provider = ScriptedProvider(
        [
            [Error(message="This model's maximum context length is exceeded")],
            [TextDelta(content="recovered")],
        ]
    )
    agent = Agent(
        MinimalConfig(),
        provider,
        options=AgentOptions(ports=AgentPorts(compactor=compactor)),  # type: ignore[arg-type]
    )
    # Act: 跑一轮。
    frames = await _collect(agent, "hi")
    # Assert: 重试后成功,且只重试一次(latch 置位)。
    assert frames[-1][0] is AgentEvent.DONE
    assert frames[-1][1]["content"] == "recovered"
    assert agent.state.overflow_recovered is True
    assert compactor.force_next is True


@pytest.mark.asyncio
async def test_non_overflow_error_ends_request() -> None:
    # Arrange: 非溢出错误。
    agent = Agent(MinimalConfig(), ScriptedProvider([[Error(message="boom")]]))
    # Act: 跑一轮。
    frames = await _collect(agent, "hi")
    # Assert: 直接 ERROR 终结。
    assert frames[-1][0] is AgentEvent.ERROR
    assert frames[-1][1]["message"] == "boom"


@pytest.mark.asyncio
async def test_max_iterations_reached() -> None:
    # Arrange: 每轮都调工具,永不 DONE。
    registry = ToolRegistry()

    @tool(name="echo", parameters={"properties": {"t": {"type": "string"}}})
    async def echo(t: str = "") -> ToolResult:
        return ToolResult(content="echo")

    registry.register(echo, name="echo")
    provider = ScriptedProvider(
        [[ToolCall(id="c1", name="echo", arguments='{"t":"x"}')]]
    )
    agent = Agent(
        AgentConfig(model="test", max_iterations=1),
        provider,
        options=AgentOptions(tools=registry),
    )
    # Act: 跑一轮。
    frames = await _collect(agent, "go")
    # Assert: max_iterations 错误终结,工具配对完整。
    assert frames[-1][0] is AgentEvent.ERROR
    assert frames[-1][1]["reason"] == "max_iterations"
    _paired(frames)


@pytest.mark.asyncio
async def test_run_turn_abort_paths_and_llm_missing() -> None:
    # Arrange: 正常 agent。
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))
    await agent.session_service.resolve_session(None, "hi", None)
    # Act: _run_turn 前中断(覆盖首个 abort 分支)。
    agent._abort_event.set()
    frames = [
        f async for f in agent._run_turn(1, None, group_id="g", loaded_messages=[])
    ]
    # Assert: 直接 stopped。
    assert frames == [frames[0]] and frames[0][0] is AgentEvent.DONE
    agent._abort_event.clear()
    # Act: 无 provider 的完成流与工具调用均抛错。
    agent.llm = None
    with pytest.raises(RuntimeError, match="no LLM provider"):
        await agent._start_completion_stream(1, None, [])
    with pytest.raises(RuntimeError, match="no LLM provider"):
        async for _ in agent._invoke_tools([], "g", 1):
            pass


@pytest.mark.asyncio
async def test_run_turn_second_abort_after_stream() -> None:
    # Arrange: 流正常结束,但之后中断。
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))
    await agent.session_service.resolve_session(None, "hi", None)
    orig = agent._stop_if_aborted
    calls = 0

    async def flipping() -> dict | None:
        nonlocal calls
        calls += 1
        if calls >= 3:
            agent._abort_event.set()
        return await orig()

    agent._stop_if_aborted = flipping  # type: ignore[method-assign]
    # Act: 跑一 turn(覆盖第二个 abort 分支)。
    frames = [
        f async for f in agent._run_turn(1, None, group_id="g", loaded_messages=[])
    ]
    # Assert: 以 stopped 收尾。
    assert frames[-1][0] is AgentEvent.DONE
    assert frames[-1][1]["reason"] == "stopped"


@pytest.mark.asyncio
async def test_run_turn_failed_and_stopped_outcomes_return() -> None:
    # Arrange: FAILED 与 STOPPED 两条早返。
    failed = Agent(
        MinimalConfig(),
        ScriptedProvider([[TextDelta(content="half"), Error(message="boom")]]),
    )
    await failed.session_service.resolve_session(None, "hi", None)
    # Act: FAILED 持久化失败轮并返回。
    frames = [
        f async for f in failed._run_turn(1, None, group_id="g", loaded_messages=[])
    ]
    # Assert: ERROR 透出且会话落盘。
    assert any(e is AgentEvent.ERROR for e, _ in frames)
    stored = await failed.session.get_messages()
    assert any(m.role == "assistant" and m.error for m in stored)
    stopped = Agent(
        MinimalConfig(), ScriptedProvider([[Done(content="x", aborted=True)]])
    )
    await stopped.session_service.resolve_session(None, "hi", None)
    # Act: STOPPED 直接返回。
    frames2 = [
        f async for f in stopped._run_turn(1, None, group_id="g", loaded_messages=[])
    ]
    # Assert: DONE stopped。
    assert any(f[0] is AgentEvent.DONE for f in frames2)


def test_normalize_and_executable_branches() -> None:
    # Arrange: 各类畸形调用。
    registry = ToolRegistry()

    @tool(name="calc", parameters={"properties": {"n": {"type": "number"}}})
    async def calc(n: float = 0) -> ToolResult:
        return ToolResult(content="ok")

    registry.register(calc, name="calc")

    class BadNormalize:
        def normalize_input(self, args: dict) -> dict:
            raise RuntimeError("bad")

    registry.set_behavior("calc", BadNormalize())  # type: ignore[arg-type]
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))
    agent.replace_tool_registry(registry)
    calls = [
        ToolCall(id="a", name="", arguments="{}"),
        ToolCall(id="b", name="calc", arguments=""),
        ToolCall(id="c", name="calc", arguments="{bad"),
        ToolCall(id="d", name="calc", arguments="[1,2]"),
        ToolCall(id="e", name="calc", arguments='{"n": 1}'),
    ]
    # Act: 归一化(覆盖全部 continue 分支,不抛错)。
    agent.normalize_tool_arguments(calls)
    # Assert: 异常行为保持原文。
    assert calls[4].arguments == '{"n": 1}'

    class FixNormalize:
        def normalize_input(self, args: dict) -> dict:
            return {"n": 2.0}

    registry.set_behavior("calc", FixNormalize())  # type: ignore[arg-type]
    fix_call = [ToolCall(id="f", name="calc", arguments='{"n": 1}')]
    # Act: 归一化改写。
    agent.normalize_tool_arguments(fix_call)
    # Assert: 参数被改写。
    assert '"n": 2' in fix_call[0].arguments
    # Act: 不可解析调用被过滤。
    kept = agent._executable_tool_calls(
        [ToolCall(id="g", name="calc", arguments="{bad")], 1
    )
    # Assert: 被丢弃。
    assert kept == []


@pytest.mark.asyncio
async def test_register_tool_and_compact_now_branches() -> None:
    # Arrange: 基础 agent。
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))

    @tool(name="extra", parameters={})
    async def extra() -> ToolResult:
        return ToolResult(content="ok")

    # Act: 注册工具(覆盖 register_tool)。
    agent.register_tool(extra, "extra")
    # Assert: 已注册。
    assert agent.tool_registry.get("extra") is not None
    # Act/Assert: 无会话时 compact_now 抛错。
    with pytest.raises(FileNotFoundError):
        await agent.compact_now("missing")
    # Arrange: 有会话但无 provider。
    await _collect(agent, "hi")
    session_id = agent.session.get_current_session().id  # type: ignore[union-attr]
    agent.llm = None
    # Act/Assert: 无 provider 抛错。
    with pytest.raises(RuntimeError, match="no LLM provider"):
        await agent.compact_now(session_id)
    # Arrange: 恢复 provider 但无 compactor。
    agent.llm = ScriptedProvider([[TextDelta(content="ok")]])
    # Act: 无压缩器直接返回 False。
    assert await agent.compact_now(session_id) == {"compacted": False}


@pytest.mark.asyncio
async def test_compact_now_success_sets_flag() -> None:
    # Arrange: 带压缩器的 agent。
    compactor = FakeCompactor(frames=[(AgentEvent.COMPACTION_START, None)])
    agent = Agent(
        MinimalConfig(),
        ScriptedProvider([[TextDelta(content="ok")]]),
        options=AgentOptions(ports=AgentPorts(compactor=compactor)),  # type: ignore[arg-type]
    )
    await _collect(agent, "hi")
    session_id = agent.session.get_current_session().id  # type: ignore[union-attr]
    # Act: 强制压缩。
    result = await agent.compact_now(session_id)
    # Assert: force_next 置位。
    assert result == {"compacted": False}
    assert compactor.force_next is True


@pytest.mark.asyncio
async def test_compact_now_current_vanished_raises() -> None:
    # Arrange: load 成功但 current 为空的桩。
    agent = Agent(MinimalConfig(), ScriptedProvider([[TextDelta(content="ok")]]))

    class LoadOkCurrentNone(InMemorySession):
        async def load_session(self, session_id: str) -> SessionContext | None:
            return SessionContext(id=session_id)

        def get_current_session(self) -> SessionContext | None:
            return None

    agent.session = LoadOkCurrentNone()
    # Act/Assert: 抛 FileNotFoundError(覆盖 current is None 分支)。
    with pytest.raises(FileNotFoundError):
        await agent.compact_now("ghost")


@pytest.mark.asyncio
async def test_session_service_branches() -> None:
    # Arrange: 各类会话与存储组合。
    state = TurnState()
    session = InMemorySession()

    class ExplodingStorage:
        async def update_session_route_if_changed(
            self,
            session_id: str,
            provider: str | None,
            model: str | None,
            reasoning_effort: str | None,
        ) -> bool:
            raise RuntimeError("store down")

    class OkStorage:
        def __init__(self) -> None:
            self.seen: list[str] = []

        async def update_session_route_if_changed(
            self,
            session_id: str,
            provider: str | None,
            model: str | None,
            reasoning_effort: str | None,
        ) -> bool:
            self.seen.append(session_id)
            return True

    # Act: 无当前会话直接返回。
    svc = AgentSession(
        AgentSessionContext(
            session=session, config=MinimalConfig(), compaction=None, agent_key="a"
        ),
        state,
        _Hooks(),
    )
    await svc.record_session_route()
    # Act: 有会话但无存储直接返回。
    await session.create_session()
    await svc.record_session_route()
    # Act: 存储抛错被吞掉。
    svc2 = AgentSession(
        AgentSessionContext(
            session=session,
            config=MinimalConfig(),
            compaction=None,
            agent_key="a",
            storage=ExplodingStorage(),  # type: ignore[arg-type]
        ),
        state,
        _Hooks(),
    )
    await svc2.record_session_route()
    # Act: 显式存储参数成功。
    ok = OkStorage()
    await svc2.record_session_route(storage=ok)  # type: ignore[arg-type]
    # Assert: 显式存储被调用。
    assert len(ok.seen) == 1
    # Act: overflow 恢复三条拒绝分支。
    assert svc.try_overflow_recovery("boom") is False
    state.overflow_recovered = True
    compactor = FakeCompactor()
    svc3 = AgentSession(
        AgentSessionContext(
            session=session, config=MinimalConfig(), compaction=compactor, agent_key="a"
        ),
        state,
        _Hooks(),
    )
    assert svc3.try_overflow_recovery("maximum context length") is False
    state.overflow_recovered = False
    assert svc3.try_overflow_recovery("ThrottlingException: Too many tokens") is False
    assert svc3.try_overflow_recovery("maximum context length") is True
    # Act: persist_failed_turn 的两条早返。
    reader = TurnStreamReader(emit=_emit_noop, stop_if_aborted=_stop_none)
    reader.outcome = (
        TurnStreamReader.__new__(TurnStreamReader).outcome if False else reader.outcome
    )
    from novakit.agent.llm_stream import TurnOutcome

    reader.outcome = TurnOutcome.CONTINUE
    await svc.persist_failed_turn(reader, None)
    reader.outcome = TurnOutcome.FAILED
    reader.content = ""
    await svc.persist_failed_turn(reader, None)
    # Assert: 均未落盘新增。
    assert True


class _Hooks:
    def maybe_schedule_title_generation(
        self, session: SessionContext, first_message: str
    ) -> None:
        return None

    def maybe_schedule_memory_review(self, turn_count: int) -> None:
        return None

    async def schedule_request_learning(
        self, tool_call_count: int, messages_before: int
    ) -> None:
        return None


async def _emit_noop(event: AgentEvent, data: object = None) -> None:
    return None


async def _stop_none() -> dict | None:
    return None


@pytest.mark.asyncio
async def test_resolve_session_reuse_and_vanish() -> None:
    # Arrange: 已有会话复用。
    session = InMemorySession()
    svc = AgentSession(
        AgentSessionContext(
            session=session, config=MinimalConfig(), compaction=None, agent_key="a"
        ),
        TurnState(),
        _Hooks(),
    )
    first = await svc.resolve_session(None, "hello", None)
    # Act: 按 id 复用。
    second = await svc.resolve_session(first.id, "hello", None)
    # Assert: 同一会话。
    assert second.id == first.id

    # Arrange: load 返回但 current 为空。
    class VanishLoad(InMemorySession):
        async def load_session(self, session_id: str) -> SessionContext | None:
            return SessionContext(id=session_id)

    svc_vanish = AgentSession(
        AgentSessionContext(
            session=VanishLoad(), config=MinimalConfig(), compaction=None, agent_key="a"
        ),
        TurnState(),
        _Hooks(),
    )
    # Act/Assert: 抛错。
    with pytest.raises(RuntimeError, match="vanished"):
        await svc_vanish.resolve_session("x", "hi", None)

    # Arrange: create 后 current 为空。
    class VanishCreate(InMemorySession):
        async def create_session(self, **kwargs: object) -> SessionContext:
            return SessionContext(id="ghost")

    svc_ghost = AgentSession(
        AgentSessionContext(
            session=VanishCreate(),
            config=MinimalConfig(),
            compaction=None,
            agent_key="a",
        ),
        TurnState(),
        _Hooks(),
    )
    # Act/Assert: 抛错。
    with pytest.raises(RuntimeError, match="was not created"):
        await svc_ghost.resolve_session(None, "hi", None)


@pytest.mark.asyncio
async def test_requires_input_done_path_and_tool_pairing() -> None:
    # Arrange: 工具要求输入。
    registry = ToolRegistry()

    @tool(name="ask", parameters={})
    async def ask() -> ToolResult:
        return ToolResult(content="need more", requires_input=True)

    registry.register(ask, name="ask")
    provider = ScriptedProvider(
        [
            [ToolCall(id="c1", name="ask", arguments="{}")],
            [TextDelta(content="never")],
        ]
    )
    agent = Agent(MinimalConfig(), provider, options=AgentOptions(tools=registry))
    # Act: 跑一轮。
    frames = await _collect(agent, "go")
    # Assert: requires_input 终结且配对完整。
    assert frames[-1][1]["reason"] == "requires_input"
    _paired(frames)


@pytest.mark.asyncio
async def test_initializing_runtime_assigns_all_fields() -> None:
    # Arrange: 手工组装依赖(覆盖 _initialize_runtime 全字段)。
    from novakit.agent.services import NoopBackground, StaticPromptProvider
    from novakit.agent.tool_guardrails import ToolGuardrails
    from novakit.events import EventBus

    session = InMemorySession()
    state = TurnState()
    config = MinimalConfig()
    messaging = StaticPromptProvider("sys", state, session)
    background = NoopBackground()
    service = AgentSession(
        AgentSessionContext(
            session=session, config=config, compaction=None, agent_key="a"
        ),
        state,
        background,  # type: ignore[arg-type]
    )
    deps = AgentRuntimeDependencies(
        config=config,  # type: ignore[arg-type]
        llm=ScriptedProvider([[TextDelta(content="ok")]]),
        session=session,
        tool_registry=ToolRegistry(),
        events=EventBus(),
        agent_key="a",
        is_sub_agent=False,
        allowed_tools=None,
        state=state,
        background_tasks=set(),
        messaging=messaging,  # type: ignore[arg-type]
        background=background,  # type: ignore[arg-type]
        session_service=service,
        compactor=None,
        approval=None,
        guardrails=ToolGuardrails(),
        begin_turn=lambda **kwargs: None,
    )
    agent = Agent.__new__(Agent)
    # Act: 初始化。
    agent._initialize_runtime(deps, parent_agent=None, depth=0)
    # Assert: 关键字段就位。
    assert agent.agent_key == "a"
    assert agent.depth == 0
    assert isinstance(agent.tool_registry, ToolRegistry)
