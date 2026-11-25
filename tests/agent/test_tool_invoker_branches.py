"""ToolInvoker 与 guardrails 的分支覆盖。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from novakit.agent.tool_guardrails import GuardrailAction, ToolGuardrails
from novakit.agent.tool_invoker import (
    CANCELLED_TOOL_CONTENT,
    REJECTED_TOOL_CONTENT,
    ToolInvoker,
    ToolOutcome,
    _downscale_all,
    has_parsable_arguments,
    parse_tool_arguments,
)
from novakit.events import AgentEvent, AgentEventFrame
from novakit.llm import FakerLLMProvider, ToolCall, ToolResult
from novakit.session import InMemorySession
from novakit.tools import PreExecutionCheck, ToolRegistry
from novakit.tools.approval import InMemoryApprovalManager
from novakit.tools.behavior import DefaultToolBehavior


def _session_with_current() -> InMemorySession:
    # Arrange: 带当前会话的内存会话(同步构造后由调用方 create)。
    return InMemorySession()


async def _make_invoker(
    registry: ToolRegistry,
    session: InMemorySession,
    approval: InMemoryApprovalManager | None = None,
    guardrails: ToolGuardrails | None = None,
    abort_event: asyncio.Event | None = None,
    llm: FakerLLMProvider | None = None,
) -> tuple[ToolInvoker, list[AgentEventFrame], asyncio.Event]:
    # Arrange: 标准 invoker 与事件收集器。
    abort = abort_event or asyncio.Event()
    emitted: list[AgentEventFrame] = []

    async def emit(event: AgentEvent, data: object = None) -> None:
        emitted.append((event, data))

    async def stop() -> dict | None:
        if abort.is_set():
            return {"reason": "stopped", "content": "Stopped by user"}
        return None

    invoker = ToolInvoker(
        registry=registry,
        session=session,
        approval=approval,
        guardrails=guardrails,
        llm=llm or FakerLLMProvider(seed=1),
        model="fake",
        provider="test",
        abort_event=abort,
        emit=emit,
        emit_approval=emit,
        stop_if_aborted=stop,
    )
    return invoker, emitted, abort


def _paired(frames: list[AgentEventFrame]) -> None:
    # Assert: 每个 TOOL_CALL 都有终态 TOOL_RESULT。
    calls = [d.id for e, d in frames if e is AgentEvent.TOOL_CALL]
    results = [d["tool_call_id"] for e, d in frames if e is AgentEvent.TOOL_RESULT]
    assert sorted(calls) == sorted(results)


def _register(
    registry: ToolRegistry, name: str, func: object, behavior: object = None
) -> None:
    registry.register(func, name)  # type: ignore[arg-type]
    if behavior is not None:
        registry.set_behavior(name, behavior)  # type: ignore[arg-type]


class _Reject(DefaultToolBehavior):
    async def before_execute(self, args: dict, ctx: object) -> PreExecutionCheck:  # type: ignore[override]
        return PreExecutionCheck(allowed=False, reject_reason="nope")


class _ImageBehavior(DefaultToolBehavior):
    def postprocess(self, raw_content: str) -> tuple[str, list | None]:
        return raw_content, ["dGVzdA=="]


class _HeartbeatApprove:
    def __init__(self, manager: InMemoryApprovalManager) -> None:
        self._manager = manager

    def pre_request(self, *args: object, **kwargs: object) -> str:
        return self._manager.pre_request(*args, **kwargs)  # type: ignore[arg-type]

    def resolve(self, *args: object, **kwargs: object) -> bool:
        return self._manager.resolve(*args, **kwargs)  # type: ignore[arg-type]

    async def wait_with_heartbeat(
        self,
        req_id: str,
        heartbeat_interval: int = 15,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[bool | None, None]:
        yield None
        yield True


class _DenyApprove:
    def __init__(self, manager: InMemoryApprovalManager) -> None:
        self._manager = manager

    def pre_request(self, *args: object, **kwargs: object) -> str:
        return self._manager.pre_request(*args, **kwargs)  # type: ignore[arg-type]

    def resolve(self, *args: object, **kwargs: object) -> bool:
        return self._manager.resolve(*args, **kwargs)  # type: ignore[arg-type]

    async def wait_with_heartbeat(
        self,
        req_id: str,
        heartbeat_interval: int = 15,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncGenerator[bool | None, None]:
        yield False


class _NeedsApproval(DefaultToolBehavior):
    async def before_execute(self, args: dict, ctx: object) -> PreExecutionCheck:  # type: ignore[override]
        return PreExecutionCheck(approval_request={"id": "req-x"})


def test_guardrail_failure_halt_and_reset() -> None:
    # Arrange: 连续失败阈值为 2。
    guardrails = ToolGuardrails(max_same_failure=2)
    # Act: 两次失败后应 HALT。
    assert guardrails.observe("shell", {"c": "x"}, False) is GuardrailAction.ALLOW
    assert guardrails.observe("shell", {"c": "x"}, False) is GuardrailAction.HALT
    # Act: 成功清掉失败计数。
    guardrails.reset()
    assert guardrails.observe("shell", {"c": "x"}, False) is GuardrailAction.ALLOW
    assert guardrails.observe("shell", {"c": "x"}, True) is GuardrailAction.ALLOW
    assert guardrails.observe("shell", {"c": "x"}, False) is GuardrailAction.ALLOW


def test_guardrail_write_resets_no_write_window() -> None:
    # Arrange: 读窗口为 1。
    guardrails = ToolGuardrails(max_no_write_reads=1)
    # Act: 写操作后读计数清零。
    assert guardrails.observe("read", {"p": "a"}, True) is GuardrailAction.ALLOW
    assert guardrails.observe("write", {"p": "a"}, True) is GuardrailAction.ALLOW
    assert guardrails.observe("read", {"p": "b"}, True) is GuardrailAction.ALLOW
    assert guardrails.observe("read", {"p": "c"}, True) is GuardrailAction.WARN


def test_parse_and_predicate_branches() -> None:
    # Arrange: 三种参数形态。
    # Act/Assert: dict 透传、非法 JSON 抛错、谓词分支。
    assert parse_tool_arguments({"a": 1}) == {"a": 1}
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_tool_arguments("{bad")
    assert (
        has_parsable_arguments(ToolCall(id="1", name="t", arguments={"a": 1})) is True
    )
    assert (
        has_parsable_arguments(ToolCall(id="1", name="t", arguments='{"a":1}')) is True
    )
    assert has_parsable_arguments(ToolCall(id="1", name="t", arguments="{bad")) is False
    assert _downscale_all(["dGVzdA=="]) == ["dGVzdA=="]


@pytest.mark.asyncio
async def test_precheck_rejection_keeps_pairing() -> None:
    # Arrange: before_execute 直接拒绝。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def never() -> ToolResult:
        raise AssertionError("must not run")

    _register(registry, "blocked", never, _Reject())
    invoker, _, _ = await _make_invoker(registry, session)
    # Act: 执行。
    frames = [
        f
        async for f in invoker.run([ToolCall(id="c1", name="blocked", arguments="{}")])
    ]
    # Assert: 拒绝以 tool 结果终结,配对完整。
    assert [e for e, _ in frames] == [AgentEvent.TOOL_CALL, AgentEvent.TOOL_RESULT]
    assert frames[-1][1]["result"].denied is True
    assert "nope" in frames[-1][1]["result"].content
    _paired(frames)


@pytest.mark.asyncio
async def test_approval_heartbeat_then_allowed() -> None:
    # Arrange: 先心跳再放行。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def ok() -> ToolResult:
        return ToolResult(content="ran")

    _register(registry, "g", ok, _NeedsApproval())
    manager = InMemoryApprovalManager()
    invoker, _, _ = await _make_invoker(
        registry, session, approval=_HeartbeatApprove(manager)
    )  # type: ignore[arg-type]
    # Act: 执行。
    frames = [
        f async for f in invoker.run([ToolCall(id="c1", name="g", arguments="{}")])
    ]
    # Assert: 心跳透出且最终成功,配对完整。
    assert AgentEvent.APPROVAL_HEARTBEAT in [e for e, _ in frames]
    assert frames[-1][1]["result"].content == "ran"
    _paired(frames)


@pytest.mark.asyncio
async def test_approval_denied_keeps_pairing() -> None:
    # Arrange: 审批拒绝。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def ok() -> ToolResult:
        raise AssertionError("must not run")

    _register(registry, "g", ok, _NeedsApproval())
    manager = InMemoryApprovalManager()
    invoker, _, _ = await _make_invoker(
        registry, session, approval=_DenyApprove(manager)
    )  # type: ignore[arg-type]
    # Act: 执行。
    frames = [
        f async for f in invoker.run([ToolCall(id="c1", name="g", arguments="{}")])
    ]
    # Assert: 拒绝内容提示换路,配对完整。
    kinds = [e for e, _ in frames]
    assert kinds == [
        AgentEvent.TOOL_CALL,
        AgentEvent.APPROVAL_REQUIRED,
        AgentEvent.APPROVAL_RESULT,
        AgentEvent.TOOL_RESULT,
    ]
    assert REJECTED_TOOL_CONTENT in frames[-1][1]["result"].content
    assert frames[-1][1]["result"].denied is True
    _paired(frames)


@pytest.mark.asyncio
async def test_tool_failure_and_requires_input_and_unknown() -> None:
    # Arrange: 失败、需输入、未知工具三个调用顺序执行。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def fail() -> ToolResult:
        return ToolResult(success=False, content="bad", error="bad")

    async def need() -> ToolResult:
        return ToolResult(content="ask", requires_input=True)

    _register(registry, "fail", fail)
    _register(registry, "need", need)
    invoker, _, _ = await _make_invoker(registry, session)
    calls = [
        ToolCall(id="c1", name="fail", arguments="{}"),
        ToolCall(id="c2", name="need", arguments="{}"),
        ToolCall(id="c3", name="ghost", arguments="{}"),
    ]
    # Act: 执行(遇到 requires_input 即停,第三个不跑)。
    frames = [f async for f in invoker.run(calls)]
    # Assert: 失败继续、需输入以 DONE 终结。
    assert invoker.outcome is ToolOutcome.NEEDS_INPUT
    assert frames[-1][0] is AgentEvent.DONE
    assert frames[-1][1]["reason"] == "requires_input"
    _paired(
        [f for f in frames if f[0] in (AgentEvent.TOOL_CALL, AgentEvent.TOOL_RESULT)]
    )
    # Arrange: 未知工具单独跑。
    invoker2, _, _ = await _make_invoker(registry, session)
    frames2 = [
        f async for f in invoker2.run([ToolCall(id="u1", name="ghost", arguments="{}")])
    ]
    # Assert: 未知工具转为失败结果。
    assert "Unknown tool" in frames2[-1][1]["result"].content
    _paired(frames2)


@pytest.mark.asyncio
async def test_tool_images_postprocess_downscales() -> None:
    # Arrange: postprocess 返回图片。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def shot() -> ToolResult:
        return ToolResult(content="pic")

    _register(registry, "shot", shot, _ImageBehavior())
    invoker, _, _ = await _make_invoker(registry, session)
    # Act: 执行。
    frames = [
        f async for f in invoker.run([ToolCall(id="c1", name="shot", arguments="{}")])
    ]
    # Assert: tool 消息带图,配对完整。
    assert frames[-1][1]["result"].content == "pic"
    stored = await session.get_messages()
    assert stored[-1].images == ["dGVzdA=="]
    _paired(frames)


@pytest.mark.asyncio
async def test_tool_exception_becomes_failed_result() -> None:
    # Arrange: 工具抛错。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def boom() -> ToolResult:
        raise RuntimeError("kaboom")

    _register(registry, "boom", boom)
    invoker, _, _ = await _make_invoker(registry, session)
    # Act: 执行。
    frames = [
        f async for f in invoker.run([ToolCall(id="c1", name="boom", arguments="{}")])
    ]
    # Assert: 转为失败结果,配对完整。
    assert frames[-1][1]["result"].success is False
    assert "kaboom" in frames[-1][1]["result"].content
    _paired(frames)


@pytest.mark.asyncio
async def test_guardrail_halt_rewrites_result() -> None:
    # Arrange: max_same_call=0 使首次成功即 HALT。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def ok() -> ToolResult:
        return ToolResult(content="ran")

    _register(registry, "loop", ok)
    invoker, _, _ = await _make_invoker(
        registry, session, guardrails=ToolGuardrails(max_same_call=0)
    )
    # Act: 执行。
    frames = [
        f async for f in invoker.run([ToolCall(id="c1", name="loop", arguments="{}")])
    ]
    # Assert: 被改写为换路提示。
    assert "repeating" in frames[-1][1]["result"].content
    assert frames[-1][1]["result"].success is False
    _paired(frames)


@pytest.mark.asyncio
async def test_implicit_arguments_injection() -> None:
    # Arrange: 需要 turn_context 与 session_id 的工具。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()
    seen: dict[str, object] = {}

    async def with_ctx(turn_context: object = None) -> ToolResult:
        seen["ctx"] = turn_context
        return ToolResult(content="ok")

    async def with_sid(session_id: str = "") -> ToolResult:
        seen["sid"] = session_id
        return ToolResult(content="ok")

    _register(registry, "with_ctx", with_ctx)
    _register(registry, "with_sid", with_sid)
    invoker, _, _ = await _make_invoker(registry, session)
    # Act: 执行两个调用。
    frames = [
        f
        async for f in invoker.run(
            [
                ToolCall(id="c1", name="with_ctx", arguments="{}"),
                ToolCall(id="c2", name="with_sid", arguments="{}"),
            ]
        )
    ]
    # Assert: 隐式参数已注入,配对完整。
    assert seen["ctx"] is not None
    assert seen["sid"] == session.get_current_session().id  # type: ignore[union-attr]
    _paired(frames)


@pytest.mark.asyncio
async def test_abort_before_announce_cancels_pairing() -> None:
    # Arrange: 提前中断。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def ok() -> ToolResult:
        return ToolResult(content="ran")

    _register(registry, "ok", ok)
    abort = asyncio.Event()
    abort.set()
    invoker, _, _ = await _make_invoker(registry, session, abort_event=abort)
    calls = [ToolCall(id="c1", name="ok", arguments="{}")]
    # Act: 执行。
    frames = [f async for f in invoker.run(calls)]
    # Assert: 中断以取消结果终结,配对完整。
    assert invoker.outcome is ToolOutcome.STOPPED
    assert frames[-1][0] is AgentEvent.DONE
    results = [d for e, d in frames if e is AgentEvent.TOOL_RESULT]
    assert len(results) == 1
    assert results[0]["result"].content == CANCELLED_TOOL_CONTENT


@pytest.mark.asyncio
async def test_abort_after_announce_cancels_pairing() -> None:
    # Arrange: emit 后再中断(覆盖第二处 stop 检查)。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def ok() -> ToolResult:
        return ToolResult(content="ran")

    _register(registry, "ok", ok)
    abort = asyncio.Event()
    invoker, _, _ = await _make_invoker(registry, session, abort_event=abort)
    orig_stop = invoker._stop_if_aborted
    calls = 0

    async def flipping() -> dict | None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            abort.set()
        return await orig_stop()

    invoker._stop_if_aborted = flipping  # type: ignore[method-assign]
    # Act: 执行。
    frames = [
        f async for f in invoker.run([ToolCall(id="c1", name="ok", arguments="{}")])
    ]
    # Assert: 配对完整且停机。
    assert invoker.outcome is ToolOutcome.STOPPED
    _paired(
        [f for f in frames if f[0] in (AgentEvent.TOOL_CALL, AgentEvent.TOOL_RESULT)]
    )


@pytest.mark.asyncio
async def test_execute_with_abort_tool_wins_and_abort_wins() -> None:
    # Arrange: 快工具与慢工具。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def fast() -> ToolResult:
        return ToolResult(content="fast")

    _register(registry, "fast", fast)
    invoker, _, _ = await _make_invoker(registry, session)
    # Act: 正常执行走 tool_task 分支。
    result = await invoker.execute_with_abort(
        ToolCall(id="c1", name="fast", arguments="{}"), {}
    )
    # Assert: 成功返回。
    assert result is not None and result.content == "fast"

    # Arrange: 慢工具 + 已中断。
    async def slow() -> ToolResult:
        await asyncio.sleep(5)
        return ToolResult(content="slow")

    _register(registry, "slow", slow)
    abort = asyncio.Event()
    abort.set()
    invoker2, _, _ = await _make_invoker(registry, session, abort_event=abort)
    # Act: 中断抢先。
    result2 = await invoker2.execute_with_abort(
        ToolCall(id="c2", name="slow", arguments="{}"), {}
    )
    # Assert: 返回 None。
    assert result2 is None


@pytest.mark.asyncio
async def test_execute_with_abort_exception_maps_to_failure() -> None:
    # Arrange: 直接 execute 抛错的打桩(覆盖 result.exception 分支)。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()
    invoker, _, _ = await _make_invoker(registry, session)

    async def broken(
        tool_call: ToolCall, arguments: dict, approval_request_id: str = ""
    ) -> ToolResult:
        raise RuntimeError("nope")

    invoker.execute = broken  # type: ignore[method-assign]
    # Act: 走异常映射分支。
    result = await invoker.execute_with_abort(
        ToolCall(id="c1", name="x", arguments="{}"), {}
    )
    # Assert: 失败结果。
    assert result is not None and result.success is False


@pytest.mark.asyncio
async def test_persist_cancelled_skips_executed() -> None:
    # Arrange: 第一个已执行,第二个待取消。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()
    invoker, _, _ = await _make_invoker(registry, session)
    invoker._executed_ids.add("c1")
    calls = [
        ToolCall(id="c1", name="a", arguments="{}"),
        ToolCall(id="c2", name="b", arguments="{}"),
    ]
    # Act: 持久化取消。
    frames = [f async for f in invoker.persist_cancelled(calls, group_id="g")]
    # Assert: 只为 c2 产出终态。
    assert len(frames) == 1
    assert frames[0][1]["tool_call_id"] == "c2"


@pytest.mark.asyncio
async def test_run_without_approval_manager_refuses() -> None:
    # Arrange: 需要审批但无 manager(覆盖 approval is None 分支)。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def ok() -> ToolResult:
        raise AssertionError("must not run")

    _register(registry, "g", ok, _NeedsApproval())
    invoker, _, _ = await _make_invoker(registry, session, approval=None)
    # Act: 执行。
    frames = [
        f async for f in invoker.run([ToolCall(id="c1", name="g", arguments="{}")])
    ]
    # Assert: 拒绝并配对完整。
    assert [e for e, _ in frames] == [
        AgentEvent.TOOL_CALL,
        AgentEvent.APPROVAL_REQUIRED,
        AgentEvent.APPROVAL_RESULT,
        AgentEvent.TOOL_RESULT,
    ]
    _paired(frames)


@pytest.mark.asyncio
async def test_run_execute_none_persists_cancelled() -> None:
    # Arrange: execute_with_abort 直接返回 None(覆盖 run 内取消分支)。
    session = _session_with_current()
    await session.create_session()
    registry = ToolRegistry()

    async def ok() -> ToolResult:
        return ToolResult(content="ran")

    _register(registry, "ok", ok)
    invoker, _, _ = await _make_invoker(registry, session)

    async def none_abort(
        tool_call: ToolCall, arguments: dict, approval_request_id: str = ""
    ) -> ToolResult | None:
        return None

    invoker.execute_with_abort = none_abort  # type: ignore[method-assign]
    # Act: 执行。
    frames = [
        f async for f in invoker.run([ToolCall(id="c1", name="ok", arguments="{}")])
    ]
    # Assert: 中断终结且配对完整。
    assert invoker.outcome is ToolOutcome.STOPPED
    assert frames[-1][0] is AgentEvent.DONE
    _paired(
        [f for f in frames if f[0] in (AgentEvent.TOOL_CALL, AgentEvent.TOOL_RESULT)]
    )
