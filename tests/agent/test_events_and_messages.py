"""EventBus 与消息纯函数的分支覆盖。"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator

import pytest

from novakit.agent import TurnOutcome, TurnStreamReader
from novakit.agent.messages import (
    build_user_message,
    convert_to_llm_messages,
    count_user_turns,
)
from novakit.agent.services import NoopBackground, StaticPromptProvider
from novakit.agent.state import TurnState
from novakit.events import AgentEvent, EventBus, done_payload, error_payload
from novakit.llm.provider import (
    ChatStreamEvent,
    Done,
    ToolInputDelta,
    ToolInputEnd,
    ToolInputError,
    ToolInputStart,
)
from novakit.session import InMemorySession
from novakit.session.models import Message


def _message(
    role: str,
    content: str,
    tool_call_id: str | None = None,
    tool_calls: list[dict[str, str]] | None = None,
    **extra: object,
) -> Message:
    # Arrange: 最小可用的持久化消息。
    base: dict[str, object] = {
        "id": f"{role}-{content[:4] or 'x'}",
        "session_id": "s",
        "role": role,
        "content": content,
    }
    if tool_call_id is not None:
        base["tool_call_id"] = tool_call_id
    if tool_calls is not None:
        base["tool_calls"] = tool_calls
    base.update(extra)
    return Message(**base)  # type: ignore[arg-type]


def _chunks(items: list[ChatStreamEvent]) -> AsyncIterator[ChatStreamEvent]:
    async def gen() -> AsyncIterator[ChatStreamEvent]:
        for item in items:
            yield item

    return gen()


async def _drive(
    reader: TurnStreamReader, items: list[ChatStreamEvent]
) -> list[tuple[AgentEvent, object]]:
    # Act: 消费一组 provider 事件。
    return [pair async for pair in reader.consume(_chunks(items))]


def _reader() -> tuple[TurnStreamReader, list[tuple[AgentEvent, object]]]:
    # Arrange: 用列表记录 emit 的同步回调。
    seen: list[tuple[AgentEvent, object]] = []

    async def emit(event: AgentEvent, data: object = None) -> None:
        seen.append((event, data))

    async def stop() -> dict | None:
        return None

    return TurnStreamReader(emit=emit, stop_if_aborted=stop), seen


@pytest.mark.asyncio
async def test_event_bus_delivers_sync_async_and_swallows_failure() -> None:
    # Arrange: 同步、异步、抛错三种 handler。
    bus = EventBus()
    seen: list[tuple[AgentEvent, object]] = []

    def sync_handler(event: AgentEvent, data: object = None) -> None:
        seen.append((event, data))

    async def async_handler(event: AgentEvent, data: object = None) -> None:
        seen.append((event, data))

    def bad_handler(event: AgentEvent, data: object = None) -> None:
        raise RuntimeError("boom")

    async def bad_async(event: AgentEvent, data: object = None) -> None:
        raise RuntimeError("boom")

    bus.on(AgentEvent.TEXT_DELTA, sync_handler)
    bus.on(AgentEvent.TEXT_DELTA, async_handler)
    bus.on(AgentEvent.TEXT_DELTA, bad_handler)
    bus.on(AgentEvent.TEXT_DELTA, bad_async)
    # Act: 广播一次。
    await bus.emit(AgentEvent.TEXT_DELTA, "hi")
    # Assert: 失败的 handler 不影响其他人。
    assert (AgentEvent.TEXT_DELTA, "hi") in seen
    assert len([s for s in seen if s[1] == "hi"]) == 2


@pytest.mark.asyncio
async def test_event_bus_off_removes_and_ignores_unknown() -> None:
    # Arrange: 注册后移除。
    bus = EventBus()
    seen: list[tuple[AgentEvent, object]] = []

    def handler(event: AgentEvent, data: object = None) -> None:
        seen.append((event, data))

    bus.on(AgentEvent.DONE, handler)
    # Act: 移除并再次触发。
    bus.off(AgentEvent.DONE, handler)
    await bus.emit(AgentEvent.DONE, done_payload("completed", "ok"))
    # Assert: 不再收到。
    assert seen == []
    # Act: 对未订阅事件执行 off 不抛错。
    bus.off(AgentEvent.ERROR, handler)
    assert bus._handlers.get(AgentEvent.ERROR, []) == []


def test_done_and_error_payload_shapes() -> None:
    # Arrange/Act: 构造终止负载。
    done = done_payload("completed", "ok")
    err = error_payload("llm_error", "boom")
    # Assert: 字段齐全。
    assert done == {"reason": "completed", "content": "ok"}
    assert err == {"reason": "llm_error", "message": "boom"}


@pytest.mark.asyncio
async def test_tool_input_error_is_forwarded() -> None:
    # Arrange: 未覆盖的 ToolInputError 分支。
    reader, _ = _reader()
    chunk = ToolInputError(id="c1", name="write", raw="{bad")
    # Act: 消费。
    yielded = await _drive(reader, [chunk, Done()])
    # Assert: 透传且不影响继续。
    assert yielded == [(AgentEvent.TOOL_INPUT_ERROR, chunk)]
    assert reader.outcome is TurnOutcome.CONTINUE


@pytest.mark.asyncio
async def test_reasoning_elapsed_none_without_start() -> None:
    # Arrange: 从未开始推理的 reader。
    reader, _ = _reader()
    # Act/Assert: 直接查询为 None(覆盖 early-return)。
    assert reader._reasoning_elapsed() is None


@pytest.mark.asyncio
async def test_log_usage_branches() -> None:
    # Arrange: 三种用量组合。
    reader, _ = _reader()
    # Act: 无用量直接返回。
    await _drive(reader, [Done(content="a")])
    assert reader.tokens_input is None
    # Act: 有输入、无缓存命中。
    await _drive(reader, [Done(content="b", tokens_input=100, tokens_output=5)])
    assert reader.tokens_input == 100
    assert reader.cache_read_tokens is None
    # Act: 有缓存命中率。
    await _drive(
        reader,
        [Done(content="c", tokens_input=100, cache_read_tokens=25)],
    )
    # Assert: 缓存被吸收。
    assert reader.cache_read_tokens == 25


@pytest.mark.asyncio
async def test_tool_input_end_and_delta_forward() -> None:
    # Arrange: 补齐 end/delta 的 emit 路径。
    reader, seen = _reader()
    start = ToolInputStart(id="c1", name="w")
    delta = ToolInputDelta(id="c1", name="w", delta="{}")
    end = ToolInputEnd(id="c1", name="w")
    # Act: 消费。
    yielded = await _drive(reader, [start, delta, end, Done()])
    # Assert: yield 与 emit 均有。
    assert [e for e, _ in yielded] == [
        AgentEvent.TOOL_INPUT_START,
        AgentEvent.TOOL_INPUT_DELTA,
        AgentEvent.TOOL_INPUT_END,
    ]
    assert (AgentEvent.TOOL_INPUT_END, end) in seen


def test_build_user_message_image_branches() -> None:
    # Arrange: 1x1 PNG 的 dataURL、远端引用、空引用、超预算由补丁触发。
    tiny = base64.b64encode(bytes([137, 80, 78, 71, 13, 10, 26, 10])).decode()
    local = f"data:image/png;base64,{tiny}"
    attachments: list[dict] = [
        {
            "type": "image",
            "content": [
                {"type": "image", "image": local},
                {"type": "text", "image": local},
                {"type": "image", "image": ""},
                {"type": "image", "image": "https://example.com/x.png"},
                {"type": "image", "image": local},
            ],
        },
        {"type": "other", "content": [{"type": "text", "text": "ignore"}]},
        {"type": "document", "content": [{"type": "image", "text": "skip"}]},
        {"type": "document", "content": [{"type": "text", "text": "doc"}]},
    ]
    # Act: 折叠附件。
    text, images, shown = build_user_message("q", attachments, "shown")
    # Assert: 本地小图保留、远端与空跳过、文档前置、display 透传。
    assert shown == "shown"
    assert text == "doc\n\nq"
    assert len(images) == 2
    assert images[0] == images[1]


def test_build_user_message_drops_oversize_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: 把预算压到 0,任何图片都超限。
    import novakit.agent.messages as messages_mod

    monkeypatch.setattr(messages_mod, "MAX_IMAGE_BYTES", 0)
    monkeypatch.setattr(messages_mod, "MAX_IMAGES_BYTES", 0)
    tiny = base64.b64encode(b"0123456789").decode()
    attachments = [{"type": "image", "content": [{"type": "image", "image": tiny}]}]
    # Act: 折叠。
    text, images, _ = build_user_message("q", attachments)
    # Assert: 图片被丢弃但文本保留。
    assert text == "q"
    assert images == []


def test_convert_carries_images_reasoning_meta_model() -> None:
    # Arrange: 携带全字段的 assistant + 配对的 tool 结果。
    call = {"id": "k1", "name": "read"}
    assistant = _message(
        "assistant",
        "out",
        tool_calls=[call],
        images=["img"],
        reasoning_content="why",
        provider_meta={"sig": "s"},
        model="m",
    )
    result = _message("tool", "ok", tool_call_id="k1")
    stray = _message("assistant", "stray", tool_calls=[{"id": "ghost", "name": "x"}])
    # Act: 转换。
    converted = convert_to_llm_messages([assistant, result, stray])
    # Assert: 字段透传,未配对的调用被裁剪。
    assert converted[0].images == ["img"]
    assert converted[0].reasoning_content == "why"
    assert converted[0].provider_meta == {"sig": "s"}
    assert converted[0].model == "m"
    assert converted[1].tool_call_id == "k1"
    assert converted[2].tool_calls == []


def test_count_user_turns_none_and_empty() -> None:
    # Arrange: 无 role 与空列表。
    # Act/Assert: 均为 0。
    assert count_user_turns([]) == 0
    assert count_user_turns([{"role": "system"}]) == 0
    assert count_user_turns([{"nobody": 1}]) == 0


@pytest.mark.asyncio
async def test_static_provider_reads_session_when_no_preload() -> None:
    # Arrange: 会话中有历史,不预加载。
    session = InMemorySession()
    await session.create_session()
    await session.add_message(role="user", content="hi")
    provider = StaticPromptProvider("sys", TurnState(), session)
    # Act: 不传 loaded_messages(覆盖 46 行)。
    messages = await provider.build_messages()
    # Assert: 系统提示 + 历史。
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == "sys"


def test_noop_background_hooks_do_nothing() -> None:
    # Arrange: 默认空实现。
    hooks = NoopBackground()
    session = InMemorySession().get_current_session()
    # Act/Assert: 调用不抛错(append 覆盖无-op 分支)。
    hooks.maybe_schedule_title_generation(session, "hi")  # type: ignore[arg-type]
    hooks.maybe_schedule_memory_review(3)


@pytest.mark.asyncio
async def test_noop_schedule_request_learning() -> None:
    # Arrange: 空学习钩子。
    hooks = NoopBackground()
    # Act/Assert: 等待完成。
    await hooks.schedule_request_learning(0, 0)
