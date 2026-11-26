"""Direct tests for the portable background-task manager."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from novakit.tasks import (
    BackgroundTaskManager,
    TaskExecutionContext,
    TaskExecutionResult,
    TaskExecutor,
    TaskLimitError,
    TaskRecord,
    resolve_foreground_wait,
)


class FunctionExecutor(TaskExecutor):
    """Wrap a coroutine as a registered task executor."""

    def __init__(
        self,
        kind: str,
        run: object,
        *,
        unlimited: bool = False,
    ) -> None:
        self.kind = kind
        self.unlimited = unlimited
        self._run = run

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: TaskExecutionContext,
    ) -> TaskExecutionResult:
        return await self._run(arguments, context)  # type: ignore[operator]


@pytest.fixture
def make_executor() -> object:
    return FunctionExecutor


@pytest.mark.asyncio
async def test_manager_runs_executor_and_tracks_output() -> None:
    manager = BackgroundTaskManager(max_output_chars=32)

    async def execute(
        arguments: Mapping[str, object], context: TaskExecutionContext
    ) -> TaskExecutionResult:
        context.write_output(str(arguments["output"]))
        context.update_progress(2.0, "done")
        context.set_metadata("answer", 42)
        return TaskExecutionResult(success=True, result="done", exit_code=0)

    manager.register_executor(FunctionExecutor("example", execute))
    task = manager.submit(
        "example",
        {"output": "executor output"},
        session_id="session-a",
        label="Example task",
        timeout_seconds=2,
        background=True,
    )

    completed = await manager.wait(task.task_id, "session-a", timeout=2)

    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.result == "done"
    assert completed.output_tail == "executor output"
    assert completed.progress == 1.0
    assert completed.metadata == {"answer": 42}
    assert manager.get(task.task_id, "session-a") is not None
    assert manager.get(task.task_id, "session-b") is None
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_and_session_quota() -> None:
    manager = BackgroundTaskManager(max_per_session=1)
    started = asyncio.Event()

    async def execute(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        started.set()
        await asyncio.Event().wait()
        return TaskExecutionResult(success=True)

    manager.register_executor(FunctionExecutor("wait", execute))
    task = manager.submit(
        "wait",
        {},
        session_id="session-a",
        label="Wait forever",
        timeout_seconds=30,
        background=True,
    )
    await started.wait()

    with pytest.raises(TaskLimitError):
        manager.submit(
            "wait",
            {},
            session_id="session-a",
            label="Exceeds quota",
            timeout_seconds=30,
            background=True,
        )

    assert not await manager.cancel(task.task_id, "session-b")
    assert await manager.cancel(task.task_id, "session-a")
    cancelled = manager.get(task.task_id, "session-a")
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert not await manager.cancel(task.task_id, "session-a")
    await manager.shutdown()


@pytest.mark.asyncio
async def test_timeout_bounds_output_and_result() -> None:
    manager = BackgroundTaskManager(max_output_chars=8)

    async def execute(
        _arguments: Mapping[str, object], context: TaskExecutionContext
    ) -> TaskExecutionResult:
        context.write_output("0123456789abcdef")
        await asyncio.sleep(1)
        return TaskExecutionResult(success=True, result="x" * 20)

    manager.register_executor(FunctionExecutor("slow", execute))
    task = manager.submit(
        "slow",
        {},
        session_id="session-a",
        label="Slow task",
        timeout_seconds=1,
        background=True,
    )
    record = manager.get(task.task_id, "session-a")
    assert record is not None
    record.timeout_seconds = 0
    manager._notify(record)

    completed = await manager.wait(task.task_id, "session-a", timeout=2)

    assert completed is not None
    assert completed.status == "timed_out"
    assert completed.output_tail == "89abcdef"
    assert completed.output_truncated is True
    await manager.shutdown()


@pytest.mark.asyncio
async def test_result_truncation_and_executor_failure() -> None:
    manager = BackgroundTaskManager(max_output_chars=8)

    async def long_result(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        return TaskExecutionResult(success=True, result="x" * 20)

    async def broken(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        raise RuntimeError("boom")

    manager.register_executor(FunctionExecutor("long", long_result))
    manager.register_executor(FunctionExecutor("broken", broken))
    long_task = manager.submit(
        "long", {}, session_id="s", label="long", timeout_seconds=1, background=True
    )
    broken_task = manager.submit(
        "broken", {}, session_id="s", label="broken", timeout_seconds=1, background=True
    )

    long_done = await manager.wait(long_task.task_id, "s", timeout=2)
    broken_done = await manager.wait(broken_task.task_id, "s", timeout=2)

    assert long_done is not None
    assert long_done.result == "x" * 8
    assert long_done.output_truncated is True
    assert broken_done is not None
    assert broken_done.status == "failed"
    assert broken_done.error == "boom"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_mark_background_lifts_timeout_and_keeps_ownership() -> None:
    manager = BackgroundTaskManager()
    started = asyncio.Event()

    async def execute(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        started.set()
        await asyncio.sleep(0.05)
        return TaskExecutionResult(success=True)

    manager.register_executor(FunctionExecutor("work", execute))
    task = manager.submit(
        "work",
        {},
        session_id="s",
        label="work",
        timeout_seconds=1,
        background=False,
    )
    await started.wait()

    detached = manager.mark_background(task.task_id, "s")
    assert detached is not None
    assert detached.background is True
    assert detached.timeout_seconds is None
    assert manager.mark_background(task.task_id, "other") is None
    completed = await manager.wait(task.task_id, "s", timeout=2)
    assert completed is not None
    assert completed.status == "succeeded"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_listeners_are_isolated_and_completion_fires_once() -> None:
    manager = BackgroundTaskManager()
    updates: list[TaskRecord] = []
    completions: list[TaskRecord] = []

    def broken_listener(_record: TaskRecord) -> None:
        raise RuntimeError("observer failed")

    manager.set_listener(broken_listener)
    manager.set_completion_listener(completions.append)

    async def execute(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        return TaskExecutionResult(success=True)

    manager.register_executor(FunctionExecutor("ok", execute))
    task = manager.submit(
        "ok", {}, session_id="s", label="ok", timeout_seconds=1, background=True
    )
    completed = await manager.wait(task.task_id, "s", timeout=2)
    assert completed is not None
    manager._notify(completed)
    assert len(completions) == 1

    manager.set_listener(updates.append)
    manager._notify(completed)
    assert updates == [completed]
    manager.set_completion_listener(None)
    await manager.shutdown()


@pytest.mark.asyncio
async def test_wait_timeout_and_cancel_for_session() -> None:
    manager = BackgroundTaskManager()
    started = asyncio.Event()

    async def execute(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        started.set()
        await asyncio.Event().wait()
        return TaskExecutionResult(success=True)

    manager.register_executor(FunctionExecutor("wait", execute))
    task = manager.submit(
        "wait", {}, session_id="s", label="wait", timeout_seconds=None, background=True
    )
    await started.wait()
    assert await manager.wait(task.task_id, "s", timeout=0) is None
    assert await manager.wait("missing", "s", timeout=0) is None

    await manager.cancel_for_session("s")
    record = manager.get(task.task_id, "s")
    assert record is not None
    assert record.status == "cancelled"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_retention_and_retained_cap_evict_terminal_records() -> None:
    manager = BackgroundTaskManager(retention_seconds=1, max_retained_tasks=1)

    async def execute(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        return TaskExecutionResult(success=True)

    manager.register_executor(FunctionExecutor("ok", execute))
    first = manager.submit(
        "ok", {}, session_id="s", label="one", timeout_seconds=1, background=True
    )
    assert (await manager.wait(first.task_id, "s", timeout=2)) is not None
    second = manager.submit(
        "ok", {}, session_id="s", label="two", timeout_seconds=1, background=True
    )
    assert (await manager.wait(second.task_id, "s", timeout=2)) is not None

    assert manager.get(first.task_id, "s") is None
    assert [record.task_id for record in manager.list_all()] == [second.task_id]
    assert manager.list_for_session("s")[0].task_id == second.task_id
    await manager.shutdown()


def test_register_executor_validation_and_reregistration() -> None:
    manager = BackgroundTaskManager()

    async def execute(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        return TaskExecutionResult(success=True)

    with pytest.raises(TypeError, match="TaskExecutor"):
        manager.register_executor(execute)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="kind"):
        manager.register_executor(FunctionExecutor("  ", execute))

    manager.register_executor(FunctionExecutor("kind", execute, unlimited=True))
    assert "kind" in manager._unlimited_kinds
    manager.register_executor(FunctionExecutor("kind", execute))
    assert "kind" not in manager._unlimited_kinds


def test_submit_unknown_kind_and_label_normalization() -> None:
    manager = BackgroundTaskManager()
    with pytest.raises(KeyError, match="No executor"):
        manager.submit(
            "missing",
            {},
            session_id="s",
            label="x",
            timeout_seconds=1,
            background=True,
        )


def test_task_record_snapshot_shape() -> None:
    record = TaskRecord(
        task_id="t1",
        kind="kind",
        session_id="s1",
        label="label",
        metadata={"nested": 1},
    )
    snapshot = record.to_dict()
    without_output = record.to_dict(include_output=False)

    assert snapshot["task_id"] == "t1"
    assert snapshot["output_tail"] == ""
    assert snapshot["metadata"] == {"nested": 1}
    assert "output_tail" not in without_output


@pytest.mark.asyncio
async def test_unlimited_executor_skips_pool_and_empty_output_is_ignored() -> None:
    manager = BackgroundTaskManager(max_concurrent=1, max_foreground_concurrent=1)
    seen: list[int] = []

    async def execute(
        _arguments: Mapping[str, object], context: TaskExecutionContext
    ) -> TaskExecutionResult:
        context.write_output("")
        seen.append(manager._semaphore._value)
        return TaskExecutionResult(success=True)

    manager.register_executor(FunctionExecutor("free", execute, unlimited=True))
    task = manager.submit(
        "free", {}, session_id="s", label="free", timeout_seconds=1, background=True
    )
    completed = await manager.wait(task.task_id, "s", timeout=2)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.output_bytes == 0
    assert seen == [1]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_completion_listener_failure_is_isolated() -> None:
    manager = BackgroundTaskManager()

    def broken(_record: TaskRecord) -> None:
        raise RuntimeError("observer failed")

    manager.set_completion_listener(broken)

    async def execute(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        return TaskExecutionResult(success=True)

    manager.register_executor(FunctionExecutor("ok", execute))
    task = manager.submit(
        "ok", {}, session_id="s", label="ok", timeout_seconds=1, background=True
    )
    completed = await manager.wait(task.task_id, "s", timeout=2)
    assert completed is not None
    assert completed.status == "succeeded"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_wait_without_completion_event_and_cancel_without_task() -> None:
    manager = BackgroundTaskManager()
    record = TaskRecord(
        task_id="manual",
        kind="kind",
        session_id="s",
        label="manual",
        status="running",
    )
    manager._records[record.task_id] = record

    snapshot = await manager.wait(record.task_id, "s", timeout=0)
    assert snapshot is not None
    assert snapshot.status == "running"
    assert await manager.cancel(record.task_id, "s") is True
    assert manager.get(record.task_id, "s").status == "cancelled"  # type: ignore[union-attr]
    manager._finish(record, "failed")
    assert record.status == "cancelled"
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cancel_finishes_record_when_task_suppresses_cancellation() -> None:
    manager = BackgroundTaskManager()
    started = asyncio.Event()

    async def execute(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        started.set()
        await asyncio.Event().wait()
        return TaskExecutionResult(success=True)

    manager.register_executor(FunctionExecutor("wait", execute))
    task = manager.submit(
        "wait", {}, session_id="s", label="wait", timeout_seconds=None, background=True
    )
    await started.wait()

    original = manager._tasks[task.task_id]

    async def suppress_cancellation() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return

    replacement = asyncio.create_task(suppress_cancellation())
    manager._tasks[task.task_id] = replacement
    assert await manager.cancel(task.task_id, "s") is True
    original.cancel()
    with pytest.raises(asyncio.CancelledError):
        await original
    assert manager.get(task.task_id, "s").status == "cancelled"  # type: ignore[union-attr]
    await manager.shutdown()


@pytest.mark.asyncio
async def test_task_cancelled_before_start_is_not_run() -> None:
    manager = BackgroundTaskManager()
    ran = False

    async def execute(
        _arguments: Mapping[str, object], _context: TaskExecutionContext
    ) -> TaskExecutionResult:
        nonlocal ran
        ran = True
        return TaskExecutionResult(success=True)

    manager.register_executor(FunctionExecutor("never", execute))
    task = manager.submit(
        "never", {}, session_id="s", label="never", timeout_seconds=1, background=True
    )
    record = manager.get(task.task_id, "s")
    assert record is not None
    manager._records[task.task_id].status = "cancelled"
    await asyncio.sleep(0)
    assert ran is False
    await manager.shutdown()


def test_stale_retention_evicts_records_and_completion_events() -> None:
    manager = BackgroundTaskManager(retention_seconds=1)
    record = TaskRecord(
        task_id="old",
        kind="kind",
        session_id="s",
        label="old",
        status="succeeded",
        finished_at_ms=1,
    )
    manager._records[record.task_id] = record
    manager._completion_events[record.task_id] = asyncio.Event()

    assert manager.list_all() == []
    assert record.task_id not in manager._completion_events


@pytest.mark.parametrize(
    ("requested", "timeout", "default", "maximum", "expected"),
    [
        (None, None, 10, 30, 10),
        ("5", None, 10, 30, 5),
        ("bad", None, 10, 30, 10),
        (100, None, 10, 30, 30),
        (20, 7, 10, 30, 7),
        (0, None, 10, 30, 1),
    ],
)
def test_resolve_foreground_wait(
    requested: object,
    timeout: int | None,
    default: int,
    maximum: int,
    expected: int,
) -> None:
    assert (
        resolve_foreground_wait(
            requested,
            timeout,
            default=default,
            maximum=maximum,
        )
        == expected
    )
