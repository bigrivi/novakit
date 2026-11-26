"""10 Background tasks: submit, progress, listeners, pool, quota, timeout, retention.

What you'll learn:
- Register an executor with ``register_executor`` and submit
  foreground/background tasks with ``submit``
- Wait with ``wait`` and move a foreground task to the background
  with ``mark_background``
- What progress updates and state/completion listeners report
- How a full pool and a per-session quota surface as ``TaskLimitError``
- How ``timeout``/``cancel`` end tasks, and how retention keeps
  only the newest records

Run:
    python examples/10_background_tasks.py
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from novakit.tasks import (
    BackgroundTaskManager,
    TaskExecutionContext,
    TaskExecutionResult,
    TaskExecutor,
    TaskLimitError,
    TaskRecord,
)


class EchoExecutor(TaskExecutor):
    """Uppercase its input while reporting output, progress, and metadata."""

    kind = "echo"

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: TaskExecutionContext,
    ) -> TaskExecutionResult:
        """Echo the input with bounded output and progress updates."""
        text = str(arguments.get("text", ""))
        context.write_output(text)
        context.update_progress(0.5, "halfway")
        context.update_progress(1.0, "done")
        context.set_metadata("length", len(text))
        return TaskExecutionResult(success=True, result=text.upper(), exit_code=0)


class SleepExecutor(TaskExecutor):
    """Block on an event so the demo controls exactly when it finishes."""

    kind = "sleep"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(
        self,
        arguments: Mapping[str, object],
        context: TaskExecutionContext,
    ) -> TaskExecutionResult:
        """Wait until released, then succeed."""
        self.started.set()
        await self.release.wait()
        return TaskExecutionResult(success=True, result="slept")


async def show_run_and_listeners() -> None:
    """Submit, wait, and observe progress plus state/completion listeners."""
    print("- submit + wait: result, output tail, progress, metadata, listeners")
    manager = BackgroundTaskManager()
    updates: list[TaskRecord] = []
    completions: list[TaskRecord] = []
    manager.set_listener(updates.append)
    manager.set_completion_listener(completions.append)
    manager.register_executor(EchoExecutor())

    task = manager.submit(
        "echo",
        {"text": "hello"},
        session_id="s-demo",
        label="Echo hello",
        timeout_seconds=10,
        background=True,
    )
    print(f"  submitted task_id={task.task_id} background={task.background}")
    done = await manager.wait(task.task_id, "s-demo", timeout=5)
    if done is not None:
        print(
            f"  status={done.status} result={done.result} output_tail={done.output_tail!r}"
        )
        print(f"  progress={done.progress} metadata={done.metadata}")
    print(f"  get same session: {manager.get(task.task_id, 's-demo') is not None}")
    print(f"  get other session: {manager.get(task.task_id, 'other-session')}")
    print(f"  state updates={len(updates)} completions={len(completions)}")
    print(
        f"  list_for_session={[r.task_id for r in manager.list_for_session('s-demo')]}"
    )
    await manager.shutdown()


async def show_concurrency_and_pool() -> None:
    """Bound the global pool and detach a foreground task to the background."""
    print("- pool: two running tasks fill the pool, the third is refused")
    manager = BackgroundTaskManager(
        max_concurrent=2, max_foreground_concurrent=2, max_per_session=10
    )
    gate = SleepExecutor()
    manager.register_executor(gate)
    first = manager.submit(
        "sleep",
        {},
        session_id="s-pool",
        label="Hold one",
        timeout_seconds=30,
        background=True,
    )
    second = manager.submit(
        "sleep",
        {},
        session_id="s-pool",
        label="Hold two",
        timeout_seconds=30,
        background=True,
    )
    for _ in range(100):
        one = manager.get(first.task_id, "s-pool")
        two = manager.get(second.task_id, "s-pool")
        if (
            one is not None
            and two is not None
            and one.status == two.status == "running"
        ):
            break
        await asyncio.sleep(0.05)
    first_record = manager.get(first.task_id, "s-pool")
    second_record = manager.get(second.task_id, "s-pool")
    print(f"  first={first_record.status if first_record is not None else None}")
    print(f"  second={second_record.status if second_record is not None else None}")
    try:
        manager.submit(
            "sleep",
            {},
            session_id="s-pool",
            label="Over capacity",
            timeout_seconds=30,
            background=True,
        )
        print("  third submit outcome: accepted (the pool still had room)")
    except TaskLimitError as exc:
        print(f"  third submit outcome: refused ({exc})")
        print("  (a full pool surfaces the refusal as TaskLimitError)")
    gate.release.set()
    for task_id in (first.task_id, second.task_id):
        done = await manager.wait(task_id, "s-pool", timeout=5)
        print(f"  released {task_id[:8]}... status={done.status if done else None}")

    manager.register_executor(EchoExecutor())
    for i in range(4):
        quick = manager.submit(
            "echo",
            {"text": f"job-{i}"},
            session_id="s-pool",
            label=f"Job {i}",
            timeout_seconds=10,
            background=True,
        )
        done = await manager.wait(quick.task_id, "s-pool", timeout=5)
        print(
            f"  quick job-{i} status={done.status if done else None} result={done.result if done else None}"
        )

    # Foreground submits carry a timeout by default; moving to the
    # background clears it, which suits long-running tasks.
    foreground = manager.submit(
        "echo",
        {"text": "fg"},
        session_id="s-pool",
        label="Foreground",
        timeout_seconds=10,
        background=False,
    )
    detached = manager.mark_background(foreground.task_id, "s-pool")
    if detached is not None:
        print(
            f"  detached to background={detached.background} timeout={detached.timeout_seconds}"
        )
    done = await manager.wait(foreground.task_id, "s-pool", timeout=5)
    print(f"  foreground-turned-background status={done.status if done else None}")
    await manager.shutdown()


async def show_quota_timeout_cancel() -> None:
    """Enforce the session quota, then show timeout and cancel paths."""
    print("- quota: second task in a quota-1 session is refused, then cancel/timeout")
    manager = BackgroundTaskManager(max_per_session=1)
    sleeper = SleepExecutor()
    manager.register_executor(sleeper)
    first = manager.submit(
        "sleep",
        {},
        session_id="s-quota",
        label="Long sleep",
        timeout_seconds=30,
        background=True,
    )
    await asyncio.wait_for(sleeper.started.wait(), timeout=5)
    try:
        manager.submit(
            "sleep",
            {},
            session_id="s-quota",
            label="Over quota",
            timeout_seconds=30,
            background=True,
        )
        print("  over-quota submit outcome: accepted (quota still had room)")
    except TaskLimitError as exc:
        print(f"  over-quota submit outcome: refused ({exc})")
        print("  (a full session quota surfaces the refusal as TaskLimitError)")
    print(f"  cancel returns: {await manager.cancel(first.task_id, 's-quota')}")
    cancelled = manager.get(first.task_id, "s-quota")
    print(f"  status after cancel: {cancelled.status if cancelled else None}")

    manager.register_executor(EchoExecutor())
    quick = manager.submit(
        "echo",
        {"text": "after"},
        session_id="s-quota",
        label="After cancel",
        timeout_seconds=10,
        background=True,
    )
    done = await manager.wait(quick.task_id, "s-quota", timeout=5)
    print(f"  after cancel: status={done.status if done else None}")
    print(
        f"  wait again (cached): {(await manager.wait(quick.task_id, 's-quota', timeout=0)) is not None}"
    )
    await manager.shutdown()

    print("- timeout: hanging task exceeds its runtime limit")
    timeouts = BackgroundTaskManager()
    hang = SleepExecutor()
    timeouts.register_executor(hang)
    slow = timeouts.submit(
        "sleep",
        {},
        session_id="s-timeout",
        label="Hanging",
        timeout_seconds=1,
        background=True,
    )
    timed = await timeouts.wait(slow.task_id, "s-timeout", timeout=5)
    if timed is not None:
        print(f"  status={timed.status} error={timed.error}")
    await timeouts.shutdown()


async def show_retention() -> None:
    """Keep only the newest records and list them newest-first per session."""
    print("- retention: max 2 records, oldest evicted, newest-first listing")
    manager = BackgroundTaskManager(max_retained_tasks=2)
    manager.register_executor(EchoExecutor())
    ids: list[str] = []
    for i in range(3):
        task = manager.submit(
            "echo",
            {"text": f"m{i}"},
            session_id="s-keep",
            label=f"Kept {i}",
            timeout_seconds=10,
            background=True,
        )
        ids.append(task.task_id)
        await manager.wait(task.task_id, "s-keep", timeout=5)
    print(
        f"  oldest still kept: {manager.get(ids[0], 's-keep') is not None} (False means evicted)"
    )
    print(f"  list_all={[r.task_id for r in manager.list_all()]}")
    print(
        f"  list_for_session={[r.task_id for r in manager.list_for_session('s-keep')]}"
    )
    await manager.shutdown()


async def main() -> None:
    """Demo background tasks: configure, run, then observe the results."""
    print("== 10 background tasks ==")
    # Each show_* builds its own manager and registers the executors it needs.
    await show_run_and_listeners()
    await show_concurrency_and_pool()
    await show_quota_timeout_cancel()
    await show_retention()
    print("10 background tasks: done")


if __name__ == "__main__":
    asyncio.run(main())
