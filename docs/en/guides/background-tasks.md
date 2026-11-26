# Background tasks

`novakit.tasks.BackgroundTaskManager` (`novakit/tasks/manager.py`) provides
executor registration, separate foreground/background concurrency pools,
per-session quotas, timeouts, cancellation, progress, listeners, retention,
and bounded output. It has no process-global singleton and no built-in
registry — hosts supply executors implementing the `TaskExecutor` protocol
(`novakit/tasks/models.py`: `kind` plus `execute(...)`).

## Submit and wait

```python
from novakit.tasks import BackgroundTaskManager

manager = BackgroundTaskManager(max_concurrent=2)
# A host executor implements kind and execute(...).
record = manager.submit(
    "host_kind",
    {"argument": "value"},
    session_id=session.id,
    label="Host work",
    timeout_seconds=30,
    background=True,
)
completed = await manager.wait(record.task_id, session.id, timeout=5)
```

Key behavior:

- `submit` enforces per-session quotas (`TaskLimitError` when exceeded) and
  routes to the foreground or background pool.
- `wait(task_id, session_id, timeout=...)` resolves foreground waits via
  `resolve_foreground_wait`; background tasks report progress through
  listeners instead of blocking the turn.
- `cancel(task_id, session_id)` cancels one task; `cancel_for_session(...)`
  cancels everything owned by a session.

## Process helper

`novakit.tasks.process.run_process` executes an explicit argv/cwd/environment,
streams merged output, and kills the process tree on cancellation:

```python
from novakit.tasks import run_process

result = await run_process(["echo", "hello"], cwd="/tmp")
print(result.returncode, result.output)
```

`kill_process_tree` is the cross-platform termination primitive used on
cancel. There is no shell default — the caller passes the exact argv.

## Retention and output bounds

The manager applies `DEFAULT_MAX_RETAINED_TASKS`, `DEFAULT_RETENTION_SECONDS`,
and `DEFAULT_MAX_OUTPUT_CHARS` so long-running hosts stay bounded. Terminal
statuses are listed in `TERMINAL_STATUSES`; each `TaskRecord` carries
`TaskStatus`, progress, and truncated output.

## API reference

::: novakit.tasks.BackgroundTaskManager
::: novakit.tasks.TaskExecutor
::: novakit.tasks.TaskRecord
::: novakit.tasks.run_process
