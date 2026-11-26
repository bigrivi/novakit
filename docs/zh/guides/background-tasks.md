# 后台任务

`novakit.tasks.BackgroundTaskManager`（`novakit/tasks/manager.py`）提供
执行器注册、前台/后台分离的并发池、按会话配额、超时、取消、进度、监听器、保留
和有界输出。它没有进程全局单例，也没有内置注册表——宿主提供实现
`TaskExecutor` 协议的执行器（`novakit/tasks/models.py`：`kind` 加 `execute(...)`）。

## 提交并等待

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

关键行为：

- `submit` 执行按会话配额（超限时抛出 `TaskLimitError`），并
  路由到前台或后台池。
- `wait(task_id, session_id, timeout=...)` 通过 `resolve_foreground_wait`
  处理前台等待；后台任务改为通过监听器汇报进度，而不阻塞轮次。
- `cancel(task_id, session_id)` 取消单个任务；`cancel_for_session(...)`
  取消该会话拥有的全部任务。

## 进程辅助函数

`novakit.tasks.process.run_process` 执行显式的 argv/工作目录/环境，
流式合并输出，并在取消时杀死整个进程树：

```python
from novakit.tasks import run_process

result = await run_process(["echo", "hello"], cwd="/tmp")
print(result.returncode, result.output)
```

`kill_process_tree` 是取消时使用的跨平台终止原语。没有默认 shell——
调用方传入确切的 argv。

## 保留与输出上限

管理器应用 `DEFAULT_MAX_RETAINED_TASKS`、`DEFAULT_RETENTION_SECONDS`
和 `DEFAULT_MAX_OUTPUT_CHARS`，使长期运行的宿主保持有界。终态
列于 `TERMINAL_STATUSES`；每个 `TaskRecord` 携带 `TaskStatus`、
进度和截断后的输出。

## API 参考

::: novakit.tasks.BackgroundTaskManager
::: novakit.tasks.TaskExecutor
::: novakit.tasks.TaskRecord
::: novakit.tasks.run_process
