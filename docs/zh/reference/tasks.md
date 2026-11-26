# 任务参考

可移植的进程内任务生命周期、配额与有界输出。从
`novakit.tasks` 导入。

## 管理器

::: novakit.tasks.BackgroundTaskManager
::: novakit.tasks.TaskLimitError
::: novakit.tasks.resolve_foreground_wait

`BackgroundTaskManager(max_concurrent=..., ...)` 注册执行器，
并将工作路由到独立的前台/后台线程池，附带按会话配额。
超出配额时 `submit` 抛出 `TaskLimitError`；
`resolve_foreground_wait` 计算有效的前台等待时间。

## 执行器契约与记录

::: novakit.tasks.TaskExecutor
::: novakit.tasks.TaskExecutionContext
::: novakit.tasks.TaskExecutionResult
::: novakit.tasks.TaskRecord
::: novakit.tasks.TaskStatus

宿主实现 `TaskExecutor`（`kind` 加 `execute(...)`）；管理器以
`TaskExecutionContext` 驱动它，并记录 `TaskExecutionResult`。
`TaskRecord` 携带状态（`TaskStatus`）、进度和有界输出；
`TERMINAL_STATUSES` 列出已完成状态。

## 进程辅助函数

::: novakit.tasks.run_process
::: novakit.tasks.kill_process_tree

`run_process(argv, cwd=..., env=...)` 执行显式的 argv，流式合并输出，
取消时通过 `kill_process_tree` 杀死整个进程树。
