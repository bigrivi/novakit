# Tasks reference

Portable in-process task lifecycle, quotas, and bounded output. Import from
`novakit.tasks`.

## Manager

::: novakit.tasks.BackgroundTaskManager
::: novakit.tasks.TaskLimitError
::: novakit.tasks.resolve_foreground_wait

`BackgroundTaskManager(max_concurrent=..., ...)` registers executors and
routes work to separate foreground/background pools with per-session quotas.
`submit` raises `TaskLimitError` when a quota is exceeded;
`resolve_foreground_wait` computes the effective foreground wait.

## Executor contract and records

::: novakit.tasks.TaskExecutor
::: novakit.tasks.TaskExecutionContext
::: novakit.tasks.TaskExecutionResult
::: novakit.tasks.TaskRecord
::: novakit.tasks.TaskStatus

Hosts implement `TaskExecutor` (`kind` plus `execute(...)`); the manager
drives it with a `TaskExecutionContext` and records a `TaskExecutionResult`.
`TaskRecord` carries status (`TaskStatus`), progress, and bounded output;
`TERMINAL_STATUSES` lists the finished states.

## Process helper

::: novakit.tasks.run_process
::: novakit.tasks.kill_process_tree

`run_process(argv, cwd=..., env=...)` executes an explicit argv, streams
merged output, and kills the process tree on cancellation via
`kill_process_tree`.
