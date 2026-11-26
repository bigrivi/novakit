"""Portable in-process task lifecycle, quotas, and bounded output."""

from novakit.tasks.manager import (
    DEFAULT_FOREGROUND_WAIT_SECONDS,
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_MAX_FOREGROUND_CONCURRENT,
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_MAX_PER_SESSION,
    DEFAULT_MAX_RETAINED_TASKS,
    DEFAULT_RETENTION_SECONDS,
    BackgroundTaskManager,
    TaskLimitError,
    resolve_foreground_wait,
)
from novakit.tasks.models import (
    TERMINAL_STATUSES,
    TaskExecutionContext,
    TaskExecutionResult,
    TaskExecutor,
    TaskRecord,
    TaskStatus,
)
from novakit.tasks.process import kill_process_tree, run_process

__all__ = [
    "DEFAULT_FOREGROUND_WAIT_SECONDS",
    "DEFAULT_MAX_CONCURRENT",
    "DEFAULT_MAX_FOREGROUND_CONCURRENT",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MAX_PER_SESSION",
    "DEFAULT_MAX_RETAINED_TASKS",
    "DEFAULT_RETENTION_SECONDS",
    "TERMINAL_STATUSES",
    "BackgroundTaskManager",
    "TaskExecutionContext",
    "TaskExecutionResult",
    "TaskExecutor",
    "TaskLimitError",
    "TaskRecord",
    "TaskStatus",
    "kill_process_tree",
    "resolve_foreground_wait",
    "run_process",
]
