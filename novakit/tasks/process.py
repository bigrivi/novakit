"""Shared child-process plumbing for process-backed task executors."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys

from novakit.tasks.models import TaskExecutionContext, TaskExecutionResult


def kill_process_tree(pid: int) -> None:
    """Kill a process and its children across supported platforms.

    Args:
        pid: Process identifier to terminate.
    """
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


async def run_process(
    command: list[str],
    *,
    cwd: str,
    context: TaskExecutionContext,
    environment: dict[str, str] | None = None,
) -> TaskExecutionResult:
    """Run a command and stream merged output into a task context.

    Cancellation kills the whole process tree so a cancelled task does not leave
    orphaned children holding its pipes open.

    Args:
        command: Argument vector to execute.
        cwd: Explicit working directory for the child process.
        context: Task output and progress sink.
        environment: Optional child environment; None inherits the parent.

    Returns:
        The process exit result.
    """
    if sys.platform == "win32":
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    try:
        if process.stdout is not None:
            while chunk := await process.stdout.read(8192):
                context.write_output(chunk.decode("utf-8", errors="replace"))
        return_code = await process.wait()
    except asyncio.CancelledError:
        kill_process_tree(process.pid)
        try:
            await asyncio.wait_for(process.wait(), timeout=1)
        except TimeoutError:
            pass
        raise

    return TaskExecutionResult(
        success=return_code == 0,
        exit_code=return_code,
        error=None if return_code == 0 else f"Process exited with code {return_code}",
    )


__all__ = ["kill_process_tree", "run_process"]
