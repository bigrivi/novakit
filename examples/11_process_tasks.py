"""11 Local process tasks: explicit argv/cwd/env, merged output, tree kill.

What you'll learn:
- Start a child process with explicit ``argv``/``cwd``/``environment``
- How stdout and stderr stream merged into one task context
- How a non-zero exit becomes a failed result
- How timeout/cancel kills the whole process tree

Run:
    python examples/11_process_tasks.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

from novakit.tasks import TaskExecutionResult, run_process
from novakit.tasks.process import kill_process_tree


class CollectingContext:
    """Minimal ``TaskExecutionContext`` keeping output, progress, metadata."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.progress: list[tuple[float | None, str | None]] = []
        self.metadata: dict[str, object] = {}

    def write_output(self, text: str) -> None:
        """Append one streamed chunk."""
        self.chunks.append(text)

    def update_progress(
        self, progress: float | None, message: str | None = None
    ) -> None:
        """Record one progress update."""
        self.progress.append((progress, message))

    def set_metadata(self, key: str, value: object) -> None:
        """Record one metadata entry."""
        self.metadata[key] = value

    @property
    def output(self) -> str:
        """All streamed output concatenated."""
        return "".join(self.chunks)


def child_env(extra: dict[str, str]) -> dict[str, str]:
    """Minimal explicit child environment: PATH plus demo values only."""
    env: dict[str, str] = {}
    path = os.environ.get("PATH")
    if path:
        env["PATH"] = path
    if sys.platform == "win32":
        system_root = os.environ.get("SYSTEMROOT")
        if system_root:
            env["SYSTEMROOT"] = system_root
    return {**env, **extra}


async def show_merged_output(tmp: str) -> None:
    """Stream merged stdout/stderr from an explicit argv, cwd, and env."""
    print("- merged output: stdout + stderr flow into one stream")
    context = CollectingContext()
    argv = [
        sys.executable,
        "-c",
        "import os; print('STDOUT-LINE'); "
        "import sys; print('STDERR-LINE', file=sys.stderr); "
        "print('ENV:' + os.environ.get('NOVAKIT_EXAMPLE', ''))",
    ]
    print(f"  argv={argv[1:]} cwd={tmp} NOVAKIT_EXAMPLE=hello")
    result = await run_process(
        argv,
        cwd=tmp,
        context=context,
        environment=child_env({"NOVAKIT_EXAMPLE": "hello"}),
    )
    print(f"  success={result.success} exit_code={result.exit_code}")
    print(f"  output={context.output.strip().splitlines()}")
    print("  (both STDOUT-LINE and STDERR-LINE appear: stderr is merged)")


async def show_nonzero_exit(tmp: str) -> None:
    """Report a non-zero exit as a failed result."""
    print("- exit code: non-zero exit becomes a failed result")
    result: TaskExecutionResult = await run_process(
        [sys.executable, "-c", "raise SystemExit(3)"],
        cwd=tmp,
        context=CollectingContext(),
        environment=child_env({}),
    )
    print(
        f"  success={result.success} exit_code={result.exit_code} error={result.error!r}"
    )


async def show_timeout_kills_tree(tmp: str) -> None:
    """Time out a sleeping child; cancellation kills its process tree."""
    print("- timeout: cancelling the wait kills the whole process tree")
    context = CollectingContext()
    try:
        await asyncio.wait_for(
            run_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cwd=tmp,
                context=context,
                environment=child_env({}),
            ),
            timeout=2,
        )
        print("  sleeping child: completed before the timeout")
    except TimeoutError:
        print("  sleeping child hit the 2s timeout; asyncio cancelled the wait")
        print("  (run_process kills the child tree on cancellation)")


async def show_direct_tree_kill() -> None:
    """Kill a spawned sleeper through ``kill_process_tree`` and reap it."""
    print("- tree kill: kill_process_tree reaps a spawned sleeper directly")
    if sys.platform == "win32":
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)"
        )
    else:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            start_new_session=True,
        )
    print(f"  spawned pid={process.pid} returncode={process.returncode}")
    if process.pid is not None:
        kill_process_tree(process.pid)
    await asyncio.wait_for(process.wait(), timeout=5)
    print(f"  returncode after kill: {process.returncode}")


async def main() -> None:
    """Demo local process tasks: configure, run, then observe the results."""
    print("== 11 process tasks ==")
    # The temp directory serves as cwd and child_env builds the explicit
    # environment; each show_* runs its step and prints the outcome.
    with tempfile.TemporaryDirectory() as tmp:
        await show_merged_output(tmp)
        await show_nonzero_exit(tmp)
        await show_timeout_kills_tree(tmp)
    await show_direct_tree_kill()
    print("11 process tasks: done")


if __name__ == "__main__":
    asyncio.run(main())
