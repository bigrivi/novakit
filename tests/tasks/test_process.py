"""Direct tests for portable child-process execution."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from novakit.tasks.process import kill_process_tree, run_process


class _Context:
    def __init__(self) -> None:
        self.output: list[str] = []

    def write_output(self, text: str) -> None:
        self.output.append(text)

    def update_progress(
        self, progress: float | None, message: str | None = None
    ) -> None:
        pass

    def set_metadata(self, key: str, value: object) -> None:
        pass


@pytest.mark.asyncio
async def test_run_process_streams_output_and_returns_success(tmp_path: Path) -> None:
    context = _Context()

    result = await run_process(
        [sys.executable, "-c", "print('hello')"],
        cwd=str(tmp_path),
        context=context,
    )

    assert result.success is True
    assert result.exit_code == 0
    assert "hello" in "".join(context.output)


@pytest.mark.asyncio
async def test_run_process_reports_nonzero_exit(tmp_path: Path) -> None:
    result = await run_process(
        [sys.executable, "-c", "raise SystemExit(3)"],
        cwd=str(tmp_path),
        context=_Context(),
    )

    assert result.success is False
    assert result.exit_code == 3
    assert result.error == "Process exited with code 3"


@pytest.mark.asyncio
async def test_cancelled_process_kills_tree(tmp_path: Path) -> None:
    process = MagicMock()
    process.pid = 4321
    process.stdout = MagicMock()
    process.stdout.read = AsyncMock(side_effect=asyncio.CancelledError)
    process.wait = AsyncMock(side_effect=TimeoutError)

    with (
        patch(
            "novakit.tasks.process.asyncio.create_subprocess_exec",
            AsyncMock(return_value=process),
        ),
        patch("novakit.tasks.process.kill_process_tree") as kill,
        pytest.raises(asyncio.CancelledError),
    ):
        await run_process(
            ["sleep", "10"],
            cwd=str(tmp_path),
            context=_Context(),
        )

    kill.assert_called_once_with(4321)


def test_kill_process_tree_uses_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "getpgid", lambda pid: pid + 1)
    kill_group = MagicMock()
    monkeypatch.setattr(os, "killpg", kill_group)

    kill_process_tree(10)

    kill_group.assert_called_once_with(11, signal.SIGKILL)


def test_kill_process_tree_falls_back_to_single_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    def missing_group(pid: int) -> int:
        raise ProcessLookupError

    monkeypatch.setattr(os, "getpgid", missing_group)
    kill = MagicMock()
    monkeypatch.setattr(os, "kill", kill)

    kill_process_tree(10)

    kill.assert_called_once_with(10, signal.SIGKILL)


def test_kill_process_tree_windows_uses_taskkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0, raising=False)
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)

    kill_process_tree(10)

    run.assert_called_once_with(
        ["taskkill", "/F", "/T", "/PID", "10"],
        capture_output=True,
        creationflags=0,
    )


@pytest.mark.asyncio
async def test_run_process_windows_branch_without_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0, raising=False)
    process = MagicMock()
    process.pid = 1
    process.stdout = None
    process.wait = AsyncMock(return_value=0)
    create = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    result = await run_process(
        ["echo", "hello"],
        cwd=str(tmp_path),
        context=_Context(),
    )

    assert result.success is True
    assert create.await_args.kwargs["creationflags"] == 0
