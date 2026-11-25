"""Offline tests for novakit.llm.policy and novakit.llm.request_hook."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from novakit.llm import request_hook as rh
from novakit.llm.policy import RetryBudget, RetryDecision, StreamPolicy, TransportPolicy
from novakit.llm.provider import RETRY_STATUS_CODES
from novakit.llm.request_hook import (
    RequestHookError,
    resolve_hook_path,
    run_request_hook,
    run_session_hook,
)


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    rh._cache.clear()
    yield
    rh._cache.clear()


def _budget(**kwargs: object) -> RetryBudget:
    params: dict[str, object] = {
        "use_retry": True,
        "retry_429": True,
        "honor_retry_after": False,
        "statuses": RETRY_STATUS_CODES,
    }
    params.update(kwargs)
    return RetryBudget(**params)  # type: ignore[arg-type]


class TestPolicies:
    def test_stream_policy_stores_switches(self) -> None:
        policy = StreamPolicy(
            line_limit=64,
            max_content_chars=100,
            max_tool_arg_chars=50,
            swallow_cancel=True,
            cancel_keeps_content=False,
        )
        assert policy.line_limit == 64
        assert policy.max_content_chars == 100
        assert policy.cancel_keeps_content is False

    def test_transport_policy_defaults(self) -> None:
        first = TransportPolicy(
            use_retry=True,
            trust_env=True,
            honor_retry_after=False,
            guard_non_json=True,
        )
        second = TransportPolicy(
            use_retry=True,
            trust_env=True,
            honor_retry_after=False,
            guard_non_json=True,
        )
        assert first.retry_429 is True
        assert first.connector_kwargs == {}
        assert first.connector_kwargs is not second.connector_kwargs

    def test_retry_decision_fields(self) -> None:
        decision = RetryDecision(wait_seconds=1.5, attempt=1, total=3)
        assert (decision.wait_seconds, decision.attempt, decision.total) == (1.5, 1, 3)


class TestRetryBudget:
    def test_total_attempts_is_longest_budget(self) -> None:
        assert _budget().total_attempts == 6
        assert _budget(retry_429_attempts=1).total_attempts == 3
        assert _budget(use_retry=False).total_attempts == 1
        assert _budget(use_retry=False, retry_429_attempts=9).total_attempts == 1

    def test_general_attempts_follows_retry_switch(self) -> None:
        assert _budget().general_attempts == 3
        assert _budget(use_retry=False).general_attempts == 1

    def test_connection_failure_disabled_is_none(self) -> None:
        assert _budget(use_retry=False).connection_failure() is None

    def test_connection_failure_backoff_then_none(self) -> None:
        budget = _budget()
        first = budget.connection_failure()
        second = budget.connection_failure()
        assert first is not None and (first.wait_seconds, first.attempt) == (1.0, 1)
        assert second is not None and (second.wait_seconds, second.attempt) == (2.0, 2)
        assert budget.connection_failure() is None

    def test_response_failure_ignores_retry_after_by_default(self) -> None:
        budget = _budget()
        decision = budget.response_failure(503, retry_after=30.0, terminal_429=False)
        assert decision is not None and decision.wait_seconds == 1.0

    def test_response_failure_honours_retry_after(self) -> None:
        budget = _budget(honor_retry_after=True)
        decision = budget.response_failure(503, retry_after=7.0, terminal_429=False)
        assert decision is not None and decision.wait_seconds == 7.0

    def test_response_failure_without_header_uses_backoff(self) -> None:
        budget = _budget(honor_retry_after=True)
        decision = budget.response_failure(503, retry_after=None, terminal_429=False)
        assert decision is not None and decision.wait_seconds == 1.0

    def test_response_failure_exhausts_general_budget(self) -> None:
        budget = _budget()
        assert (
            budget.response_failure(500, retry_after=None, terminal_429=False)
            is not None
        )
        assert (
            budget.response_failure(500, retry_after=None, terminal_429=False)
            is not None
        )
        assert (
            budget.response_failure(500, retry_after=None, terminal_429=False) is None
        )

    def test_response_failure_ignores_unknown_status(self) -> None:
        assert (
            _budget().response_failure(400, retry_after=5.0, terminal_429=False) is None
        )

    def test_rate_path_uses_header_or_schedule(self) -> None:
        budget = _budget()
        first = budget.response_failure(429, retry_after=None, terminal_429=False)
        second = budget.response_failure(429, retry_after=5.0, terminal_429=False)
        assert first is not None and first.wait_seconds == 1.0
        assert second is not None and second.wait_seconds == 5.0
        assert second.attempt == 2 and second.total == 6

    def test_terminal_429_never_retries(self) -> None:
        assert (
            _budget().response_failure(429, retry_after=1.0, terminal_429=True) is None
        )

    def test_rate_path_exhausts_attempts(self) -> None:
        budget = _budget()
        for _ in range(5):
            assert (
                budget.response_failure(429, retry_after=None, terminal_429=False)
                is not None
            )
        assert (
            budget.response_failure(429, retry_after=None, terminal_429=False) is None
        )

    def test_rate_path_disabled_is_none(self) -> None:
        assert (
            _budget(retry_429=False).response_failure(
                429, retry_after=None, terminal_429=False
            )
            is None
        )
        assert (
            _budget(use_retry=False).response_failure(
                429, retry_after=None, terminal_429=False
            )
            is None
        )

    def test_rate_total_budget_caps_long_waits(self) -> None:
        budget = _budget(rate_total_budget=3.0)
        assert (
            budget.response_failure(429, retry_after=None, terminal_429=False)
            is not None
        )
        assert (
            budget.response_failure(429, retry_after=None, terminal_429=False)
            is not None
        )
        assert (
            budget.response_failure(429, retry_after=99.0, terminal_429=False) is None
        )


def _write_hook(tmp_path: Path, name: str, body: str) -> str:
    target = tmp_path / name
    target.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(target)


ECHO_HOOK = """\
    import json, sys
    ctx = json.load(sys.stdin)
    assert ctx["session_id"] == "sess-1"
    print(json.dumps({"headers": {"x-opencode-session": "ses-from-hook"}}))
    """


class TestResolveHookPath:
    def test_empty_values_disable_hook(self, tmp_path: Path) -> None:
        assert resolve_hook_path("") is None
        assert resolve_hook_path(None) is None
        assert resolve_hook_path("   ") is None

    def test_absolute_path_is_kept(self) -> None:
        assert resolve_hook_path("/abs/z.py") == "/abs/z.py"

    def test_home_is_expanded(self) -> None:
        assert resolve_hook_path("~/hooks/z.py") == str(Path.home() / "hooks" / "z.py")

    def test_relative_path_uses_base_dir(self, tmp_path: Path) -> None:
        assert resolve_hook_path("hooks/z.py", tmp_path) == str(
            tmp_path / "hooks" / "z.py"
        )

    def test_relative_path_defaults_to_cwd(self, tmp_path: Path) -> None:
        import os

        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert resolve_hook_path("hooks/z.py") == str(tmp_path / "hooks" / "z.py")
        finally:
            os.chdir(old)


class TestRunRequestHook:
    def test_success_returns_headers(self, tmp_path: Path) -> None:
        hook = _write_hook(tmp_path, "ok.py", ECHO_HOOK)
        assert run_request_hook(hook, "sess-1") == {
            "x-opencode-session": "ses-from-hook"
        }

    def test_values_are_coerced_to_strings(self, tmp_path: Path) -> None:
        hook = _write_hook(
            tmp_path, "ok.py", 'import json; print(json.dumps({"headers": {"n": 1}}))\n'
        )
        assert run_request_hook(hook, None) == {"n": "1"}

    def test_empty_path_raises(self) -> None:
        with pytest.raises(RequestHookError):
            run_request_hook("", "s")

    def test_unfrozen_command_uses_executable(self) -> None:
        assert rh._hook_command("/h/z.py") == [sys.executable, "/h/z.py"]

    def test_frozen_command_uses_run_code_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        assert rh._hook_command("/h/z.py") == [sys.executable, "--_run-code", "/h/z.py"]

    def test_missing_interpreter_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        hook = _write_hook(tmp_path, "ok.py", ECHO_HOOK)
        monkeypatch.setattr(sys, "executable", "/nonexistent/python")
        with pytest.raises(RequestHookError, match="not found"):
            run_request_hook(hook, "s")

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _slow(*args: object, **kwargs: object) -> object:
            raise subprocess.TimeoutExpired(cmd="hook", timeout=1)

        monkeypatch.setattr(subprocess, "run", _slow)
        with pytest.raises(RequestHookError, match="timed out"):
            run_request_hook("/h/z.py", "s", timeout=1)

    def test_os_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _broken(*args: object, **kwargs: object) -> object:
            raise OSError("noexec")

        monkeypatch.setattr(subprocess, "run", _broken)
        with pytest.raises(RequestHookError, match="failed to start"):
            run_request_hook("/h/z.py", "s")

    def test_nonzero_exit_with_detail(self, tmp_path: Path) -> None:
        hook = _write_hook(
            tmp_path, "bad.py", "import sys; sys.stderr.write('nope\\n'); sys.exit(3)\n"
        )
        with pytest.raises(RequestHookError, match=r"exited 3.*nope"):
            run_request_hook(hook, "s")

    def test_nonzero_exit_without_detail(self, tmp_path: Path) -> None:
        hook = _write_hook(tmp_path, "bad.py", "import sys; sys.exit(3)\n")
        with pytest.raises(RequestHookError, match="exited 3"):
            run_request_hook(hook, "s")

    def test_empty_output_raises(self, tmp_path: Path) -> None:
        hook = _write_hook(tmp_path, "empty.py", "print('  ')\n")
        with pytest.raises(RequestHookError, match="no output"):
            run_request_hook(hook, "s")

    def test_bad_json_raises(self, tmp_path: Path) -> None:
        hook = _write_hook(tmp_path, "bad.py", 'print("not json")\n')
        with pytest.raises(RequestHookError, match="not JSON"):
            run_request_hook(hook, "s")

    def test_wrong_shape_raises(self, tmp_path: Path) -> None:
        lst = _write_hook(tmp_path, "lst.py", "import json; print(json.dumps([1]))\n")
        with pytest.raises(RequestHookError, match="headers"):
            run_request_hook(lst, "s")
        bad_headers = _write_hook(
            tmp_path, "bh.py", 'import json; print(json.dumps({"headers": [1]}))\n'
        )
        with pytest.raises(RequestHookError, match="headers"):
            run_request_hook(bad_headers, "s")

    def test_win32_creationflags_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        hook = _write_hook(tmp_path, "ok.py", ECHO_HOOK)
        assert run_request_hook(hook, "sess-1") == {
            "x-opencode-session": "ses-from-hook"
        }


class TestRunSessionHook:
    def test_empty_path_raises(self) -> None:
        with pytest.raises(RequestHookError):
            run_session_hook("", "s")

    def test_results_cached_per_session(self, tmp_path: Path) -> None:
        hook = _write_hook(
            tmp_path,
            "count.py",
            """\
            import json, sys
            ctx = json.load(sys.stdin)
            print(json.dumps({"headers": {"x-s": ctx["session_id"]}}))
            """,
        )
        first = run_session_hook(hook, "a")
        second = run_session_hook(hook, "a")
        other = run_session_hook(hook, "b")
        assert first == second == {"x-s": "a"}
        assert other == {"x-s": "b"}

    def test_cached_result_is_a_copy(self, tmp_path: Path) -> None:
        hook = _write_hook(tmp_path, "ok.py", ECHO_HOOK)
        headers = run_session_hook(hook, "sess-1")
        headers["x-opencode-session"] = "mutated"
        assert run_session_hook(hook, "sess-1")["x-opencode-session"] == "ses-from-hook"

    def test_relative_path_resolves_against_base_dir(self, tmp_path: Path) -> None:
        _write_hook(tmp_path, "rel.py", ECHO_HOOK)
        assert run_session_hook("rel.py", "sess-1", base_dir=tmp_path) == {
            "x-opencode-session": "ses-from-hook"
        }

    def test_eviction_keeps_cache_bounded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(rh, "_MAX_CACHED_SESSIONS", 2)
        hook = _write_hook(
            tmp_path,
            "plain.py",
            'import json; print(json.dumps({"headers": {"x-ok": "yes"}}))\n',
        )
        run_session_hook(hook, "a")
        run_session_hook(hook, "b")
        assert len(rh._cache) == 2
        run_session_hook(hook, "c")
        assert len(rh._cache) == 2
        assert run_session_hook(hook, "a") == {"x-ok": "yes"}
        assert len(rh._cache) == 2
