"""Offline tests for novakit.llm.http_provider.

Uses fake connectors, sessions and responses only; no network is touched.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from email.utils import formatdate
from typing import Any

import aiohttp
import pytest

from novakit.llm import http_provider as hp
from novakit.llm.accumulator import StreamAccumulator
from novakit.llm.framing import SSEFramer
from novakit.llm.http_provider import HttpProvider, sleep_unless_aborted
from novakit.llm.policy import StreamPolicy, TransportPolicy
from novakit.llm.provider import (
    ChatStreamEvent,
    Done,
    Error,
    RetryNotice,
    TextDelta,
)
from novakit.llm.stream_driver import StreamParser

_ResponseItem = object


class _FakeContent:
    """Minimal stand-in for ``resp.content``."""

    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._lines = list(lines or [])
        self.readline_limits: list[int | None] = []

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._generate()

    async def _generate(self) -> AsyncIterator[bytes]:
        for line in self._lines:
            yield line

    async def readline(self, max_line_length: int | None = None) -> bytes:
        """Return the next buffered line, honouring the length cap argument."""
        self.readline_limits.append(max_line_length)
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeResponse:
    """Offline replacement for ``aiohttp.ClientResponse``."""

    def __init__(
        self,
        status: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
        payload: Any = None,
        content_type: str = "application/json",
        json_error: bool = False,
        lines: list[bytes] | None = None,
    ) -> None:
        self.status = status
        self._text = text
        self.headers = headers or {}
        self._payload = payload
        self.content_type = content_type
        self._json_error = json_error
        self.content = _FakeContent(lines)
        self.released = False
        self.closed_resp = False

    async def text(self) -> str:
        return self._text

    async def release(self) -> None:
        self.released = True

    async def json(self) -> Any:
        if self._json_error:
            raise ValueError("not json")
        if self._payload is not None:
            return self._payload
        return json.loads(self._text)

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def close(self) -> None:
        self.closed_resp = True


class _FakeSession:
    """Replays a script of responses or exceptions for ``post``."""

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.posts = 0
        self.last_kwargs: dict[str, Any] = {}
        self.closed = False

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.posts += 1
        self.last_kwargs = dict(kwargs)
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, _FakeResponse)
        return item

    async def close(self) -> None:
        self.closed = True


class _HangingSession(_FakeSession):
    """A session whose ``post`` never returns, for abort-race tests."""

    def __init__(self) -> None:
        super().__init__([])

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.posts += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The post finished racing the abort; the caller ignores the result.
            return _FakeResponse(status=200, text="{}")
        raise AssertionError("unreachable")


class _FailOnCancelSession(_FakeSession):
    """A post whose teardown fails when the abort cancels it."""

    def __init__(self) -> None:
        super().__init__([])

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.posts += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise ConnectionError("socket torn down") from None
        raise AssertionError("unreachable")


class _FakeConnector:
    """Offline replacement for ``aiohttp.TCPConnector``."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _Parser(StreamParser):
    """Minimal parser; the driver behaviour is tested elsewhere."""

    def feed(
        self, event: dict[str, Any], acc: StreamAccumulator
    ) -> list[ChatStreamEvent]:
        return []

    def build_done(self, acc: StreamAccumulator) -> Done:
        return Done(content=acc.content, tool_calls=[])


class _Provider(HttpProvider):
    framer = SSEFramer()
    stream_policy = StreamPolicy(
        line_limit=None,
        max_content_chars=1000,
        max_tool_arg_chars=1000,
        swallow_cancel=True,
        cancel_keeps_content=False,
    )
    transport_policy = TransportPolicy(
        use_retry=True,
        trust_env=True,
        honor_retry_after=False,
        guard_non_json=True,
    )
    trace_label = "fake"
    timeout_seconds = 5

    def _prepare_request(
        self,
        messages: list[Any],
        model: str,
        stream: bool,
        tools: list[dict[str, Any]] | None,
        session_id: str | None,
        reasoning_effort: str | None,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        return ("http://x.test/v1", {"Authorization": "Bearer k"}, {"model": model})

    def _parse_response(self, data: dict[str, Any]) -> Done:
        return Done(content=str(data.get("text", "")), tool_calls=[])

    def _new_parser(self, model: str) -> StreamParser:
        return _Parser()

    async def count_tokens(self, text: str, model: str | None = None) -> int:
        return len(text.split())


class _NoGuardProvider(_Provider):
    transport_policy = TransportPolicy(
        use_retry=True,
        trust_env=True,
        honor_retry_after=False,
        guard_non_json=False,
    )


class _NoRetryProvider(_Provider):
    transport_policy = TransportPolicy(
        use_retry=False,
        trust_env=True,
        honor_retry_after=False,
        guard_non_json=True,
    )


class _HonorProvider(_Provider):
    transport_policy = TransportPolicy(
        use_retry=True,
        trust_env=True,
        honor_retry_after=True,
        guard_non_json=True,
    )


class _LimitedProvider(_Provider):
    stream_policy = StreamPolicy(
        line_limit=64,
        max_content_chars=1000,
        max_tool_arg_chars=1000,
        swallow_cancel=True,
        cancel_keeps_content=False,
    )


class _KeepContentProvider(_Provider):
    stream_policy = StreamPolicy(
        line_limit=None,
        max_content_chars=1000,
        max_tool_arg_chars=1000,
        swallow_cancel=True,
        cancel_keeps_content=True,
    )


class _ReraiseProvider(_Provider):
    stream_policy = StreamPolicy(
        line_limit=None,
        max_content_chars=1000,
        max_tool_arg_chars=1000,
        swallow_cancel=False,
        cancel_keeps_content=False,
    )


class _LocalProvider(_Provider):
    transport_policy = TransportPolicy(
        use_retry=False,
        trust_env=False,
        honor_retry_after=False,
        guard_non_json=True,
    )


def _ok_response(text: str = '{"text": "hi"}') -> _FakeResponse:
    return _FakeResponse(status=200, text=text)


async def _collect(
    gen: AsyncGenerator[object, None],
) -> list[object]:
    """Drain an async generator into a list."""
    return [item async for item in gen]


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fast(seconds: float, abort: asyncio.Event | None) -> bool:
        return False

    monkeypatch.setattr(hp, "sleep_unless_aborted", _fast)


def _abort_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _aborted(seconds: float, abort: asyncio.Event | None) -> bool:
        return True

    monkeypatch.setattr(hp, "sleep_unless_aborted", _aborted)


def _patch_open(
    monkeypatch: pytest.MonkeyPatch,
    provider_cls: type[_Provider],
    session: _FakeSession,
) -> _FakeConnector:
    connector = _FakeConnector()
    monkeypatch.setattr(
        provider_cls, "_open_session", lambda self: (connector, session)
    )
    return connector


class TestSleepUnlessAborted:
    @pytest.mark.asyncio
    async def test_no_abort_event_waits_full_time(self) -> None:
        assert await sleep_unless_aborted(0.01, None) is False

    @pytest.mark.asyncio
    async def test_nonpositive_delay_returns_event_state(self) -> None:
        assert await sleep_unless_aborted(0, asyncio.Event()) is False
        assert await sleep_unless_aborted(-1, asyncio.Event()) is False
        set_event = asyncio.Event()
        set_event.set()
        assert await sleep_unless_aborted(0, set_event) is True

    @pytest.mark.asyncio
    async def test_abort_during_wait_wins(self) -> None:
        abort = asyncio.Event()

        async def _setter() -> None:
            await asyncio.sleep(0.01)
            abort.set()

        setter = asyncio.ensure_future(_setter())
        assert await sleep_unless_aborted(30.0, abort) is True
        assert setter.done()

    @pytest.mark.asyncio
    async def test_quiet_wait_serves_full_delay(self) -> None:
        assert await sleep_unless_aborted(0.01, asyncio.Event()) is False


class TestSmallHelpers:
    def test_make_connector_forwards_kwargs(self) -> None:
        provider = _Provider()

        async def _run() -> None:
            connector = provider._make_connector()
            assert isinstance(connector, aiohttp.TCPConnector)
            await connector.close()

        asyncio.run(_run())

    def test_build_http_error_message(self) -> None:
        assert "empty response" in _Provider._build_http_error_message(
            "http://x", 500, "  "
        )
        assert (
            _Provider._build_http_error_message("http://x", 500, "  busy  ")
            == "HTTP 500 from http://x: busy"
        )

    def test_post_kwargs_omits_empty_headers(self) -> None:
        assert _Provider._post_kwargs({}, {"a": 1}) == {"json": {"a": 1}}
        assert _Provider._post_kwargs({"h": "v"}, {"a": 1}) == {
            "json": {"a": 1},
            "headers": {"h": "v"},
        }


class TestRetryAfterHeader:
    def _resp(self, value: str | None) -> _FakeResponse:
        headers = {} if value is None else {"retry-after": value}
        return _FakeResponse(status=503, headers=headers)

    def test_missing_header_is_none(self) -> None:
        assert _Provider._retry_after_header(self._resp(None)) is None

    def test_seconds_header(self) -> None:
        assert _Provider._retry_after_header(self._resp("2.5")) == 2.5

    def test_seconds_header_is_capped(self) -> None:
        assert _Provider._retry_after_header(self._resp("999")) == 60.0

    def test_negative_seconds_clamp_to_zero(self) -> None:
        assert _Provider._retry_after_header(self._resp("-3")) == 0.0

    def test_garbage_header_is_none(self) -> None:
        assert _Provider._retry_after_header(self._resp("soon")) is None

    def test_http_date_header(self) -> None:
        import time as _time

        value = formatdate(_time.time() + 30)
        wait = _Provider._retry_after_header(self._resp(value))
        assert wait is not None and 0.0 < wait <= 30.0

    def test_past_http_date_clamps_to_zero(self) -> None:
        assert _Provider._retry_after_header(self._resp("21 Oct 2015 07:28:00")) == 0.0

    def test_unparsable_date_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            hp, "parsedate_to_datetime", lambda raw: (_ for _ in ()).throw(TypeError())
        )
        assert _Provider._retry_after_header(self._resp("weird")) is None


class TestRetryReason:
    def test_empty_body_has_no_reason(self) -> None:
        assert _Provider._retry_reason("") == ""
        assert _Provider._retry_reason("   ") == ""

    def test_plain_text_uses_first_line(self) -> None:
        assert _Provider._retry_reason("busy\ntry later") == "busy"

    def test_long_line_is_truncated(self) -> None:
        assert _Provider._retry_reason("x" * 300) == "x" * 200

    def test_nested_error_message_is_unwrapped(self) -> None:
        body = '{"error": {"message": "Endpoint is unavailable.\\nmore"}}'
        assert _Provider._retry_reason(body) == "Endpoint is unavailable."

    def test_string_error_is_used(self) -> None:
        assert _Provider._retry_reason('{"error": "boom"}') == "boom"

    def test_top_level_message_is_used(self) -> None:
        assert _Provider._retry_reason('{"message": "hello"}') == "hello"

    def test_unshaped_json_falls_back_to_text(self) -> None:
        assert _Provider._retry_reason('{"error": {}}') == '{"error": {}}'
        assert _Provider._retry_reason("[1, 2]") == "[1, 2]"
        assert _Provider._retry_reason('{"error": {"message": 123}}') == (
            '{"error": {"message": 123}}'
        )


class TestPostWithRetry:
    @pytest.mark.asyncio
    async def test_success_uses_default_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider, session = _Provider(), _FakeSession([_ok_response()])
        out = await _collect(
            provider._post_with_retry(session, "http://x", {"h": "v"}, {"m": 1}, None)
        )
        assert out == [session._script[0]]
        assert session.last_kwargs["headers"] == {"h": "v"}
        assert session.last_kwargs["json"] == {"m": 1}
        timeout = session.last_kwargs["timeout"]
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == provider.timeout_seconds

    @pytest.mark.asyncio
    async def test_explicit_timeout_is_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        session = _FakeSession([_ok_response()])
        timeout = aiohttp.ClientTimeout(total=9)
        out = await _collect(
            provider._post_with_retry(
                session, "http://x", {}, {"m": 1}, None, timeout=timeout
            )
        )
        assert len(out) == 1
        assert session.last_kwargs["timeout"] is timeout
        assert "headers" not in session.last_kwargs

    @pytest.mark.asyncio
    async def test_abort_while_posting_yields_none(self) -> None:
        provider, abort = _Provider(), asyncio.Event()
        task = asyncio.ensure_future(
            _collect(
                provider._post_with_retry(_HangingSession(), "http://x", {}, {}, abort)
            )
        )
        await asyncio.sleep(0.02)
        abort.set()
        assert await asyncio.wait_for(task, timeout=1.0) == [None]

    @pytest.mark.asyncio
    async def test_abort_with_failing_teardown_still_yields_none(self) -> None:
        provider, abort = _Provider(), asyncio.Event()
        task = asyncio.ensure_future(
            _collect(
                provider._post_with_retry(
                    _FailOnCancelSession(), "http://x", {}, {}, abort
                )
            )
        )
        await asyncio.sleep(0.02)
        abort.set()
        assert await asyncio.wait_for(task, timeout=1.0) == [None]

    @pytest.mark.asyncio
    async def test_unset_abort_event_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        out = await _collect(
            provider._post_with_retry(
                _FakeSession([_ok_response()]), "http://x", {}, {}, asyncio.Event()
            )
        )
        assert len(out) == 1

    @pytest.mark.asyncio
    async def test_connection_error_retries_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        ok = _ok_response()
        session = _FakeSession([TimeoutError("down"), ok])
        out = await _collect(
            provider._post_with_retry(session, "http://x", {}, {}, None)
        )
        assert isinstance(out[0], RetryNotice)
        assert out[0].reason == "down"
        assert out[0].attempt == 1
        assert out[1] is ok

    @pytest.mark.asyncio
    async def test_connection_error_abort_during_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _abort_sleep(monkeypatch)
        provider = _Provider()
        out = await _collect(
            provider._post_with_retry(
                _FakeSession([TimeoutError("down")]), "http://x", {}, {}, None
            )
        )
        assert isinstance(out[0], RetryNotice)
        assert out[1] is None

    @pytest.mark.asyncio
    async def test_connection_error_without_retry_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _NoRetryProvider()
        with pytest.raises(aiohttp.ClientConnectionError):
            await _collect(
                provider._post_with_retry(
                    _FakeSession([aiohttp.ClientConnectionError("dead")]),
                    "http://x",
                    {},
                    {},
                    None,
                )
            )

    @pytest.mark.asyncio
    async def test_unexpected_post_error_reraises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        with pytest.raises(ValueError, match="weird"):
            await _collect(
                provider._post_with_retry(
                    _FakeSession([ValueError("weird")]), "http://x", {}, {}, None
                )
            )

    @pytest.mark.asyncio
    async def test_status_500_retries_then_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        ok = _ok_response()
        bad = _FakeResponse(status=500, text="busy\nlater")
        out = await _collect(
            provider._post_with_retry(_FakeSession([bad, ok]), "http://x", {}, {}, None)
        )
        assert isinstance(out[0], RetryNotice)
        assert out[0].reason == "busy"
        assert out[1] is ok
        assert bad.released is True

    @pytest.mark.asyncio
    async def test_status_500_without_retry_returns_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _NoRetryProvider()
        bad = _FakeResponse(status=500, text="busy")
        out = await _collect(
            provider._post_with_retry(_FakeSession([bad]), "http://x", {}, {}, None)
        )
        assert out == [bad]

    @pytest.mark.asyncio
    async def test_non_retryable_status_returns_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        bad = _FakeResponse(status=400, text="bad request")
        out = await _collect(
            provider._post_with_retry(_FakeSession([bad]), "http://x", {}, {}, None)
        )
        assert out == [bad]

    @pytest.mark.asyncio
    async def test_retryable_429_retries_then_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        ok = _ok_response()
        limited = _FakeResponse(
            status=429,
            text='{"error": {"message": "rate limit exceeded"}}',
            headers={"retry-after": "2"},
        )
        out = await _collect(
            provider._post_with_retry(
                _FakeSession([limited, ok]), "http://x", {}, {}, None
            )
        )
        assert isinstance(out[0], RetryNotice)
        assert out[0].reason == "rate limit exceeded"
        assert out[0].wait_seconds == 2.0
        assert out[1] is ok

    @pytest.mark.asyncio
    async def test_terminal_429_is_returned_without_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        quota = _FakeResponse(
            status=429, text='{"error": {"message": "insufficient_quota"}}'
        )
        out = await _collect(
            provider._post_with_retry(_FakeSession([quota]), "http://x", {}, {}, None)
        )
        assert out == [quota]
        assert quota.released is False

    @pytest.mark.asyncio
    async def test_honor_retry_after_overrides_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _HonorProvider()
        ok = _ok_response()
        bad = _FakeResponse(status=503, text="slow", headers={"retry-after": "4"})
        out = await _collect(
            provider._post_with_retry(_FakeSession([bad, ok]), "http://x", {}, {}, None)
        )
        assert isinstance(out[0], RetryNotice)
        assert out[0].wait_seconds == 4.0
        assert out[1] is ok

    @pytest.mark.asyncio
    async def test_ignored_retry_after_keeps_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        ok = _ok_response()
        bad = _FakeResponse(status=503, text="slow", headers={"retry-after": "4"})
        out = await _collect(
            provider._post_with_retry(_FakeSession([bad, ok]), "http://x", {}, {}, None)
        )
        assert isinstance(out[0], RetryNotice)
        assert out[0].wait_seconds == 1.0
        assert out[1] is ok

    @pytest.mark.asyncio
    async def test_status_abort_during_backoff_yields_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _abort_sleep(monkeypatch)
        provider = _Provider()
        out = await _collect(
            provider._post_with_retry(
                _FakeSession([_FakeResponse(status=500, text="busy")]),
                "http://x",
                {},
                {},
                asyncio.Event(),
            )
        )
        assert isinstance(out[0], RetryNotice)
        assert out[1] is None

    @pytest.mark.asyncio
    async def test_mixed_budgets_can_exhaust_the_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        script: list[object] = [
            _FakeResponse(status=500, text="busy"),
            _FakeResponse(status=500, text="busy"),
            _FakeResponse(status=429, text="rate limit exceeded"),
            _FakeResponse(status=429, text="rate limit exceeded"),
            _FakeResponse(status=429, text="rate limit exceeded"),
            _FakeResponse(status=429, text="rate limit exceeded"),
        ]
        with pytest.raises(RuntimeError, match="Failed after 6 attempts"):
            await _collect(
                provider._post_with_retry(
                    _FakeSession(script), "http://x", {}, {}, None
                )
            )


class TestSessionAndTimeout:
    @pytest.mark.asyncio
    async def test_open_session_returns_real_pair(self) -> None:
        provider = _Provider()
        connector, session = provider._open_session()
        assert isinstance(connector, aiohttp.TCPConnector)
        assert isinstance(session, aiohttp.ClientSession)
        await session.close()
        await connector.close()

    @pytest.mark.asyncio
    async def test_open_session_without_trust_env(self) -> None:
        provider = _LocalProvider()
        connector, session = provider._open_session()
        assert isinstance(session, aiohttp.ClientSession)
        await session.close()
        await connector.close()

    def test_stream_timeout_defaults_to_idle_only(self) -> None:
        from novakit.llm.provider import STREAM_IDLE_TIMEOUT_SECONDS

        timeout = _Provider()._stream_timeout(None)
        assert timeout.total is None
        assert timeout.sock_read == STREAM_IDLE_TIMEOUT_SECONDS
        assert timeout.sock_connect == STREAM_IDLE_TIMEOUT_SECONDS

    def test_stream_timeout_honours_explicit_total(self) -> None:
        assert _Provider()._stream_timeout(45).total == 45

    @pytest.mark.asyncio
    async def test_read_line_factory_without_limit_uses_iterator(self) -> None:
        resp = _FakeResponse(lines=[b"a\n"])
        factory = _Provider()._read_line_factory(resp)
        assert await factory() == b"a\n"

    @pytest.mark.asyncio
    async def test_read_line_factory_with_limit_uses_readline(self) -> None:
        resp = _FakeResponse(lines=[b"a\n"])
        factory = _LimitedProvider()._read_line_factory(resp)
        assert await factory() == b"a\n"
        assert resp.content.readline_limits == [64]


class TestChat:
    @pytest.mark.asyncio
    async def test_chat_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        session = _FakeSession([_ok_response()])
        connector = _patch_open(monkeypatch, _Provider, session)
        out = await provider.chat([{"role": "user"}], model="m")
        assert isinstance(out, Done) and out.content == "hi"
        assert session.closed is True
        assert connector.closed is True

    @pytest.mark.asyncio
    async def test_chat_aborted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _Provider()
        session = _HangingSession()
        _patch_open(monkeypatch, _Provider, session)
        abort = asyncio.Event()
        abort.set()
        out = await provider.chat([{"role": "user"}], abort_event=abort)
        assert isinstance(out, Done) and out.aborted is True

    @pytest.mark.asyncio
    async def test_chat_drops_retry_notices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        session = _FakeSession([_FakeResponse(status=503, text="slow"), _ok_response()])
        _patch_open(monkeypatch, _Provider, session)
        out = await provider.chat([{"role": "user"}])
        assert isinstance(out, Done) and out.content == "hi"

    @pytest.mark.asyncio
    async def test_chat_non_200_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        session = _FakeSession([_FakeResponse(status=400, text="bad")])
        _patch_open(monkeypatch, _Provider, session)
        out = await provider.chat([{"role": "user"}])
        assert isinstance(out, Error)
        assert "HTTP 400" in out.message and "bad" in out.message

    @pytest.mark.asyncio
    async def test_chat_guards_non_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        session = _FakeSession([_FakeResponse(status=200, text="<html>")])
        _patch_open(monkeypatch, _Provider, session)
        out = await provider.chat([{"role": "user"}])
        assert isinstance(out, Error) and out.message == "unexpected response from API"

    @pytest.mark.asyncio
    async def test_chat_without_guard_surfaces_decode_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _NoGuardProvider()
        session = _FakeSession([_FakeResponse(status=200, text="{oops")])
        _patch_open(monkeypatch, _NoGuardProvider, session)
        out = await provider.chat([{"role": "user"}])
        assert isinstance(out, Error)

    @pytest.mark.asyncio
    async def test_chat_transport_failure_is_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _NoRetryProvider()
        session = _FakeSession([aiohttp.ClientConnectionError("dead")])
        _patch_open(monkeypatch, _NoRetryProvider, session)
        out = await provider.chat([{"role": "user"}])
        assert isinstance(out, Error) and "dead" in out.message

    @pytest.mark.asyncio
    async def test_chat_parse_failure_is_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        provider = _Provider()
        session = _FakeSession([_ok_response()])
        _patch_open(monkeypatch, _Provider, session)

        def _boom(data: dict[str, Any]) -> Done:
            raise ValueError("parse boom")

        monkeypatch.setattr(provider, "_parse_response", _boom)
        out = await provider.chat([{"role": "user"}])
        assert isinstance(out, Error) and "parse boom" in out.message


async def _drive_ok(**kwargs: Any) -> AsyncGenerator[ChatStreamEvent, None]:
    yield TextDelta(content="hi")
    acc = kwargs["accumulator"]
    assert isinstance(acc, StreamAccumulator)
    yield Done(content=acc.content, tool_calls=[])


async def _drive_cancel(**kwargs: Any) -> AsyncGenerator[ChatStreamEvent, None]:
    if False:
        yield
    raise asyncio.CancelledError


async def _drive_cancel_after_text(
    **kwargs: Any,
) -> AsyncGenerator[ChatStreamEvent, None]:
    acc = kwargs["accumulator"]
    assert isinstance(acc, StreamAccumulator)
    acc.add_text("part")
    raise asyncio.CancelledError
    yield TextDelta(content="never")


async def _drive_boom(**kwargs: Any) -> AsyncGenerator[ChatStreamEvent, None]:
    yield TextDelta(content="x")
    raise ValueError("boom")


class TestChatStream:
    @pytest.mark.asyncio
    async def test_stream_forwards_notices_then_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        monkeypatch.setattr(hp, "drive_stream", _drive_ok)
        provider = _Provider()
        session = _FakeSession([_FakeResponse(status=503, text="slow"), _ok_response()])
        connector = _patch_open(monkeypatch, _Provider, session)
        out = await _collect(provider.chat_stream([{"role": "user"}]))
        assert isinstance(out[0], RetryNotice)
        assert isinstance(out[1], TextDelta)
        assert isinstance(out[2], Done)
        assert session.closed is True
        assert connector.closed is True

    @pytest.mark.asyncio
    async def test_stream_aborted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _Provider()
        _patch_open(monkeypatch, _Provider, _HangingSession())
        abort = asyncio.Event()
        abort.set()
        out = await _collect(
            provider.chat_stream([{"role": "user"}], abort_event=abort)
        )
        assert len(out) == 1
        assert isinstance(out[0], Done) and out[0].aborted is True

    @pytest.mark.asyncio
    async def test_stream_non_200_is_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        session = _FakeSession([_FakeResponse(status=500, text="down")])
        _patch_open(monkeypatch, _NoRetryProvider, session)
        out = await _collect(_NoRetryProvider().chat_stream([{"role": "user"}]))
        assert len(out) == 1
        assert isinstance(out[0], Error) and "HTTP 500" in out[0].message

    @pytest.mark.asyncio
    async def test_stream_cancel_drops_content_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        monkeypatch.setattr(hp, "drive_stream", _drive_cancel_after_text)
        provider = _Provider()
        session = _FakeSession([_ok_response()])
        _patch_open(monkeypatch, _Provider, session)
        out = await _collect(provider.chat_stream([{"role": "user"}]))
        assert len(out) == 1
        assert isinstance(out[0], Done)
        assert out[0].aborted is True and out[0].content == ""

    @pytest.mark.asyncio
    async def test_stream_cancel_keeps_content_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        monkeypatch.setattr(hp, "drive_stream", _drive_cancel_after_text)
        provider = _KeepContentProvider()
        session = _FakeSession([_ok_response()])
        _patch_open(monkeypatch, _KeepContentProvider, session)
        out = await _collect(provider.chat_stream([{"role": "user"}]))
        assert len(out) == 1
        assert isinstance(out[0], Done)
        assert out[0].aborted is True and out[0].content == "part"

    @pytest.mark.asyncio
    async def test_stream_cancel_reraises_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        monkeypatch.setattr(hp, "drive_stream", _drive_cancel)
        provider = _ReraiseProvider()
        session = _FakeSession([_ok_response()])
        _patch_open(monkeypatch, _ReraiseProvider, session)
        with pytest.raises(asyncio.CancelledError):
            await _collect(provider.chat_stream([{"role": "user"}]))
        assert session.closed is True

    @pytest.mark.asyncio
    async def test_stream_exception_becomes_parser_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _no_sleep(monkeypatch)
        monkeypatch.setattr(hp, "drive_stream", _drive_boom)
        provider = _Provider()
        session = _FakeSession([_ok_response()])
        _patch_open(monkeypatch, _Provider, session)
        out = await _collect(provider.chat_stream([{"role": "user"}]))
        assert isinstance(out[0], TextDelta)
        assert isinstance(out[1], Error) and out[1].message == "boom"
