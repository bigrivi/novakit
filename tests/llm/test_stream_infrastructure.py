"""Offline tests for stream_driver, stream_read and stream_trace."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from aiohttp import SocketTimeoutError

from novakit.llm import stream_trace as trace_module
from novakit.llm.accumulator import RunawayOutput, StreamAccumulator
from novakit.llm.framing import SSEFramer
from novakit.llm.provider import (
    ChatStreamEvent,
    Done,
    Error,
    TextDelta,
)
from novakit.llm.stream_driver import (
    StreamParser,
    drive_stream,
    no_answer_error,
    output_limit_error,
)
from novakit.llm.stream_read import (
    StreamAborted,
    StreamPoller,
    StreamTimeout,
    _settle,
)
from novakit.llm.stream_trace import StreamTrace

POLL = 0.01


def _trace(name: str = "test") -> StreamTrace:
    return StreamTrace(name, "m")


def _accumulator() -> StreamAccumulator:
    return StreamAccumulator("m", 1000, 1000)


def _reader(
    script: list[bytes | BaseException],
) -> Callable[[], Coroutine[object, object, bytes]]:
    def _factory() -> Coroutine[object, object, bytes]:
        async def _one() -> bytes:
            item = script.pop(0)
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, bytes)
            return item

        return _one()

    return _factory


class _TextParser(StreamParser):
    """Echoes ``{"text": ...}`` events; ``{"end": true}`` finishes the turn."""

    def feed(
        self, event: dict[str, Any], acc: StreamAccumulator
    ) -> list[ChatStreamEvent]:
        if event.get("end"):
            self.finished = True
            return []
        acc.add_text(event.get("text", ""))
        return [TextDelta(content=event.get("text", ""))]

    def build_done(self, acc: StreamAccumulator) -> Done:
        return Done(content=acc.content, tool_calls=[])


class _StoppedParser(_TextParser):
    """Yields its own terminal event, then stops the driver."""

    def feed(
        self, event: dict[str, Any], acc: StreamAccumulator
    ) -> list[ChatStreamEvent]:
        self.stopped = True
        return [Error(message="parser stopped")]


class _RunawayParser(_TextParser):
    def feed(
        self, event: dict[str, Any], acc: StreamAccumulator
    ) -> list[ChatStreamEvent]:
        raise RunawayOutput("ceiling hit")


class _EofErrorParser(_TextParser):
    """A Responses-style parser: EOF without a terminal event is a failure."""

    def on_eof(self, acc: StreamAccumulator) -> list[ChatStreamEvent]:
        return [Error(message="truncated")]  # type: ignore[return-value]


class _FakeResp:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TestErrorHelpers:
    def test_output_limit_error_names_reason(self) -> None:
        assert "length" in output_limit_error("length").message

    def test_no_answer_error_names_reason(self) -> None:
        err = no_answer_error("end_turn")
        assert "end_turn" in err.message


class TestParserDefaults:
    def test_default_on_eof_builds_done(self) -> None:
        parser = _TextParser()
        acc = _accumulator()
        acc.add_text("hi")
        out = list(parser.on_eof(acc))
        assert len(out) == 1
        assert isinstance(out[0], Done) and out[0].content == "hi"

    def test_default_on_exception_reports_text(self) -> None:
        parser = _TextParser()
        out = parser.on_exception(ValueError("boom"), _accumulator())
        assert isinstance(out, Error) and out.message == "boom"

    def test_parser_starts_unfinished(self) -> None:
        parser = _TextParser()
        assert parser.finished is False
        assert parser.stopped is False


class TestDriveStream:
    @pytest.mark.asyncio
    async def test_abort_yields_aborted_done_and_closes(self) -> None:
        abort = asyncio.Event()
        abort.set()
        resp = _FakeResp()
        out = [
            item
            async for item in drive_stream(
                resp=resp,  # type: ignore[arg-type]
                read_line=_reader([b"data: {}\n"]),
                abort_event=abort,
                trace=_trace(),
                accumulator=_accumulator(),
                parser=_TextParser(),
                framer=SSEFramer(),
            )
        ]
        assert len(out) == 1
        assert isinstance(out[0], Done) and out[0].aborted is True
        assert resp.closed is True

    @pytest.mark.asyncio
    async def test_empty_line_ends_peer_closed(self) -> None:
        out = [
            item
            async for item in drive_stream(
                resp=_FakeResp(),  # type: ignore[arg-type]
                read_line=_reader([b""]),
                abort_event=None,
                trace=_trace(),
                accumulator=_accumulator(),
                parser=_TextParser(),
                framer=SSEFramer(),
            )
        ]
        assert len(out) == 1
        assert isinstance(out[0], Done) and out[0].content == ""

    @pytest.mark.asyncio
    async def test_stop_iteration_ends_stream(self) -> None:
        out = [
            item
            async for item in drive_stream(
                resp=_FakeResp(),  # type: ignore[arg-type]
                read_line=_reader([StopAsyncIteration()]),
                abort_event=None,
                trace=_trace(),
                accumulator=_accumulator(),
                parser=_TextParser(),
                framer=SSEFramer(),
            )
        ]
        assert len(out) == 1 and isinstance(out[0], Done)

    @pytest.mark.asyncio
    async def test_skippable_lines_are_ignored(self) -> None:
        out = [
            item
            async for item in drive_stream(
                resp=_FakeResp(),  # type: ignore[arg-type]
                read_line=_reader([b"event: ping\n", b"\n", b""]),
                abort_event=None,
                trace=_trace(),
                accumulator=_accumulator(),
                parser=_TextParser(),
                framer=SSEFramer(),
            )
        ]
        assert len(out) == 1 and isinstance(out[0], Done)

    @pytest.mark.asyncio
    async def test_events_then_finished_builds_done(self) -> None:
        out = [
            item
            async for item in drive_stream(
                resp=_FakeResp(),  # type: ignore[arg-type]
                read_line=_reader(
                    [b'data: {"text": "hi"}\n', b'data: {"end": true}\n']
                ),
                abort_event=None,
                trace=_trace(),
                accumulator=_accumulator(),
                parser=_TextParser(),
                framer=SSEFramer(),
            )
        ]
        assert [type(item) for item in out] == [TextDelta, Done]
        assert isinstance(out[1], Done) and out[1].content == "hi"

    @pytest.mark.asyncio
    async def test_stopped_parser_ends_without_terminal(self) -> None:
        out = [
            item
            async for item in drive_stream(
                resp=_FakeResp(),  # type: ignore[arg-type]
                read_line=_reader([b'data: {"text": "x"}\n']),
                abort_event=None,
                trace=_trace(),
                accumulator=_accumulator(),
                parser=_StoppedParser(),
                framer=SSEFramer(),
            )
        ]
        assert len(out) == 1 and isinstance(out[0], Error)

    @pytest.mark.asyncio
    async def test_runaway_closes_response_and_errors(self) -> None:
        resp = _FakeResp()
        out = [
            item
            async for item in drive_stream(
                resp=resp,  # type: ignore[arg-type]
                read_line=_reader([b'data: {"text": "x"}\n']),
                abort_event=None,
                trace=_trace(),
                accumulator=_accumulator(),
                parser=_RunawayParser(),
                framer=SSEFramer(),
            )
        ]
        assert len(out) == 1
        assert isinstance(out[0], Error) and "ceiling" in out[0].message
        assert resp.closed is True

    @pytest.mark.asyncio
    async def test_eof_without_finish_uses_on_eof(self) -> None:
        out = [
            item
            async for item in drive_stream(
                resp=_FakeResp(),  # type: ignore[arg-type]
                read_line=_reader([b""]),
                abort_event=None,
                trace=_trace(),
                accumulator=_accumulator(),
                parser=_EofErrorParser(),
                framer=SSEFramer(),
            )
        ]
        assert len(out) == 1
        assert isinstance(out[0], Error) and out[0].message == "truncated"

    @pytest.mark.asyncio
    async def test_read_failure_propagates_to_caller(self) -> None:
        with pytest.raises(ValueError, match="socket died"):
            async for _ in drive_stream(
                resp=_FakeResp(),  # type: ignore[arg-type]
                read_line=_reader([ValueError("socket died")]),
                abort_event=None,
                trace=_trace(),
                accumulator=_accumulator(),
                parser=_TextParser(),
                framer=SSEFramer(),
            ):
                pass


class _Counter:
    """A read_line factory counting its invocations."""

    def __init__(
        self, behaviour: Callable[[], Coroutine[object, object, bytes]]
    ) -> None:
        self.calls = 0
        self._behaviour = behaviour

    def __call__(self) -> Coroutine[object, object, bytes]:
        self.calls += 1
        return self._behaviour()


class TestPollerQuietPeer:
    @pytest.mark.asyncio
    async def test_quiet_peer_polls_without_restarting_read(self) -> None:
        reads = _Counter(lambda: asyncio.Event().wait())  # type: ignore[return-value]
        poller = StreamPoller(reads, None, _trace(), poll_seconds=POLL)
        task = asyncio.ensure_future(poller.next_line())
        await asyncio.sleep(POLL * 8)
        assert poller.pending is True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert reads.calls == 1
        assert poller.pending is False

    @pytest.mark.asyncio
    async def test_idle_polls_then_line(self) -> None:
        async def _slow() -> bytes:
            await asyncio.sleep(POLL * 4)
            return b"late\n"

        poller = StreamPoller(_slow, None, _trace(), poll_seconds=POLL)
        assert await poller.next_line() == b"late\n"

    @pytest.mark.asyncio
    async def test_lines_in_order_then_eof(self) -> None:
        script: list[bytes | BaseException] = [b"one\n", b"two\n", b""]
        poller = StreamPoller(_reader(script), None, _trace(), poll_seconds=POLL)
        assert await poller.next_line() == b"one\n"
        assert await poller.next_line() == b"two\n"
        assert not await poller.next_line()
        assert poller.pending is False

    @pytest.mark.asyncio
    async def test_stop_async_iteration_passes_through(self) -> None:
        poller = StreamPoller(
            _reader([StopAsyncIteration()]), None, _trace(), poll_seconds=POLL
        )
        with pytest.raises(StopAsyncIteration):
            await poller.next_line()
        assert poller.pending is False


class TestPollerFailures:
    @pytest.mark.asyncio
    async def test_instant_failure_propagates_without_respins(self) -> None:
        async def _fail() -> bytes:
            raise TimeoutError

        reads = _Counter(_fail)
        poller = StreamPoller(reads, None, _trace(), poll_seconds=POLL)
        with pytest.raises(StreamTimeout):
            await poller.next_line()
        assert reads.calls == 1
        assert poller.pending is False

    @pytest.mark.asyncio
    async def test_total_timeout_message_and_cause(self) -> None:
        async def _fail() -> bytes:
            raise TimeoutError

        poller = StreamPoller(_Counter(_fail), None, _trace(), poll_seconds=POLL)
        with pytest.raises(StreamTimeout) as caught:
            await poller.next_line()
        assert "total timeout" in str(caught.value)
        assert isinstance(caught.value.__cause__, asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_socket_timeout_is_described_differently(self) -> None:
        async def _idle() -> bytes:
            raise SocketTimeoutError("Timeout on reading data from socket")

        poller = StreamPoller(_Counter(_idle), None, _trace(), poll_seconds=POLL)
        with pytest.raises(StreamTimeout) as caught:
            await poller.next_line()
        assert "socket read timeout" in str(caught.value)
        assert isinstance(caught.value.__cause__, SocketTimeoutError)

    @pytest.mark.asyncio
    async def test_externally_cancelled_read_is_a_timeout(self) -> None:
        async def _cancelled() -> bytes:
            raise asyncio.CancelledError

        poller = StreamPoller(_Counter(_cancelled), None, _trace(), poll_seconds=POLL)
        with pytest.raises(StreamTimeout) as caught:
            await poller.next_line()
        assert "cancelled" in str(caught.value)

    @pytest.mark.asyncio
    async def test_own_cancellation_reraises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _cancelled() -> bytes:
            raise asyncio.CancelledError

        class _CancellingTask:
            def cancelling(self) -> int:
                return 1

        task: asyncio.Task[bytes] = asyncio.ensure_future(_cancelled())
        await asyncio.sleep(0)
        monkeypatch.setattr(
            "novakit.llm.stream_read.asyncio.current_task",
            lambda: _CancellingTask(),
        )
        poller = StreamPoller(_reader([]), None, _trace(), poll_seconds=POLL)
        with pytest.raises(asyncio.CancelledError):
            poller._read_outcome(task)

    @pytest.mark.asyncio
    async def test_plain_read_error_propagates(self) -> None:
        async def _fail() -> bytes:
            raise ValueError("boom")

        poller = StreamPoller(_Counter(_fail), None, _trace(), poll_seconds=POLL)
        with pytest.raises(ValueError, match="boom"):
            await poller.next_line()
        assert poller.pending is False


class TestPollerAbort:
    @pytest.mark.asyncio
    async def test_abort_while_pending_raises(self) -> None:
        abort = asyncio.Event()
        reads = _Counter(lambda: asyncio.Event().wait())  # type: ignore[return-value]
        poller = StreamPoller(reads, abort, _trace(), poll_seconds=POLL)
        task = asyncio.ensure_future(poller.next_line())
        await asyncio.sleep(POLL * 3)
        assert poller.pending is True
        abort.set()
        with pytest.raises(StreamAborted):
            await asyncio.wait_for(task, timeout=1.0)
        assert poller.pending is False
        assert reads.calls == 1

    @pytest.mark.asyncio
    async def test_preset_abort_starts_nothing(self) -> None:
        abort = asyncio.Event()
        abort.set()
        reads = _Counter(lambda: asyncio.Event().wait())  # type: ignore[return-value]
        poller = StreamPoller(reads, abort, _trace(), poll_seconds=POLL)
        with pytest.raises(StreamAborted):
            await poller.next_line()
        assert reads.calls == 0

    @pytest.mark.asyncio
    async def test_no_abort_event_means_no_abort(self) -> None:
        poller = StreamPoller(_reader([b"line\n"]), None, _trace(), poll_seconds=POLL)
        assert await poller.next_line() == b"line\n"

    @pytest.mark.asyncio
    async def test_abort_beats_long_poll_interval(self) -> None:
        abort = asyncio.Event()
        reads = _Counter(lambda: asyncio.Event().wait())  # type: ignore[return-value]
        poller = StreamPoller(reads, abort, _trace(), poll_seconds=5.0)
        task = asyncio.ensure_future(poller.next_line())
        await asyncio.sleep(0.02)
        started = asyncio.get_running_loop().time()
        abort.set()
        with pytest.raises(StreamAborted):
            await asyncio.wait_for(task, timeout=1.0)
        assert asyncio.get_running_loop().time() - started < 0.5

    @pytest.mark.asyncio
    async def test_outer_cancel_cancels_read_and_leaves_nothing(self) -> None:
        reads = _Counter(lambda: asyncio.Event().wait())  # type: ignore[return-value]
        poller = StreamPoller(reads, None, _trace(), poll_seconds=POLL)
        before = asyncio.all_tasks()
        task = asyncio.ensure_future(poller.next_line())
        await asyncio.sleep(POLL * 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert asyncio.all_tasks() - before == set()
        assert poller.pending is False


class TestPollerCleanup:
    @pytest.mark.asyncio
    async def test_aclose_settles_finished_read(self) -> None:
        async def _ok() -> bytes:
            return b"x"

        task: asyncio.Task[bytes] = asyncio.ensure_future(_ok())
        await asyncio.sleep(0)
        poller = StreamPoller(_reader([]), None, _trace())
        poller._read_task = task
        await poller.aclose()
        assert poller.pending is False

    @pytest.mark.asyncio
    async def test_aclose_is_safe_when_idle(self) -> None:
        poller = StreamPoller(_reader([]), None, _trace())
        await poller.aclose()
        assert poller.pending is False

    @pytest.mark.asyncio
    async def test_aclose_cancels_pending_read(self) -> None:
        started = asyncio.Event()
        finished = asyncio.Event()

        async def _read() -> bytes:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()
            return b""

        poller = StreamPoller(_read, None, _trace(), poll_seconds=POLL)
        task = asyncio.ensure_future(poller.next_line())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert poller.pending is False

    @pytest.mark.asyncio
    async def test_aclose_logs_read_error_while_cancelling(self) -> None:
        async def _flaky() -> bytes:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ValueError("cleanup blew up") from None
            return b""

        poller = StreamPoller(_flaky, None, _trace(), poll_seconds=POLL)
        task = asyncio.ensure_future(poller.next_line())
        await asyncio.sleep(POLL * 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert poller.pending is False

    @pytest.mark.asyncio
    async def test_discard_handles_none_done_and_pending(self) -> None:
        poller = StreamPoller(_reader([]), None, _trace())
        poller._discard(None)

        async def _ok() -> bool:
            return True

        done_task: asyncio.Task[bool] = asyncio.ensure_future(_ok())
        await asyncio.sleep(0)
        poller._discard(done_task)

        pending_task: asyncio.Task[bool] = asyncio.ensure_future(
            asyncio.Event().wait()  # type: ignore[arg-type]
        )
        poller._discard(pending_task)
        await asyncio.sleep(0)
        assert pending_task.done()
        pending_task.cancel()
        try:
            await pending_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_settle_covers_all_task_states(self) -> None:
        pending: asyncio.Task[bytes] = asyncio.ensure_future(asyncio.Event().wait())  # type: ignore[arg-type]
        _settle(pending)
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
        _settle(pending)

        async def _fail() -> bytes:
            raise ValueError("gone")

        failed: asyncio.Task[bytes] = asyncio.ensure_future(_fail())
        await asyncio.sleep(0)
        _settle(failed)

    @pytest.mark.asyncio
    async def test_settle_never_raises_on_broken_task(self) -> None:
        class _BrokenTask:
            def done(self) -> bool:
                return True

            def cancelled(self) -> bool:
                return False

            def exception(self) -> None:
                raise ValueError("cannot retrieve")

        _settle(_BrokenTask())  # type: ignore[arg-type]


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(trace_module.time, "monotonic", lambda: fake.now)
    return fake


def _messages(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records]


class TestTraceAccounting:
    def test_reads_counted_without_logging_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        trace = StreamTrace("p", "m")
        with caplog.at_level(logging.INFO, logger=trace_module.__name__):
            trace.opened()
            for i in range(3):
                trace.line(b'data: {"i":%d}\n' % i)
        assert trace.reads == 3
        assert [m for m in _messages(caplog.records) if "data:" in m] == []

    def test_accepts_bytes_bytearray_and_str(self) -> None:
        trace = StreamTrace("p", "m")
        trace.line(b"a\n")
        trace.line(bytearray(b"b\n"))
        trace.line("c\n")
        assert trace.reads == 3

    def test_elapsed_is_monotonic(self, clock: _Clock) -> None:
        _ = clock
        assert StreamTrace("p", "m").elapsed >= 0.0

    def test_flood_logs_nothing_per_line(
        self, caplog: pytest.LogCaptureFixture, clock: _Clock
    ) -> None:
        trace = StreamTrace("p", "m")
        with caplog.at_level(logging.INFO, logger=trace_module.__name__):
            for _ in range(200):
                trace.line(b"data: x\n")
                clock.advance(trace_module._BEAT_SECONDS / 400)
        assert trace.reads == 200
        assert caplog.records == []


class TestTraceHeartbeat:
    def test_silent_peer_reports_rising_idle(
        self, caplog: pytest.LogCaptureFixture, clock: _Clock
    ) -> None:
        trace = StreamTrace("p", "m")
        with caplog.at_level(logging.INFO, logger=trace_module.__name__):
            for _ in range(25):
                trace.idle_read()
                clock.advance(1.0)
        beats = [m for m in _messages(caplog.records) if m.startswith("Stream beat")]
        assert len(beats) == 2
        assert "reads=0" in beats[0]
        assert "idle=11" in beats[0]
        assert "idle=21" in beats[1]

    def test_flooding_peer_reports_rising_reads(
        self, caplog: pytest.LogCaptureFixture, clock: _Clock
    ) -> None:
        trace = StreamTrace("p", "m")
        with caplog.at_level(logging.INFO, logger=trace_module.__name__):
            for _ in range(2000):
                trace.line(b"data: x\n")
            clock.advance(trace_module._BEAT_SECONDS)
            trace.line(b"data: x\n")
        beats = [m for m in _messages(caplog.records) if m.startswith("Stream beat")]
        assert len(beats) == 1
        assert "reads=2001" in beats[0]

    def test_no_beat_inside_interval(
        self, caplog: pytest.LogCaptureFixture, clock: _Clock
    ) -> None:
        trace = StreamTrace("p", "m")
        with caplog.at_level(logging.INFO, logger=trace_module.__name__):
            for _ in range(10):
                trace.line(b"data: x\n")
                clock.advance(trace_module._BEAT_SECONDS / 20)
        assert not [m for m in _messages(caplog.records) if "beat" in m]


class TestTraceTermination:
    def test_end_reports_reason_counts_and_elapsed(
        self, caplog: pytest.LogCaptureFixture, clock: _Clock
    ) -> None:
        trace = StreamTrace("openai-responses", "muse")
        with caplog.at_level(logging.INFO, logger=trace_module.__name__):
            for _ in range(7):
                trace.line(b"data: x\n")
            trace.idle_read()
            clock.advance(2.5)
            trace.end("peer closed", content=42)
        assert len(caplog.records) == 1
        end = caplog.records[0].getMessage()
        assert "reason=peer closed" in end
        assert "reads=7" in end
        assert "content=42" in end

    def test_mark_names_event_with_read_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        trace = StreamTrace("p", "m")
        with caplog.at_level(logging.INFO, logger=trace_module.__name__):
            trace.line(b"data: [DONE]\n")
            trace.mark("[DONE] read past, loop continues")
        marks = [m for m in _messages(caplog.records) if m.startswith("Stream mark")]
        assert len(marks) == 1
        assert "after 1 lines" in marks[0]
