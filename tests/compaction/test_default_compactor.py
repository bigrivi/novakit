"""DefaultCompactor tests: frame order, persistence, breaker, and force."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from novakit.agent.compaction import CompactionError
from novakit.agent.ports import Compactor
from novakit.compaction.compactor import DefaultCompactor, compact
from novakit.compaction.config import CompactionConfig
from novakit.compaction.store import MemoryCompactionStore
from novakit.compaction.summary import CONTINUATION_INSTRUCTION
from novakit.events import AgentEvent
from novakit.session.memory import InMemorySession

COMPACTION_DIR = Path(__file__).resolve().parents[2] / "novakit" / "compaction"


class StubResolver:
    """Deterministic token source: 1 token per 4 chars, fixed window."""

    def __init__(self, limit: int = 1000) -> None:
        self.limit = limit

    def estimate_messages(self, messages: list, model: str) -> int:
        total = 0
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content", "") or ""
            else:
                content = getattr(message, "content", "") or ""
            total += max(1, len(content) // 4)
        return total

    def context_limit(self, model: str, provider: str) -> int:
        return self.limit


def default_config() -> CompactionConfig:
    return CompactionConfig(
        output_reserve_tokens=200,
        summary_reserve_tokens=100,
        summary_keep_tokens=100,
        max_consecutive_failures=3,
    )


async def make_sessions(
    count: int = 10, content_len: int = 400
) -> tuple[InMemorySession, Any, list]:
    sessions = InMemorySession()
    session = await sessions.create_session(first_message="hi")
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        await sessions.add_message(
            role=role,
            content=f"m{index}-" + "x" * content_len,
            time_created=1000 + index * 1000,
        )
    messages = await sessions.get_messages(session.id)
    return sessions, session, messages


def make_compactor(
    sessions: InMemorySession,
    resolver: StubResolver | None = None,
    config: CompactionConfig | None = None,
) -> DefaultCompactor:
    return DefaultCompactor(
        "stub-model",
        "stub-provider",
        store=MemoryCompactionStore(sessions),
        config=config or default_config(),
        resolver=resolver or StubResolver(),
    )


class StreamStub:
    """Stand-in for stream_text_once with scripted results and delta relay."""

    def __init__(
        self, results: list[str | None], chunks: list[str] | None = None
    ) -> None:
        self.results = results
        self.chunks = chunks or []
        self.calls = 0

    async def __call__(self, llm: Any, **kwargs: Any) -> str | None:
        self.calls += 1
        on_delta = kwargs.get("on_delta")
        if on_delta is not None:
            for chunk in self.chunks:
                on_delta(chunk)
        return self.results.pop(0) if self.results else None


async def collect(
    compactor: DefaultCompactor, messages: list, session: Any
) -> tuple[list, list]:
    emitted: list = []
    yielded: list = []

    async def emit(event: Any, data: Any) -> None:
        emitted.append((event, data))

    async for event, data in compactor.maybe_compact(
        messages,
        session,
        object(),
        emit,  # type: ignore[arg-type]
    ):
        yielded.append((event, data))
    return emitted, yielded


@pytest.mark.asyncio
async def test_quiet_history_emits_only_context_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "novakit.compaction.summary.stream_text_once", StreamStub(["unused"])
    )
    sessions, session, messages = await make_sessions(count=2, content_len=40)
    compactor = make_compactor(sessions)
    emitted, yielded = await collect(compactor, messages, session)
    assert [event for event, _ in yielded] == [AgentEvent.CONTEXT_UPDATE]
    assert emitted == yielded
    assert compactor.compacted is False
    payload = yielded[0][1]
    assert payload["limit"] == 700
    assert payload["percent"] == int(payload["used"] / 700 * 100)


@pytest.mark.asyncio
async def test_compaction_frame_order_and_dual_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "novakit.compaction.summary.stream_text_once",
        StreamStub(["folded summary"], chunks=["folded ", "summary"]),
    )
    sessions, session, messages = await make_sessions()
    compactor = make_compactor(sessions)
    emitted, yielded = await collect(compactor, messages, session)

    kinds = [event for event, _ in yielded]
    assert kinds == [
        AgentEvent.CONTEXT_UPDATE,
        AgentEvent.COMPACTION_START,
        AgentEvent.COMPACTION_DELTA,
        AgentEvent.COMPACTION_DELTA,
        AgentEvent.COMPACTION_END,
        AgentEvent.CONTEXT_UPDATE,
    ]
    # Deltas stream to the caller only; everything else is emit+yield dual.
    assert emitted == [
        frame for frame in yielded if frame[0] != AgentEvent.COMPACTION_DELTA
    ]
    start = yielded[1][1]
    assert start == {
        "message_count": len(messages),
        "token_count": yielded[1][1]["token_count"],
        "reason": "threshold",
    }
    assert yielded[4][1] == start
    assert yielded[2][1] == {"delta": "folded "}
    assert yielded[5][1]["used"] < yielded[0][1]["used"]
    assert compactor.compacted is True


@pytest.mark.asyncio
async def test_persistence_and_summary_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "novakit.compaction.summary.stream_text_once", StreamStub(["did things"])
    )
    sessions, session, messages = await make_sessions()
    old_ids = {m.id for m in messages[:-1]}
    recent_min = min(m.time_created for m in messages[-1:])
    compactor = make_compactor(sessions)
    await collect(compactor, messages, session)

    visible = await sessions.get_messages(session.id)
    summary = visible[0]
    assert summary.summary == 1
    assert summary.content.startswith("[Previous conversation summary]\ndid things")
    assert summary.content.endswith(CONTINUATION_INSTRUCTION)
    assert summary.time_created == recent_min - 1
    assert not (old_ids & {m.id for m in visible})
    assert len(visible) == 2  # summary + the kept recent message
    assert session.compacted_at is not None


@pytest.mark.asyncio
async def test_orphan_tool_responses_follow_their_declarant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "novakit.compaction.summary.stream_text_once", StreamStub(["folded"])
    )
    sessions = InMemorySession()
    session = await sessions.create_session(first_message="hi")
    await sessions.add_message(role="user", content="u-" + "x" * 400, time_created=1000)
    await sessions.add_message(
        role="assistant",
        content="calling",
        tool_calls=[{"id": "tc1", "name": "read"}],
        time_created=2000,
    )
    await sessions.add_message(
        role="tool", content="small", tool_call_id="tc1", time_created=3000
    )
    await sessions.add_message(
        role="user", content="next-" + "y" * 400, time_created=4000
    )
    messages = await sessions.get_messages(session.id)
    tool_id = messages[2].id
    store = MemoryCompactionStore(sessions)

    assert (
        await compact(
            session.id,
            store,
            object(),  # type: ignore[arg-type]
            messages=messages,
            split_index=2,
            config=default_config(),
            resolver=StubResolver(),
        )
        is True
    )

    visible = await sessions.get_messages(session.id)
    assert tool_id not in {m.id for m in visible}
    assert visible[0].summary == 1


@pytest.mark.asyncio
async def test_summary_failure_raises_without_layer2_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "novakit.compaction.summary.stream_text_once", StreamStub([None, None])
    )
    sessions, session, messages = await make_sessions()
    before = [(m.id, m.content) for m in messages]
    compactor = make_compactor(sessions)

    with pytest.raises(CompactionError):
        await collect(compactor, messages, session)

    after = [(m.id, m.content) for m in await sessions.get_messages(session.id)]
    assert after == before
    assert session.compacted_at is None
    assert compactor.consecutive_failures == 1
    assert compactor.compacted is False


@pytest.mark.asyncio
async def test_breaker_blocks_auto_but_forced_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "novakit.compaction.summary.stream_text_once", StreamStub([None, None, None])
    )
    config = CompactionConfig(
        output_reserve_tokens=200,
        summary_reserve_tokens=100,
        summary_keep_tokens=100,
        max_consecutive_failures=1,
    )
    sessions, session, messages = await make_sessions()
    compactor = make_compactor(sessions, config=config)

    with pytest.raises(CompactionError):
        await collect(compactor, messages, session)
    assert compactor.consecutive_failures == 1

    _, yielded = await collect(compactor, messages, session)
    assert [event for event, _ in yielded] == [AgentEvent.CONTEXT_UPDATE]
    assert compactor.force_next is False

    compactor.force_next = True
    with pytest.raises(CompactionError):
        await collect(compactor, messages, session)
    assert compactor.consecutive_failures == 2
    assert compactor.force_next is False


@pytest.mark.asyncio
async def test_forced_success_clears_streak_and_reports_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    async def scripted(llm: Any, **kwargs: Any) -> str | None:
        calls["n"] += 1
        # The first compaction burns both the attempt and its retry.
        return None if calls["n"] <= 2 else "recovered"

    monkeypatch.setattr("novakit.compaction.summary.stream_text_once", scripted)
    sessions, session, messages = await make_sessions()
    compactor = make_compactor(sessions)
    with pytest.raises(CompactionError):
        await collect(compactor, messages, session)
    assert compactor.consecutive_failures == 1

    compactor.force_next = True
    _, yielded = await collect(compactor, messages, session)
    assert yielded[1][1]["reason"] == "overflow"
    assert compactor.consecutive_failures == 0
    assert compactor.compacted is True


@pytest.mark.asyncio
async def test_unlimited_breaker_never_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "novakit.compaction.summary.stream_text_once", StreamStub(["ok"])
    )
    unlimited = CompactionConfig(
        output_reserve_tokens=200,
        summary_reserve_tokens=100,
        summary_keep_tokens=100,
        max_consecutive_failures=0,
    )
    sessions, session, messages = await make_sessions()
    compactor = make_compactor(sessions, config=unlimited)
    compactor.consecutive_failures = 10
    _, yielded = await collect(compactor, messages, session)
    assert AgentEvent.COMPACTION_START in [event for event, _ in yielded]


@pytest.mark.asyncio
async def test_layer1_snip_persists_while_breaker_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "novakit.compaction.summary.stream_text_once", StreamStub(["unused"])
    )
    config = CompactionConfig(
        output_reserve_tokens=200,
        summary_reserve_tokens=100,
        summary_keep_tokens=100,
        snip_tool_output_token_budget=100,
        max_consecutive_failures=1,
    )
    sessions = InMemorySession()
    session = await sessions.create_session(first_message="hi")
    for index in range(10):
        await sessions.add_message(
            role="user",
            content=f"u{index}-" + "x" * 400,
            time_created=1000 + index * 1000,
        )
    await sessions.add_message(role="tool", content="v" * 3000, time_created=20000)
    messages = await sessions.get_messages(session.id)
    compactor = make_compactor(sessions, config=config)
    compactor.consecutive_failures = 1

    _, yielded = await collect(compactor, messages, session)

    assert [event for event, _ in yielded] == [AgentEvent.CONTEXT_UPDATE]
    assert "chars snipped" in messages[-1].content
    stored = await sessions.get_messages(session.id)
    snipped = [m for m in stored if "chars snipped" in m.content]
    assert snipped, "layer 1 must persist snips even while the breaker is open"


@pytest.mark.asyncio
async def test_context_emit_failure_never_aborts_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "novakit.compaction.summary.stream_text_once", StreamStub(["ok"])
    )
    sessions, session, messages = await make_sessions()
    compactor = make_compactor(sessions)
    calls = {"n": 0}
    seen: list = []

    async def flaky_emit(event: Any, data: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("pill failed")
        seen.append((event, data))

    yielded = [
        (event, data)
        async for event, data in compactor.maybe_compact(
            messages,
            session,
            object(),
            flaky_emit,  # type: ignore[arg-type]
        )
    ]
    # The first pill is lost with its failed emit (same as the reference
    # behavior); the turn still compacts instead of aborting.
    assert yielded[0][0] == AgentEvent.COMPACTION_START
    assert AgentEvent.COMPACTION_END in [event for event, _ in yielded]
    assert compactor.compacted is True


def test_matches_compactor_protocol_shape() -> None:
    import inspect

    assert issubclass(Compactor, object)
    params = list(inspect.signature(DefaultCompactor.maybe_compact).parameters)
    assert params == ["self", "messages", "session", "llm", "emit", "tools"]
    compactor = DefaultCompactor(
        "m", "p", store=MemoryCompactionStore(InMemorySession())
    )
    assert compactor.compacted is False
    assert compactor.force_next is False
    assert callable(compactor.maybe_compact)


def test_compaction_modules_stay_standalone() -> None:
    for path in sorted(COMPACTION_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(
                    name.name == "nova" or name.name.startswith("nova.")
                    for name in node.names
                ), path
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not (node.module == "nova" or node.module.startswith("nova.")), (
                    path
                )
        text = path.read_text(encoding="utf-8")
        assert "NOVA_HOME" not in text, path
        assert "os.environ" not in text and "os.getenv" not in text, path
