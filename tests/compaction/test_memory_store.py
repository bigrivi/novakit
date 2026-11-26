"""MemoryCompactionStore tests: persistence behind the compactor."""

from __future__ import annotations

import pytest

from novakit.compaction.store import CompactionStore, MemoryCompactionStore
from novakit.session.memory import InMemorySession


async def make_store() -> tuple[InMemorySession, MemoryCompactionStore, str]:
    sessions = InMemorySession()
    session = await sessions.create_session(first_message="hi")
    return sessions, MemoryCompactionStore(sessions), session.id


def test_store_is_structurally_complete() -> None:
    for name in (
        "update_message_content",
        "add_summary",
        "mark_messages_compacted_by_ids",
        "update_session_compacted_at",
        "get_messages",
    ):
        assert callable(getattr(CompactionStore, name)), name
        assert callable(getattr(MemoryCompactionStore, name)), name


@pytest.mark.asyncio
async def test_add_summary_backdates_and_restores_current() -> None:
    sessions, store, session_id = await make_store()
    await sessions.add_message(role="user", content="hello")
    current_before = sessions.get_current_session()

    summary = await store.add_summary(session_id, "folded", 500)

    assert summary.summary == 1
    assert summary.content == "folded"
    assert summary.time_created == 500
    assert sessions.get_current_session() is current_before
    visible = await store.get_messages(session_id)
    assert [m.id for m in visible] == [
        m.id for m in await sessions.get_messages(session_id)
    ]


@pytest.mark.asyncio
async def test_add_summary_unknown_session_raises() -> None:
    _, store, _ = await make_store()
    with pytest.raises(KeyError):
        await store.add_summary("missing", "folded", 1)


@pytest.mark.asyncio
async def test_update_message_content_mirrors_data() -> None:
    sessions, store, _ = await make_store()
    message = await sessions.add_message(role="tool", content="raw-output")

    await store.update_message_content(message.id, "trimmed")

    assert message.content == "trimmed"
    assert message.data == "trimmed"
    assert await sessions.update_message_content("nope", "x") is False


@pytest.mark.asyncio
async def test_mark_compacted_hides_from_reads() -> None:
    sessions, store, session_id = await make_store()
    first = await sessions.add_message(role="user", content="one")
    second = await sessions.add_message(role="user", content="two")

    assert await sessions.mark_messages_compacted_by_ids(session_id, []) == 0
    assert await sessions.mark_messages_compacted_by_ids(session_id, [first.id]) == 1

    visible = await store.get_messages(session_id)
    assert [m.id for m in visible] == [second.id]


@pytest.mark.asyncio
async def test_update_session_compacted_at() -> None:
    sessions, store, session_id = await make_store()
    await store.update_session_compacted_at(session_id, 1234)
    assert sessions.get_current_session() is not None
    assert sessions.get_current_session().compacted_at == 1234  # type: ignore[union-attr]
    assert await sessions.update_session_compacted_at("missing", 1) is False


@pytest.mark.asyncio
async def test_add_message_summary_and_time_created_params() -> None:
    sessions, _, _ = await make_store()
    message = await sessions.add_message(
        role="assistant", content="folded", summary=True, time_created=42
    )
    assert message.summary == 1
    assert message.time_created == 42
    plain = await sessions.add_message(role="user", content="hi")
    assert plain.summary == 0
    assert plain.time_created > 42
