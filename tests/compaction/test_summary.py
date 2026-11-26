"""Summary unit tests: transcript rendering and the summariser call."""

from __future__ import annotations

from typing import Any

import pytest

from novakit.compaction.summary import (
    NEW_SUMMARY_ANCHOR,
    PREVIOUS_SUMMARY_ANCHOR,
    SUMMARY_MESSAGE_MAX_CHARS,
    format_for_summary,
    generate_summary,
)


def test_format_truncates_ordinary_but_keeps_summaries() -> None:
    long_text = "v" * (SUMMARY_MESSAGE_MAX_CHARS + 500)
    messages = [
        {"role": "user", "content": long_text},
        {"role": "assistant", "content": long_text, "summary": 1},
        {"role": "assistant", "tool_calls": [{"id": "c1", "name": "read"}]},
    ]
    rendered = format_for_summary(messages)
    assert f"[user]: {long_text[:SUMMARY_MESSAGE_MAX_CHARS]}" in rendered
    assert long_text[: SUMMARY_MESSAGE_MAX_CHARS + 1] not in rendered.split("\n")[0]
    assert f"[assistant]: {long_text}" in rendered
    assert "[assistant]: (tool calls)" in rendered


class StreamStub:
    """Stand-in for stream_text_once capturing prompt and deltas."""

    def __init__(self, results: list[str | None], emit_parts: bool = False) -> None:
        self.results = results
        self.emit_parts = emit_parts
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, llm: Any, **kwargs: Any) -> str | None:
        self.calls.append(kwargs)
        on_delta = kwargs.get("on_delta")
        if self.emit_parts and on_delta is not None and self.results:
            on_delta("partial")
        return self.results.pop(0) if self.results else None


@pytest.mark.asyncio
async def test_generate_summary_success_uses_new_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StreamStub(["  hello  "])
    monkeypatch.setattr("novakit.compaction.summary.stream_text_once", stub)
    result = await generate_summary("conv", object(), "m")  # type: ignore[arg-type]
    assert result == "hello"
    assert len(stub.calls) == 1
    prompt = stub.calls[0]["messages"][1].content
    assert NEW_SUMMARY_ANCHOR in prompt
    assert PREVIOUS_SUMMARY_ANCHOR not in prompt


@pytest.mark.asyncio
async def test_generate_summary_previous_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StreamStub(["ok"])
    monkeypatch.setattr("novakit.compaction.summary.stream_text_once", stub)
    await generate_summary("conv", object(), "m", has_previous_summary=True)  # type: ignore[arg-type]
    assert PREVIOUS_SUMMARY_ANCHOR in stub.calls[0]["messages"][1].content


@pytest.mark.asyncio
async def test_generate_summary_retries_once_then_gives_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StreamStub([None, None])
    monkeypatch.setattr("novakit.compaction.summary.stream_text_once", stub)
    assert await generate_summary("conv", object(), "m") is None  # type: ignore[arg-type]
    assert len(stub.calls) == 2


@pytest.mark.asyncio
async def test_generate_summary_retry_can_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = StreamStub([None, "recovered"])
    monkeypatch.setattr("novakit.compaction.summary.stream_text_once", stub)
    assert await generate_summary("conv", object(), "m") == "recovered"  # type: ignore[arg-type]
    assert len(stub.calls) == 2


@pytest.mark.asyncio
async def test_generate_summary_partial_emit_skips_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deltas: list[str] = []
    stub = StreamStub([None], emit_parts=True)
    monkeypatch.setattr("novakit.compaction.summary.stream_text_once", stub)
    result = await generate_summary("conv", object(), "m", on_delta=deltas.append)  # type: ignore[arg-type]
    assert result is None
    assert deltas == ["partial"]
    assert len(stub.calls) == 1
