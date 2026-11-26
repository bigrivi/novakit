"""Policy unit tests: thresholds, anchors, splits, snips, and injection."""

from __future__ import annotations

import pytest

from novakit.compaction.config import CompactionConfig
from novakit.compaction.policy import (
    SNIP_MARKER,
    NovakitTokenResolver,
    TokenResolver,
    compaction_threshold,
    estimate_context_tokens,
    estimate_tokens,
    evaluate_compaction,
    find_split_point,
    first_safe_boundary,
    get_context_limit,
    should_compact,
    snip_old_tool_results,
)


class StubResolver:
    """Deterministic token source: 1 token per 4 chars, fixed window."""

    def __init__(self, limit: int = 1000) -> None:
        self.limit = limit
        self.estimate_calls = 0

    def estimate_messages(self, messages: list, model: str) -> int:
        self.estimate_calls += 1
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


def make_message(
    role: str,
    content: str,
    message_id: str = "m",
    created: int = 1000,
    **extra: object,
) -> dict:
    message: dict = {
        "id": message_id,
        "role": role,
        "content": content,
        "time_created": created,
    }
    message.update(extra)
    return message


@pytest.fixture
def resolver() -> StubResolver:
    return StubResolver(limit=1000)


@pytest.fixture
def config() -> CompactionConfig:
    return CompactionConfig(
        output_reserve_tokens=200,
        summary_reserve_tokens=100,
        summary_keep_tokens=100,
        max_consecutive_failures=3,
    )


def test_threshold_absolute_with_half_cap(config: CompactionConfig) -> None:
    assert compaction_threshold(1000, config) == 700
    small = CompactionConfig(output_reserve_tokens=200, summary_reserve_tokens=100)
    assert compaction_threshold(400, small) == 200
    assert compaction_threshold(1, small) == 1


def test_should_compact_fires_at_threshold(config: CompactionConfig) -> None:
    assert should_compact(700, 1000, config) is True
    assert should_compact(699, 1000, config) is False


def test_anchor_covers_prompt_but_stale_anchor_is_skipped(
    resolver: StubResolver,
) -> None:
    old = [make_message("user", "x" * 400, "u1", created=1000)]
    anchor = make_message(
        "assistant",
        "ok",
        "a1",
        created=2000,
        tokens_input=900,
        tokens_output=10,
    )
    tail = [make_message("user", "y" * 40, "u2", created=3000)]
    messages = [*old, anchor, *tail]
    assert estimate_context_tokens(messages, "m", None, resolver) == 900 + 10 + 10

    stale = estimate_context_tokens(messages, "m", 2500, resolver)
    assert stale == estimate_tokens(messages, "m", resolver) == (100 + 1 + 10)

    anchor_no_output = make_message(
        "assistant", "ok", "a2", created=2000, tokens_input=900
    )
    anchored = estimate_context_tokens([anchor_no_output], "m", None, resolver)
    assert anchored == 900 + estimate_tokens([anchor_no_output], "m", resolver)


def test_first_safe_boundary_never_starts_on_tool() -> None:
    messages = [
        make_message("user", "a", "u1"),
        make_message("assistant", "call", "a1"),
        make_message("tool", "result", "t1"),
        make_message("user", "b", "u2"),
    ]
    assert first_safe_boundary(messages, 2) == 3
    assert first_safe_boundary(messages, 0) == 0
    all_tool = [make_message("tool", "r", f"t{i}") for i in range(3)]
    assert first_safe_boundary(all_tool, 1) == 0


def test_find_split_point_budget_force_and_summary_guard(
    resolver: StubResolver,
) -> None:
    small = [make_message("user", "hi", "u1")]
    assert find_split_point(small, resolver=resolver) == 0
    assert find_split_point(small, force=True, resolver=resolver) == 0

    big = [make_message("user", "x" * 400, f"u{i}", created=i) for i in range(10)]
    split = find_split_point(big, keep_tokens=100, resolver=resolver)
    assert split > 0

    summaries = [
        {**make_message("assistant", "s", f"s{i}", created=i), "summary": 1}
        for i in range(10)
    ]
    assert find_split_point(summaries, keep_tokens=1, resolver=resolver) == 0


def test_evaluate_plan_needs_splittable_over_threshold(
    resolver: StubResolver, config: CompactionConfig
) -> None:
    big = [
        make_message("user", "x" * 400, f"u{i}", created=1000 + i) for i in range(10)
    ]
    plan = evaluate_compaction(
        "s1", big, None, "m", "p", config=config, resolver=resolver
    )
    assert plan.over_threshold is True
    assert plan.needs_compaction is True
    assert plan.split_index > 0
    assert plan.token_count >= 700

    small = [make_message("user", "hi", "u1")]
    calm = evaluate_compaction(
        "s1", small, None, "m", "p", config=config, resolver=resolver
    )
    assert calm.needs_compaction is False
    assert calm.over_threshold is False

    empty = evaluate_compaction("s1", [], None, "m", "p", config=config)
    assert empty.needs_compaction is False

    forced = evaluate_compaction(
        "s1", small, None, "m", "p", force=True, config=config, resolver=resolver
    )
    assert forced.over_threshold is True


def test_snip_walks_new_to_old_with_head_tail_and_marker(
    resolver: StubResolver,
) -> None:
    big = "v" * 2000
    messages = [
        make_message("tool", big, "old", created=1000),
        make_message("user", "mid", "u1", created=2000),
        make_message("tool", big, "new", created=3000),
    ]
    snip_old_tool_results(
        messages,
        max_chars=200,
        preserve_last_n_messages=1,
        tool_output_token_budget=10_000,
        resolver=resolver,
    )
    assert SNIP_MARKER not in messages[2]["content"]
    trimmed = messages[0]["content"]
    assert SNIP_MARKER in trimmed
    assert trimmed.startswith("v" * 50)
    assert trimmed.endswith("v" * 150)
    assert "Full output" not in trimmed


def test_snip_sink_pointer_none_and_oserror() -> None:
    big = "v" * 2000
    seen: list[tuple[str, str]] = []

    def good_sink(message_id: str, content: str) -> str | None:
        seen.append((message_id, content))
        return "/tmp/offload.txt"

    messages = [make_message("tool", big, "t1", created=1000)]
    snip_old_tool_results(
        messages, max_chars=200, preserve_last_n_messages=0, sink=good_sink
    )
    assert "Full output: /tmp/offload.txt" in messages[0]["content"]
    assert seen == [("t1", big)]

    skipped = [make_message("tool", big, "t2", created=1000)]
    snip_old_tool_results(
        skipped,
        max_chars=200,
        preserve_last_n_messages=0,
        sink=lambda _i, _c: None,
    )
    assert SNIP_MARKER in skipped[0]["content"]
    assert "Full output" not in skipped[0]["content"]

    def broken_sink(message_id: str, content: str) -> str | None:
        raise OSError("disk gone")

    failed = [make_message("tool", big, "t3", created=1000)]
    snip_old_tool_results(
        failed, max_chars=200, preserve_last_n_messages=0, sink=broken_sink
    )
    assert SNIP_MARKER in failed[0]["content"]


def test_resolver_and_config_injection(
    resolver: StubResolver, config: CompactionConfig
) -> None:
    messages = [make_message("user", "x" * 400, "u1", created=1000)]
    plan = evaluate_compaction(
        "s1", messages, None, "m", "p", config=config, resolver=resolver
    )
    assert plan.model_max_tokens == 1000
    assert resolver.estimate_calls > 0
    assert get_context_limit("m", "p", resolver) == 1000


def test_novakit_resolver_delegates_to_tokenizer() -> None:
    resolver: TokenResolver = NovakitTokenResolver()
    messages = [{"role": "user", "content": "hello"}]
    assert resolver.estimate_messages(messages, "unknown") > 0
    assert resolver.context_limit("gpt-4o", "openai") == 128000
