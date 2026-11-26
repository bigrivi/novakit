"""Pure compaction planning: estimates, split points and thresholds."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from novakit.compaction.config import CompactionConfig
from novakit.compaction.messages import (
    content_of,
    created_at_of,
    id_of,
    is_summary,
    role_of,
    tokens_input_of,
    tokens_output_of,
)
from novakit.llm.tokenizer import estimate_messages_tokens, resolve_context_limit

log = logging.getLogger(__name__)


SNIP_MARKER = "chars snipped"


@dataclass(frozen=True)
class CompactionPlan:
    """Pure decision output: no IO, no mutation, no message payload."""

    session_id: str
    model_max_tokens: int
    token_count: int
    message_count: int
    needs_compaction: bool
    split_index: int = 0
    over_threshold: bool = False


class TokenResolver(Protocol):
    """Token estimation and context limits for compaction planning."""

    def estimate_messages(self, messages: list, model: str) -> int:
        """Estimate tokens for *messages* taken in isolation.

        Args:
            messages: History to estimate.
            model: Model id the estimate is planned against.

        Returns:
            Estimated token count.
        """
        ...

    def context_limit(self, model: str, provider: str) -> int:
        """Return the context window to plan against.

        Args:
            model: Model id to look up.
            provider: Provider the model runs on.

        Returns:
            Context window in tokens.
        """
        ...


class NovakitTokenResolver:
    """Default resolver backed by :mod:`novakit.llm.tokenizer`."""

    def estimate_messages(self, messages: list, model: str) -> int:
        """Estimate tokens for *messages* with the bundled tokenizer.

        Args:
            messages: History to estimate.
            model: Model id the estimate is planned against.

        Returns:
            Estimated token count.
        """
        return estimate_messages_tokens(messages, model)

    def context_limit(self, model: str, provider: str) -> int:
        """Return the provider-stated context window for *model*.

        Args:
            model: Model id to look up.
            provider: Provider the model runs on.

        Returns:
            Context window in tokens.
        """
        return resolve_context_limit(model, provider)


_DEFAULT_RESOLVER = NovakitTokenResolver()


def _use_resolver(resolver: TokenResolver | None) -> TokenResolver:
    return resolver if resolver is not None else _DEFAULT_RESOLVER


#: Offload hook for trimmed tool output: called with (message_id, content),
#: returns a readable pointer (e.g. a path) or None when offloading is skipped.
ToolOutputSink = Callable[[str, str], str | None]


def estimate_tokens(
    messages: list, model: str = "unknown", resolver: TokenResolver | None = None
) -> int:
    """Character-based token estimate for *messages* taken in isolation.

    Use this whenever messages are being compared against each other or against
    a budget of their own - split ratios, growth since a compaction, per-message
    trimming. It must stay independent of any API-reported total, otherwise a
    subset would appear to weigh as much as the whole prompt.

    Pure computation: reads *messages* without mutating them.

    Args:
        messages: History to estimate.
        model: Model id the estimate is planned against.
        resolver: Token estimator to use; the bundled default when None.

    Returns:
        Estimated token count.
    """
    return _use_resolver(resolver).estimate_messages(messages, model)


def estimate_context_tokens(
    messages: list,
    model: str = "unknown",
    compacted_at: int | None = None,
    resolver: TokenResolver | None = None,
) -> int:
    """Absolute context size of *messages*, anchored on the provider's accounting.

    ``tokens_input`` on an assistant message is the exact prompt size the API
    charged for the request that produced it, so it already covers the system
    prompt, the tool schemas and the cached prefix - none of which character
    heuristics can see. Only the anchor's own output and the messages appended
    after it are estimated, so the error stops accumulating over a session.

    An anchor only describes the *current* history while no compaction has run
    since. Its ``tokens_input`` counts whatever was in the prompt at the time,
    including messages compaction has since removed, so anchoring on one that
    predates *compacted_at* re-counts the history that was just dropped and the
    reported size never falls. Those are skipped and the character estimate is
    used until a post-compaction call records a fresh anchor.

    ``compacted_at`` is the moment of the last successful compaction rather than a
    message timestamp: the messages that survive a compaction are older than the
    summary that replaced them, so ordering cannot tell a stale anchor from a
    fresh one.

    Pure computation: reads *messages* without mutating them.

    Args:
        messages: History to measure.
        model: Model id the estimate is planned against.
        compacted_at: Last successful compaction time, used to skip stale anchors.
        resolver: Token estimator to use; the bundled default when None.

    Returns:
        Absolute context size in tokens.
    """
    anchor_floor = compacted_at or 0
    anchor_index = -1
    anchor_prompt_tokens = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        reported = tokens_input_of(message)
        if reported and created_at_of(message) > anchor_floor:
            anchor_index = index
            anchor_prompt_tokens = reported
            break

    if anchor_index < 0:
        return estimate_tokens(messages, model, resolver)

    anchor_message = messages[anchor_index]
    anchor_output_tokens = tokens_output_of(anchor_message) or estimate_tokens(
        [anchor_message], model, resolver
    )
    appended_after_anchor = messages[anchor_index + 1 :]
    return (
        anchor_prompt_tokens
        + anchor_output_tokens
        + estimate_tokens(appended_after_anchor, model, resolver)
    )


def get_context_limit(
    model: str, provider: str, resolver: TokenResolver | None = None
) -> int:
    """The context window to plan against, as the provider states it.

    Thin wrapper kept so compaction callers need not reach into the tokenizer.
    Pure computation: no IO and no mutation.

    Args:
        model: Model id to look up.
        provider: Provider the model runs on.
        resolver: Limit source to use; the bundled default when None.

    Returns:
        Context window in tokens.
    """
    return _use_resolver(resolver).context_limit(model, provider)


def snip_old_tool_results(
    messages: list,
    max_chars: int = 2000,
    preserve_last_n_messages: int = 12,
    tool_output_token_budget: int = 50000,
    sink: ToolOutputSink | None = None,
    resolver: TokenResolver | None = None,
) -> list:
    """Layer 1: trim tool results under a reverse token budget.

    Walking from the newest message backwards and spending a token budget on
    tool output keeps the results the current task actually depends on intact,
    whatever their number, and only trims once the budget runs out. A fixed
    "last N messages" window cannot do that: N recent messages may be a few
    hundred tokens or a repository-wide grep.

    Trimmed output keeps its tail, where commands put their verdict. When *sink*
    is given, the full text is offered to it (which may persist the text and
    return a readable pointer); a ``None`` return or an :exc:`OSError` only
    warns and the snip continues without a pointer.

    Mutates *messages* in place and returns the same list, so callers holding
    the history observe the trimmed text.

    Args:
        messages: History to trim; tool messages are rewritten in place.
        max_chars: Maximum characters kept per trimmed tool result.
        preserve_last_n_messages: Trailing messages kept verbatim while the
            token budget allows.
        tool_output_token_budget: Reverse token budget spent on tool output
            from newest to oldest.
        sink: Offload hook receiving ``(message_id, content)``; its return
            becomes the readable pointer, or None to skip offloading.
        resolver: Token estimator to use; the bundled default when None.

    Returns:
        The same *messages* list, with old tool results trimmed.
    """
    resolved = _use_resolver(resolver)
    keep_verbatim_from = max(0, len(messages) - preserve_last_n_messages)
    spent_tokens = 0

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if role_of(message) != "tool":
            continue

        content = content_of(message)
        if not isinstance(content, str) or not content:
            continue

        tokens = resolved.estimate_messages([message], "unknown")
        within_budget = spent_tokens + tokens <= tool_output_token_budget
        if index >= keep_verbatim_from and within_budget:
            spent_tokens += tokens
            continue
        if len(content) <= max_chars:
            spent_tokens += tokens
            continue

        pointer = ""
        if sink is not None:
            try:
                offloaded = sink(id_of(message), content)
            except OSError as error:
                log.warning("[Compaction] could not offload tool output: %s", error)
                offloaded = None
            if offloaded:
                pointer = f" Full output: {offloaded}"
            else:
                log.warning(
                    "[Compaction] tool output offload skipped for message %s",
                    id_of(message),
                )
        head = content[: max_chars // 4]
        tail = content[-(max_chars * 3 // 4) :]
        omitted = len(content) - len(head) - len(tail)
        new_content = f"{head}\n[... {omitted} {SNIP_MARKER} ...{pointer}]\n{tail}"
        if isinstance(message, dict):
            message["content"] = new_content
        else:
            message.content = new_content
        spent_tokens += resolved.estimate_messages([message], "unknown")

    return messages


def find_split_point(
    messages: list,
    keep_tokens: int = 20000,
    force: bool = False,
    resolver: TokenResolver | None = None,
) -> int:
    """Find a split point leaving roughly ``keep_tokens`` of recent history.

    The budget is absolute rather than a share of the history. A share grew with
    the history, so a single turn that dumped a large tool result left a
    permanently higher floor for the rest of the session, and every later
    compaction inherited it.

    Returns 0 when nothing should be compacted: the whole history already fits the
    budget, no boundary is safe, or the portion that would be folded in is nothing
    but earlier summaries.

    ``force`` skips the budget check. A request the provider rejected as too large
    can be that small a history when the fixed prompt - system prompt, tool
    schemas - is what does not fit, and folding the history is then the only
    remedy available. Declining to try would guarantee the retry failed too.

    Pure computation: reads *messages* without mutating them.

    Args:
        messages: History to split.
        keep_tokens: Recent history budget to leave untouched.
        force: Skip the budget check when a provider rejection already proved
            the history too large.
        resolver: Token estimator to use; the bundled default when None.

    Returns:
        Split index, or 0 when nothing should be compacted.
    """
    total = estimate_tokens(messages, resolver=resolver)
    if total <= keep_tokens and not force:
        # Already within budget. With a ratio this could not happen, so it needs
        # saying: compacting here would shave off a message or two, leave the
        # threshold exceeded, and repeat on every call until the breaker tripped.
        return 0

    running = 0
    target = 0
    for i in range(len(messages) - 1, -1, -1):
        running += estimate_tokens([messages[i]], resolver=resolver)
        if running >= keep_tokens:
            target = i
            break

    # The walk can only reach the budget at index 0 when the first message alone
    # outweighs it — a giant pasted prompt, say. That reported "unsplittable" for
    # exactly the history that most needed compacting, so start the boundary at 1
    # and let the boundary search validate it.
    split = first_safe_boundary(messages, max(1, target))
    if split > 0 and all(is_summary(message) for message in messages[:split]):
        # Folding summary-into-summary rewrites the summary without freeing any
        # context, and because the hard limit is still exceeded it would repeat
        # on every model call. Refusing here keeps that out of the failure
        # counter (returning False from compact would trip the circuit breaker).
        return 0
    return split


def first_safe_boundary(messages: list, target: int) -> int:
    """The first index at or after *target* that a split may leave in the past.

    A tool response must never start the recent portion. Its declaring assistant
    message would be in the compacted portion, leaving an orphan tool message that
    breaks the assistant->tool pairing contract and makes the provider reject the
    request. So tool messages are not eligible boundaries.

    The search moves forward from the budget target rather than backward to a
    user message. Preferring a user boundary reads better, but nothing depends on
    it: the turn's opening user message is itself summarised, so the model still
    learns what was asked. Moving backward instead meant the distance to the
    previous user turn decided how much was kept, which for a session that had
    compacted down to a single exchange kept almost everything, and for a long
    agentic turn kept 4x the budget.

    Returns 0 when no eligible boundary exists at or after *target*, in which case
    the whole remaining history is a tool run and the split falls back to the
    earliest eligible index rather than onto a tool message.

    Pure computation: reads *messages* without mutating them.

    Args:
        messages: History to search.
        target: Budget-derived index the boundary must not precede.

    Returns:
        Split index, or 0 when no eligible boundary exists.
    """
    count = len(messages)
    for index in range(min(target, count), count):
        if role_of(messages[index]) != "tool":
            return index
    for index in range(min(target, count) - 1, -1, -1):
        if role_of(messages[index]) != "tool":
            return index
    return 0


def compaction_threshold(
    model_max_tokens: int, config: CompactionConfig | None = None
) -> int:
    """Token headroom that must stay free for the reply and the summary request.

    The reserves are absolute rather than a fraction of the window: what they
    pay for - one model reply and one summarisation request - costs the same
    whether the window is 32k or 1M. A ratio would starve small windows and
    waste most of a large one.

    Small windows still need a guard: a flat 24k reserve would leave a 26k
    window with almost nothing usable and compact on every request, so the
    reserve never takes more than half of the window.

    Pure computation: no IO and no mutation.

    Args:
        model_max_tokens: Provider-stated context window.
        config: Threshold tuning; defaults are used when None.

    Returns:
        Token count at or above which compaction should run.
    """
    cfg = config or CompactionConfig()
    requested_reserve = cfg.output_reserve_tokens + cfg.summary_reserve_tokens
    reserve = min(requested_reserve, model_max_tokens // 2)
    return max(1, model_max_tokens - reserve)


def should_compact(
    total_tokens: int,
    model_max_tokens: int,
    config: CompactionConfig | None = None,
) -> bool:
    """Decide whether compaction should run, on token pressure alone.

    The trigger is the absolute context size against the compaction threshold, so
    the reserve stays free for the reply and the summarisation request however
    large the carried prefix grows. It is deliberately not the growth since the
    last compaction: bounding the total by the threshold also bounds the growth
    (growth <= total), so a separate growth trigger would be subsumed.

    Pure computation: no IO and no mutation.

    Args:
        total_tokens: Absolute context size to test.
        model_max_tokens: Provider-stated context window.
        config: Threshold tuning; defaults are used when None.

    Returns:
        True when compaction should run.
    """
    return total_tokens >= compaction_threshold(model_max_tokens, config)


def evaluate_compaction(
    session_id: str,
    messages: list,
    last_compacted_at: int | None,
    model: str = "gpt-4o",
    provider: str = "ollama",
    force: bool = False,
    config: CompactionConfig | None = None,
    resolver: TokenResolver | None = None,
) -> CompactionPlan:
    """Pure decision step: no IO, no mutation, no message payload in the result.

    A plan only asks for compaction when the history can also be split safely,
    so callers never announce a compaction that would turn into a no-op.

    ``force`` skips the token-pressure test. The provider rejecting a request as
    too large is a better signal than any estimate, and it arrives precisely when
    the estimate was wrong.

    Args:
        session_id: Session the plan is for.
        messages: History under consideration; read without mutation.
        last_compacted_at: Last successful compaction time, used to skip stale
            usage anchors.
        model: Model id the plan is for.
        provider: Provider the model runs on.
        force: Skip the token-pressure test after a provider rejection.
        config: Threshold and keep tuning; defaults are used when None.
        resolver: Token estimator to use; the bundled default when None.

    Returns:
        The compaction decision with split point and token counts.
    """
    cfg = config or CompactionConfig()
    model_max_tokens = get_context_limit(model, provider, resolver)

    if not messages:
        return CompactionPlan(session_id, model_max_tokens, 0, 0, False)

    token_count = estimate_context_tokens(messages, model, last_compacted_at, resolver)
    threshold = compaction_threshold(model_max_tokens, cfg)
    over_threshold = force or should_compact(
        total_tokens=token_count,
        model_max_tokens=model_max_tokens,
        config=cfg,
    )
    # The budget never takes more than half the threshold: after a compaction the
    # kept portion plus the summary must leave room below the threshold, or the
    # next call would find the total already over it and compact on every turn.
    keep_tokens = min(cfg.summary_keep_tokens, threshold // 2)
    split_index = (
        find_split_point(
            messages, keep_tokens=keep_tokens, force=force, resolver=resolver
        )
        if over_threshold
        else 0
    )
    return CompactionPlan(
        session_id,
        model_max_tokens,
        token_count,
        len(messages),
        over_threshold and split_index > 0,
        split_index,
        over_threshold,
    )


__all__ = [
    "SNIP_MARKER",
    "CompactionPlan",
    "NovakitTokenResolver",
    "TokenResolver",
    "ToolOutputSink",
    "compaction_threshold",
    "estimate_context_tokens",
    "estimate_tokens",
    "evaluate_compaction",
    "find_split_point",
    "first_safe_boundary",
    "get_context_limit",
    "should_compact",
    "snip_old_tool_results",
]
