"""Compaction orchestration: prepare, run, and the controller that gates it."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from novakit.agent.compaction import CompactionError
from novakit.compaction.config import CompactionConfig
from novakit.compaction.messages import (
    compacted_at_of,
    contains_summary,
    content_of,
    created_at_of,
    id_of,
    role_of,
    session_id_of,
    set_compacted_at,
    tool_call_id_of,
    tool_call_ids_of,
)
from novakit.compaction.policy import (
    SNIP_MARKER,
    CompactionPlan,
    TokenResolver,
    ToolOutputSink,
    compaction_threshold,
    estimate_context_tokens,
    estimate_tokens,
    evaluate_compaction,
    find_split_point,
    get_context_limit,
    snip_old_tool_results,
)
from novakit.compaction.store import CompactionStore
from novakit.compaction.summary import (
    CONTINUATION_INSTRUCTION,
    format_for_summary,
    generate_summary,
)
from novakit.events import AgentEvent, AgentEventFrame
from novakit.llm import LLMProvider

if TYPE_CHECKING:
    from novakit.session.models import SessionContext

log = logging.getLogger(__name__)


async def prepare_compaction(
    session_id: str,
    messages: list,
    last_compacted_at: int | None,
    store: CompactionStore,
    model: str = "gpt-4o",
    provider: str = "ollama",
    force: bool = False,
    config: CompactionConfig | None = None,
    resolver: TokenResolver | None = None,
    sink: ToolOutputSink | None = None,
) -> CompactionPlan:
    """Evaluate *messages*, run Layer 1 snipping when needed, then re-evaluate.

    The caller owns *messages*; Layer 1 trims them in place so the caller's copy
    stays consistent with what was written back to the store.

    Args:
        session_id: Session the plan is for.
        messages: Caller-owned history, trimmed in place by Layer 1.
        last_compacted_at: Last successful compaction time for the estimate.
        store: Persistence seam receiving snipped tool output.
        model: Model id the plan is for.
        provider: Provider the model runs on.
        force: Skip the token-pressure test after a provider rejection.
        config: Threshold and snip tuning; defaults are used when None.
        resolver: Token estimator to use; the bundled default when None.
        sink: Offload hook for trimmed tool output; no offloading when None.

    Returns:
        The compaction decision with split point and token counts.
    """
    plan = evaluate_compaction(
        session_id,
        messages,
        last_compacted_at,
        model,
        provider,
        force=force,
        config=config,
        resolver=resolver,
    )
    if not plan.over_threshold:
        return plan

    # Layer 1 answers to token pressure alone. Gating it on Layer 2's split
    # point would leave a runaway tool loop untouched whenever the history
    # cannot be split safely - exactly the case where trimming is the only
    # defence left.
    await snip_tool_results_in_store(
        store, session_id, messages, config=config, resolver=resolver, sink=sink
    )
    return evaluate_compaction(
        session_id,
        messages,
        last_compacted_at,
        model,
        provider,
        force=force,
        config=config,
        resolver=resolver,
    )


async def run_compaction_plan(
    plan: CompactionPlan,
    store: CompactionStore,
    llm: LLMProvider,
    model: str = "gpt-4o",
    provider: str = "ollama",
    messages: list | None = None,
    tools: list[dict] | None = None,
    on_delta: Callable[[str], None] | None = None,
    config: CompactionConfig | None = None,
    resolver: TokenResolver | None = None,
) -> bool:
    """Execute a prepared compaction plan against caller-owned *messages*.

    Args:
        plan: Decision from :func:`prepare_compaction`.
        store: Persistence seam for the summary and compaction marks.
        llm: Provider used to generate the summary.
        model: Model id to summarise with.
        provider: Provider the model runs on.
        messages: Caller-owned history to compact; loaded from the store
            when None.
        tools: Tool schemas forwarded to the summary request.
        on_delta: Called with each summary chunk as it arrives.
        config: Threshold and keep tuning; defaults are used when None.
        resolver: Token estimator to use; the bundled default when None.

    Returns:
        True when history was compacted, False when the plan asked for nothing.

    Raises:
        CompactionError: When summary generation failed; the session is left
            untouched.
    """
    if not plan.needs_compaction:
        return False
    return await compact(
        plan.session_id,
        store,
        llm,
        model,
        provider,
        messages=messages,
        split_index=plan.split_index or None,
        tools=tools,
        on_delta=on_delta,
        config=config,
        resolver=resolver,
    )


async def snip_tool_results_in_store(
    store: CompactionStore,
    session_id: str,
    messages: list,
    config: CompactionConfig | None = None,
    resolver: TokenResolver | None = None,
    sink: ToolOutputSink | None = None,
) -> None:
    """Layer 1: trim old tool results stored in *store*.

    Trims *messages* in place, then persists only the messages that were
    actually snipped; untrimmed messages are never written back.

    Args:
        store: Persistence seam receiving snipped tool output.
        session_id: Session the messages belong to.
        messages: History to trim in place.
        config: Snip tuning; defaults are used when None.
        resolver: Token estimator to use; the bundled default when None.
        sink: Offload hook for trimmed tool output; no offloading when None.
    """
    cfg = config or CompactionConfig()
    snip_old_tool_results(
        messages,
        max_chars=cfg.snip_max_chars,
        preserve_last_n_messages=cfg.snip_preserve_last_n_messages,
        tool_output_token_budget=cfg.snip_tool_output_token_budget,
        sink=sink,
        resolver=resolver,
    )

    for message in messages:
        if role_of(message) != "tool":
            continue
        content = content_of(message)
        if SNIP_MARKER in content:
            await store.update_message_content(id_of(message), content)


async def compact(
    session_id: str,
    store: CompactionStore,
    llm: LLMProvider,
    model: str = "gpt-4o",
    provider: str = "ollama",
    messages: list | None = None,
    split_index: int | None = None,
    tools: list[dict] | None = None,
    on_delta: Callable[[str], None] | None = None,
    config: CompactionConfig | None = None,
    resolver: TokenResolver | None = None,
) -> bool:
    """Run session compaction (Layer 2).

    A failed summary leaves the session untouched: writing the failure text as a
    summary would poison the context it is supposed to compress, and marking the
    old messages compacted would discard them for nothing.

    Args:
        session_id: Session to compact.
        store: Persistence seam for the summary and compaction marks.
        llm: Provider used to generate the summary.
        model: Model id to summarise with.
        provider: Provider the model runs on.
        messages: History to compact; loaded from the store when None.
        split_index: Boundary from a prepared plan; recomputed when None.
        tools: Tool schemas forwarded to the summary request.
        on_delta: Called with each summary chunk as it arrives.
        config: Threshold and keep tuning; defaults are used when None.
        resolver: Token estimator to use; the bundled default when None.

    Returns:
        True when history was compacted, False when there was nothing safe
        to fold in.

    Raises:
        CompactionError: When summary generation failed; no summary is
            written and no message is marked compacted.
    """
    pending = messages if messages is not None else await store.get_messages(session_id)

    if not pending:
        return False

    before_tokens = estimate_tokens(pending, model, resolver)
    cfg = config or CompactionConfig()
    split = (
        split_index
        if split_index is not None
        else find_split_point(
            pending,
            keep_tokens=min(
                cfg.summary_keep_tokens,
                compaction_threshold(get_context_limit(model, provider, resolver), cfg)
                // 2,
            ),
            resolver=resolver,
        )
    )
    if split <= 0:
        return False

    old = pending[:split]
    recent = pending[split:]
    old_text = format_for_summary(old)

    log.info(
        f"[Compaction] session={session_id}, before={len(pending)} msgs, {before_tokens} tokens, split at={split}"
    )

    summary = await generate_summary(
        old_text,
        llm,
        model,
        has_previous_summary=contains_summary(old),
        tools=tools,
        on_delta=on_delta,
    )
    if summary is None:
        log.warning(
            "[Compaction] session=%s aborted: summary generation failed", session_id
        )
        raise CompactionError(f"summary generation failed for session {session_id}")

    now_ms = int(time.time() * 1000)
    # The summary stands in for the history it replaces, so it is stamped at
    # that boundary rather than at "now". The prompt then reads in
    # chronological order (the summary leads), the next compaction finds it
    # inside the portion it folds in - keeping exactly one summary instead of
    # piling a new one beside it - and it is not counted as growth since the
    # last compaction.
    boundary_ms = min(created_at_of(m) for m in recent) - 1 if recent else now_ms
    await store.add_summary(
        session_id,
        f"[Previous conversation summary]\n{summary}\n\n{CONTINUATION_INSTRUCTION}",
        boundary_ms,
    )
    # Also compact tool responses whose declaring assistant was compacted.
    compacted_tc_ids = set()
    for m in old:
        if role_of(m) != "assistant":
            continue
        compacted_tc_ids.update(tool_call_ids_of(m))

    orphan_ids = []
    for m in recent:
        if role_of(m) != "tool":
            continue
        if tool_call_id_of(m) in compacted_tc_ids:
            orphan_ids.append(id_of(m))

    if orphan_ids:
        log.info(
            "[Compaction] also compacting %d orphaned tool responses", len(orphan_ids)
        )

    old_ids = [id_of(m) for m in old]
    await store.mark_messages_compacted_by_ids(session_id, old_ids + orphan_ids)
    await store.update_session_compacted_at(session_id, now_ms)

    after_tokens = estimate_tokens(recent, model, resolver)
    log.info(
        f"[Compaction] session={session_id}, compacted={len(old)} msgs, after={len(recent) + 1} msgs, {after_tokens} tokens"
    )
    return True


class DefaultCompactor:
    """Per-agent compaction policy: what to run, and when to stop trying.

    Owns the consecutive-failure counter, so the circuit breaker state lives
    next to the compaction logic it guards rather than on the agent. A forced
    run is latched in ``force_next`` and consumed by the next
    :meth:`maybe_compact` call; ``compacted`` reports whether that call
    actually changed history.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        *,
        store: CompactionStore,
        config: CompactionConfig | None = None,
        resolver: TokenResolver | None = None,
        sink: ToolOutputSink | None = None,
    ) -> None:
        """Bind a model route and its persistence to one gating instance.

        Args:
            model: Model id the summary request runs with.
            provider: Provider the model runs on.
            store: Persistence seam for snips, summaries, and marks.
            config: Threshold, breaker, and snip tuning; defaults when None.
            resolver: Token estimator to use; the bundled default when None.
            sink: Offload hook for trimmed tool output; no offloading when None.
        """
        self.model = model
        self.provider = provider
        self._store = store
        self._config = config or CompactionConfig()
        self._resolver = resolver
        self._sink = sink
        self.consecutive_failures = 0
        self.compacted = False
        # Set by the caller when the provider rejected a request as too large, so
        # the next plan ignores token pressure. Consumed on the next run.
        self.force_next = False

    def summarising_allowed(self) -> bool:
        """Whether another summary attempt may run under the breaker limit.

        Returns:
            True when the consecutive-failure streak is below the configured
            limit, or when no limit is configured.
        """
        limit = self._config.max_consecutive_failures
        if limit <= 0:
            return True
        if self.consecutive_failures < limit:
            return True
        log.warning(
            "Auto-compaction disabled for this agent after %d consecutive failures",
            self.consecutive_failures,
        )
        return False

    async def plan(
        self,
        messages: list,
        session: SessionContext | dict[str, Any] | None,
        force: bool = False,
    ) -> CompactionPlan | None:
        """Decide whether Layer 2 should run, after Layer 1 has had its chance.

        Layer 1 lives inside :func:`prepare_compaction` and never calls a model, so it
        runs even while the circuit breaker is open; only the summarisation step
        is gated.

        Args:
            messages: Caller-owned history, trimmed in place by Layer 1.
            session: Session context or raw session dict the plan is for.
            force: Skip the token-pressure test and the breaker gate after a
                provider rejection.

        Returns:
            A runnable plan, or None when no compaction is needed or the
            breaker holds an unforced run.
        """
        plan = await prepare_compaction(
            session_id=session_id_of(session) or "",
            messages=messages,
            last_compacted_at=compacted_at_of(session),
            store=self._store,
            model=self.model,
            provider=self.provider,
            force=force,
            config=self._config,
            resolver=self._resolver,
            sink=self._sink,
        )
        if not plan.needs_compaction:
            return None
        # The breaker guards speculative auto-compaction: repeated attempts that
        # only the token estimate triggered. A forced run is not speculative. It
        # follows a provider rejection that only compaction can clear, and the
        # caller latches it to at most one attempt per user request, so it cannot
        # loop. Gating it here would block the one action that can make the
        # pending request succeed, leaving the session to re-fail every request
        # until a human intervenes.
        if not force and not self.summarising_allowed():
            return None
        return plan

    def record_result(
        self, compacted: bool, session: SessionContext | dict[str, Any] | None
    ) -> None:
        """Fold one compaction outcome into the consecutive-failure streak.

        Forced (overflow-driven) runs share this streak with automatic ones: the
        counter measures whether summarisation currently works, which is true
        regardless of what triggered the attempt, and a success clears it either
        way. Only the gate in :meth:`plan` exempts forced runs; keeping a single
        counter means the breaker re-arms from a forced success and no second
        piece of state can drift out of sync.

        Args:
            compacted: Whether the attempt changed history.
            session: Session context or raw session dict stamped on success.
        """
        if compacted:
            self.consecutive_failures = 0
            set_compacted_at(session, int(time.time() * 1000))
            return
        self.consecutive_failures += 1
        log.warning("Compaction failed (%d consecutive)", self.consecutive_failures)

    async def maybe_compact(
        self,
        messages: list,
        session: SessionContext | dict[str, Any] | None,
        llm: LLMProvider,
        emit: Callable[..., Awaitable[None]],
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[AgentEventFrame, None]:
        """Compact if needed, announcing it around the summarisation call.

        ``compacted`` records whether history actually changed, so the caller
        knows when to reload it. ``tools`` is forwarded to the summary call so
        the request is shaped like an agent turn (see
        :mod:`novakit.llm.oneshot`).

        Emits a context-update frame first, then compaction-start, one
        compaction-delta per summary chunk, compaction-end, and a final
        context-update when history changed.

        Args:
            messages: Caller-owned history to evaluate and compact in place.
            session: Session context or raw session dict being compacted.
            llm: Provider used to generate the summary.
            emit: Callback fanning each frame out to subscribers.
            tools: Tool schemas forwarded to the summary request.

        Yields:
            Context and compaction event frames in stream order.

        Raises:
            CompactionError: When summary generation failed; the failure is
                counted before it surfaces.
        """
        self.compacted = False
        # A forced run is consumed here so the latch cannot survive a turn that
        # ends up not needing a plan at all.
        force = self.force_next
        self.force_next = False
        # Always emit current context usage so the TUI can render the ctx bar
        # with the same numbers the compaction decision uses.
        try:
            compacted_at = compacted_at_of(session)
            used = estimate_context_tokens(
                messages, self.model, compacted_at, self._resolver
            )
            # Report against the compaction threshold, not the raw window. The
            # reserve is headroom the history can never use (the reply plus the
            # summarisation request), so a raw-window denominator shows the bar
            # short of full at the very moment compaction fires. Deriving it from
            # compaction_threshold keeps the figure and the trigger in lockstep.
            limit = compaction_threshold(
                get_context_limit(self.model, self.provider, self._resolver),
                self._config,
            )
            percent = int(used / limit * 100) if limit else 0
            ctx_payload = {"used": used, "limit": limit, "percent": percent}
            await emit(AgentEvent.CONTEXT_UPDATE, ctx_payload)
            yield AgentEvent.CONTEXT_UPDATE, ctx_payload
        except Exception:
            # The context pill is informational; never abort a turn for it.
            pass

        plan = await self.plan(messages, session, force=force)
        if plan is None:
            return

        payload = {
            "message_count": plan.message_count,
            "token_count": plan.token_count,
            "reason": "overflow" if force else "threshold",
        }
        await emit(AgentEvent.COMPACTION_START, payload)
        yield AgentEvent.COMPACTION_START, payload

        # The summary is written by a nested (awaited) call, so its chunks land
        # on a callback rather than arriving from this generator. Bridge them
        # through a queue so partial text reaches the client as it is produced.
        deltas: asyncio.Queue[str | None] = asyncio.Queue()

        async def summarise() -> bool:
            try:
                return await run_compaction_plan(
                    plan,
                    store=self._store,
                    llm=llm,
                    model=self.model,
                    provider=self.provider,
                    messages=messages,
                    tools=tools,
                    on_delta=deltas.put_nowait,
                    config=self._config,
                    resolver=self._resolver,
                )
            finally:
                # Close the drain below even if summarising raised, so the
                # generator cannot deadlock waiting on a missing sentinel.
                deltas.put_nowait(None)

        task = asyncio.create_task(summarise())
        while True:
            chunk = await deltas.get()
            if chunk is None:
                break
            yield AgentEvent.COMPACTION_DELTA, {"delta": chunk}
        try:
            compacted = await task
        except CompactionError:
            # Count the failure before surfacing it, so the streak reflects it
            # even though the turn ends here.
            self.record_result(False, session)
            raise
        self.record_result(compacted, session)

        await emit(AgentEvent.COMPACTION_END, payload)
        yield AgentEvent.COMPACTION_END, payload
        self.compacted = compacted
        if compacted and session is not None:
            try:
                fresh = await self._store.get_messages(session_id_of(session) or "")
                used_after = estimate_context_tokens(
                    fresh, self.model, compacted_at_of(session), self._resolver
                )
                limit_after = compaction_threshold(
                    get_context_limit(self.model, self.provider, self._resolver),
                    self._config,
                )
                percent_after = (
                    int(used_after / limit_after * 100) if limit_after else 0
                )
                ctx_after = {
                    "used": used_after,
                    "limit": limit_after,
                    "percent": percent_after,
                }
                await emit(AgentEvent.CONTEXT_UPDATE, ctx_after)
                yield AgentEvent.CONTEXT_UPDATE, ctx_after
            except Exception:
                # Informational frame; the compaction already succeeded.
                pass


__all__ = [
    "DefaultCompactor",
    "compact",
    "prepare_compaction",
    "run_compaction_plan",
    "snip_tool_results_in_store",
]
