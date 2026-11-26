"""Compaction budget configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactionConfig:
    """Token budgets and guards for context compaction.

    Attributes:
        output_reserve_tokens: Headroom kept free for the model reply.
        summary_reserve_tokens: Headroom kept free for the summarisation request.
        snip_max_chars: Maximum characters kept per trimmed tool result.
        snip_tool_output_token_budget: Reverse token budget for intact tool output.
        snip_preserve_last_n_messages: Newest messages exempt from snipping.
        summary_keep_tokens: Target size of the recent history kept verbatim.
        max_consecutive_failures: Breaker limit for summary attempts; <= 0 means
            no limit.
    """

    output_reserve_tokens: int = 16000
    summary_reserve_tokens: int = 8000
    snip_max_chars: int = 2000
    snip_tool_output_token_budget: int = 50000
    snip_preserve_last_n_messages: int = 12
    summary_keep_tokens: int = 20000
    max_consecutive_failures: int = 3


__all__ = ["CompactionConfig"]
