"""Pure per-million-token pricing math without catalogue or settings access."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

INPUT = "input"
OUTPUT = "output"
CACHE_READ = "cache_read"

_PER_MILLION = 1_000_000


def numeric_prices(raw: object) -> dict[str, float]:
    """Extract numeric prices from one catalogue cost block.

    Args:
        raw: Untrusted catalogue cost value.

    Returns:
        Numeric price entries, excluding booleans and non-numeric values.
    """
    if not isinstance(raw, dict):
        return {}
    prices: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        prices[str(key)] = float(value)
    return prices


def _price_sum(prices: Mapping[str, float]) -> float:
    return prices.get(INPUT, 0.0) + prices.get(OUTPUT, 0.0)


def cheapest_prices(candidates: Iterable[object]) -> dict[str, float] | None:
    """Choose the lowest input-plus-output price from catalogue candidates.

    Args:
        candidates: Raw cost blocks from providers serving the same model.

    Returns:
        The cheapest numeric price block, or None when no candidate has prices.
    """
    best: dict[str, float] | None = None
    for raw in candidates:
        prices = numeric_prices(raw)
        if not prices:
            continue
        if best is None or _price_sum(prices) < _price_sum(best):
            best = prices
    return best


def estimate_cost_from_prices(
    prices: Mapping[str, float] | None,
    tokens_input: int | None,
    tokens_output: int | None,
    cache_read_tokens: int | None = None,
) -> float | None:
    """Estimate one call's dollar cost from a known price block.

    Args:
        prices: Dollars per million tokens, or None when unknown.
        tokens_input: Provider-reported prompt tokens.
        tokens_output: Provider-reported completion tokens.
        cache_read_tokens: Prompt tokens served from cache.

    Returns:
        Estimated dollars, or None when the price is unknown.
    """
    if prices is None:
        return None
    return (
        (tokens_input or 0) / _PER_MILLION * prices.get(INPUT, 0.0)
        + (tokens_output or 0) / _PER_MILLION * prices.get(OUTPUT, 0.0)
        + (cache_read_tokens or 0) / _PER_MILLION * prices.get(CACHE_READ, 0.0)
    )


def usage_cost(
    rows: list[dict[str, Any]],
    prices_for_model: Callable[[str], Mapping[str, float] | None],
) -> dict[str, Any]:
    """Annotate usage rows with cost and summarize priced/unpriced models.

    Args:
        rows: Aggregated rows containing model, input, output, and cache_read.
        prices_for_model: Host callback resolving a model to its price block.

    Returns:
        Total priced cost, whether every row was priced, and unpriced row count.
    """
    total = 0.0
    unpriced = 0
    for row in rows:
        cost = estimate_cost_from_prices(
            prices_for_model(str(row.get("model") or "")),
            int(row.get("input") or 0),
            int(row.get("output") or 0),
            int(row.get("cache_read") or 0),
        )
        if cost is None:
            unpriced += 1
            continue
        row["cost"] = cost
        total += cost
    return {"total": total, "priced": unpriced == 0, "unpriced_models": unpriced}


__all__ = [
    "CACHE_READ",
    "INPUT",
    "OUTPUT",
    "cheapest_prices",
    "estimate_cost_from_prices",
    "numeric_prices",
    "usage_cost",
]
