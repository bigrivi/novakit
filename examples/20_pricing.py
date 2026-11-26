"""Pricing: per-million-token math, cheapest provider, and usage totals.

What you'll learn:
  - How per-million-token prices turn token counts into dollars.
  - How to pick the cheapest provider from a price catalog.
  - How usage_cost annotates priced rows and reports unpriced models.

Run:
    python examples/20_pricing.py
"""

from __future__ import annotations

from novakit.pricing import (
    CACHE_READ,
    INPUT,
    OUTPUT,
    cheapest_prices,
    estimate_cost_from_prices,
    numeric_prices,
    usage_cost,
)


def main() -> None:
    # Prepare data: a weekend side project served 1M input, 500K output,
    # and 250K cache-read tokens — small enough to price by hand.
    print(f"price keys: input={INPUT!r} output={OUTPUT!r} cache_read={CACHE_READ!r}")
    raw_prices = {INPUT: 1, OUTPUT: 2.5, CACHE_READ: True, "note": "free"}
    print(f"raw catalog entry: {raw_prices}")

    # Call SDK: numeric_prices keeps real numbers and drops booleans/notes,
    # so a messy catalog still yields usable per-million prices.
    clean = numeric_prices(raw_prices)
    print(f"numeric prices: {clean}")
    print(f"numeric prices (None catalog): {numeric_prices(None)}")

    # Prepare data: three providers quote input+output per-million prices
    # for the same model class.
    candidates = [{INPUT: 5, OUTPUT: 5}, {INPUT: 1, OUTPUT: 2}, {INPUT: 2, OUTPUT: 2}]
    print(f"candidate quotes (input+output sums 10, 3, 4): {candidates}")

    # Call SDK: cheapest compares input-plus-output sums, ignoring junk rows.
    cheapest = cheapest_prices(candidates)
    print(f"cheapest provider quote: {cheapest}")
    print(f"cheapest of nothing: {cheapest_prices([])}")
    print(f"cheapest of junk rows: {cheapest_prices([None, {}, 'bad'])}")

    # Call SDK: cost scales linearly per million tokens; at $1/M input,
    # $2/M output, $0.50/M cache-read, the weekend traffic costs $2.125.
    prices = {INPUT: 1.0, OUTPUT: 2.0, CACHE_READ: 0.5}
    weekend_cost = estimate_cost_from_prices(prices, 1_000_000, 500_000, 250_000)
    print(f"weekend traffic cost: ${weekend_cost}")
    print("math: 1M x $1 + 0.5M x $2 + 0.25M x $0.50 = $2.125")
    print(f"zero traffic cost: ${estimate_cost_from_prices(prices, None, None, None)}")
    print(f"unknown model cost: {estimate_cost_from_prices(None, 1, 1, 1)}")
    partial = estimate_cost_from_prices({INPUT: 1.0}, 1_000_000, 1_000_000)
    print(f"quote with only input priced (1M in + 1M out): ${partial}")

    # Prepare data: a usage export with one priced model, one unknown model,
    # and one free model (empty price table).
    rows = [
        {"model": "known", "input": 1_000_000, "output": 0, "cache_read": 0},
        {"model": "unknown", "input": 10, "output": 20, "cache_read": 0},
        {"model": "free", "input": 0, "output": 0, "cache_read": 0},
    ]
    catalog = {"known": {INPUT: 2.0}, "free": {}}

    # Call SDK: usage_cost stamps a "cost" onto priced rows, leaves unknown
    # rows untouched, and summarizes the unpriced remainder.
    summary = usage_cost(rows, lambda model: catalog.get(model))
    print("usage rows after costing:")
    for row in rows:
        print(f"  {row}")
    print(f"usage summary: total=${summary['total']} {summary}")

    priced_rows = [{"model": "known", "input": 500_000, "output": 0, "cache_read": 0}]
    priced = usage_cost(priced_rows, lambda model: catalog.get(model))
    print(f"half-traffic row: {priced_rows[0]}")
    print(f"fully-priced summary: {priced}")


if __name__ == "__main__":
    main()
