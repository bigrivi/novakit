"""Direct tests for portable pricing math."""

from __future__ import annotations

import pytest

from novakit.pricing import (
    CACHE_READ,
    INPUT,
    OUTPUT,
    cheapest_prices,
    estimate_cost_from_prices,
    numeric_prices,
    usage_cost,
)


def test_numeric_prices_filters_untrusted_values() -> None:
    assert numeric_prices(None) == {}
    assert numeric_prices(
        {
            INPUT: 1,
            OUTPUT: 2.5,
            CACHE_READ: True,
            "note": "free",
            "extra": 3,
        }
    ) == {INPUT: 1.0, OUTPUT: 2.5, "extra": 3.0}


def test_cheapest_prices_uses_input_plus_output() -> None:
    assert cheapest_prices([]) is None
    assert cheapest_prices([None, {}, "bad"]) is None
    assert cheapest_prices(
        [
            {INPUT: 5, OUTPUT: 5},
            {INPUT: 1, OUTPUT: 2},
            {INPUT: 2, OUTPUT: 2},
        ]
    ) == {INPUT: 1.0, OUTPUT: 2.0}


def test_estimate_cost_math_and_unknown_price() -> None:
    prices = {INPUT: 1.0, OUTPUT: 2.0, CACHE_READ: 0.5}
    assert estimate_cost_from_prices(prices, 1_000_000, 500_000, 250_000) == 2.125
    assert estimate_cost_from_prices(prices, None, None, None) == 0.0
    assert estimate_cost_from_prices(None, 1, 1, 1) is None


def test_usage_cost_annotates_rows_and_counts_unpriced() -> None:
    rows = [
        {"model": "known", "input": 1_000_000, "output": 0, "cache_read": 0},
        {"model": "unknown", "input": 10, "output": 20, "cache_read": 0},
        {"model": "free", "input": 0, "output": 0, "cache_read": 0},
    ]
    prices = {"known": {INPUT: 2.0}, "free": {}}

    result = usage_cost(rows, lambda model: prices.get(model))

    assert rows[0]["cost"] == pytest.approx(2.0)
    assert "cost" not in rows[1]
    assert rows[2]["cost"] == 0.0
    assert result == {"total": 2.0, "priced": False, "unpriced_models": 1}
