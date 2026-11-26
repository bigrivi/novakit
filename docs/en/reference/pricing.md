# Pricing reference

Pure per-million-token cost math with unknown-price handling. `novakit.pricing`
has no catalogue access and no settings lookup — the host resolves model
prices and passes them in. Catalogue loading and model discovery remain host
responsibilities.

::: novakit.pricing.estimate_cost_from_prices
::: novakit.pricing.cheapest_prices
::: novakit.pricing.numeric_prices
::: novakit.pricing.usage_cost

Example — price one call, tolerating an unknown model:

```python
from novakit.pricing import estimate_cost_from_prices

prices = {"input": 3.0, "output": 15.0, "cache_read": 0.3}  # $/M tokens
cost = estimate_cost_from_prices(prices, tokens_input=1200, tokens_output=300)
print(cost)  # dollars, or None when prices is None
```

Example — annotate aggregated usage rows, summarizing priced coverage:

```python
from novakit.pricing import usage_cost

rows = [{"model": "my-model", "input": 1200, "output": 300, "cache_read": 0}]
summary = usage_cost(rows, prices_for_model=lambda model: {"input": 3.0, "output": 15.0})
print(summary)  # {"total": ..., "priced": True, "unpriced_models": 0}
```

Semantics:

- `numeric_prices(raw)` extracts numeric entries from one catalogue cost
  block, skipping booleans and non-numeric values.
- `cheapest_prices(candidates)` picks the lowest input-plus-output price
  block across providers serving the same model, or `None` when none is
  priced.
- `estimate_cost_from_prices(...)` returns dollars or `None` when the price
  block is unknown — never zero-for-unknown.
- `usage_cost(rows, prices_for_model)` annotates each priced row with
  `row["cost"]` and reports `{"total", "priced", "unpriced_models"}`.
