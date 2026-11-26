# 计价参考

纯粹的按百万 token 成本计算，含未知价格处理。`novakit.pricing`
不访问价格目录，也不读取配置——宿主解析模型价格后传入。
目录加载和模型发现仍是宿主的职责。

::: novakit.pricing.estimate_cost_from_prices
::: novakit.pricing.cheapest_prices
::: novakit.pricing.numeric_prices
::: novakit.pricing.usage_cost

示例——为一次调用计价，容忍未知模型：

```python
from novakit.pricing import estimate_cost_from_prices

prices = {"input": 3.0, "output": 15.0, "cache_read": 0.3}  # $/M tokens
cost = estimate_cost_from_prices(prices, tokens_input=1200, tokens_output=300)
print(cost)  # dollars, or None when prices is None
```

示例——为聚合用量行标注成本，并汇总已定价覆盖率：

```python
from novakit.pricing import usage_cost

rows = [{"model": "my-model", "input": 1200, "output": 300, "cache_read": 0}]
summary = usage_cost(rows, prices_for_model=lambda model: {"input": 3.0, "output": 15.0})
print(summary)  # {"total": ..., "priced": True, "unpriced_models": 0}
```

语义：

- `numeric_prices(raw)` 从单个目录成本块中提取数值条目，
  跳过布尔值和非数值。
- `cheapest_prices(candidates)` 在服务同一模型的多个供应商中选取
  输入加输出价格最低的价格块；无定价时返回 `None`。
- `estimate_cost_from_prices(...)` 返回美元金额，价格块未知时返回
  `None`——未知永不计为零。
- `usage_cost(rows, prices_for_model)` 为每个已定价的行标注
  `row["cost"]`，并报告 `{"total", "priced", "unpriced_models"}`。
