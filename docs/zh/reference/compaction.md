# 压缩参考

可移植的上下文压缩规划、摘要生成与持久化。从
`novakit.compaction` 导入。

## 压缩器

::: novakit.compaction.DefaultCompactor

`DefaultCompactor(model, provider, store, config=...)` 通过历史规划、
安全切分、工具结果裁剪、带重试的摘要生成和熔断器实现
`Compactor` 端口。模块辅助函数 `prepare_compaction`、
`run_compaction_plan`、`compact` 和 `snip_tool_results_in_store`
分别暴露各个阶段。

## 配置与策略

::: novakit.compaction.CompactionConfig
::: novakit.compaction.TokenResolver
::: novakit.compaction.NovakitTokenResolver
::: novakit.compaction.ToolOutputSink

`CompactionConfig` 保存 token 预算和熔断上限（见
[压缩](../guides/compaction.md)）。`TokenResolver` 提供 token
估算和上下文上限（`NovakitTokenResolver` 是内置的模型表默认实现）；
`ToolOutputSink` 可在裁剪前保存完整的工具输出。

## 持久化

::: novakit.compaction.CompactionStore
::: novakit.compaction.MemoryCompactionStore

`CompactionStore` 持久化内容更新、摘要、压缩标记和时间戳。
`MemoryCompactionStore(session)` 是基于 `InMemorySession` 的
独立适配器。

## 摘要与消息辅助函数

::: novakit.compaction.generate_summary
::: novakit.compaction.format_for_summary
::: novakit.compaction.find_split_point
::: novakit.compaction.first_safe_boundary
::: novakit.compaction.should_compact

`generate_summary` 通过 `format_for_summary` 渲染历史并对瞬时失败
重试；`find_split_point` / `first_safe_boundary` 定位永不拆散
工具调用对的安全切分点；`should_compact` / `evaluate_compaction`
判断何时需要折叠。
