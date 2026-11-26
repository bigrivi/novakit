# 项目指令

`novakit.prompt.discover_project_instructions`
（`novakit/prompt/project_instructions.py`）从宿主提供的全局文件和
按目录的项目文件构建合并后的单个指令块。`novakit`
从不自己读取 `HOME`——宿主通过 `ProjectInstructionConfig`
提供全局目录和策略。

## 发现

```python
from pathlib import Path

from novakit.prompt import ProjectInstructionConfig, discover_project_instructions

instructions = discover_project_instructions(
    workspace,
    ProjectInstructionConfig(global_dir=Path("/host/instructions")),
)
```

分层与优先级：

1. **全局层**——从 `global_dir` 中按 `global_filenames` 优先级最多选择一个文件
   （`GLOBAL_FILENAMES = ("AGENTS.override.md", "AGENTS.md")`），不做筛查。
   `global_dir=None` 表示跳过。
2. **项目层**——从 git 根目录到工作区，每目录按 `project_filenames` 优先级
   最多选择一个文件（`PROJECT_FILENAMES = ("AGENTS.override.md", "AGENTS.md", "CLAUDE.md")`），
   按根到叶合并。git 根目录链之外的文件会被忽略。
3. **筛查**——每个项目文件用 `has_threats` 检查配置的 `threat_categories`
   （默认值 `PROJECT_THREAT_CATEGORIES = ["ignore_previous", "role_hijack"]`，
   集合故意收窄，以免正常安全规则误报）。被标记的文件会被丢弃，
   并与其命中的类别一起列在 `ProjectInstructions.flagged` 中。
4. **字节预算**——合并后的块从最宽泛的一端截断到 `max_block_bytes`
   （默认值 `MAX_BLOCK_BYTES = 32 * 1024` UTF-8 字节）。

结果是一个 `ProjectInstructions(text, sources, flagged)` 值，
未找到任何内容时为 `None`。`sources` 按根到叶列出每个有贡献的文件以供展示；
`flagged` 列出被丢弃的文件。

## 渲染

```python
from novakit.prompt import render_project_instructions

section = render_project_instructions(instructions)
```

`render_project_instructions` 将文本、来源和被标记文件格式化为单个
提示词片段（为空时返回 `None`）。完整提示词组装请结合
`novakit.prompt.composition` 中的产品中立辅助函数——见
[参考：提示词](../reference/prompt.md)。

## API 参考

::: novakit.prompt.ProjectInstructionConfig
::: novakit.prompt.ProjectInstructions
::: novakit.prompt.discover_project_instructions
