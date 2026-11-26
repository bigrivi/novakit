# 提示词参考

可移植的提示词发现、筛查与片段组装。从 `novakit.prompt`
导入。

## 发现与筛查

::: novakit.prompt.discover_project_instructions
::: novakit.prompt.ProjectInstructionConfig
::: novakit.prompt.ProjectInstructions
::: novakit.prompt.scan_text
::: novakit.prompt.has_threats

`discover_project_instructions(workspace, config=None)` 合并全局文件
（宿主 `global_dir`，不筛查）与从 git 根目录向下逐目录的项目文件，
用 `has_threats`/`scan_text` 按配置的威胁类别筛查项目文件，
并截断到 `max_block_bytes`。分层与默认值见
[项目指令](../guides/project-instructions.md)（`GLOBAL_FILENAMES`、
`PROJECT_FILENAMES`、`PROJECT_THREAT_CATEGORIES`、
`MAX_BLOCK_BYTES`）。

## 组装

::: novakit.prompt.join_sections
::: novakit.prompt.render_project_instructions
::: novakit.prompt.PromptEnvironment
::: novakit.prompt.format_available_skills
::: novakit.prompt.format_subagent_roster

`join_sections(*sections)` 用空行连接非空片段；
`render_project_instructions` 格式化带来源和标旗文件的指令块；
`PromptEnvironment(...).render()` 渲染标准环境要点列表；
`format_available_skills` 和 `format_subagent_roster` 分别格式化
精简技能列表和子智能体名单。产品人设、委派策略和最终片段顺序
仍由宿主决定。
