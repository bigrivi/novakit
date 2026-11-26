# 工具函数参考

无用户目录或数据库默认值的独立辅助函数。根目录或目标目录
始终由宿主提供。

## 技能

`novakit.skills` 在调用方提供的根目录下解析并扫描 `SKILL.md`
文件，提供大小写不敏感的目录。

::: novakit.skills.SkillCatalog
::: novakit.skills.SkillDocument
::: novakit.skills.SkillSummary
::: novakit.skills.SkillInstallResult
::: novakit.skills.load_skill_document
::: novakit.skills.scan_skills_dir

示例——扫描目录并按名称查找技能：

```python
from pathlib import Path

from novakit.skills import SkillCatalog, scan_skills_dir

summaries = scan_skills_dir(Path("/host/skills"))
catalog = SkillCatalog(summaries)
skill = catalog.get("my-skill")  # case-insensitive
```

`load_skill_document(skill_md_path, skills_dir)` 解析单个 `SKILL.md`
文件（校验其位于根目录内）；`scan_skills_dir` 收集可安装的摘要；
`SkillParseError` 报告格式错误的 frontmatter。

## 检查点

`novakit.checkpoints` 在显式根路径或根目录提供器下捕获并恢复文件
前像。它从不自行选择家目录。

::: novakit.checkpoints.CheckpointStore
::: novakit.checkpoints.TurnRecorder
::: novakit.checkpoints.CheckpointSummary
::: novakit.checkpoints.RestoreReport

示例——记录一轮，然后回滚到该轮：

```python
from pathlib import Path

from novakit.checkpoints import CheckpointStore

store = CheckpointStore(Path("/host/checkpoints"))
recorder = store.begin_turn(session_id, message_id, "user turn", workspace)
# ... run tools, recording pre-images through the recorder ...
report = store.restore(session_id, message_id, workspace)
```

将 `store.begin_turn` 接入 `AgentPorts(begin_turn=...)` 即可每轮自动
捕获检查点。`CheckpointSummary` 描述一个回滚点；
`RestoreReport` 列出已恢复、已删除和已跳过的文件。
