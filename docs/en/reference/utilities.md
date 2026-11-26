# Utilities reference

Standalone helpers with no user-directory or database defaults. The host
always supplies the root or directory.

## Skills

`novakit.skills` parses and scans `SKILL.md` files under a caller-supplied
root and provides a case-insensitive catalog.

::: novakit.skills.SkillCatalog
::: novakit.skills.SkillDocument
::: novakit.skills.SkillSummary
::: novakit.skills.SkillInstallResult
::: novakit.skills.load_skill_document
::: novakit.skills.scan_skills_dir

Example — scan a directory and look a skill up by name:

```python
from pathlib import Path

from novakit.skills import SkillCatalog, scan_skills_dir

summaries = scan_skills_dir(Path("/host/skills"))
catalog = SkillCatalog(summaries)
skill = catalog.get("my-skill")  # case-insensitive
```

`load_skill_document(skill_md_path, skills_dir)` parses one `SKILL.md` file
(validating it stays inside the root); `scan_skills_dir` collects installable
summaries; `SkillParseError` reports malformed frontmatter.

## Checkpoints

`novakit.checkpoints` captures and restores file pre-images under an explicit
root path or root provider. It never chooses a home directory.

::: novakit.checkpoints.CheckpointStore
::: novakit.checkpoints.TurnRecorder
::: novakit.checkpoints.CheckpointSummary
::: novakit.checkpoints.RestoreReport

Example — record one turn, then rewind to it:

```python
from pathlib import Path

from novakit.checkpoints import CheckpointStore

store = CheckpointStore(Path("/host/checkpoints"))
recorder = store.begin_turn(session_id, message_id, "user turn", workspace)
# ... run tools, recording pre-images through the recorder ...
report = store.restore(session_id, message_id, workspace)
```

Wire `store.begin_turn` into `AgentPorts(begin_turn=...)` to capture
checkpoints automatically each turn. `CheckpointSummary` describes one rewind
point; `RestoreReport` lists restored, deleted, and skipped files.
