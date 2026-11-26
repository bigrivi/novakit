# Project instructions

`novakit.prompt.discover_project_instructions`
(`novakit/prompt/project_instructions.py`) builds one merged instruction block
from host-supplied global files and per-directory project files. `novakit`
never reads `HOME` itself — the host supplies the global directory and policy
through `ProjectInstructionConfig`.

## Discovery

```python
from pathlib import Path

from novakit.prompt import ProjectInstructionConfig, discover_project_instructions

instructions = discover_project_instructions(
    workspace,
    ProjectInstructionConfig(global_dir=Path("/host/instructions")),
)
```

Layering and precedence:

1. **Global layer** — at most one file from `global_dir`, chosen by
   `global_filenames` precedence (`GLOBAL_FILENAMES =
   ("AGENTS.override.md", "AGENTS.md")`), unscreened. `global_dir=None`
   skips it.
2. **Project layer** — from the git root down to the workspace, at most one
   file per directory by `project_filenames` precedence
   (`PROJECT_FILENAMES = ("AGENTS.override.md", "AGENTS.md", "CLAUDE.md")`),
   merged root-to-leaf. Files outside the git root chain are ignored.
3. **Screening** — each project file is checked with `has_threats` for the
   configured `threat_categories` (default `PROJECT_THREAT_CATEGORIES =
   ["ignore_previous", "role_hijack"]`, a deliberately narrow set so normal
   safety rules do not false-positive). Flagged files are dropped and listed
   in `ProjectInstructions.flagged` with their matched category.
4. **Byte budget** — the merged block is truncated from the broadest end to
   `max_block_bytes` (default `MAX_BLOCK_BYTES = 32 * 1024` UTF-8 bytes).

The result is a `ProjectInstructions(text, sources, flagged)` value, or
`None` when nothing was found. `sources` lists every contributing file
root-to-leaf for display; `flagged` lists dropped files.

## Rendering

```python
from novakit.prompt import render_project_instructions

section = render_project_instructions(instructions)
```

`render_project_instructions` formats text, sources, and flagged files as one
prompt section (or `None` when empty). For full prompt assembly, combine with
the product-neutral helpers in `novakit.prompt.composition` — see
[Reference: prompt](../reference/prompt.md).

## API reference

::: novakit.prompt.ProjectInstructionConfig
::: novakit.prompt.ProjectInstructions
::: novakit.prompt.discover_project_instructions
