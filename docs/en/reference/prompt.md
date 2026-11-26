# Prompt reference

Portable prompt discovery, screening, and section composition. Import from
`novakit.prompt`.

## Discovery and screening

::: novakit.prompt.discover_project_instructions
::: novakit.prompt.ProjectInstructionConfig
::: novakit.prompt.ProjectInstructions
::: novakit.prompt.scan_text
::: novakit.prompt.has_threats

`discover_project_instructions(workspace, config=None)` merges the global
file (host `global_dir`, unscreened) with per-directory project files from
the git root down, screens project files with `has_threats`/`scan_text`
against the configured threat categories, and truncates to `max_block_bytes`.
See [Project instructions](../guides/project-instructions.md) for layering
and defaults (`GLOBAL_FILENAMES`, `PROJECT_FILENAMES`,
`PROJECT_THREAT_CATEGORIES`, `MAX_BLOCK_BYTES`).

## Composition

::: novakit.prompt.join_sections
::: novakit.prompt.render_project_instructions
::: novakit.prompt.PromptEnvironment
::: novakit.prompt.format_available_skills
::: novakit.prompt.format_subagent_roster

`join_sections(*sections)` joins non-empty sections with a blank line;
`render_project_instructions` formats an instruction block with sources and
flagged files; `PromptEnvironment(...).render()` renders the standard
environment bullet list; `format_available_skills` and
`format_subagent_roster` format the abbreviated skill list and sub-agent
roster. Product persona, delegation policy, and final section order remain
with the host.
