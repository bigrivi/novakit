"""Direct tests for portable prompt composition helpers."""

from __future__ import annotations

from types import SimpleNamespace

from novakit.prompt import (
    ProjectInstructions,
    PromptEnvironment,
    format_available_skills,
    format_subagent_roster,
    join_sections,
    render_project_instructions,
)


def test_environment_rendering() -> None:
    environment = PromptEnvironment(
        date="2026-11-26 Thursday",
        home="/home/user",
        workspace="/repo",
        platform="Darwin",
        shell="zsh",
    )
    assert environment.render() == (
        "# Environment\n"
        "- Current date: 2026-11-26 Thursday\n"
        "- Home: /home/user\n"
        "- Workspace: /repo\n"
        "- Platform: Darwin\n"
        "- Shell: zsh"
    )


def test_join_sections_preserves_order_and_skips_empty() -> None:
    assert join_sections(" first ", None, "", "second") == " first \n\nsecond"
    assert join_sections(None, "  ") == "  "


def test_project_instruction_rendering_variants() -> None:
    assert render_project_instructions(None) is None
    assert render_project_instructions(ProjectInstructions(text="")) is None
    rendered = render_project_instructions(
        ProjectInstructions(
            text="Use uv.",
            sources=["/repo/AGENTS.md"],
            flagged=["/repo/pkg/AGENTS.md (role_hijack)"],
        )
    )
    assert rendered == (
        "## Project Instructions\n\n"
        "Use uv.\n\n"
        "In effect from: /repo/AGENTS.md\n\n"
        "Omitted as potential injection: /repo/pkg/AGENTS.md (role_hijack)"
    )
    flagged_only = render_project_instructions(
        ProjectInstructions(text="", flagged=["/repo/AGENTS.md"])
    )
    assert flagged_only is not None
    assert "Omitted as potential injection" in flagged_only


def test_subagent_roster_formatting() -> None:
    records = [
        {"key": "helper", "posture": "read_only", "description": "Helps"},
        {"key": "worker", "description": "  "},
        {"key": "", "description": "ignored"},
    ]
    assert format_subagent_roster(records) == (
        "- `helper` (read-only): Helps\n- `worker` (full access): (no description)"
    )


def test_available_skills_formatting_and_limits() -> None:
    assert format_available_skills(None).startswith("- No skills")
    skills = [
        SimpleNamespace(name="alpha", description="First sentence. More detail."),
        SimpleNamespace(name="beta", description="x" * 100),
        SimpleNamespace(name="", description=""),
    ]
    rendered = format_available_skills(skills)
    assert "- alpha: First sentence" in rendered
    assert "More detail" not in rendered
    assert "- beta: " in rendered
    assert "- unknown-skill: (no description)" in rendered
    assert "Descriptions above are abbreviated" in rendered
