"""Reusable prompt-section composition without product persona or policy."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from novakit.prompt.project_instructions import ProjectInstructions


@dataclass(frozen=True)
class PromptEnvironment:
    """Values rendered into a host's environment prompt section."""

    date: str
    home: str
    workspace: str
    platform: str
    shell: str

    def render(self) -> str:
        """Render the standard environment bullet list."""
        return (
            "# Environment\n"
            f"- Current date: {self.date}\n"
            f"- Home: {self.home}\n"
            f"- Workspace: {self.workspace}\n"
            f"- Platform: {self.platform}\n"
            f"- Shell: {self.shell}"
        )


def join_sections(*sections: str | None) -> str:
    """Join non-empty prompt sections with a blank line.

    Args:
        *sections: Ordered section text; empty sections are omitted.

    Returns:
        The combined prompt text without leading or trailing blank lines.
    """
    return "\n\n".join(section for section in sections if section)


def render_project_instructions(
    instructions: ProjectInstructions | None,
) -> str | None:
    """Render instruction text, sources, and individually flagged files.

    Args:
        instructions: Discovered project instruction block.

    Returns:
        A prompt section, or None when there is nothing to report.
    """
    if instructions is None:
        return None
    if not instructions.text.strip() and not instructions.flagged:
        return None
    lines = ["## Project Instructions", ""]
    if instructions.text.strip():
        lines.append(instructions.text)
        lines.append("")
    if instructions.sources:
        lines.append("In effect from: " + ", ".join(instructions.sources))
        lines.append("")
    if instructions.flagged:
        lines.append(
            "Omitted as potential injection: " + "; ".join(instructions.flagged)
        )
    return "\n".join(lines).rstrip()


def format_subagent_roster(records: Iterable[dict[str, Any]]) -> str:
    """Format persisted child records as a compact roster.

    Args:
        records: Child agent rows containing key, posture, and description.

    Returns:
        Newline-separated roster entries, or an empty string when none are usable.
    """
    lines: list[str] = []
    for child in records:
        key = str(child.get("key") or "").strip()
        if not key:
            continue
        access = "read-only" if child.get("posture") == "read_only" else "full access"
        description = str(child.get("description") or "").strip() or "(no description)"
        lines.append(f"- `{key}` ({access}): {description}")
    return "\n".join(lines)


def format_available_skills(skills: Sequence[object] | None) -> str:
    """Format abbreviated skill summaries for a system prompt.

    Args:
        skills: Skill summaries exposing name and description attributes.

    Returns:
        A compact skill list, including the host-neutral retrieval reminder.
    """
    if not skills:
        return "- No skills currently installed in the runtime catalog."

    lines: list[str] = []
    for skill in skills:
        name = str(getattr(skill, "name", "") or "").strip() or "unknown-skill"
        description = (
            str(getattr(skill, "description", "") or "").strip() or "(no description)"
        )
        first_sentence = description.split(". ")[0].split("\n")[0].strip()
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:77].rstrip() + "..."
        lines.append(f"- {name}: {first_sentence}")
    lines.append(
        "- Descriptions above are abbreviated. Call `list_skills` for full "
        "details, or `load_skill` with the exact name once you have chosen."
    )
    return "\n".join(lines)


__all__ = [
    "PromptEnvironment",
    "format_available_skills",
    "format_subagent_roster",
    "join_sections",
    "render_project_instructions",
]
