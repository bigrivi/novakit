"""Portable prompt discovery and untrusted-text screening."""

from novakit.prompt.composition import (
    PromptEnvironment,
    format_available_skills,
    format_subagent_roster,
    join_sections,
    render_project_instructions,
)
from novakit.prompt.project_instructions import (
    GLOBAL_FILENAMES,
    MAX_BLOCK_BYTES,
    PROJECT_FILENAMES,
    PROJECT_THREAT_CATEGORIES,
    ProjectInstructionConfig,
    ProjectInstructions,
    discover_project_instructions,
)
from novakit.prompt.threat_patterns import (
    THREAT_PATTERNS,
    has_threats,
    scan_text,
)

__all__ = [
    "GLOBAL_FILENAMES",
    "MAX_BLOCK_BYTES",
    "PROJECT_FILENAMES",
    "PROJECT_THREAT_CATEGORIES",
    "THREAT_PATTERNS",
    "ProjectInstructionConfig",
    "ProjectInstructions",
    "PromptEnvironment",
    "discover_project_instructions",
    "format_available_skills",
    "format_subagent_roster",
    "has_threats",
    "join_sections",
    "render_project_instructions",
    "scan_text",
]
