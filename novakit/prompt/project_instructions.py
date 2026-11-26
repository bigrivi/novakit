"""Portable project-instruction discovery and screening.

The host supplies the global instruction directory and filename policy. Project
files are discovered from the git root down to the workspace, screened before
they enter the prompt, merged root-to-leaf, and bounded by a byte budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from novakit.prompt.threat_patterns import has_threats, scan_text

log = logging.getLogger(__name__)

GLOBAL_FILENAMES = ("AGENTS.override.md", "AGENTS.md")
PROJECT_FILENAMES = ("AGENTS.override.md", "AGENTS.md", "CLAUDE.md")
# Only categories a legitimate convention file should never contain. A full
# screen would flag normal safety rules ("never reveal your system prompt"
# trips prompt_leak) and drop the whole file.
PROJECT_THREAT_CATEGORIES = ["ignore_previous", "role_hijack"]
MAX_BLOCK_BYTES = 32 * 1024


@dataclass(frozen=True)
class ProjectInstructionConfig:
    """Host policy for instruction discovery and screening.

    Attributes:
        global_dir: Host-owned global instruction directory, or None to skip it.
        global_filenames: Global filename precedence, highest first.
        project_filenames: Per-directory project filename precedence.
        threat_categories: High-confidence categories used to screen project files.
        max_block_bytes: Maximum UTF-8 bytes retained in the merged block.
    """

    global_dir: Path | None = None
    global_filenames: tuple[str, ...] = GLOBAL_FILENAMES
    project_filenames: tuple[str, ...] = PROJECT_FILENAMES
    threat_categories: tuple[str, ...] = tuple(PROJECT_THREAT_CATEGORIES)
    max_block_bytes: int = MAX_BLOCK_BYTES


@dataclass
class ProjectInstructions:
    """Merged instruction block plus where it came from.

    ``sources`` lists every file that contributed, root-to-leaf, so the prompt
    can show the user what is in effect. ``flagged`` lists project files that
    were dropped as potential injection, with the matched category.
    """

    text: str
    sources: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)


def _read_file(path: Path) -> str | None:
    """Read one file; unreadable means skipped, never fatal."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        log.warning("Skipping unreadable instruction file %s: %s", path, exc)
        return None
    if not text.strip():
        return None
    return text


def _first_present(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _find_git_root(start: Path) -> Path | None:
    """Nearest ancestor (including self) containing ``.git``."""
    current = start
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def discover_project_instructions(
    workspace: str,
    config: ProjectInstructionConfig | None = None,
) -> ProjectInstructions | None:
    """Build the merged instruction block for a workspace.

    Args:
        workspace: Workspace directory whose instruction chain should be loaded.
        config: Host filename, global-directory, screening, and size policy.

    Returns:
        Merged instructions and provenance, or None when nothing was found.
    """
    cfg = config or ProjectInstructionConfig()
    blocks: list[tuple[str, str]] = []
    flagged: list[str] = []
    seen: set[Path] = set()

    def take(path: Path, screen: bool) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        text = _read_file(path)
        if text is None:
            return
        if screen and has_threats(text, categories=list(cfg.threat_categories)):
            categories = sorted(
                {
                    hit["category"]
                    for hit in scan_text(text, categories=list(cfg.threat_categories))
                }
            )
            flagged.append(f"{path} ({', '.join(categories)})")
            log.warning("Dropping flagged instruction file %s", path)
            return
        blocks.append((str(path), text))

    # Global layer: the user's own, at most one file, unscreened.
    if cfg.global_dir is not None:
        global_hit = _first_present(cfg.global_dir, cfg.global_filenames)
        if global_hit is not None:
            take(global_hit, screen=False)

    # Project layer: git root down to the workspace, at most one per level.
    try:
        workdir = Path(workspace).expanduser().resolve()
    except OSError:
        workdir = Path(workspace).expanduser()
    if workdir.is_dir():
        git_root = _find_git_root(workdir)
        if git_root is None:
            levels = [workdir]
        else:
            levels = []
            current = workdir
            while True:
                levels.append(current)
                if current == git_root:
                    break
                current = current.parent
            levels.reverse()
        for level in levels:
            hit = _first_present(level, cfg.project_filenames)
            if hit is not None:
                take(hit, screen=True)

    if not blocks and not flagged:
        return None

    sources = [path for path, _ in blocks]
    chunks = [f"<!-- from {path} -->\n{text}" for path, text in blocks]
    notice = (
        f"[Earlier (broader) project instructions truncated at "
        f"{cfg.max_block_bytes // 1024} KiB]"
    )
    body = "\n\n".join(chunks)
    if len(body.encode("utf-8")) > cfg.max_block_bytes:
        # Trim from the root end so the most specific file survives.
        kept: list[str] = []
        kept_sources: list[str] = []
        for (path, _), chunk in zip(reversed(blocks), reversed(chunks), strict=True):
            trial = "\n\n".join([chunk, *kept, notice])
            if kept and len(trial.encode("utf-8")) > cfg.max_block_bytes:
                break
            kept.insert(0, chunk)
            kept_sources.insert(0, path)
        sources = kept_sources
        body = "\n\n".join([*kept, notice])

    return ProjectInstructions(text=body, sources=sources, flagged=flagged)


__all__ = [
    "GLOBAL_FILENAMES",
    "MAX_BLOCK_BYTES",
    "PROJECT_FILENAMES",
    "PROJECT_THREAT_CATEGORIES",
    "ProjectInstructionConfig",
    "ProjectInstructions",
    "discover_project_instructions",
]
