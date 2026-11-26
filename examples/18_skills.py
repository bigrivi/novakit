"""Skills: write SKILL.md packages, scan them, and query case-insensitively.

What you'll learn:
  - How a SKILL.md frontmatter block maps to a skill document.
  - How directory scans skip invalid packages and dedupe names.
  - How the catalog answers case-insensitive lookups.

Run:
    python examples/18_skills.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from novakit.skills import (
    SkillCatalog,
    load_skill_document,
    scan_skills_dir,
)


def _write_skill(
    root: Path,
    directory: str,
    *,
    name: str,
    description: str,
    allowed_tools: str = "[read, grep]",
) -> Path:
    skill_dir = root / directory
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"allowed-tools: {allowed_tools}\n"
        "---\n\n"
        "# Instructions\n\nDo the work.\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    # Prepare data: skill packages live in tempdir so nothing touches home;
    # includes a duplicate name and a broken package to show scan handling.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "skills"
        review_md = _write_skill(
            root, "review", name="review", description="Review code carefully."
        )
        _write_skill(
            root,
            "Plan",
            name="Plan",
            description="Draft an implementation plan.",
            allowed_tools="[read]",
        )
        _write_skill(
            root,
            "second-review",
            name="REVIEW",
            description="Duplicate name, loses dedup.",
        )
        broken = root / "broken"
        broken.mkdir(parents=True)
        (broken / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")

        # Call SDK: single-document load parses frontmatter into fields.
        document = load_skill_document(review_md, skills_dir=root)
        print("SKILL.md frontmatter -> document:")
        print(f"  name: {document.name}")
        print(f"  description: {document.description}")
        print(f"  allowed_tools: {document.allowed_tools}")
        print(f"  body: {document.body_content!r}")

        # Call SDK: the scan skips the broken package (no frontmatter) and
        # keeps only one of the two "review" names, so output stays clean.
        summaries = scan_skills_dir(root)
        print(f"scan names: {sorted(s.name for s in summaries)}")
        print(f"scan missing dir: {scan_skills_dir(root / 'missing')}")
        for summary in summaries:
            print(f"  {summary.name}: {summary.description}")

        # Call SDK: catalog lookup ignores case, which is why "REVIEW" and
        # "review" resolve to the same entry.
        catalog = SkillCatalog()
        catalog.replace(summaries)
        print("catalog queries:")
        print(f"  REVIEW -> {catalog.get('REVIEW')}")
        print(f"  review is REVIEW: {catalog.get('review') is catalog.get('REVIEW')}")
        print(f"  plan -> {catalog.get('plan')}")
        print(f"  PLAN is plan: {catalog.get('PLAN') is catalog.get('plan')}")
        print(f"  missing -> {catalog.get('missing')}")
        print(f"catalog list: {[s.name for s in catalog.list()]}")


if __name__ == "__main__":
    main()
