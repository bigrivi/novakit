"""Direct tests for portable skill scanning and catalog behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from novakit.skills import (
    SkillCatalog,
    SkillParseError,
    SkillSummary,
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


def test_load_skill_document_parses_frontmatter(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    path = _write_skill(
        root,
        "review",
        name="review",
        description="Review code.",
    )

    document = load_skill_document(path, skills_dir=root)

    assert document.name == "review"
    assert document.description == "Review code."
    assert document.allowed_tools == ("read", "grep")
    assert document.body_content == "# Instructions\n\nDo the work.\n"


def test_load_skill_document_rejects_paths_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    outside = _write_skill(tmp_path, "outside", name="outside", description="Nope")

    with pytest.raises(SkillParseError, match="outside skills directory"):
        load_skill_document(outside, skills_dir=root)


def test_scan_skills_dir_skips_invalid_documents(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "valid", name="valid", description="Valid skill")
    invalid = root / "invalid"
    invalid.mkdir(parents=True)
    (invalid / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")

    summaries = scan_skills_dir(root)

    assert [summary.name for summary in summaries] == ["valid"]


def test_catalog_is_case_insensitive_and_deduplicates() -> None:
    first = SkillSummary("Review", "first", "/first", "/first/SKILL.md")
    second = SkillSummary("review", "second", "/second", "/second/SKILL.md")
    catalog = SkillCatalog()

    catalog.replace([first, second])

    assert catalog.list() == [first]
    assert catalog.get("REVIEW") is first
    assert catalog.get("Review") is first
    assert catalog.get("missing") is None


def test_load_skill_document_rejects_missing_and_wrong_filename(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()

    with pytest.raises(SkillParseError, match="not found"):
        load_skill_document(root / "missing" / "SKILL.md", skills_dir=root)

    wrong = root / "README.md"
    wrong.write_text("---\nname: wrong\n---\n", encoding="utf-8")
    with pytest.raises(SkillParseError, match="Unexpected skill file name"):
        load_skill_document(wrong, skills_dir=root)


def test_scan_missing_root_returns_empty(tmp_path: Path) -> None:
    assert scan_skills_dir(tmp_path / "missing") == []


def test_scan_skips_hidden_files_and_duplicate_names(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "first", name="duplicate", description="first")
    _write_skill(root, "second", name="DUPLICATE", description="second")
    _write_skill(root / ".hidden", "hidden", name="hidden", description="hidden")
    (root / "plain-file").write_text("ignored", encoding="utf-8")
    (root / "missing-skill-md").mkdir()

    summaries = scan_skills_dir(root)

    assert [summary.name for summary in summaries] == ["duplicate"]
    assert summaries[0].description == "first"
