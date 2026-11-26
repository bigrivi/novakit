"""Tests for release tag/version consistency checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_version_tag import (
    package_version,
    project_version,
    validate_tag,
)


def test_validate_tag_accepts_matching_versions() -> None:
    assert validate_tag("v0.1.0", "0.1.0", "0.1.0") == "0.1.0"
    assert validate_tag("v1.2.3-rc1", "1.2.3-rc1", "1.2.3-rc1") == "1.2.3-rc1"


@pytest.mark.parametrize("tag", ["0.1.0", "v0.1", "release-v0.1.0", "v0.1.0/evil"])
def test_validate_tag_rejects_malformed_tags(tag: str) -> None:
    with pytest.raises(ValueError, match="invalid release tag"):
        validate_tag(tag, "0.1.0", "0.1.0")


def test_validate_tag_rejects_mismatches() -> None:
    with pytest.raises(ValueError, match="version mismatch"):
        validate_tag("v0.1.0", "0.1.0", "0.2.0")
    with pytest.raises(ValueError, match="does not match"):
        validate_tag("v0.2.0", "0.1.0", "0.1.0")


def test_version_readers(tmp_path: Path) -> None:
    module = tmp_path / "version.py"
    module.write_text('__version__: str = "1.2.3"\n', encoding="utf-8")
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")

    assert package_version(module) == "1.2.3"
    assert project_version(project) == "1.2.3"


def test_version_readers_reject_missing_values(tmp_path: Path) -> None:
    module = tmp_path / "version.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    project = tmp_path / "pyproject.toml"
    project.write_text("[project]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="__version__"):
        package_version(module)
    with pytest.raises(ValueError, match=r"project\.version"):
        project_version(project)
