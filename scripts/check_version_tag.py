"""Validate that a release tag matches package and project versions."""

from __future__ import annotations

import argparse
import ast
import re
import tomllib
from pathlib import Path

_TAG_PATTERN = re.compile(r"^v(?P<version>\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)$")


def package_version(path: Path) -> str:
    """Read ``__version__`` from a Python module without importing it.

    Args:
        path: Python source file containing the version assignment.

    Returns:
        The declared package version.

    Raises:
        ValueError: When no literal string ``__version__`` assignment exists.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            ) and isinstance(node.value, ast.Constant):
                value = node.value.value
                if isinstance(value, str):
                    return value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ValueError(f"{path} does not declare a literal string __version__")


def project_version(path: Path) -> str:
    """Read the project version from ``pyproject.toml``.

    Args:
        path: Path to the TOML project file.

    Returns:
        The declared project version.

    Raises:
        ValueError: When the project version is missing or not a string.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    value = data.get("project", {}).get("version")
    if not isinstance(value, str):
        raise ValueError(f"{path} does not declare project.version")
    return value


def validate_tag(tag: str, package: str, project: str) -> str:
    """Validate one release tag against both version declarations.

    Args:
        tag: Git tag, expected in ``vX.Y.Z`` form.
        package: Version declared by the Python package.
        project: Version declared by the project metadata.

    Returns:
        The version shared by the tag and both declarations.

    Raises:
        ValueError: When the tag is malformed or any version differs.
    """
    match = _TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ValueError(f"invalid release tag {tag!r}; expected vX.Y.Z")
    version = match.group("version")
    if package != project:
        raise ValueError(f"version mismatch: package={package!r}, project={project!r}")
    if version != package:
        raise ValueError(
            f"tag {tag!r} does not match package/project version {package!r}"
        )
    return version


def main(argv: list[str] | None = None) -> int:
    """Run the release-tag check from the command line.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Zero when the tag and versions agree.

    Raises:
        SystemExit: When validation fails or the tag is omitted.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Git tag, for example v0.1.0")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing pyproject.toml and novakit/",
    )
    args = parser.parse_args(argv)
    try:
        version = validate_tag(
            args.tag,
            package_version(args.root / "novakit" / "version.py"),
            project_version(args.root / "pyproject.toml"),
        )
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        parser.error(str(error))
    print(f"Release tag {args.tag} matches version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
