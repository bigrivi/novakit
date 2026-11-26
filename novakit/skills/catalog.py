"""In-memory skill catalog."""

from __future__ import annotations

from novakit.skills.models import SkillSummary


class SkillCatalog:
    """Case-insensitive in-memory catalog of discovered skills."""

    def __init__(self) -> None:
        self._items: dict[str, SkillSummary] = {}
        self._keys_by_name: dict[str, str] = {}

    def replace(self, summaries: list[SkillSummary]) -> None:
        """Replace the catalog, keeping the first case-insensitive duplicate.

        Args:
            summaries: Discovered skill summaries.
        """
        items: dict[str, SkillSummary] = {}
        keys_by_name: dict[str, str] = {}
        for summary in sorted(
            summaries, key=lambda item: (item.name.lower(), item.path.lower())
        ):
            lookup_key = summary.name.lower()
            if lookup_key in keys_by_name:
                continue
            items[summary.name] = summary
            keys_by_name[lookup_key] = summary.name
        self._items = items
        self._keys_by_name = keys_by_name

    def list(self) -> list[SkillSummary]:
        """Return all catalog entries in deterministic name order."""
        return list(self._items.values())

    def get(self, skill_name: str) -> SkillSummary | None:
        """Look up a skill by exact or case-insensitive name.

        Args:
            skill_name: Requested skill name.

        Returns:
            The matching summary, or None when absent.
        """
        direct = self._items.get(skill_name)
        if direct is not None:
            return direct
        mapped_name = self._keys_by_name.get(skill_name.lower())
        if mapped_name is None:
            return None
        return self._items.get(mapped_name)
