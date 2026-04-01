"""Stopword manager with category-based selection and full/half-width expansion."""

from __future__ import annotations

import string
from collections.abc import Iterable, Sequence

from patiroha.stopwords.catalog import CATEGORIES, NPL_CATEGORIES, PATENT_CATEGORIES


def _get_expanded_set(word_list: Iterable[str]) -> frozenset[str]:
    """Expand half-width ASCII to full-width equivalents and return a frozenset."""
    expanded: set[str] = set(word_list)
    hankaku = string.ascii_letters + string.digits
    zenkaku = (
        "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
        "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
        "０１２３４５６７８９"
    )
    trans = str.maketrans(hankaku, zenkaku)
    for w in word_list:
        if any(c in hankaku for c in w):
            expanded.add(w.translate(trans))
    return frozenset(expanded)


def get_stopwords(mode: str = "patent") -> frozenset[str]:
    """Get stopwords as a frozenset.

    Args:
        mode: "patent" (default) excludes NPL terms, "npl" includes all.

    Returns:
        Frozenset of stopwords with half/full-width variants.
    """
    categories = NPL_CATEGORIES if mode == "npl" else PATENT_CATEGORIES
    words: list[str] = []
    for cat in categories:
        words.extend(CATEGORIES[cat])
    return _get_expanded_set(words)


def list_categories() -> dict[str, int]:
    """List all available stopword categories with word counts.

    Returns:
        Dict mapping category name to number of words (before expansion).

    Example:
        >>> list_categories()
        {"general": 90, "patent_terms": 100, "structure": 60, ...}
    """
    return {name: len(words) for name, words in CATEGORIES.items()}


def list_words(category: str) -> list[str]:
    """List all words in a specific stopword category.

    Args:
        category: Category name (e.g. "general", "patent_terms", "chemistry").

    Returns:
        Sorted list of words in the category (before half/full-width expansion).

    Raises:
        KeyError: If category name is not found.

    Example:
        >>> list_words("chemistry")[:3]
        ["含有", "含有量", "反応"]
    """
    if category not in CATEGORIES:
        available = ", ".join(sorted(CATEGORIES.keys()))
        raise KeyError(f"Unknown category: {category!r}. Available: {available}")
    return sorted(CATEGORIES[category])


class StopwordManager:
    """Flexible stopword manager with category-based include/exclude.

    Args:
        include: Category names to include. If None, uses all patent categories.
        exclude: Category names to exclude from the include set.
    """

    def __init__(
        self,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> None:
        if include is None:
            self._categories = list(PATENT_CATEGORIES)
        else:
            self._categories = [c for c in include if c in CATEGORIES]

        if exclude:
            self._categories = [c for c in self._categories if c not in exclude]

        self._added: set[str] = set()
        self._removed: set[str] = set()

    def add(self, words: Iterable[str]) -> None:
        """Add custom stopwords."""
        self._added.update(words)
        self._removed -= set(words)

    def remove(self, words: Iterable[str]) -> None:
        """Remove words from the stopword set (e.g. analysis target terms)."""
        self._removed.update(words)
        self._added -= set(words)

    def build(self) -> frozenset[str]:
        """Build the final stopword frozenset."""
        base: list[str] = []
        for cat in self._categories:
            base.extend(CATEGORIES[cat])
        base.extend(self._added)
        expanded = set(_get_expanded_set(base))
        expanded -= self._removed
        return frozenset(expanded)

    def list_active_words(self) -> dict[str, list[str]]:
        """List words in each active category, plus custom additions.

        Returns:
            Dict mapping category name (or "custom") to sorted word list.

        Example:
            >>> mgr = StopwordManager(include=["general"])
            >>> mgr.add(["自社用語"])
            >>> mgr.list_active_words().keys()
            dict_keys(["general", "custom"])
        """
        result: dict[str, list[str]] = {}
        for cat in self._categories:
            result[cat] = sorted(CATEGORIES[cat])
        if self._added:
            result["custom"] = sorted(self._added)
        return result

    def summary(self) -> str:
        """Return a human-readable summary of the current configuration.

        Returns:
            Multi-line string showing categories, counts, and customizations.
        """
        lines = ["StopwordManager Configuration:"]
        lines.append(f"  Active categories: {', '.join(self._categories)}")
        total = 0
        for cat in self._categories:
            n = len(CATEGORIES[cat])
            total += n
            lines.append(f"    {cat}: {n} words")
        if self._added:
            lines.append(f"  Custom added: {len(self._added)} words — {', '.join(sorted(self._added))}")
            total += len(self._added)
        if self._removed:
            lines.append(f"  Removed: {len(self._removed)} words — {', '.join(sorted(self._removed))}")
        built = self.build()
        lines.append(f"  Total after expansion: {len(built)} words")
        return "\n".join(lines)

    @property
    def categories(self) -> list[str]:
        """Currently selected categories."""
        return list(self._categories)

    @property
    def added_words(self) -> frozenset[str]:
        """Custom words added by the user."""
        return frozenset(self._added)

    @property
    def removed_words(self) -> frozenset[str]:
        """Words explicitly removed."""
        return frozenset(self._removed)
