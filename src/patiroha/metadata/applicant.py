"""Applicant name normalization.

Splits delimited applicant fields and removes corporate entity suffixes.
"""

from __future__ import annotations

import re

# Corporate entity patterns to remove
_CORPORATE_ENTITIES = [
    "株式会社",
    "有限会社",
    "合資会社",
    "合名会社",
    "合同会社",
    "（株）",
    "㈱",
    "（有）",
    r"\bInc\.?",
    r"\bLtd\.?",
    r"\bCo\.?",
    r"\bCorp\.?",
    r"\bLLC\b",
    r"\bGmbH\b",
    r"\bAG\b",
    r"\bBV\b",
    r"\bB\.V\.",
    r"\bS\.A\.",
    r"\bS\.p\.A\.",
]

_CORPORATE_PATTERN = re.compile("|".join(_CORPORATE_ENTITIES))


def normalize_applicant(text: str, delimiter: str = ";") -> list[str]:
    """Normalize applicant names by splitting and removing corporate entity suffixes.

    Args:
        text: Raw applicant field text.
        delimiter: Separator character between applicant names.

    Returns:
        List of normalized applicant name strings.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    names: list[str] = []
    for part in text.split(delimiter):
        name = part.strip()
        if not name:
            continue
        # Remove corporate entity suffixes
        name = _CORPORATE_PATTERN.sub("", name).strip()
        # Remove leading/trailing whitespace and common separators
        name = re.sub(r"^[\s,;]+|[\s,;]+$", "", name)
        if name:
            names.append(name)

    return names
