# Copyright 2026 しばやま (shibayamalicht)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Applicant name normalization.

Splits delimited applicant fields and removes corporate entity suffixes.
"""

from __future__ import annotations

import re
import unicodedata

# Corporate entity patterns to remove.
# English suffixes are anchored with \b on BOTH sides (and matched
# case-insensitively) so that "Co"/"Corp"/"AG" etc. only match standalone tokens
# and never eat into a company body (e.g. Corporation, Cobalt, Continental).
# Longer alternatives precede their prefixes. Parenthesized Japanese forms are
# written in their NFKC-normalized (half-width paren) shape; input is normalized
# before matching.
_CORPORATE_ENTITIES = [
    "株式会社",
    "有限会社",
    "合資会社",
    "合名会社",
    "合同会社",
    r"\(株\)",
    "㈱",
    r"\(有\)",
    r"(?i:\bCorp(?:oration)?\b)\.?",
    r"(?i:\bIncorporated\b|\bInc\b)\.?",
    r"(?i:\bLtd\b)\.?",
    r"(?i:\bCo\b)\.?",
    r"(?i:\bLLC\b)",
    r"(?i:\bGmbH\b)",
    r"(?i:\bAG\b)",
    r"(?i:\bB\.?V\b)\.?",
    r"(?i:\bS\.?A\b)\.?",
    r"(?i:\bS\.?p\.?A\b)\.?",
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

    text = unicodedata.normalize("NFKC", text)

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
