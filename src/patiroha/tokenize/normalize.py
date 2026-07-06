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

"""Text normalization utilities."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def normalize_text(text: object) -> str:
    """Normalize text with NFKC, whitespace cleanup, and special character fixes.

    Args:
        text: Input text (non-string values are converted).

    Returns:
        Normalized text string.
    """
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("µ", "μ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_html(text: str) -> str:
    """Remove HTML/XML tags from text.

    Args:
        text: Input text possibly containing HTML tags.

    Returns:
        Text with HTML tags removed.
    """
    return re.sub(r"<[^>]+>", " ", text)
