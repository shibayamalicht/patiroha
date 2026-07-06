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

"""Automatic column mapping for patent data."""

from __future__ import annotations

import re
from collections.abc import Sequence

import pandas as pd

# Default keyword mappings for common patent columns
DEFAULT_MAPPINGS: dict[str, list[str]] = {
    "title": ["title", "発明の名称", "名称", "タイトル", "invention_title"],
    "abstract": ["abstract", "要約", "抄録", "概要", "要約書"],
    "claims": ["claims", "請求の範囲", "請求項", "クレーム"],
    "applicant": ["applicant", "出願人", "権利者", "特許権者"],
    "inventor": ["inventor", "発明者"],
    "ipc": ["ipc", "IPC", "国際特許分類", "FI"],
    "date": ["date", "出願日", "公開日", "filing_date", "publication_date"],
    "app_num": ["app_num", "出願番号", "application_number", "公開番号"],
}


def smart_map_columns(
    df: pd.DataFrame,
    mappings: dict[str, list[str]] | None = None,
) -> dict[str, str | None]:
    """Automatically map DataFrame columns to standard patent field names.

    Uses keyword matching (exact then substring) to find the best column match.

    Args:
        df: DataFrame with patent data.
        mappings: Custom keyword mappings. If None, uses DEFAULT_MAPPINGS.

    Returns:
        Dict mapping standard field names to actual column names (None if not found).
    """
    if mappings is None:
        mappings = DEFAULT_MAPPINGS

    columns = list(df.columns)
    result: dict[str, str | None] = {}

    for field_name, keywords in mappings.items():
        matched = _find_column(columns, keywords)
        result[field_name] = matched

    return result


def _find_column(columns: Sequence[str], keywords: list[str]) -> str | None:
    """Find a column matching the given keywords.

    Matching is case-insensitive. ASCII keywords use boundary-aware substring
    matching so that e.g. "date" does not match "candidate_id" and "applicant"
    does not match "non_applicant_flag". Longer (more specific) keywords win.
    """
    # 1) Exact match (case-insensitive)
    for kw in keywords:
        for col in columns:
            if kw.lower() == str(col).lower():
                return str(col)

    # 2) Boundary-aware substring match, most-specific (longest) keyword first
    for kw in sorted(keywords, key=len, reverse=True):
        for col in columns:
            c = str(col)
            if kw.isascii():
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(kw)}(?![A-Za-z0-9])", c, re.IGNORECASE):
                    return c
            elif kw in c:
                return c

    return None
