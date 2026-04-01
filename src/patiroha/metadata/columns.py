"""Automatic column mapping for patent data."""

from __future__ import annotations

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
    """Find a column matching the given keywords."""
    # Exact match first
    for kw in keywords:
        for col in columns:
            if kw == str(col):
                return str(col)

    # Substring match
    for kw in keywords:
        for col in columns:
            if kw in str(col):
                return str(col)

    return None
