"""Robust date parsing for patent application dates."""

from __future__ import annotations

from typing import Any

import pandas as pd


def parse_date(series: pd.Series[Any]) -> pd.Series[Any]:
    """Parse dates from a Series using multiple fallback strategies.

    Tries these strategies in order:
    1. Automatic parsing (pandas default)
    2. YYYYMMDD format
    3. Year-only format
    4. Excel serial date (numeric > 30000)

    Args:
        series: Pandas Series containing date values in various formats.

    Returns:
        Series with datetime64 values (NaT for unparseable entries).
    """
    # Strategy 1: Automatic parsing
    parsed: pd.Series[Any] = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() > 0.5:
        return parsed

    # Strategy 2: YYYYMMDD format
    parsed = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    if parsed.notna().mean() > 0.5:
        return parsed

    # Strategy 3: Year-only format
    parsed = pd.to_datetime(series, format="%Y", errors="coerce")
    if parsed.notna().mean() > 0.5:
        return parsed

    # Strategy 4: Excel serial date
    try:
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().sum() > 0 and numeric_series.mean() > 30000:
            parsed = pd.to_datetime(numeric_series, unit="D", origin="1899-12-30", errors="coerce")
            return parsed
    except Exception:
        pass

    return parsed
