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
    # Numeric dtype needs explicit handling: pd.to_datetime treats integers as
    # nanoseconds since 1970, silently corrupting year / YYYYMMDD / Excel-serial
    # values (common after read_excel of numeric date columns). Route by magnitude.
    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        non_na = numeric_series.dropna()
        if not non_na.empty:
            median = float(non_na.median())
            as_str = numeric_series.astype("Int64").astype(str).replace("<NA>", "")
            if median >= 1_000_000:  # 8-digit YYYYMMDD
                return pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")
            if 1000 <= median <= 9999:  # 4-digit bare year
                return pd.to_datetime(as_str, format="%Y", errors="coerce")
            if 20000 <= median <= 80000:  # Excel serial (days since 1899-12-30)
                return pd.to_datetime(numeric_series, unit="D", origin="1899-12-30", errors="coerce")
        # Fall through: stringify and let the generic strategies try.
        series = numeric_series.astype("Int64").astype(str).replace("<NA>", "")

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
