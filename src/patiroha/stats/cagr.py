"""Compound Annual Growth Rate (CAGR) and trend analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from patiroha._types import CAGRResult


def calculate_cagr(df: pd.DataFrame, year_col: str = "year") -> CAGRResult:
    """Calculate CAGR and trend direction from year-based patent counts.

    Trend classification (based on linear regression slope):
    - slope > 0.5: Steep rise
    - 0 < slope <= 0.5: Growth
    - -0.5 <= slope <= 0: Decline
    - slope < -0.5: Collapse

    Args:
        df: DataFrame containing a year column.
        year_col: Name of the year column.

    Returns:
        CAGRResult with growth_rate and trend string.
    """
    if year_col not in df.columns:
        return CAGRResult(growth_rate=None, trend=None)

    years = df[year_col].dropna().astype(int)
    if years.empty:
        return CAGRResult(growth_rate=None, trend=None)

    counts = years.value_counts().sort_index()
    if len(counts) < 2:
        return CAGRResult(growth_rate=0.0, trend="Stable")

    y_vals = counts.index.values.astype(float)
    c_vals = counts.values.astype(float)

    # Trend via linear regression
    try:
        coeffs = np.polyfit(y_vals, c_vals, 1)
        slope = float(coeffs[0])
        if slope > 0.5:
            trend = "急上昇"
        elif slope > 0:
            trend = "増加傾向"
        elif slope > -0.5:
            trend = "減少傾向"
        else:
            trend = "失速"
    except Exception:
        trend = "不明"

    # CAGR: (end/start)^(1/n) - 1
    try:
        start_val = float(c_vals[0]) if c_vals[0] > 0 else 1.0
        end_val = float(c_vals[-1])
        n_years = max(1, int(y_vals[-1] - y_vals[0]))
        cagr = (end_val / start_val) ** (1 / n_years) - 1
    except Exception:
        cagr = 0.0

    return CAGRResult(growth_rate=cagr, trend=trend)
