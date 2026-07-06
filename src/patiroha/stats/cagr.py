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

"""Compound Annual Growth Rate (CAGR) and trend analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from patiroha._types import CAGRResult


def calculate_cagr(df: pd.DataFrame, year_col: str = "year") -> CAGRResult:
    """Calculate CAGR and trend direction from year-based patent counts.

    Trend classification (based on linear regression slope, with a small
    dead-band around zero so flat data is reported as flat):
    - slope > 0.5: Steep rise (急上昇)
    - eps < slope <= 0.5: Growth (増加傾向)
    - |slope| <= eps: Flat (横ばい)
    - -0.5 < slope < -eps: Decline (減少傾向)
    - slope <= -0.5: Collapse (失速)
    Single-year input returns ("横ばい", growth_rate 0.0).

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
        return CAGRResult(growth_rate=0.0, trend="横ばい")

    y_vals = counts.index.values.astype(float)
    c_vals = counts.values.astype(float)

    # Trend via linear regression
    try:
        coeffs = np.polyfit(y_vals, c_vals, 1)
        slope = float(coeffs[0])
        # Dead-band around zero so flat data isn't flipped by floating-point noise.
        eps = 1e-9
        if slope > 0.5:
            trend = "急上昇"
        elif slope > eps:
            trend = "増加傾向"
        elif slope >= -eps:
            trend = "横ばい"
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
