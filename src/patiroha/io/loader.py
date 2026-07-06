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

"""CSV/Excel file loader with automatic encoding detection.

Provides robust file loading for Japanese patent data files.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

# shift_jis omitted: cp932 is a strict superset, so a shift_jis-only branch is
# unreachable. cp932 and euc-jp can BOTH decode some byte streams without error,
# so they are disambiguated by a mojibake heuristic (see _load_csv).
_CSV_ENCODINGS = ["utf-8", "utf-8-sig", "cp932", "euc-jp"]


def load_patent_data(path: str | Path) -> pd.DataFrame:
    """Load patent data from CSV or Excel file.

    For CSV files, tries multiple Japanese-compatible encodings.
    For Excel files, uses openpyxl.

    Args:
        path: Path to the data file (.csv, .xlsx, .xls).

    Returns:
        DataFrame with the loaded data.

    Raises:
        ValueError: If file format is unsupported or all encoding attempts fail.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _load_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return _load_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv, .xlsx, or .xls")


def _halfwidth_katakana_count(text: str) -> int:
    """Count half-width katakana (a strong signal of cp932/euc-jp mojibake)."""
    return sum(1 for ch in text if "｡" <= ch <= "ﾟ")


def _read_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(StringIO(text))
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}") from e


def _load_csv(path: Path) -> pd.DataFrame:
    """Load CSV, auto-detecting the encoding.

    UTF-8 is unambiguous, so the first clean UTF-8 decode is accepted as-is.
    cp932 and euc-jp can both decode the same bytes without raising (cp932 may
    lossily turn EUC-JP bytes into half-width katakana garbage), so the candidate
    with the fewest half-width katakana characters is chosen.
    """
    raw = path.read_bytes()
    errors = []

    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return _read_csv_text(raw.decode(encoding))
        except (UnicodeDecodeError, UnicodeError) as e:
            errors.append(f"{encoding}: {e}")

    best: tuple[int, str] | None = None  # (mojibake_score, decoded_text)
    for encoding in ("cp932", "euc-jp"):
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError) as e:
            errors.append(f"{encoding}: {e}")
            continue
        score = _halfwidth_katakana_count(text)
        if best is None or score < best[0]:
            best = (score, text)

    if best is not None:
        return _read_csv_text(best[1])

    raise ValueError(f"Could not read CSV with any supported encoding. Tried: {', '.join(_CSV_ENCODINGS)}")


def _load_excel(path: Path) -> pd.DataFrame:
    """Load Excel file."""
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}") from e
