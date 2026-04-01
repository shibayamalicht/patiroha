"""CSV/Excel file loader with automatic encoding detection.

Provides robust file loading for Japanese patent data files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_CSV_ENCODINGS = ["utf-8", "utf-8-sig", "cp932", "shift_jis", "euc-jp"]


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


def _load_csv(path: Path) -> pd.DataFrame:
    """Try loading CSV with multiple encodings."""
    errors = []
    for encoding in _CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError) as e:
            errors.append(f"{encoding}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}") from e

    raise ValueError(f"Could not read CSV with any supported encoding. Tried: {', '.join(_CSV_ENCODINGS)}")


def _load_excel(path: Path) -> pd.DataFrame:
    """Load Excel file."""
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Failed to read Excel file: {e}") from e
