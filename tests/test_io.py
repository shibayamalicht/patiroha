"""Tests for patiroha.io."""

import tempfile
from pathlib import Path

import pytest

from patiroha.io import load_patent_data


def test_load_csv_utf8():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("title,abstract\n特許1,要約1\n特許2,要約2\n")
        path = f.name

    df = load_patent_data(path)
    assert len(df) == 2
    assert "title" in df.columns
    Path(path).unlink()


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_patent_data("/nonexistent/file.csv")


def test_load_unsupported_format():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name

    with pytest.raises(ValueError, match="Unsupported"):
        load_patent_data(path)
    Path(path).unlink()
