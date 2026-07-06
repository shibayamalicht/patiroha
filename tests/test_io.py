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
