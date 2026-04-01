"""Tests for patiroha.metadata."""

import pandas as pd

from patiroha.metadata import extract_ipc, normalize_applicant, parse_date, smart_map_columns


class TestExtractIPC:
    def test_full_ipc_code(self):
        result = extract_ipc("B32B 27/00; C08L 1/02")
        assert "b32b27/00" in result
        assert "c08l1/02" in result

    def test_main_class_only(self):
        result = extract_ipc("B32B")
        assert "b32b" in result

    def test_empty_input(self):
        assert extract_ipc("") == []
        assert extract_ipc(123) == []  # type: ignore[arg-type]

    def test_custom_delimiter(self):
        result = extract_ipc("A61K 35/56, H01L 21/02", delimiter=",")
        assert len(result) == 2

    def test_nfkc_normalization(self):
        result = extract_ipc("Ｂ３２Ｂ 27/00")
        assert len(result) >= 1


class TestParseDate:
    def test_standard_format(self):
        series = pd.Series(["2020-01-15", "2021-03-20"])
        result = parse_date(series)
        assert result.notna().all()

    def test_yyyymmdd_format(self):
        series = pd.Series(["20200115", "20210320"])
        result = parse_date(series)
        assert result.notna().all()

    def test_year_only(self):
        series = pd.Series(["2020", "2021", "2022"])
        result = parse_date(series)
        assert result.notna().all()


class TestNormalizeApplicant:
    def test_basic_split(self):
        result = normalize_applicant("大日本印刷株式会社;凸版印刷株式会社")
        assert len(result) == 2
        assert "大日本印刷" in result
        assert "凸版印刷" in result

    def test_remove_corporate_entity(self):
        result = normalize_applicant("トヨタ自動車株式会社")
        assert result == ["トヨタ自動車"]

    def test_english_corporate(self):
        result = normalize_applicant("Apple Inc.")
        assert len(result) == 1
        assert "Apple" in result[0]

    def test_empty_input(self):
        assert normalize_applicant("") == []
        assert normalize_applicant(123) == []  # type: ignore[arg-type]


class TestSmartMapColumns:
    def test_exact_match(self):
        df = pd.DataFrame(columns=["title", "abstract", "出願人", "ipc"])
        result = smart_map_columns(df)
        assert result["title"] == "title"
        assert result["abstract"] == "abstract"

    def test_substring_match(self):
        df = pd.DataFrame(columns=["発明の名称_JP", "要約_JP"])
        result = smart_map_columns(df)
        assert result["title"] == "発明の名称_JP"

    def test_no_match(self):
        df = pd.DataFrame(columns=["col_a", "col_b"])
        result = smart_map_columns(df)
        assert result["title"] is None
