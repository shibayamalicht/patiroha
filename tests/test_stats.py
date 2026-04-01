"""Tests for patiroha.stats."""

import numpy as np
import pandas as pd

from patiroha.stats import (
    calculate_cagr,
    calculate_diversity,
    calculate_hhi,
    find_representatives,
    find_representatives_mmr,
    find_similar,
)
from patiroha.stats.hhi import calculate_entropy, calculate_gini


class TestHHI:
    def test_competitive_market(self):
        counts = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        result = calculate_hhi(counts)
        assert result.value <= 0.10 + 1e-9
        assert "競争的" in result.status or "中程度" in result.status

    def test_oligopoly(self):
        counts = [80, 10, 5, 3, 2]
        result = calculate_hhi(counts)
        assert result.value >= 0.18
        assert "寡占的" in result.status

    def test_empty(self):
        result = calculate_hhi([])
        assert result.value == 0.0
        assert "データ不足" in result.status


class TestEntropy:
    def test_uniform(self):
        counts = [10, 10, 10, 10]
        e = calculate_entropy(counts)
        assert abs(e - 2.0) < 0.01  # log2(4) = 2

    def test_single(self):
        counts = [100]
        assert calculate_entropy(counts) == 0.0

    def test_empty(self):
        assert calculate_entropy([]) == 0.0


class TestGini:
    def test_perfect_equality(self):
        counts = [10, 10, 10, 10]
        assert calculate_gini(counts) == 0.0

    def test_inequality(self):
        counts = [1, 1, 1, 100]
        g = calculate_gini(counts)
        assert g > 0.5

    def test_empty(self):
        assert calculate_gini([]) == 0.0


class TestDiversity:
    def test_all_metrics(self):
        counts = [50, 30, 10, 5, 3, 2]
        div = calculate_diversity(counts)
        assert div.hhi > 0
        assert div.entropy > 0
        assert div.gini > 0
        assert div.n_entities == 6
        assert "寡占的" in div.hhi_status or "中程度" in div.hhi_status


class TestCAGR:
    def test_growth(self):
        df = pd.DataFrame({"year": [2018, 2019, 2020, 2021, 2022] * 10})
        result = calculate_cagr(df)
        assert result.growth_rate is not None

    def test_missing_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = calculate_cagr(df)
        assert result.growth_rate is None

    def test_single_year(self):
        df = pd.DataFrame({"year": [2020, 2020]})
        result = calculate_cagr(df)
        assert result.growth_rate == 0.0
        assert result.trend == "Stable"


def _make_test_data(n: int = 20, dim: int = 10):
    np.random.seed(42)
    vectors = np.random.randn(n, dim).astype(np.float64)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    df = pd.DataFrame(
        {
            "title": [f"Patent {i}" for i in range(n)],
            "abstract": [f"Abstract of patent {i}" for i in range(n)],
        }
    )
    return vectors, df


class TestRepresentatives:
    def test_basic_extraction(self):
        vectors, df = _make_test_data()
        reps = find_representatives(vectors, df, n=3)
        assert len(reps) == 3
        assert all(r.title.startswith("Patent") for r in reps)
        # Scores should be descending
        assert reps[0].score >= reps[1].score >= reps[2].score

    def test_empty(self):
        vectors = np.array([]).reshape(0, 10)
        df = pd.DataFrame()
        reps = find_representatives(vectors, df)
        assert reps == []


class TestMMR:
    def test_returns_n_results(self):
        vectors, df = _make_test_data()
        reps = find_representatives_mmr(vectors, df, n=5, diversity=0.5)
        assert len(reps) == 5

    def test_diversity_affects_selection(self):
        vectors, df = _make_test_data()
        # Low diversity = similar to centroid
        reps_low = find_representatives_mmr(vectors, df, n=5, diversity=0.0)
        # High diversity = more spread out
        reps_high = find_representatives_mmr(vectors, df, n=5, diversity=0.9)
        # At least some different selections
        titles_low = {r.title for r in reps_low}
        titles_high = {r.title for r in reps_high}
        assert titles_low != titles_high or len(vectors) <= 5

    def test_empty(self):
        vectors = np.array([]).reshape(0, 10)
        df = pd.DataFrame()
        assert find_representatives_mmr(vectors, df) == []


class TestFindSimilar:
    def test_self_is_most_similar(self):
        vectors, df = _make_test_data()
        similar = find_similar(vectors[0], vectors, df, n=3)
        assert len(similar) == 3
        assert similar[0].index == 0
        assert abs(similar[0].score - 1.0) < 0.01

    def test_empty(self):
        vectors = np.array([]).reshape(0, 10)
        df = pd.DataFrame()
        assert find_similar(np.zeros(10), vectors, df) == []
