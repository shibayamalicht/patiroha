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

"""Regression tests for the v1.0.1 bug-fix pass.

Each test pins a previously-confirmed bug so it cannot silently return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# metadata.applicant — corporate-suffix over-matching (#1, #9, N25)
# --------------------------------------------------------------------------- #
class TestApplicantSuffix:
    def test_does_not_corrupt_company_body(self):
        from patiroha.metadata import normalize_applicant

        assert normalize_applicant("Microsoft Corporation") == ["Microsoft"]
        assert normalize_applicant("Cobalt Inc.") == ["Cobalt"]
        assert normalize_applicant("Continental AG") == ["Continental"]
        assert normalize_applicant("Cosmetics Co.") == ["Cosmetics"]

    def test_case_insensitive_english(self):
        from patiroha.metadata import normalize_applicant

        assert normalize_applicant("apple inc.") == ["apple"]

    def test_halfwidth_paren_normalized(self):
        from patiroha.metadata import normalize_applicant

        assert normalize_applicant("ABC(株)") == ["ABC"]
        assert normalize_applicant("ABC（株）") == ["ABC"]

    def test_existing_behavior_preserved(self):
        from patiroha.metadata import normalize_applicant

        assert normalize_applicant("トヨタ自動車株式会社;ソニー株式会社") == ["トヨタ自動車", "ソニー"]


# --------------------------------------------------------------------------- #
# stopwords (#2, #10, N3, N18)
# --------------------------------------------------------------------------- #
class TestStopwords:
    def test_expanded_set_accepts_generator(self):
        from patiroha.stopwords.manager import _get_expanded_set

        out = _get_expanded_set(w for w in ["abc"])
        assert "ａｂｃ" in out

    def test_invalid_mode_raises(self):
        from patiroha.stopwords import get_stopwords

        with pytest.raises(ValueError):
            get_stopwords("not-a-mode")

    def test_remove_drops_fullwidth_variant(self):
        from patiroha.stopwords import StopwordManager

        mgr = StopwordManager(include=["patent_terms"])
        mgr.remove(["PCT"])
        built = mgr.build()
        assert "PCT" not in built
        assert "ＰＣＴ" not in built

    def test_catalog_has_no_duplicate_counts(self):
        from patiroha.stopwords import list_categories

        counts = list_categories()
        assert counts["misc"] == 128
        assert counts["npl"] == 422


# --------------------------------------------------------------------------- #
# stats (#7, N5, N20)
# --------------------------------------------------------------------------- #
class TestStats:
    def test_hhi_accepts_series_and_ndarray(self):
        from patiroha.stats import calculate_diversity, calculate_hhi

        assert calculate_hhi(pd.Series([3, 1, 1])).value > 0
        assert calculate_hhi(np.array([3, 1, 1])).value > 0
        assert calculate_diversity(pd.Series([5, 3, 2])).n_entities == 3

    def test_single_year_is_japanese(self):
        from patiroha.stats import calculate_cagr

        assert calculate_cagr(pd.DataFrame({"year": [2020, 2020]})).trend == "横ばい"

    def test_flat_data_is_flat_not_growth(self):
        from patiroha.stats import calculate_cagr

        df = pd.DataFrame({"year": [2019, 2019, 2020, 2020, 2021, 2021]})
        assert calculate_cagr(df).trend == "横ばい"


# --------------------------------------------------------------------------- #
# metadata.dates — numeric Series (N1)
# --------------------------------------------------------------------------- #
class TestParseDateNumeric:
    def test_year_integers(self):
        from patiroha.metadata import parse_date

        out = parse_date(pd.Series([2019, 2020, 2021]))
        assert out.dt.year.tolist() == [2019, 2020, 2021]

    def test_yyyymmdd_integers(self):
        from patiroha.metadata import parse_date

        assert parse_date(pd.Series([20200115])).iloc[0] == pd.Timestamp("2020-01-15")

    def test_excel_serial_integers(self):
        from patiroha.metadata import parse_date

        assert parse_date(pd.Series([43831])).iloc[0] == pd.Timestamp("2020-01-01")

    def test_string_years_still_work(self):
        from patiroha.metadata import parse_date

        assert parse_date(pd.Series(["2019", "2020"])).dt.year.tolist() == [2019, 2020]


# --------------------------------------------------------------------------- #
# metadata.columns — substring mis-mapping (N2, N11)
# --------------------------------------------------------------------------- #
class TestSmartMapColumns:
    def test_embedded_keyword_not_matched(self):
        from patiroha.metadata import smart_map_columns

        # "date" must not match "candidate_id"
        assert smart_map_columns(pd.DataFrame(columns=["candidate_id"]))["date"] is None
        # but the real date column wins
        assert (
            smart_map_columns(pd.DataFrame(columns=["candidate_id", "出願日"]))["date"] == "出願日"
        )

    def test_applicant_not_mid_word(self):
        from patiroha.metadata import smart_map_columns

        assert (
            smart_map_columns(pd.DataFrame(columns=["applicantid", "出願人"]))["applicant"]
            == "出願人"
        )

    def test_underscore_compound_still_matches(self):
        from patiroha.metadata import smart_map_columns

        assert smart_map_columns(pd.DataFrame(columns=["filing_date"]))["date"] == "filing_date"

    def test_case_insensitive(self):
        from patiroha.metadata import smart_map_columns

        assert smart_map_columns(pd.DataFrame(columns=["Ipc_class"]))["ipc"] == "Ipc_class"


# --------------------------------------------------------------------------- #
# metadata.ipc (N10, N15, N16)
# --------------------------------------------------------------------------- #
class TestIPC:
    def test_trailing_junk_rejected(self):
        from patiroha.metadata import parse_ipc

        assert parse_ipc("h01l31/0725abc").section == ""

    def test_collects_all_codes_without_delimiter(self):
        from patiroha.metadata import extract_ipc

        assert extract_ipc("B32B27/00C08L1/02") == ["b32b27/00", "c08l1/02"]

    def test_single_digit_subgroup(self):
        from patiroha.metadata import parse_ipc

        ipc = parse_ipc("h01l31/0")
        assert ipc.group == "31"
        assert ipc.subgroup == "0"

    def test_normal_codes_preserved(self):
        from patiroha.metadata import extract_ipc

        assert extract_ipc("B32B 27/00; C08L 1/02") == ["b32b27/00", "c08l1/02"]


# --------------------------------------------------------------------------- #
# tokenize.filters — chemical formulae (N4)
# --------------------------------------------------------------------------- #
class TestFilters:
    def test_preserves_chemical_formulae(self):
        from patiroha.tokenize import apply_ngram_filters

        assert apply_ngram_filters("CO2") == "CO2"
        assert apply_ngram_filters("H2O") == "H2O"
        assert apply_ngram_filters("B12") == "B12"
        assert apply_ngram_filters("ISO9001") == "ISO9001"


# --------------------------------------------------------------------------- #
# embeddings.tfidf — small corpus (N8)
# --------------------------------------------------------------------------- #
class TestTfidf:
    def test_small_corpus_does_not_crash(self):
        from patiroha.embeddings import build_tfidf

        matrix, _ = build_tfidf(["特許 出願 技術", "発明 装置 方法", "システム 制御 処理"])
        assert matrix.shape[0] == 3


# --------------------------------------------------------------------------- #
# clustering.labeling — noise label kept on empty vocab (#6, N9)
# --------------------------------------------------------------------------- #
class TestLabelingNoise:
    def test_tfidf_keeps_noise_label(self):
        from patiroha.clustering import auto_label

        labels = np.array([0, 0, 1, -1])
        assert auto_label(["", "", "", ""], labels, method="tfidf").get(-1) is not None

    def test_c_tfidf_keeps_noise_label(self):
        from patiroha.clustering import auto_label

        labels = np.array([0, 0, 1, -1])
        assert auto_label(["", "", "", ""], labels, method="c-tfidf").get(-1) is not None


# --------------------------------------------------------------------------- #
# clustering.spatial (N13, N14)
# --------------------------------------------------------------------------- #
class TestSpatial:
    def test_noise_not_a_neighbor(self):
        from patiroha.clustering import generate_spatial_summary

        df = pd.DataFrame(
            {
                "cluster": [-1, -1, 0, 0, 1, 1],
                "x": [0, 0.1, 0.2, 0.3, 9, 9.1],
                "y": [0, 0.1, 0, 0.1, 9, 9.1],
            }
        )
        out = generate_spatial_summary(df, "cluster", "x", "y", {-1: "NOISE", 0: "C0", 1: "C1"})
        assert "NOISE" not in out

    def test_missing_y_col_guarded(self):
        from patiroha.clustering import generate_spatial_summary

        df = pd.DataFrame({"cluster": [0, 1], "umap_x": [0, 1]})
        assert generate_spatial_summary(df, "cluster", "umap_x", "umap_y") == "空間データなし"


# --------------------------------------------------------------------------- #
# package surface (N19, N26)
# --------------------------------------------------------------------------- #
class TestPackage:
    def test_version_bumped(self):
        import patiroha

        assert patiroha.__version__ == "1.0.1"

    def test_lazy_names_in_dir(self):
        import patiroha

        for name in ("SBERTEmbedder", "build_landscape", "PatentPipeline"):
            assert name in dir(patiroha)


# --------------------------------------------------------------------------- #
# pipeline.run guard (N23)
# --------------------------------------------------------------------------- #
class TestPipelineRunGuard:
    def test_run_without_input_raises_value_error(self):
        from patiroha import PatentPipeline

        with pytest.raises(ValueError):
            PatentPipeline().run()


# --------------------------------------------------------------------------- #
# network.cooccurrence — argument validation (N27)
# --------------------------------------------------------------------------- #
def _skip_if_no_networkx():
    try:
        import networkx  # noqa: F401
    except ImportError:
        pytest.skip("networkx not installed")


class TestCooccurrenceValidation:
    def test_invalid_algorithm_raises(self):
        _skip_if_no_networkx()
        from patiroha import build_cooccurrence_graph, detect_communities

        g = build_cooccurrence_graph([["a", "b"], ["a", "c"], ["b", "c"]], top_n=10, threshold=-1)
        with pytest.raises(ValueError):
            detect_communities(g, algorithm="typo")

    def test_invalid_centrality_raises(self):
        _skip_if_no_networkx()
        from patiroha import build_cooccurrence_graph, get_hub_keywords

        g = build_cooccurrence_graph([["a", "b"], ["a", "c"], ["b", "c"]], top_n=10, threshold=-1)
        with pytest.raises(ValueError):
            get_hub_keywords(g, centrality="typo")


# --------------------------------------------------------------------------- #
# network.cooccurrence — document-frequency-based metrics (#3)
# --------------------------------------------------------------------------- #
class TestCooccurrenceDocFrequency:
    def test_intra_document_duplicates_do_not_distort(self):
        _skip_if_no_networkx()
        from patiroha import build_cooccurrence_graph

        # "A" appears 3x in doc 0 but in only 2 documents overall; "B" in 2 docs;
        # they co-occur in 2 docs -> true Jaccard = 2 / (2 + 2 - 2) = 1.0
        g = build_cooccurrence_graph([["A", "A", "A", "B"], ["A", "B"]], top_n=10, threshold=-1)
        assert g.nodes["A"]["size"] == 2  # document frequency, not raw 4
        assert abs(g["A"]["B"]["weight"] - 1.0) < 1e-9

    def test_jaccard_never_exceeds_one(self):
        _skip_if_no_networkx()
        from patiroha import build_cooccurrence_graph

        g = build_cooccurrence_graph(
            [["x", "x", "y"], ["x", "y", "z"], ["x", "z"]],
            top_n=10,
            threshold=-1,
            similarity="jaccard",
        )
        for _, _, data in g.edges(data=True):
            assert data["weight"] <= 1.0 + 1e-9

    def test_dice_and_cosine_are_set_based(self):
        _skip_if_no_networkx()
        from patiroha import build_cooccurrence_graph

        docs = [["x", "x", "y"], ["x", "y", "z"], ["x", "z"]]
        # df(x)=3, df(y)=2, df(z)=2; cooccur(x,y)=2
        dice = build_cooccurrence_graph(docs, top_n=10, threshold=-1, similarity="dice")
        cosine = build_cooccurrence_graph(docs, top_n=10, threshold=-1, similarity="cosine")
        assert abs(dice["x"]["y"]["weight"] - (2 * 2) / (3 + 2)) < 1e-9
        assert abs(cosine["x"]["y"]["weight"] - 2 / (3 * 2) ** 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# clustering.landscape — KMeans clamp & dtype (N7, N12)
# --------------------------------------------------------------------------- #
def _skip_if_no_clustering():
    try:
        import hdbscan  # noqa: F401
        import umap  # noqa: F401
    except ImportError:
        pytest.skip("umap/hdbscan not installed")


class TestLandscape:
    def test_kmeans_clamps_n_clusters(self):
        _skip_if_no_clustering()
        from patiroha import build_landscape

        vectors = np.random.default_rng(0).random((5, 32)).astype(np.float64)
        result = build_landscape(vectors, method="kmeans", n_neighbors=4, n_clusters=8)
        assert result.labels.dtype == np.intp
        assert result.coords.dtype == np.float64
