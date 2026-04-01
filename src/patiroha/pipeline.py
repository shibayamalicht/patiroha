"""High-level patent analysis pipeline.

Chains preprocessing → embedding → clustering → labeling in one call.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from patiroha.io import load_patent_data
from patiroha.metadata import extract_ipc, normalize_applicant, parse_date, smart_map_columns
from patiroha.stopwords import StopwordManager, get_stopwords
from patiroha.tokenize import extract_keywords


@dataclass
class AnalysisResult:
    """Container for the full pipeline output."""

    df: pd.DataFrame
    col_map: dict[str, str | None]
    stopwords: frozenset[str]
    keywords_col: str = "keywords"
    vectors: npt.NDArray[np.float64] | None = None
    cluster_labels: npt.NDArray[np.intp] | None = None
    cluster_coords: npt.NDArray[np.float64] | None = None
    cluster_names: dict[int, str] = field(default_factory=dict)


class PatentPipeline:
    """Configurable pipeline for end-to-end patent analysis.

    Usage:
        pipe = PatentPipeline()
        result = pipe.run("data.csv")

        # Or step by step:
        pipe = PatentPipeline()
        pipe.load("data.csv")
        pipe.preprocess()
        pipe.extract_keywords()
        pipe.embed()
        pipe.cluster()
        result = pipe.result
    """

    def __init__(
        self,
        # Stopword config
        stopword_mode: str = "patent",
        stopword_include: Sequence[str] | None = None,
        stopword_exclude: Sequence[str] | None = None,
        extra_stopwords: Iterable[str] | None = None,
        remove_stopwords: Iterable[str] | None = None,
        # Keyword extraction config
        pos_tags: Sequence[str] = ("名詞",),
        min_keyword_length: int = 2,
        # Embedding config
        sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        text_columns: Sequence[str] | None = None,
        column_weights: dict[str, float] | None = None,
        # Clustering config
        cluster_method: Literal["hdbscan", "kmeans"] = "hdbscan",
        min_cluster_size: int = 15,
        min_samples: int = 10,
        n_clusters: int = 8,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        umap_metric: str = "cosine",
        cluster_metric: str = "euclidean",
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        # Labeling config
        label_method: Literal["tfidf", "c-tfidf"] = "tfidf",
        label_top_n: int = 3,
    ) -> None:
        self._stopword_mode = stopword_mode
        self._stopword_include = stopword_include
        self._stopword_exclude = stopword_exclude
        self._extra_stopwords = extra_stopwords
        self._remove_stopwords = remove_stopwords
        self._pos_tags = pos_tags
        self._min_keyword_length = min_keyword_length
        self._sbert_model = sbert_model
        self._text_columns = text_columns
        self._column_weights = column_weights
        self._cluster_method = cluster_method
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples
        self._n_clusters = n_clusters
        self._n_neighbors = n_neighbors
        self._min_dist = min_dist
        self._umap_metric = umap_metric
        self._cluster_metric = cluster_metric
        self._cluster_selection_method = cluster_selection_method
        self._label_method = label_method
        self._label_top_n = label_top_n

        self._df: pd.DataFrame | None = None
        self._col_map: dict[str, str | None] = {}
        self._stopwords: frozenset[str] = frozenset()
        self._vectors: npt.NDArray[np.float64] | None = None
        self._labels: npt.NDArray[np.intp] | None = None
        self._coords: npt.NDArray[np.float64] | None = None
        self._cluster_names: dict[int, str] = {}

    def _build_stopwords(self) -> frozenset[str]:
        if self._stopword_include or self._stopword_exclude:
            mgr = StopwordManager(
                include=list(self._stopword_include) if self._stopword_include else None,
                exclude=list(self._stopword_exclude) if self._stopword_exclude else None,
            )
            if self._extra_stopwords:
                mgr.add(self._extra_stopwords)
            if self._remove_stopwords:
                mgr.remove(self._remove_stopwords)
            return mgr.build()
        else:
            sw = set(get_stopwords(self._stopword_mode))
            if self._extra_stopwords:
                sw.update(self._extra_stopwords)
            if self._remove_stopwords:
                sw -= set(self._remove_stopwords)
            return frozenset(sw)

    def load(self, path: str | Path) -> PatentPipeline:
        """Load patent data from file."""
        self._df = load_patent_data(path)
        return self

    def load_df(self, df: pd.DataFrame) -> PatentPipeline:
        """Load patent data from an existing DataFrame."""
        self._df = df.copy()
        return self

    def preprocess(self) -> PatentPipeline:
        """Run metadata preprocessing: column mapping, date parsing, IPC, applicant."""
        assert self._df is not None, "Call load() first"
        df = self._df

        self._col_map = smart_map_columns(df)

        date_col = self._col_map.get("date")
        if date_col and date_col in df.columns:
            df["parsed_date"] = parse_date(df[date_col])
            df["year"] = df["parsed_date"].dt.year

        ipc_col = self._col_map.get("ipc")
        if ipc_col and ipc_col in df.columns:
            df["ipc_list"] = df[ipc_col].apply(lambda x: extract_ipc(str(x)))

        app_col = self._col_map.get("applicant")
        if app_col and app_col in df.columns:
            df["applicant_list"] = df[app_col].apply(lambda x: normalize_applicant(str(x)))

        return self

    def extract_kw(self) -> PatentPipeline:
        """Extract keywords from abstract/title."""
        assert self._df is not None, "Call load() first"
        self._stopwords = self._build_stopwords()

        text_col = self._col_map.get("abstract") or self._col_map.get("title")
        if text_col and text_col in self._df.columns:
            self._df["keywords"] = self._df[text_col].apply(
                lambda x: extract_keywords(
                    str(x),
                    stopwords=self._stopwords,
                    pos_tags=self._pos_tags,
                    min_length=self._min_keyword_length,
                )
            )
        return self

    def embed(
        self,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PatentPipeline:
        """Generate SBERT embeddings."""
        assert self._df is not None, "Call load() first"
        from patiroha.embeddings.sbert import SBERTEmbedder

        embedder = SBERTEmbedder(self._sbert_model)

        text_cols = self._text_columns
        if text_cols is None:
            candidates = ["title", "abstract", "claims"]
            text_cols = [self._col_map.get(c, c) for c in candidates if self._col_map.get(c)]
            if not text_cols:
                text_cols = [c for c in ["title", "abstract"] if c in self._df.columns]

        self._vectors = embedder.encode(
            self._df,
            text_columns=text_cols,
            column_weights=self._column_weights,
            progress_callback=progress_callback,
        )
        return self

    def cluster(
        self,
        progress_callback: Callable[[float], None] | None = None,
    ) -> PatentPipeline:
        """Run UMAP + clustering and auto-labeling."""
        assert self._vectors is not None, "Call embed() first"
        from patiroha.clustering import auto_label
        from patiroha.clustering.landscape import build_landscape

        result = build_landscape(
            self._vectors,
            method=self._cluster_method,
            n_neighbors=self._n_neighbors,
            min_dist=self._min_dist,
            umap_metric=self._umap_metric,
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            n_clusters=self._n_clusters,
            cluster_metric=self._cluster_metric,
            cluster_selection_method=self._cluster_selection_method,
            progress_callback=progress_callback,
        )
        self._labels = result.labels
        self._coords = result.coords

        self._df["cluster"] = result.labels
        self._df["umap_x"] = result.coords[:, 0]
        self._df["umap_y"] = result.coords[:, 1]

        text_col = self._col_map.get("abstract") or self._col_map.get("title")
        if text_col and text_col in self._df.columns:
            self._cluster_names = auto_label(
                self._df[text_col].astype(str),
                result.labels,
                stopwords=self._stopwords,
                top_n=self._label_top_n,
                method=self._label_method,
            )
            self._df["cluster_label"] = [self._cluster_names.get(int(lbl), "不明") for lbl in result.labels]

        return self

    def run(
        self,
        path: str | Path | None = None,
        df: pd.DataFrame | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> AnalysisResult:
        """Run the full pipeline: load → preprocess → keywords → embed → cluster.

        Args:
            path: Path to data file. Provide either path or df.
            df: Existing DataFrame. Provide either path or df.
            progress_callback: Optional progress callback.

        Returns:
            AnalysisResult with all computed data.
        """
        if path:
            self.load(path)
        elif df is not None:
            self.load_df(df)

        self.preprocess()
        self.extract_kw()
        self.embed(progress_callback=progress_callback)
        self.cluster(progress_callback=progress_callback)

        return self.result

    @property
    def result(self) -> AnalysisResult:
        """Get the current analysis result."""
        assert self._df is not None
        return AnalysisResult(
            df=self._df,
            col_map=self._col_map,
            stopwords=self._stopwords,
            vectors=self._vectors,
            cluster_labels=self._labels,
            cluster_coords=self._coords,
            cluster_names=self._cluster_names,
        )
