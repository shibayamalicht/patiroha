"""Automatic cluster labeling strategies.

Supports standard TF-IDF mean and c-TF-IDF (BERTopic-style class-based TF-IDF).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer

from patiroha.stopwords import get_stopwords
from patiroha.tokenize.japanese import tokenize_for_tfidf


def auto_label(
    texts: Iterable[str],
    labels: npt.NDArray[np.intp],
    stopwords: Iterable[str] | None = None,
    top_n: int = 3,
    method: Literal["tfidf", "c-tfidf"] = "tfidf",
    noise_label: str = "ノイズ / 小クラスタ",
    label_format: str = "[{id}] {terms}",
) -> dict[int, str]:
    """Generate automatic labels for clusters.

    Args:
        texts: Iterable of document texts corresponding to labels.
        labels: Cluster label array (same length as texts, -1 = noise).
        stopwords: Stopwords for tokenization. If None, uses patent defaults.
        top_n: Number of top terms to include in each label.
        method: "tfidf" (per-cluster mean TF-IDF) or "c-tfidf" (class-based TF-IDF).
        noise_label: Display label for noise cluster (-1).
        label_format: Format string with {id} and {terms} placeholders.

    Returns:
        Dict mapping cluster IDs to label strings.
    """
    sw = frozenset(stopwords) if stopwords is not None else get_stopwords()
    text_list = list(texts)
    tokenized = [tokenize_for_tfidf(t, stopwords=sw) for t in text_list]

    unique_labels = sorted(set(labels))
    labels_map: dict[int, str] = {}

    if method == "c-tfidf":
        labels_map = _c_tfidf_labels(tokenized, labels, unique_labels, top_n, noise_label, label_format)
    else:
        labels_map = _tfidf_mean_labels(tokenized, labels, unique_labels, top_n, noise_label, label_format)

    return labels_map


def _tfidf_mean_labels(
    tokenized: list[str],
    labels: npt.NDArray[np.intp],
    unique_labels: list[int],
    top_n: int,
    noise_label: str,
    label_format: str,
) -> dict[int, str]:
    """Standard TF-IDF: fit on all docs, then take per-cluster mean vector."""
    vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
    try:
        tfidf_matrix = vectorizer.fit_transform(tokenized)
    except ValueError:
        return {int(lid): f"Cluster {lid}" for lid in unique_labels if lid != -1}

    feature_names = np.array(vectorizer.get_feature_names_out())
    labels_map: dict[int, str] = {}

    for cluster_id in unique_labels:
        lid = int(cluster_id)
        if lid == -1:
            labels_map[lid] = noise_label
            continue

        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not indices:
            labels_map[lid] = f"Cluster {lid}"
            continue

        mean_vector = np.array(tfidf_matrix[indices].mean(axis=0)).flatten()
        top_indices = np.argsort(mean_vector)[::-1][:top_n]
        top_terms = [feature_names[i] for i in top_indices if mean_vector[i] > 0]

        if top_terms:
            labels_map[lid] = label_format.format(id=lid, terms=", ".join(top_terms))
        else:
            labels_map[lid] = f"Cluster {lid}"

    return labels_map


def _c_tfidf_labels(
    tokenized: list[str],
    labels: npt.NDArray[np.intp],
    unique_labels: list[int],
    top_n: int,
    noise_label: str,
    label_format: str,
) -> dict[int, str]:
    """c-TF-IDF: concatenate docs per cluster, then fit TF-IDF on cluster-level docs."""
    cluster_docs: list[str] = []
    cluster_ids: list[int] = []

    for cid in unique_labels:
        lid = int(cid)
        if lid == -1:
            continue
        indices = [i for i, lbl in enumerate(labels) if lbl == cid]
        if indices:
            merged = " ".join(tokenized[i] for i in indices)
            cluster_docs.append(merged)
            cluster_ids.append(lid)

    labels_map: dict[int, str] = {}
    if -1 in unique_labels:
        labels_map[-1] = noise_label

    if not cluster_docs:
        return labels_map

    vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_docs)
    except ValueError:
        return {lid: f"Cluster {lid}" for lid in cluster_ids}

    feature_names = np.array(vectorizer.get_feature_names_out())

    for row_idx, lid in enumerate(cluster_ids):
        row = np.array(tfidf_matrix[row_idx].todense()).flatten()
        top_indices = np.argsort(row)[::-1][:top_n]
        top_terms = [feature_names[i] for i in top_indices if row[i] > 0]

        if top_terms:
            labels_map[lid] = label_format.format(id=lid, terms=", ".join(top_terms))
        else:
            labels_map[lid] = f"Cluster {lid}"

    return labels_map
