"""Clustering pipelines: UMAP + HDBSCAN and KMeans.

Provides dimensionality reduction via UMAP followed by density-based (HDBSCAN) or
centroid-based (KMeans) clustering, with flexible parameter configuration.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from patiroha._lazy import require
from patiroha._types import LandscapeResult


def build_landscape(
    vectors: npt.NDArray[np.float64],
    method: Literal["hdbscan", "kmeans"] = "hdbscan",
    # UMAP parameters
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    umap_metric: str = "cosine",
    random_state: int = 42,
    # HDBSCAN parameters
    min_cluster_size: int = 15,
    min_samples: int = 10,
    cluster_metric: str = "euclidean",
    cluster_selection_method: Literal["eom", "leaf"] = "eom",
    # KMeans parameters
    n_clusters: int = 8,
    # Common
    progress_callback: Callable[[float], None] | None = None,
) -> LandscapeResult:
    """Run UMAP dimensionality reduction followed by clustering.

    Args:
        vectors: Input embedding matrix of shape (n_samples, n_features).
        method: Clustering method — "hdbscan" (density-based) or "kmeans" (centroid-based).
        n_neighbors: UMAP local neighborhood size.
        min_dist: UMAP minimum distance between points.
        n_components: UMAP output dimensions (2 for visualization, higher for analysis).
        umap_metric: Distance metric for UMAP (e.g. "cosine", "euclidean").
        random_state: Random seed for reproducibility.
        min_cluster_size: (HDBSCAN) Minimum points to form a cluster.
        min_samples: (HDBSCAN) Minimum samples for core points.
        cluster_metric: (HDBSCAN) Distance metric for clustering.
        cluster_selection_method: (HDBSCAN) "eom" (excess of mass) or "leaf".
        n_clusters: (KMeans) Number of clusters.
        progress_callback: Optional callback for progress tracking (0.0-1.0).

    Returns:
        LandscapeResult with cluster labels, 2D coordinates, and summary stats.
    """
    umap_mod = require("umap", "clustering")

    if progress_callback:
        progress_callback(0.1)

    reducer = umap_mod.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric=umap_metric,
    )
    coords = reducer.fit_transform(vectors)

    if progress_callback:
        progress_callback(0.6)

    if method == "hdbscan":
        hdbscan_mod = require("hdbscan", "clustering")
        clusterer = hdbscan_mod.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=cluster_metric,
            cluster_selection_method=cluster_selection_method,
        )
        labels = clusterer.fit_predict(coords)
    elif method == "kmeans":
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(coords)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}. Use 'hdbscan' or 'kmeans'.")

    if progress_callback:
        progress_callback(1.0)

    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))

    return LandscapeResult(
        labels=labels,
        coords=coords,
        n_clusters=n_found,
        noise_count=noise_count,
    )
