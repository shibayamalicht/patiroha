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
    # UMAP returns float32; honor the declared float64 contract on coords.
    coords = np.asarray(reducer.fit_transform(vectors), dtype=np.float64)

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
        # Clamp so n_clusters never exceeds the sample count (avoids a raw sklearn
        # ValueError on small inputs; pipeline default n_clusters is 8).
        k = max(1, min(n_clusters, len(coords)))
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(coords)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}. Use 'hdbscan' or 'kmeans'.")

    # Normalize label dtype so it matches the declared np.intp regardless of method
    # (HDBSCAN -> int64, KMeans -> int32) and platform.
    labels = np.asarray(labels, dtype=np.intp)

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
