"""Spatial cluster analysis -- centroid proximity descriptions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def generate_spatial_summary(
    df: pd.DataFrame,
    cluster_col: str,
    x_col: str,
    y_col: str,
    label_map: dict[int, str] | None = None,
) -> str:
    """Analyze spatial layout of clusters and generate a proximity description.

    Args:
        df: DataFrame with cluster assignments and 2D coordinates.
        cluster_col: Column name for cluster IDs.
        x_col: Column name for x coordinates.
        y_col: Column name for y coordinates.
        label_map: Optional mapping from cluster ID to display name.

    Returns:
        Japanese markdown text describing cluster proximity relationships.
    """
    try:
        if df.empty or cluster_col not in df.columns or x_col not in df.columns:
            return "空間データなし"

        centroids = df.groupby(cluster_col)[[x_col, y_col]].mean()
        coords = centroids.values
        labels = centroids.index.tolist()

        if len(labels) < 2:
            return "単一クラスタのため配置分析なし"

        dist_matrix = squareform(pdist(coords))

        spatial_desc = ["【クラスタ配置と近接関係】"]

        for i, label_id in enumerate(labels):
            if label_id == -1:
                continue

            dists = dist_matrix[i]
            nearest_indices = np.argsort(dists)[1:3]  # Top 2 nearest (skip self)

            current_name = label_map.get(label_id, f"Cluster {label_id}") if label_map else f"Cluster {label_id}"

            neighbors = []
            for n_idx in nearest_indices:
                n_id = labels[n_idx]
                n_name = label_map.get(n_id, f"Cluster {n_id}") if label_map else f"Cluster {n_id}"
                neighbors.append(n_name)

            if neighbors:
                spatial_desc.append(f"- 「{current_name}」の近傍: {', '.join(neighbors)}")

        return "\n".join(spatial_desc)

    except Exception as e:
        return f"空間分析計算エラー: {e}"
