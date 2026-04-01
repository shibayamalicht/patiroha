"""Keyword co-occurrence network construction with flexible metrics and analysis.

Supports multiple similarity metrics (Jaccard, cosine, PMI, log-likelihood),
community detection algorithms, and centrality measures.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from itertools import combinations
from typing import Any, Literal

from patiroha._lazy import require


def build_cooccurrence_graph(
    keyword_lists: Sequence[Sequence[str]],
    top_n: int = 50,
    threshold: float = 0.05,
    similarity: Literal["jaccard", "dice", "cosine", "pmi", "frequency"] = "jaccard",
) -> Any:
    """Build a keyword co-occurrence network.

    Args:
        keyword_lists: List of keyword lists (one per document).
        top_n: Number of top keywords to include as nodes.
        threshold: Minimum similarity score for edge creation.
        similarity: Edge weight metric:
            - "jaccard": |A∩B| / |A∪B| (default)
            - "dice": 2|A∩B| / (|A|+|B|)
            - "cosine": |A∩B| / sqrt(|A|*|B|)
            - "pmi": log2(P(A,B) / (P(A)*P(B))) — pointwise mutual information
            - "frequency": raw co-occurrence count (no normalization)

    Returns:
        networkx.Graph with 'size' node attribute and 'weight' edge attribute.
    """
    nx = require("networkx", "network")

    all_keywords = [w for kws in keyword_lists for w in kws]
    word_counts: Counter[str] = Counter(all_keywords)
    top_nodes = [w for w, _ in word_counts.most_common(top_n)]
    top_set = set(top_nodes)

    pair_counts: Counter[tuple[str, str]] = Counter()
    n_docs = len(keyword_lists)
    for kws in keyword_lists:
        valid = sorted(set(w for w in kws if w in top_set))
        if len(valid) >= 2:
            for pair in combinations(valid, 2):
                pair_counts[pair] += 1

    G = nx.Graph()
    for w in top_nodes:
        G.add_node(w, size=word_counts[w])

    for (u, v), c in pair_counts.items():
        cu, cv = word_counts[u], word_counts[v]
        weight = _compute_similarity(c, cu, cv, n_docs, similarity)
        if weight >= threshold:
            G.add_edge(u, v, weight=weight, cooccurrence=c)

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G


def _compute_similarity(
    cooccur: int,
    count_u: int,
    count_v: int,
    n_docs: int,
    method: str,
) -> float:
    if method == "jaccard":
        return cooccur / (count_u + count_v - cooccur)
    elif method == "dice":
        return 2 * cooccur / (count_u + count_v)
    elif method == "cosine":
        return cooccur / math.sqrt(count_u * count_v)
    elif method == "pmi":
        p_uv = cooccur / n_docs if n_docs > 0 else 0
        p_u = count_u / n_docs if n_docs > 0 else 0
        p_v = count_v / n_docs if n_docs > 0 else 0
        if p_u > 0 and p_v > 0 and p_uv > 0:
            return math.log2(p_uv / (p_u * p_v))
        return 0.0
    elif method == "frequency":
        return float(cooccur)
    else:
        raise ValueError(f"Unknown similarity metric: {method!r}")


def detect_communities(
    G: Any,
    algorithm: Literal["greedy_modularity", "louvain", "label_propagation"] = "greedy_modularity",
) -> dict[str, int]:
    """Detect communities in a co-occurrence graph.

    Args:
        G: networkx.Graph.
        algorithm: Community detection algorithm:
            - "greedy_modularity" (default)
            - "louvain"
            - "label_propagation"

    Returns:
        Dict mapping node names to community IDs.
    """
    nx = require("networkx", "network")

    if len(G.nodes) == 0:
        return {}

    if algorithm == "louvain":
        communities_list = nx.community.louvain_communities(G, seed=42)
    elif algorithm == "label_propagation":
        communities_list = nx.community.label_propagation_communities(G)
    else:
        communities_list = nx.community.greedy_modularity_communities(G)

    community_map: dict[str, int] = {}
    for i, comm in enumerate(communities_list):
        for node in comm:
            community_map[node] = i
    return community_map


def get_hub_keywords(
    G: Any,
    top_n: int = 10,
    centrality: Literal["degree", "betweenness", "eigenvector", "pagerank"] = "degree",
) -> list[tuple[str, float]]:
    """Get hub keywords ranked by centrality measure.

    Args:
        G: networkx.Graph.
        top_n: Number of top hub keywords to return.
        centrality: Centrality measure:
            - "degree" (default)
            - "betweenness"
            - "eigenvector"
            - "pagerank"

    Returns:
        List of (keyword, centrality_score) tuples, sorted descending.
    """
    nx = require("networkx", "network")

    if len(G.nodes) == 0:
        return []

    if centrality == "betweenness":
        scores: dict[str, float] = nx.betweenness_centrality(G)
    elif centrality == "eigenvector":
        try:
            scores = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            scores = nx.degree_centrality(G)
    elif centrality == "pagerank":
        scores = nx.pagerank(G)
    else:
        scores = nx.degree_centrality(G)

    sorted_hubs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_hubs[:top_n]
