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
        networkx.Graph with 'size' node attribute (document frequency) and
        'weight' / 'cooccurrence' edge attributes.

    Note:
        All metrics are set-based over documents. Node selection, node 'size',
        and the similarity denominators use document frequency (the number of
        documents containing a keyword), and the 'cooccurrence' count is the
        number of documents containing both endpoints. Intra-document repeats of
        a keyword therefore do not affect results, so jaccard/dice/cosine are
        strict set coefficients (e.g. jaccard is always <= 1).

        Changed in 1.0.1: node 'size' and the similarity denominators previously
        used raw token frequency (including intra-document repeats), which mixed
        document- and token-level counts. Edge 'weight' values may differ from
        1.0.0 for keyword lists that contain intra-document duplicates.
    """
    nx = require("networkx", "network")

    n_docs = len(keyword_lists)

    # Document frequency: number of documents containing each keyword.
    word_doc_counts: Counter[str] = Counter()
    for kws in keyword_lists:
        word_doc_counts.update(set(kws))

    top_nodes = [w for w, _ in word_doc_counts.most_common(top_n)]
    top_set = set(top_nodes)

    # Document co-occurrence: number of documents containing both keywords.
    pair_counts: Counter[tuple[str, str]] = Counter()
    for kws in keyword_lists:
        valid = sorted(w for w in set(kws) if w in top_set)
        if len(valid) >= 2:
            for pair in combinations(valid, 2):
                pair_counts[pair] += 1

    G = nx.Graph()
    for w in top_nodes:
        G.add_node(w, size=word_doc_counts[w])

    for (u, v), c in pair_counts.items():
        cu, cv = word_doc_counts[u], word_doc_counts[v]
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
    elif algorithm == "greedy_modularity":
        communities_list = nx.community.greedy_modularity_communities(G)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. "
            "Use 'greedy_modularity', 'louvain', or 'label_propagation'."
        )

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
    elif centrality == "degree":
        scores = nx.degree_centrality(G)
    else:
        raise ValueError(
            f"Unknown centrality: {centrality!r}. "
            "Use 'degree', 'betweenness', 'eigenvector', or 'pagerank'."
        )

    sorted_hubs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_hubs[:top_n]
