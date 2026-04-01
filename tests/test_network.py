"""Tests for patiroha.network."""

import pytest


@pytest.fixture
def sample_keyword_lists():
    return [
        ["セルロース", "ナノファイバー", "樹脂"],
        ["セルロース", "ナノファイバー", "複合材料"],
        ["光学", "フィルム", "偏光板"],
        ["光学", "フィルム", "液晶"],
        ["セルロース", "樹脂", "複合材料"],
        ["光学", "偏光板", "液晶"],
    ]


def _skip_if_no_networkx():
    try:
        import networkx  # noqa: F401
    except ImportError:
        pytest.skip("networkx not installed")


class TestBuildGraph:
    def test_basic(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=0.01)
        assert len(G.nodes) > 0
        assert len(G.edges) > 0

    def test_similarity_jaccard(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=0.01, similarity="jaccard")
        assert len(G.edges) > 0

    def test_similarity_dice(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=0.01, similarity="dice")
        assert len(G.edges) > 0

    def test_similarity_pmi(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=-10, similarity="pmi")
        assert len(G.edges) > 0

    def test_similarity_frequency(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=1, similarity="frequency")
        assert len(G.edges) > 0

    def test_unknown_similarity_raises(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph

        with pytest.raises(ValueError, match="Unknown similarity"):
            build_cooccurrence_graph(sample_keyword_lists, similarity="invalid")


class TestCommunities:
    def test_greedy(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph, detect_communities

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=0.01)
        c = detect_communities(G, algorithm="greedy_modularity")
        assert isinstance(c, dict)

    def test_louvain(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph, detect_communities

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=0.01)
        c = detect_communities(G, algorithm="louvain")
        assert isinstance(c, dict)

    def test_label_propagation(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph, detect_communities

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=0.01)
        c = detect_communities(G, algorithm="label_propagation")
        assert isinstance(c, dict)


class TestHubKeywords:
    def test_degree(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph, get_hub_keywords

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=0.01)
        hubs = get_hub_keywords(G, top_n=3, centrality="degree")
        assert len(hubs) <= 3

    def test_betweenness(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph, get_hub_keywords

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=0.01)
        hubs = get_hub_keywords(G, top_n=3, centrality="betweenness")
        assert isinstance(hubs, list)

    def test_pagerank(self, sample_keyword_lists):
        _skip_if_no_networkx()
        from patiroha.network.cooccurrence import build_cooccurrence_graph, get_hub_keywords

        G = build_cooccurrence_graph(sample_keyword_lists, top_n=10, threshold=0.01)
        hubs = get_hub_keywords(G, top_n=3, centrality="pagerank")
        assert isinstance(hubs, list)
