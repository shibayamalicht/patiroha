"""Keyword co-occurrence network analysis."""

__all__ = ["build_cooccurrence_graph", "detect_communities", "get_hub_keywords"]


def __getattr__(name: str) -> object:
    if name in ("build_cooccurrence_graph", "detect_communities", "get_hub_keywords"):
        from patiroha.network import cooccurrence

        return getattr(cooccurrence, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
