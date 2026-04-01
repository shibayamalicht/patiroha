"""Text embedding utilities (TF-IDF and SBERT)."""

from patiroha.embeddings.tfidf import build_tfidf

__all__ = ["build_tfidf", "SBERTEmbedder"]


def __getattr__(name: str) -> object:
    if name == "SBERTEmbedder":
        from patiroha.embeddings.sbert import SBERTEmbedder

        return SBERTEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
