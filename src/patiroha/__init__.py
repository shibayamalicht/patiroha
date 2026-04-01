"""patiroha — Patent text analysis toolkit for Japanese and multilingual patent documents."""

from __future__ import annotations

__version__ = "1.0.0"

# Core modules (always available)
from patiroha.clustering import auto_label, generate_spatial_summary
from patiroha.embeddings import build_tfidf
from patiroha.io import load_patent_data
from patiroha.metadata import (
    IPC_SECTIONS,
    extract_ipc,
    extract_ipc_parsed,
    normalize_applicant,
    parse_date,
    parse_ipc,
    smart_map_columns,
)
from patiroha.stats import (
    calculate_cagr,
    calculate_diversity,
    calculate_entropy,
    calculate_gini,
    calculate_hhi,
    find_representatives,
    find_representatives_mmr,
    find_similar,
)
from patiroha.stopwords import StopwordManager, get_stopwords, list_categories, list_words
from patiroha.tokenize import apply_ngram_filters, extract_keywords, normalize_text, tokenize_for_tfidf

__all__ = [
    "__version__",
    # stopwords
    "get_stopwords",
    "list_categories",
    "list_words",
    "StopwordManager",
    # tokenize
    "extract_keywords",
    "tokenize_for_tfidf",
    "normalize_text",
    "apply_ngram_filters",
    # metadata
    "extract_ipc",
    "extract_ipc_parsed",
    "parse_ipc",
    "IPC_SECTIONS",
    "parse_date",
    "normalize_applicant",
    "smart_map_columns",
    # io
    "load_patent_data",
    # embeddings
    "build_tfidf",
    "SBERTEmbedder",
    # clustering
    "build_landscape",
    "auto_label",
    "generate_spatial_summary",
    # stats
    "calculate_hhi",
    "calculate_entropy",
    "calculate_gini",
    "calculate_diversity",
    "calculate_cagr",
    "find_representatives",
    "find_representatives_mmr",
    "find_similar",
    # network
    "build_cooccurrence_graph",
    "detect_communities",
    "get_hub_keywords",
    # pipeline
    "PatentPipeline",
]


def __getattr__(name: str) -> object:
    """Lazy imports for optional-dependency features."""
    if name == "SBERTEmbedder":
        from patiroha.embeddings.sbert import SBERTEmbedder

        return SBERTEmbedder
    if name == "build_landscape":
        from patiroha.clustering.landscape import build_landscape

        return build_landscape
    if name in ("build_cooccurrence_graph", "detect_communities", "get_hub_keywords"):
        from patiroha.network import cooccurrence

        return getattr(cooccurrence, name)
    if name == "PatentPipeline":
        from patiroha.pipeline import PatentPipeline

        return PatentPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
