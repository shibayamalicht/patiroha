"""Clustering utilities (UMAP + HDBSCAN, auto-labeling, spatial analysis)."""

from patiroha.clustering.labeling import auto_label
from patiroha.clustering.spatial import generate_spatial_summary

__all__ = ["build_landscape", "auto_label", "generate_spatial_summary"]


def __getattr__(name: str) -> object:
    if name == "build_landscape":
        from patiroha.clustering.landscape import build_landscape

        return build_landscape
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
