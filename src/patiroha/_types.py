"""Shared data types for patiroha."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class IPCCode:
    """Parsed International Patent Classification code."""

    raw: str
    section: str = ""
    class_code: str = ""
    subclass: str = ""
    group: str = ""
    subgroup: str = ""


@dataclass(frozen=True)
class HHIResult:
    """Herfindahl-Hirschman Index result."""

    value: float
    status: str


@dataclass(frozen=True)
class CAGRResult:
    """Compound Annual Growth Rate result."""

    growth_rate: float | None
    trend: str | None


@dataclass(frozen=True)
class LandscapeResult:
    """UMAP + HDBSCAN clustering result."""

    labels: npt.NDArray[np.intp]
    coords: npt.NDArray[np.float64]
    n_clusters: int
    noise_count: int


@dataclass(frozen=True)
class Representative:
    """A representative patent extracted by centroid distance."""

    index: int
    title: str
    abstract: str
    score: float
    year: str | None = None
    applicant: str | None = None


@dataclass(frozen=True)
class CooccurrenceGraph:
    """Result of co-occurrence network construction."""

    node_count: int
    edge_count: int
    communities: dict[str, int] = field(default_factory=dict)
    hub_keywords: list[str] = field(default_factory=list)
