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
    """A representative patent extracted by centroid distance.

    Note:
        ``index`` is a POSITIONAL index into the embedding matrix / DataFrame
        (i.e. ``df.iloc[index]``), not a DataFrame label. For a DataFrame with a
        non-default index, use ``df.iloc[rep.index]`` — ``df.loc[rep.index]`` may
        raise or return the wrong row.
    """

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
