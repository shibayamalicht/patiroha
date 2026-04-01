"""Market concentration indices: HHI, Shannon entropy, and Gini coefficient."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from patiroha._types import HHIResult


@dataclass(frozen=True)
class DiversityResult:
    """Combined diversity metrics."""

    hhi: float
    hhi_status: str
    entropy: float
    gini: float
    n_entities: int


def calculate_hhi(counts: Sequence[int]) -> HHIResult:
    """Calculate the Herfindahl-Hirschman Index and classify market concentration.

    Uses Japan Fair Trade Commission thresholds (0-1 scale):
    - < 0.10: Competitive (dispersed)
    - < 0.18: Moderate concentration
    - >= 0.18: Oligopolistic (high concentration)

    Args:
        counts: Sequence of counts (e.g. patent applications per applicant).

    Returns:
        HHIResult with value and status string.
    """
    if not counts or sum(counts) == 0:
        return HHIResult(value=0.0, status="データ不足")

    total = sum(counts)
    shares = [c / total for c in counts]
    hhi = sum(s**2 for s in shares)

    if hhi < 0.10:
        status = "競争的 (分散)"
    elif hhi < 0.18:
        status = "中程度の集中"
    else:
        status = "寡占的 (高集中)"

    return HHIResult(value=hhi, status=status)


def calculate_entropy(counts: Sequence[int]) -> float:
    """Calculate Shannon entropy of a distribution.

    Higher entropy = more diverse/even distribution.

    Args:
        counts: Sequence of counts.

    Returns:
        Shannon entropy value (bits). 0.0 if data is insufficient.
    """
    if not counts or sum(counts) == 0:
        return 0.0
    total = sum(counts)
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def calculate_gini(counts: Sequence[int]) -> float:
    """Calculate Gini coefficient of inequality.

    0 = perfect equality, 1 = maximum inequality.

    Args:
        counts: Sequence of counts.

    Returns:
        Gini coefficient. 0.0 if data is insufficient.
    """
    if not counts or sum(counts) == 0:
        return 0.0
    sorted_counts = sorted(counts)
    n = len(sorted_counts)
    if n <= 1:
        return 0.0
    total = sum(sorted_counts)
    cumulative = 0.0
    gini_sum = 0.0
    for i, c in enumerate(sorted_counts):
        cumulative += c
        gini_sum += (2 * (i + 1) - n - 1) * c
    return gini_sum / (n * total)


def calculate_diversity(counts: Sequence[int]) -> DiversityResult:
    """Calculate all diversity metrics at once.

    Args:
        counts: Sequence of counts.

    Returns:
        DiversityResult with HHI, entropy, Gini, and entity count.
    """
    hhi_result = calculate_hhi(counts)
    return DiversityResult(
        hhi=hhi_result.value,
        hhi_status=hhi_result.status,
        entropy=calculate_entropy(counts),
        gini=calculate_gini(counts),
        n_entities=len([c for c in counts if c > 0]),
    )
