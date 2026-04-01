"""Representative patent extraction: centroid-based and MMR (diversity-aware).

Selects representative documents using centroid proximity or Maximal Marginal
Relevance (MMR) for diversity-aware selection.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from patiroha._types import Representative


def find_representatives(
    vectors: npt.NDArray[np.float64],
    df: pd.DataFrame,
    n: int = 5,
    title_col: str = "title",
    abstract_col: str = "abstract",
    year_col: str | None = "year",
    applicant_col: str | None = None,
) -> list[Representative]:
    """Find representative patents closest to the centroid of the embedding space.

    Args:
        vectors: Embedding matrix (n_patents, n_features), should be L2-normalized.
        df: DataFrame with patent metadata.
        n: Number of representative patents to extract.
        title_col: Column name for patent titles.
        abstract_col: Column name for patent abstracts.
        year_col: Column name for filing year (optional).
        applicant_col: Column name for applicant (optional).

    Returns:
        List of Representative instances, sorted by descending similarity to centroid.
    """
    if vectors.shape[0] == 0 or df.empty:
        return []

    centroid = np.mean(vectors, axis=0)
    dots = np.dot(vectors, centroid)
    top_indices = np.argsort(dots)[::-1][:n]

    return _build_representatives(top_indices, dots[top_indices], df, title_col, abstract_col, year_col, applicant_col)


def find_representatives_mmr(
    vectors: npt.NDArray[np.float64],
    df: pd.DataFrame,
    n: int = 5,
    diversity: float = 0.3,
    title_col: str = "title",
    abstract_col: str = "abstract",
    year_col: str | None = "year",
    applicant_col: str | None = None,
) -> list[Representative]:
    """Find representative patents using Maximal Marginal Relevance (MMR).

    Balances relevance (similarity to centroid) with diversity (dissimilarity
    to already-selected patents).

    Args:
        vectors: Embedding matrix (n_patents, n_features), should be L2-normalized.
        df: DataFrame with patent metadata.
        n: Number of representative patents to extract.
        diversity: Weight for diversity vs relevance (0.0 = pure relevance, 1.0 = max diversity).
        title_col: Column name for patent titles.
        abstract_col: Column name for patent abstracts.
        year_col: Column name for filing year.
        applicant_col: Column name for applicant.

    Returns:
        List of Representative instances selected via MMR.
    """
    if vectors.shape[0] == 0 or df.empty:
        return []

    centroid = np.mean(vectors, axis=0)
    relevance = np.dot(vectors, centroid)  # similarity to centroid

    selected: list[int] = []
    candidates = set(range(len(vectors)))
    selected_vectors: list[npt.NDArray[np.float64]] = []

    for _ in range(min(n, len(vectors))):
        best_idx = -1
        best_score = -float("inf")

        for idx in candidates:
            rel = float(relevance[idx])

            if selected_vectors:
                sims = [float(np.dot(vectors[idx], sv)) for sv in selected_vectors]
                max_sim = max(sims)
            else:
                max_sim = 0.0

            mmr_score = (1 - diversity) * rel - diversity * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            selected_vectors.append(vectors[best_idx])
            candidates.discard(best_idx)

    scores = relevance[np.array(selected)]
    return _build_representatives(
        np.array(selected),
        scores,
        df,
        title_col,
        abstract_col,
        year_col,
        applicant_col,
    )


def find_similar(
    query_vector: npt.NDArray[np.float64],
    vectors: npt.NDArray[np.float64],
    df: pd.DataFrame,
    n: int = 10,
    title_col: str = "title",
    abstract_col: str = "abstract",
    year_col: str | None = "year",
    applicant_col: str | None = None,
) -> list[Representative]:
    """Find patents most similar to a query vector.

    Args:
        query_vector: Query embedding vector (1D array).
        vectors: Embedding matrix (n_patents, n_features).
        df: DataFrame with patent metadata.
        n: Number of similar patents to return.
        title_col: Column name for titles.
        abstract_col: Column name for abstracts.
        year_col: Column name for year.
        applicant_col: Column name for applicant.

    Returns:
        List of Representative instances sorted by descending similarity.
    """
    if vectors.shape[0] == 0 or df.empty:
        return []

    similarities = np.dot(vectors, query_vector)
    top_indices = np.argsort(similarities)[::-1][:n]

    return _build_representatives(
        top_indices, similarities[top_indices], df, title_col, abstract_col, year_col, applicant_col
    )


def _build_representatives(
    indices: npt.NDArray[np.intp],
    scores: npt.NDArray[np.float64],
    df: pd.DataFrame,
    title_col: str,
    abstract_col: str,
    year_col: str | None,
    applicant_col: str | None,
) -> list[Representative]:
    """Build Representative objects from selected indices.

    Args:
        indices: Array of row indices into df.
        scores: Array of scores, same length as indices (one score per selected index).
        df: Source DataFrame.
    """
    representatives: list[Representative] = []
    for rank, idx in enumerate(indices):
        idx_int = int(idx)
        try:
            row = df.iloc[idx_int]
            title = str(row.get(title_col, "")) if pd.notna(row.get(title_col)) else "No Title"
            abstract = str(row.get(abstract_col, "")) if pd.notna(row.get(abstract_col)) else "No Abstract"

            year = None
            if year_col and year_col in df.columns:
                y = row.get(year_col)
                year = str(y) if pd.notna(y) else None

            applicant = None
            if applicant_col and applicant_col in df.columns:
                a = row.get(applicant_col)
                if pd.notna(a):
                    applicant = str(a)[:30]

            representatives.append(
                Representative(
                    index=idx_int,
                    title=title.replace("\n", " "),
                    abstract=abstract.replace("\n", " ")[:200],
                    score=float(scores[rank]),
                    year=year,
                    applicant=applicant,
                )
            )
        except (IndexError, KeyError):
            continue
    return representatives
