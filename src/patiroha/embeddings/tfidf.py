"""TF-IDF vectorization with Janome tokenizer integration."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from patiroha.stopwords import get_stopwords
from patiroha.tokenize.japanese import tokenize_for_tfidf


def build_tfidf(
    texts: Iterable[str],
    stopwords: Iterable[str] | None = None,
    min_df: int = 5,
    max_df: float = 0.80,
    max_features: int | None = None,
) -> tuple[spmatrix, npt.NDArray[np.str_]]:
    """Build a TF-IDF matrix from patent texts.

    Internally tokenizes text using Janome for Japanese compound noun extraction.

    Args:
        texts: Iterable of text strings to vectorize.
        stopwords: Stopword set for tokenization. If None, uses patent defaults.
        min_df: Minimum document frequency for terms.
        max_df: Maximum document frequency for terms (as fraction).
        max_features: Maximum number of features. None for unlimited.

    Returns:
        Tuple of (sparse TF-IDF matrix, feature name array).
    """
    sw = frozenset(stopwords) if stopwords is not None else get_stopwords()

    # Tokenize all texts
    tokenized = [tokenize_for_tfidf(t, stopwords=sw) for t in texts]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    tfidf_matrix = vectorizer.fit_transform(tokenized)
    feature_names = np.array(vectorizer.get_feature_names_out())

    return tfidf_matrix, feature_names
