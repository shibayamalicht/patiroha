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

    # Clamp min_df so it stays jointly satisfiable with max_df on small corpora
    # (otherwise sklearn raises an opaque "max_df < min_df" error).
    n_docs = len(tokenized)
    max_df_docs = int(max_df * n_docs) if isinstance(max_df, float) else max_df
    effective_min_df = min(min_df, max(1, max_df_docs))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=effective_min_df,
        max_df=max_df,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(tokenized)
    except ValueError:
        # Degenerate small corpus (e.g. every term unique across few docs):
        # fall back to permissive settings rather than crashing.
        vectorizer = TfidfVectorizer(max_features=max_features, min_df=1, max_df=1.0)
        try:
            tfidf_matrix = vectorizer.fit_transform(tokenized)
        except ValueError as e:
            raise ValueError(
                "build_tfidf: no usable vocabulary (texts are empty or all-stopwords)."
            ) from e
    feature_names = np.array(vectorizer.get_feature_names_out())

    return tfidf_matrix, feature_names
