"""SBERT (Sentence-BERT) embedding for patent documents.

Generates dense vector representations using Sentence-BERT models with
flexible text preparation strategies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import normalize

from patiroha._lazy import require


class SBERTEmbedder:
    """SBERT-based patent text embedder.

    Args:
        model_name: HuggingFace model name for sentence-transformers.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        st = require("sentence_transformers", "embeddings")
        self._model = st.SentenceTransformer(model_name)

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        progress_callback: Callable[[float], None] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Encode text strings into embeddings.

        Args:
            texts: Sequence of text strings.
            batch_size: Number of texts per batch.
            normalize_embeddings: If True (default), L2-normalize the output vectors.
            progress_callback: Optional callback receiving progress as float 0.0-1.0.

        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        total_batches = (len(texts) + batch_size - 1) // batch_size
        embeddings_list: list[npt.NDArray[np.float64]] = []

        for i in range(total_batches):
            batch_texts = texts[i * batch_size : (i + 1) * batch_size]
            batch_embeddings = self._model.encode(batch_texts, show_progress_bar=False)
            embeddings_list.append(batch_embeddings)

            if progress_callback is not None:
                progress_callback((i + 1) / total_batches)

        embeddings = np.vstack(embeddings_list)
        if normalize_embeddings:
            embeddings = normalize(embeddings, norm="l2")
        return embeddings

    def encode(
        self,
        df: pd.DataFrame,
        text_columns: Sequence[str],
        separator: str = " ",
        column_weights: dict[str, float] | None = None,
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        progress_callback: Callable[[float], None] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Encode DataFrame rows by concatenating specified text columns.

        Args:
            df: DataFrame containing patent data.
            text_columns: Column names to concatenate.
            separator: Separator between column texts (default: " ").
            column_weights: Optional dict mapping column names to repeat counts.
                E.g. {"title": 2, "abstract": 1} repeats title text twice for emphasis.
            batch_size: Number of rows per batch.
            normalize_embeddings: If True (default), L2-normalize the output.
            progress_callback: Optional callback for progress tracking.

        Returns:
            numpy array of shape (n_rows, embedding_dim).
        """
        texts: list[str] = []
        for _, row in df.iterrows():
            parts: list[str] = []
            for col in text_columns:
                val = row.get(col)
                part = str(val) if pd.notna(val) else ""
                if column_weights and col in column_weights:
                    repeat = max(1, int(column_weights[col]))
                    parts.extend([part] * repeat)
                else:
                    parts.append(part)
            texts.append(separator.join(parts))

        return self.encode_texts(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            progress_callback=progress_callback,
        )
