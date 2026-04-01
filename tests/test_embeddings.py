"""Tests for patiroha.embeddings (TF-IDF only — SBERT requires optional deps)."""

from patiroha.embeddings import build_tfidf


def test_build_tfidf_basic():
    texts = [
        "セルロースナノファイバー 樹脂 複合材料",
        "光学フィルム 偏光板 液晶",
        "セルロースナノファイバー 樹脂 強化",
        "光学フィルム 反射防止 コーティング",
        "樹脂 成形 射出 複合材料",
        "セルロースナノファイバー 樹脂 複合材料",
    ]
    matrix, feature_names = build_tfidf(texts, min_df=1)
    assert matrix.shape[0] == 6
    assert len(feature_names) > 0
