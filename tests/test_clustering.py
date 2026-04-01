"""Tests for patiroha.clustering (labeling and spatial — landscape requires optional deps)."""

import numpy as np
import pandas as pd

from patiroha.clustering import auto_label, generate_spatial_summary


def test_auto_label_basic():
    texts = [
        "セルロース ナノファイバー 樹脂",
        "セルロース ナノファイバー 複合材料",
        "光学 フィルム 偏光板",
        "光学 フィルム 液晶",
    ]
    labels = np.array([0, 0, 1, 1])
    result = auto_label(texts, labels, top_n=2)
    assert 0 in result
    assert 1 in result
    assert isinstance(result[0], str)


def test_auto_label_with_noise():
    texts = ["セルロース ナノファイバー", "光学 フィルム", "偏光板 液晶"]
    labels = np.array([-1, 0, 0])
    result = auto_label(texts, labels)
    assert -1 in result
    assert "ノイズ" in result[-1]


def test_auto_label_custom_noise_label():
    texts = ["セルロース ナノファイバー", "光学 フィルム", "偏光板 液晶"]
    labels = np.array([-1, 0, 0])
    result = auto_label(texts, labels, noise_label="その他")
    assert result[-1] == "その他"


def test_auto_label_custom_format():
    texts = ["セルロース ナノファイバー 樹脂", "光学 フィルム 偏光板"]
    labels = np.array([0, 1])
    result = auto_label(texts, labels, label_format="Cluster-{id}: {terms}")
    for v in result.values():
        assert v.startswith("Cluster-")


def test_auto_label_c_tfidf():
    texts = [
        "セルロース ナノファイバー 樹脂",
        "セルロース ナノファイバー 複合材料",
        "光学 フィルム 偏光板",
        "光学 フィルム 液晶",
    ]
    labels = np.array([0, 0, 1, 1])
    result = auto_label(texts, labels, method="c-tfidf")
    assert 0 in result
    assert 1 in result


def test_spatial_summary_basic():
    df = pd.DataFrame(
        {
            "cluster": [0, 0, 1, 1, 2, 2],
            "x": [0.0, 0.1, 5.0, 5.1, 10.0, 10.1],
            "y": [0.0, 0.1, 5.0, 5.1, 10.0, 10.1],
        }
    )
    result = generate_spatial_summary(df, "cluster", "x", "y")
    assert "近傍" in result


def test_spatial_summary_single_cluster():
    df = pd.DataFrame({"cluster": [0, 0], "x": [1.0, 2.0], "y": [1.0, 2.0]})
    result = generate_spatial_summary(df, "cluster", "x", "y")
    assert "単一クラスタ" in result
