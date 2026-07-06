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
