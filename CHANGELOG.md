# Changelog

## [1.0.0] - 2026-04-01

### Added

- **stopwords**: 7カテゴリ（一般、特許用語、構造、IT、化学、単位、NPL）のストップワード辞書と `StopwordManager` によるカテゴリ選択・カスタマイズ
- **tokenize**: Janome ベースの複合名詞抽出（`extract_keywords`）、N-gram フィルタ、NFKC 正規化、品詞タグ選択対応
- **metadata**: IPC 階層パース（セクション〜サブグループ）、出願日マルチフォーマット解析、出願人正規化、カラム自動マッピング
- **io**: CSV/Excel の自動エンコーディング判定ローダー
- **embeddings**: TF-IDF（Janome 統合）、SBERT エンベディング（モデル選択可、カラム重み付け対応）
- **clustering**: UMAP + HDBSCAN / KMeans、c-TF-IDF 自動ラベリング、空間近接分析
- **stats**: HHI / エントロピー / ジニ係数、CAGR / トレンド判定、重心距離 / MMR 代表特許抽出、類似特許検索
- **network**: キーワード共起ネットワーク（Jaccard / Dice / Cosine / PMI / 頻度）、Louvain 等コミュニティ検出、PageRank 等中心性分析
- **pipeline**: `PatentPipeline` による前処理→埋め込み→クラスタリング→ラベリングのワンライナー実行
