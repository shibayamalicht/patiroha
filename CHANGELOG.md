# Changelog

## [1.0.1] - 2026-06-13

バグ修正・堅牢性向上リリース。

### License

- ライセンスを **MIT から Apache License 2.0 に変更**。1.0.0 までのバージョンは引き続き MIT で利用可能。Apache 2.0 は特許ライセンスの明示的付与を含む。

### Fixed — 誤った結果 / データ破壊

- **metadata.applicant**: `normalize_applicant` の企業サフィックス除去が社名本体を削っていた問題を修正（`Microsoft Corporation`→`Microsoft`、`Cobalt`/`Continental`/`Cosmetics` 等）。英語サフィックスを両側語境界＋大小文字無視に変更し、NFKC 正規化で半角 `(株)` も除去対象に。
- **metadata.dates**: 数値型 Series の `parse_date` が「1970年起点ナノ秒」に化けていた問題を修正。年(4桁)/YYYYMMDD(8桁)/Excelシリアルを桁数で判別。
- **metadata.columns**: `smart_map_columns` の部分一致誤爆を修正（`date`⊄`candidate_id` 等）。境界付き・大小文字無視・長いキーワード優先に。
- **stopwords.manager**: `StopwordManager.remove()` が ASCII 語の全角版を消し残していた問題を修正。`get_stopwords` は不正な mode で `ValueError`。`_get_expanded_set` はジェネレータ入力でも全角展開するように。
- **tokenize.filters**: n-gram フィルタの `[A-Z]+[0-9]+` ルールが化学式・型番（`CO2`/`H2O`/`B12`/`ISO9001`）を破壊していたため除去。
- **stopwords.catalog**: `misc`/`npl` カテゴリの重複語を整理（misc 140→128、npl 423→422）。展開後総数（patent 791 / npl 1534）は不変。

### Fixed — クラッシュ / 堅牢性

- **stats.hhi**: `calculate_hhi`/`calculate_entropy`/`calculate_gini`/`calculate_diversity` が numpy 配列・pandas Series 入力でクラッシュしていた問題を修正。
- **clustering.landscape**: KMeans で `n_clusters > 件数` のときクラッシュしないようクランプ。`coords`/`labels` の dtype を宣言通り（float64 / intp）に統一。
- **embeddings.tfidf**: 小規模コーパスで `min_df` 既定値により不可解な失敗をしていた問題を、クランプ＋フォールバックで解消。
- **embeddings.sbert**: 空入力での `np.vstack([])` クラッシュを回避。出力を宣言通り float64 にキャスト。
- **io.loader**: EUC-JP CSV が cp932 で文字化けロードされていた問題を、半角カナ密度ヒューリスティックで改善。到達不能だった `shift_jis` を整理。
- **pipeline**: `run()` を引数なしで呼ぶと `ValueError`。テキスト列が見つからない場合も明示エラー。自動列検出時に `column_weights` のキーを実カラム名へ再マップ。

### Fixed — 整合性 / その他

- **clustering.labeling**: 語彙が空のとき例外フォールバックでノイズ(`-1`)ラベルが消えていた問題を両経路で修正。
- **clustering.spatial**: ノイズクラスタ(`-1`)を近傍として提示しないように。`y_col` 欠落をガード。
- **metadata.ipc**: フルIPC正規表現を末尾アンカーし末尾ゴミを排除。区切り無し連結の複数コードを全件抽出。1桁サブグループを許容。
- **stats.cagr**: 単年データのトレンドを日本語 `横ばい` に統一。slope≈0 に不感帯を導入し横ばいデータの誤分類を防止。
- **network.cooccurrence**: 不正な `algorithm`/`centrality` 文字列で `ValueError`。
- **pipeline / `_types` / `__init__`**: docstring 修正（`extract_kw`）、`Representative.index` が位置インデックスである旨を明記、遅延エクスポート名を `dir()` に公開。

### Changed（後方互換に影響）

- **network.cooccurrence**: `build_cooccurrence_graph` の類似度を**文書頻度ベースの厳密な集合係数**に統一。ノード選択・ノード `size`・類似度の分母がすべて「文書頻度（その語を含む文書数）」になり、文書内重複の影響を受けなくなった（Jaccard は常に ≤ 1）。**キーワードリストに文書内重複がある場合、エッジ `weight` の値が 1.0.0 と変わる**。ノード `size` は延べ出現数 → 文書頻度に変更。

### Notes

- キーワード抽出の重複保持（頻度シグナル）は意図的動作として据え置き。
- 共起ネットワークの戻り値（weight / size）が変わるため、patiroha を利用する下流（例: 閾値・表示・スナップショットテスト）は追従が必要。

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
