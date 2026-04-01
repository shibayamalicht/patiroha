# patiroha — Patent Text Analysis Toolkit / 特許テキスト分析ツールキット

**patiroha** is a Python library for patent information analysis, built with Japanese patents in mind. From keyword extraction to SBERT-powered technology clustering — everything a patent analyst needs, minus the GUI.

**patiroha** は、日本語特許の情報分析に特化した Python ライブラリです。キーワード抽出から SBERT による技術クラスタリングまで — 特許情報分析に必要な処理を、GUI なしのコードだけで実行できます。

> *"Why is it called patiroha?"* — **Pat**ent + **いろは** (iroha). "Iroha" is the beginning of Japanese words — and this library is where patent information analysis begins.
>
> 「いろは」は日本語の言葉のはじまり。特許情報分析は、ここから始まります。

---

## 🎯 Design Philosophy / 設計思想

- **Japanese-first, multilingual-ready** — Janome morphological analysis, 1,100+ patent-specific stopwords, full/half-width normalization. English fallback built in.
  - *日本語ファースト、多言語対応* — Janome 形態素解析、1,100語超の特許ストップワード辞書、全角半角正規化。英語フォールバック内蔵。

- **No GUI, pure library** — Returns DataFrames, numpy arrays, and dataclasses. Bring your own Plotly / Streamlit / Jupyter.
  - *GUI なし、純粋なライブラリ* — DataFrame・numpy 配列・dataclass を返すだけ。描画は好きなツールで。

- **Lightweight core, heavy options** — Core needs only pandas + janome + scikit-learn. SBERT, UMAP, NetworkX are opt-in.
  - *コアは軽量、オプションでヘビー級* — コアは pandas + janome + sklearn のみ。SBERT・UMAP・NetworkX は必要な時だけ。

- **AI-friendly codebase** — Every function has type hints, Google-style docstrings, and predictable return types. Perfect for vibe-coding with Claude, ChatGPT, or Copilot.
  - *AI フレンドリーなコードベース* — 全関数に型ヒント・Google スタイル docstring・予測可能な戻り値型。Claude / ChatGPT / Copilot でのバイブコーディングに最適。

---

## 🧩 Modules at a Glance / モジュール一覧

```
patiroha/
├── stopwords    — 7-category stopword dictionary & manager         ストップワード辞書・管理
├── tokenize     — Compound noun extraction & patent text cleanup   複合名詞抽出・定型句フィルタ
├── metadata     — IPC hierarchy, date parsing, applicant cleanup   IPC階層・日付・出願人処理
├── io           — CSV/Excel loader with encoding auto-detection    ファイル読み込み
├── embeddings   — SBERT & TF-IDF vectorization                    意味ベクトル・TF-IDF
├── clustering   — UMAP + HDBSCAN / KMeans + auto-labeling         クラスタリング・自動ラベリング
├── stats        — HHI, entropy, Gini, CAGR, MMR, similarity       統計指標・代表特許・類似検索
├── network      — Keyword co-occurrence with 5 similarity metrics  共起ネットワーク
└── pipeline     — One-liner end-to-end analysis                    ワンライナーで全自動分析
```

---

## 📦 Installation / インストール

```bash
pip install patiroha
```

That's it. You now have keyword extraction, IPC parsing, and statistical analysis.
これだけで、キーワード抽出・IPC 解析・統計分析が使えます。

### Optional extras / オプション

```bash
pip install patiroha[embeddings]   # SBERT (sentence-transformers)
pip install patiroha[clustering]   # UMAP + HDBSCAN
pip install patiroha[network]      # NetworkX (co-occurrence graphs / 共起ネットワーク)
pip install patiroha[all]          # Everything / 全部入り
```

**Requirements / 動作環境:** Python 3.9+ / Windows, macOS, Linux

---

## 🚀 Quick Start / クイックスタート

### The One-Liner (for the impatient) / せっかちな人向け

```python
from patiroha import PatentPipeline

result = PatentPipeline(min_cluster_size=10).run("patents.csv")
print(result.cluster_names)  # {0: "[0] セルロース, 樹脂", 1: "[1] 電池, 電解質", ...}
```

Done. Your 500 patents are now clustered and labeled. ☕
これで 500 件の特許がクラスタリング・ラベリング済みです。

### Step by step (for the thorough) / 丁寧にやりたい人向け

```python
import patiroha

# 1. Load data / データ読み込み
df = patiroha.load_patent_data("patents.csv")  # encoding auto-detected / 文字コード自動判定

# 2. Auto-detect columns / カラム自動判定
col_map = patiroha.smart_map_columns(df)
# => {"title": "発明の名称", "abstract": "要約", "applicant": "出願人", ...}

# 3. Extract keywords / キーワード抽出
keywords = patiroha.extract_keywords("セルロースナノファイバーを含有する樹脂組成物に関する。")
# => ["セルロースナノファイバー", "樹脂組成物"]

# 4. Parse IPC hierarchy / IPC 階層分解
ipc = patiroha.parse_ipc("H01L31/0725")
# ipc.section="h", ipc.class_code="h01", ipc.subclass="h01l", ipc.group="31", ipc.subgroup="0725"

# 5. Normalize applicants / 出願人正規化
patiroha.normalize_applicant("トヨタ自動車株式会社;ソニー株式会社")
# => ["トヨタ自動車", "ソニー"]
```

---

## 📖 Detailed Usage / 詳しい使い方

### 🔤 Stopwords / ストップワード

7 categories, ~1,100 words (before half/full-width expansion). Built for Japanese patent boilerplate.
7 カテゴリ、約 1,100 語（半角全角展開前）。日本語特許の定型表現に対応。

```python
sw = patiroha.get_stopwords()          # Patent mode / 特許モード (791 words / 語)
sw = patiroha.get_stopwords("npl")     # + academic English / + 英語学術用語 (1534 words / 語)
```

**Customize / カスタマイズ:**

```python
from patiroha import StopwordManager

mgr = StopwordManager(
    include=["general", "patent_terms"],  # Pick categories / カテゴリ選択
    exclude=["chemistry"],                 # Keep chemistry terms / 化学用語は残す
)
mgr.add(["自社製品名"])      # Add custom words / 不要語を追加
mgr.remove(["触媒"])         # Keep analysis targets / 分析対象語は除外
sw = mgr.build()             # => frozenset (O(1) lookup)
```

| Category / カテゴリ | Examples / 例 | Count / 語数 |
|----------|-------------|-------|
| `general` | する、ため、もの、および | 134 |
| `patent_terms` | 本発明、請求項、実施形態 | 145 |
| `structure` | 上部、表面、装置、フレーム | 98 |
| `it_control` | システム、データ、制御、通信 | 104 |
| `chemistry` | 溶液、触媒、樹脂、ポリマー | 104 |
| `misc` | mm、℃、株式会社、Inc | 140 |
| `npl` | abstract、study、however、論文 | 423 |

**Browse stopwords / ストップワードの中身を確認する:**

```python
import patiroha

# List all categories and their word counts / 全カテゴリと語数を一覧
patiroha.list_categories()
# => {"general": 134, "patent_terms": 145, "structure": 98, "it_control": 104,
#     "chemistry": 104, "misc": 140, "npl": 423}

# See all words in a category / 特定カテゴリの全語をリスト表示
patiroha.list_words("chemistry")
# => ["ポリマー", "モノマー", "化合物", "反応", "反応条件", "反応時間", "反応温度", ...]

patiroha.list_words("patent_terms")
# => ["PCT", "一実施例", "一実施形態", "不可能", "事件番号", ...]

# See what's active in a StopwordManager / マネージャーの現在設定を確認
mgr = StopwordManager(include=["general", "chemistry"])
mgr.add(["自社用語"])
mgr.remove(["触媒"])

# Category-by-category word list / カテゴリ別の語一覧
mgr.list_active_words()
# => {"general": ["ある", "いずれ", ...], "chemistry": ["モノマー", ...], "custom": ["自社用語"]}

# Human-readable summary / 設定サマリを表示
print(mgr.summary())
# StopwordManager Configuration:
#   Active categories: general, chemistry
#     general: 134 words
#     chemistry: 104 words
#   Custom added: 1 words — 自社用語
#   Removed: 1 words — 触媒
#   Total after expansion: 238 words
```

---

### 🔬 Keyword Extraction / キーワード抽出

Janome-based compound noun extraction, tuned for patent Japanese.
Janome 形態素解析ベースの複合名詞抽出。日本語特許向けにチューニング済み。

```python
# Basic / 基本
kw = patiroha.extract_keywords("セルロースナノファイバーを含有する樹脂組成物に関する。")
# => ["セルロースナノファイバー", "樹脂組成物"]

# Extract verbs too / 動詞も抽出
kw = patiroha.extract_keywords(text, pos_tags=("名詞", "動詞"))

# Custom rejection patterns / カスタム除外パターン
kw = patiroha.extract_keywords(text, extra_reject_patterns=[r"特定パターン"])

# Disable built-in filters / 組み込みフィルタ無効化
kw = patiroha.extract_keywords(text, disable_default_reject=True, min_length=1)
```

| Parameter / パラメータ | Default / デフォルト | Description / 説明 |
|-----------|---------|-------------|
| `stopwords` | `None` | Stopword set. `None` = patent default / ストップワード。`None` で特許デフォルト |
| `apply_filters` | `True` | Remove "図1に示す" etc. / 定型句を事前除去 |
| `clean_html` | `False` | Strip HTML tags / HTML タグ除去 |
| `pos_tags` | `("名詞",)` | POS tags to extract / 抽出する品詞タグ |
| `min_length` | `2` | Minimum character length / 最小文字数 |
| `extra_reject_patterns` | `None` | Additional regex patterns / 追加の除外正規表現 |
| `disable_default_reject` | `False` | Skip built-in filters / 組み込みフィルタを無効化 |

**What the built-in filters remove / 組み込みフィルタが除去するもの:**
- Reference symbols / 参照符号: 部材(101), 図3に示す, 請求項1
- Boilerplate / 定型句: 一実施形態において, 本明細書では
- Functional phrases / 機能句: することができる, に限定されない
- Section headers / 見出し: 用語の定義, 好適には

---

### 🏷️ Metadata / メタデータ処理

#### IPC Hierarchy / IPC 階層分解

```python
ipc = patiroha.parse_ipc("H01L31/0725")
# ipc.section    = "h"       → "電気" (Electricity)
# ipc.class_code = "h01"
# ipc.subclass   = "h01l"
# ipc.group      = "31"
# ipc.subgroup   = "0725"

from patiroha import IPC_SECTIONS
IPC_SECTIONS["h"]  # => "電気"
IPC_SECTIONS["c"]  # => "化学; 冶金"

# Batch extraction / 一括抽出
codes = patiroha.extract_ipc("B32B 27/00; C08L 1/02")
# => ["b32b27/00", "c08l1/02"]

# With full hierarchy / 階層付き
parsed = patiroha.extract_ipc_parsed("B32B 27/00; C08L 1/02")
# => [IPCCode(section="b", class_code="b32", ...), ...]
```

#### Date Parsing / 日付パース

Handles whatever format your data throws at it.
どんなフォーマットが来ても受け止めます。

```python
df["date"] = patiroha.parse_date(df["出願日"])
# Supported / 対応: "2020-01-15", "20200115", "2020", Excel serial numbers / Excel日付数値
```

#### Applicant Normalization / 出願人正規化

```python
patiroha.normalize_applicant("トヨタ自動車株式会社;ソニー株式会社")
# => ["トヨタ自動車", "ソニー"]
# Removes / 除去: 株式会社, Inc., Ltd., GmbH, Co., Corp., LLC, etc.
```

---

### 📊 Statistics / 統計分析

#### Concentration & Diversity / 集中度と多様性

```python
from patiroha import calculate_hhi, calculate_diversity, calculate_cagr

counts = df["applicant"].value_counts().tolist()

# HHI (Herfindahl-Hirschman Index / ハーフィンダール・ハーシュマン指数)
hhi = calculate_hhi(counts)
# => HHIResult(value=0.054, status="競争的 (分散)")

# All diversity metrics at once / 多様性指標を一括計算
div = calculate_diversity(counts)
# div.hhi        = 0.054   (concentration / 集中度)
# div.entropy    = 4.12    (higher = more diverse / 高い = 分散)
# div.gini       = 0.31    (higher = more unequal / 高い = 偏り)
# div.n_entities = 50      (entity count / エンティティ数)

# CAGR + Trend / 年平均成長率 + トレンド
cagr = calculate_cagr(df, year_col="year")
# => CAGRResult(growth_rate=0.123, trend="急上昇")
```

#### Representatives & Similarity / 代表特許と類似検索

```python
from patiroha import find_representatives, find_representatives_mmr, find_similar

# Closest to centroid / 重心に最も近い代表特許
reps = find_representatives(vectors, df, n=5)

# MMR: diverse representatives (not all similar to each other)
# MMR: 多様な代表特許（互いに似すぎない選択）
reps = find_representatives_mmr(vectors, df, n=5, diversity=0.3)

# Find similar patents / 類似特許検索
similar = find_similar(vectors[42], vectors, df, n=10)
```

---

### 🧠 SBERT Embeddings / SBERT 埋め込み

*Requires / 要インストール: `pip install patiroha[embeddings]`*

```python
from patiroha import SBERTEmbedder

# Default: multilingual lightweight model / デフォルト: 多言語軽量モデル
embedder = SBERTEmbedder()

# Or choose your own / モデルは自由に変更可能
embedder = SBERTEmbedder("cl-tohoku/bert-base-japanese-v3")   # Japanese-specialized / 日本語特化
embedder = SBERTEmbedder("all-MiniLM-L6-v2")                  # English, fast / 英語、高速

# Encode with column weighting / カラム重み付きエンコード
vectors = embedder.encode(
    df,
    text_columns=["title", "abstract"],
    column_weights={"title": 2},   # Repeat title 2x for emphasis / タイトルを2倍強調
    separator=" [SEP] ",            # Custom separator / カスタム区切り文字
)
# => numpy array (n_patents, 384), L2-normalized / L2正規化済み
```

| Parameter / パラメータ | Default / デフォルト | Description / 説明 |
|-----------|---------|-------------|
| `text_columns` | *(required / 必須)* | Columns to concatenate / 連結するカラム |
| `column_weights` | `None` | `{"title": 2}` repeats title 2x / タイトルを2回繰り返し |
| `separator` | `" "` | Text separator / 区切り文字 |
| `batch_size` | `128` | Batch size / バッチサイズ |
| `normalize_embeddings` | `True` | L2 normalize / L2正規化 |

---

### 🗺️ Clustering / クラスタリング

*Requires / 要インストール: `pip install patiroha[clustering]`*

```python
from patiroha import build_landscape, auto_label

# UMAP + HDBSCAN (automatic cluster count / クラスタ数自動)
result = build_landscape(vectors, min_cluster_size=10)
# result.labels    — cluster labels / クラスタラベル (numpy array)
# result.coords    — 2D UMAP coordinates / 2D UMAP 座標
# result.n_clusters — cluster count / クラスタ数
# result.noise_count — noise points / ノイズ件数

# Or KMeans (you choose the count / クラスタ数を自分で指定)
result = build_landscape(vectors, method="kmeans", n_clusters=10)

# Auto-label clusters / クラスタ自動ラベリング
names = auto_label(df["abstract"], result.labels, method="c-tfidf", top_n=3)
# => {0: "[0] セルロース, 樹脂, 複合材料", 1: "[1] 電池, 電解質, 正極"}
```

**Tuning guide / チューニングガイド:**

| Goal / 目的 | Parameter / パラメータ |
|------------|---------------------|
| Bigger clusters / 大きなクラスタにまとめたい | `min_cluster_size=30` |
| Finer clusters / 細かく分けたい | `min_cluster_size=5, min_samples=3` |
| Uniform sizes / 均一サイズにしたい | `cluster_selection_method="leaf"` |
| Fixed count / クラスタ数を決めたい | `method="kmeans", n_clusters=10` |
| Dense layout / 密集した配置にしたい | `min_dist=0.01` |
| Spread layout / 広がった配置にしたい | `min_dist=0.5, n_neighbors=30` |

<details>
<summary>Full parameter list / 全パラメータ一覧</summary>

| Parameter / パラメータ | Default / デフォルト | Description / 説明 |
|-----------|---------|-------------|
| `method` | `"hdbscan"` | `"hdbscan"` or `"kmeans"` |
| `n_neighbors` | `15` | UMAP neighborhood size / UMAP 近傍数 |
| `min_dist` | `0.1` | UMAP minimum distance / UMAP 最小距離 |
| `n_components` | `2` | UMAP output dimensions / UMAP 出力次元 |
| `umap_metric` | `"cosine"` | UMAP distance metric / UMAP 距離メトリック |
| `min_cluster_size` | `15` | HDBSCAN minimum cluster points / HDBSCAN 最小クラスタサイズ |
| `min_samples` | `10` | HDBSCAN core point threshold / HDBSCAN コアポイント閾値 |
| `cluster_metric` | `"euclidean"` | HDBSCAN distance metric / HDBSCAN 距離メトリック |
| `cluster_selection_method` | `"eom"` | `"eom"` or `"leaf"` |
| `n_clusters` | `8` | KMeans cluster count / KMeans クラスタ数 |

</details>

---

### 🔗 Co-occurrence Network / 共起ネットワーク

*Requires / 要インストール: `pip install patiroha[network]`*

```python
from patiroha import build_cooccurrence_graph, detect_communities, get_hub_keywords

# Build network / ネットワーク構築
G = build_cooccurrence_graph(
    df["keywords"].tolist(),
    top_n=40,
    threshold=0.05,
    similarity="jaccard",   # or "dice", "cosine", "pmi", "frequency"
)

# Find communities / コミュニティ検出
communities = detect_communities(G, algorithm="louvain")
# Also / 他にも: "greedy_modularity", "label_propagation"

# Find hub keywords / ハブキーワード抽出
hubs = get_hub_keywords(G, top_n=10, centrality="pagerank")
# Also / 他にも: "degree", "betweenness", "eigenvector"
```

**Which similarity metric? / どの類似度指標を使う？**

| Metric / 指標 | What it does / 何をする | When to use / いつ使う |
|--------|---------------------|---------------------|
| `jaccard` | Co-occurrence ratio / 共起割合 | Default. Balanced. / デフォルト。迷ったらこれ |
| `dice` | Emphasizes co-occurrence / 共起を重視 | Want more edges / 関連性を広く拾いたい |
| `cosine` | Absorbs frequency differences / 頻度差を吸収 | Data with high variance / 頻度差が大きいデータ |
| `pmi` | Statistical significance / 統計的有意性 | Rigorous analysis / 厳密な分析 |
| `frequency` | Raw count / 生の共起回数 | Compare absolute numbers / 絶対数で比較 |

---

### ⚡ Pipeline / パイプライン

For when you want the whole meal, not individual dishes.
単品じゃなくフルコースが欲しい時に。

```python
from patiroha import PatentPipeline

pipe = PatentPipeline(
    # Stopwords / ストップワード
    stopword_mode="patent",
    extra_stopwords=["不要語"],
    remove_stopwords=["触媒"],        # Keep for analysis / 分析対象は除外

    # Keywords / キーワード
    pos_tags=("名詞",),
    min_keyword_length=2,

    # SBERT
    sbert_model="paraphrase-multilingual-MiniLM-L12-v2",
    text_columns=["title", "abstract"],
    column_weights={"title": 2},

    # Clustering / クラスタリング
    cluster_method="hdbscan",
    min_cluster_size=10,
    umap_metric="cosine",

    # Labeling / ラベリング
    label_method="c-tfidf",           # or "tfidf"
    label_top_n=3,
)

# Full auto / 全自動実行
result = pipe.run("patents.csv")

# Or step by step / ステップ実行も可能
pipe.load("patents.csv")
pipe.preprocess()        # Date, IPC, applicant / 日付・IPC・出願人処理
pipe.extract_kw()        # Keywords / キーワード抽出
pipe.embed()             # SBERT vectors / SBERT ベクトル生成
pipe.cluster()           # UMAP + HDBSCAN / クラスタリング
result = pipe.result
```

**What's in `result` / 結果の中身:**

```python
result.df               # DataFrame with all computed columns / 全カラム付き DataFrame
result.vectors          # SBERT embeddings (n, 384) / SBERT 埋め込みベクトル
result.cluster_names    # {0: "[0] keyword1, keyword2", ...} / クラスタ名
result.cluster_labels   # numpy array of cluster IDs / クラスタID配列
result.cluster_coords   # 2D UMAP coordinates / 2D UMAP 座標
result.col_map          # Auto-detected column mapping / 自動検出カラム対応表
result.stopwords        # Stopword set used / 使用したストップワード
```

---

## 📥 Input Data Format / 入力データ形式

CSV or Excel. The following column names are auto-detected (partial match).
CSV または Excel。以下のカラム名を部分一致で自動検出します。

| Field / 項目 | Auto-detected names / 自動認識するカラム名 | Required? / 必須？ |
|-------|-------------------------------|-----------|
| Title / タイトル | `title`, `発明の名称`, `名称`, `タイトル` | Recommended / 推奨 |
| Abstract / 要約 | `abstract`, `要約`, `抄録`, `概要` | Recommended / 推奨 |
| Applicant / 出願人 | `applicant`, `出願人`, `権利者` | Optional / 任意 |
| IPC | `ipc`, `IPC`, `国際特許分類`, `FI` | Optional / 任意 |
| Date / 日付 | `date`, `出願日`, `公開日` | Optional / 任意 |

**Supported date formats / 対応する日付フォーマット:** `2020-01-15`, `20200115`, `2020`, Excel serial numbers / Excel日付数値

---

## 🎮 Examples / サンプル

Three demo scripts are included in `examples/`.
3 つのデモスクリプトが `examples/` に入っています。

```bash
# Generate 500 synthetic patents / 500件のダミー特許データを生成
python examples/generate_sample.py

# Terminal-based analysis / ターミナルで分析結果を確認
python examples/analyze.py

# Interactive dashboard in browser / ブラウザでインタラクティブなダッシュボードを表示
pip install plotly  # First time only / 初回のみ
python examples/visualize.py
# => http://localhost:8765/dashboard.html
```

**Dashboard includes / ダッシュボードの内容:**
- Filing trends by IPC section / IPC セクション別 出願トレンド
- IPC hierarchy treemap / IPC 階層ツリーマップ
- Patent cluster map (SBERT + UMAP + HDBSCAN) / 特許クラスタマップ
- Keyword co-occurrence network / キーワード共起ネットワーク
- Applicant portfolio (Entropy vs HHI) / 出願人ポートフォリオ
- Top 25 keywords / 頻出キーワード Top 25

---

## 🤖 For AI Coding Assistants / AI コーディングアシスタント向け

If you're an AI helping a user with patiroha, here's what you need to know.
AI アシスタントが patiroha を使うユーザーを支援する際のガイドです。

**Architecture / アーキテクチャ:**

`src/patiroha/` layout with lazy imports for optional dependencies. Core modules (`stopwords`, `tokenize`, `metadata`, `io`, `stats`) require only pandas + janome + scikit-learn. Optional modules (`embeddings`, `clustering`, `network`, `pipeline`) need extra packages and will raise `ImportError` with a `pip install` hint if missing.

`src/patiroha/` レイアウト。オプション依存は遅延 import。コアモジュール（`stopwords`, `tokenize`, `metadata`, `io`, `stats`）は pandas + janome + scikit-learn のみ。オプションモジュール（`embeddings`, `clustering`, `network`, `pipeline`）は追加パッケージが必要で、未インストール時は `pip install` ヒント付きの `ImportError` を送出。

**Key types / 主要な型:**
- `HHIResult`, `CAGRResult`, `LandscapeResult`, `Representative`, `IPCCode` — frozen dataclasses in `patiroha._types` / `patiroha._types` の frozen dataclass
- `DiversityResult` — frozen dataclass in `patiroha.stats.hhi` / `patiroha.stats.hhi` の frozen dataclass
- `frozenset[str]` for stopwords (O(1) lookup, immutable) / ストップワードは `frozenset`
- `npt.NDArray[np.float64]` for embeddings / 埋め込みベクトルは numpy 配列

**Common patterns / よくあるパターン:**
```python
# Browse stopwords → list_categories() → dict, list_words(cat) → list
# Stopword customization → StopwordManager.build() → frozenset
# Keyword extraction → extract_keywords(text, stopwords=sw) → list[str]
# Embeddings → SBERTEmbedder().encode(df, text_columns=[...]) → ndarray
# Clustering → build_landscape(vectors) → LandscapeResult
# Labeling → auto_label(texts, labels) → dict[int, str]
# Network → build_cooccurrence_graph(kw_lists) → nx.Graph
# Statistics → calculate_diversity(counts) → DiversityResult
# Full pipeline → PatentPipeline(...).run(path) → AnalysisResult
```

**Testing / テスト:** `pytest tests/` — 84 tests covering all modules. `ruff check` + `mypy --strict` clean. / 84テストで全モジュールをカバー。ruff・mypy strict クリーン。

---

## 👤 Author / 作者

**しばやま** ([@shibayamalicht](https://github.com/shibayamalicht))

## 📄 License / ライセンス

MIT License — Copyright (c) 2026 しばやま (shibayamalicht)
