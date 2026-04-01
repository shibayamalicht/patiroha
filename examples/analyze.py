"""
patiroha 分析デモ — 500件の特許データを一通り分析する

使い方:
    cd patiroha/
    source .venv/bin/activate
    python examples/analyze.py
"""

from pathlib import Path

import pandas as pd

import patiroha
from patiroha.stopwords import StopwordManager

DATA_PATH = Path(__file__).parent / "sample_patents_500.csv"
SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main() -> None:
    print(f"patiroha v{patiroha.__version__}")

    # ================================================================
    # 1. データ読み込み
    # ================================================================
    section("1. データ読み込み (io.load_patent_data)")

    df = patiroha.load_patent_data(DATA_PATH)
    print(f"  件数: {len(df)}")
    print(f"  カラム: {list(df.columns)}")

    # ================================================================
    # 2. カラム自動マッピング
    # ================================================================
    section("2. カラム自動マッピング (metadata.smart_map_columns)")

    col_map = patiroha.smart_map_columns(df)
    for field, col in col_map.items():
        print(f"  {field:12s} → {col}")

    # ================================================================
    # 3. メタデータ処理
    # ================================================================
    section("3. メタデータ処理 (metadata)")

    # 日付パース
    df["parsed_date"] = patiroha.parse_date(df["date"])
    df["year"] = df["parsed_date"].dt.year
    print(f"  出願年の範囲: {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"  年別件数:")
    for year, count in df["year"].value_counts().sort_index().items():
        print(f"    {year}: {count}件")

    # IPC抽出
    df["ipc_list"] = df["ipc"].apply(lambda x: patiroha.extract_ipc(str(x)))
    all_ipcs = [code for codes in df["ipc_list"] for code in codes]
    ipc_counts = pd.Series(all_ipcs).value_counts()
    print(f"\n  IPC上位5件:")
    for ipc, count in ipc_counts.head(5).items():
        print(f"    {ipc}: {count}件")

    # 出願人正規化
    df["applicant_list"] = df["applicant"].apply(lambda x: patiroha.normalize_applicant(str(x)))
    all_applicants = [a for apps in df["applicant_list"] for a in apps]
    app_counts = pd.Series(all_applicants).value_counts()
    print(f"\n  出願人上位10件:")
    for app, count in app_counts.head(10).items():
        print(f"    {app}: {count}件")

    # ================================================================
    # 4. ストップワード管理
    # ================================================================
    section("4. ストップワード (stopwords)")

    sw_default = patiroha.get_stopwords()
    print(f"  デフォルト(patent): {len(sw_default)}語")

    sw_npl = patiroha.get_stopwords("npl")
    print(f"  NPLモード: {len(sw_npl)}語")

    mgr = StopwordManager(include=["general", "patent_terms"], exclude=["chemistry"])
    mgr.add(["テスト用語"])
    sw_custom = mgr.build()
    print(f"  カスタム(general+patent_terms, chemistryを除外): {len(sw_custom)}語")
    print(f"    カテゴリ: {mgr.categories}")

    # ================================================================
    # 5. テキスト処理
    # ================================================================
    section("5. テキスト処理 (tokenize)")

    sample_text = df.iloc[0]["abstract"]
    print(f"  入力テキスト: {sample_text[:80]}...")

    # 正規化
    normalized = patiroha.normalize_text(sample_text)
    print(f"  正規化後: {normalized[:80]}...")

    # N-gramフィルタ
    filtered = patiroha.apply_ngram_filters(sample_text)
    print(f"  フィルタ後: {filtered[:80]}...")

    # キーワード抽出
    keywords = patiroha.extract_keywords(sample_text)
    print(f"  抽出キーワード: {keywords}")

    # 全文書のキーワード抽出
    print("\n  全500件のキーワード抽出中...")
    df["keywords"] = df["abstract"].apply(lambda x: patiroha.extract_keywords(str(x)))
    all_keywords = [kw for kws in df["keywords"] for kw in kws]
    kw_counts = pd.Series(all_keywords).value_counts()
    print(f"  ユニークキーワード数: {len(kw_counts)}")
    print(f"  頻出キーワード上位15件:")
    for kw, count in kw_counts.head(15).items():
        print(f"    {kw}: {count}回")

    # ================================================================
    # 6. TF-IDF
    # ================================================================
    section("6. TF-IDF (embeddings.tfidf)")

    tfidf_matrix, feature_names = patiroha.build_tfidf(
        df["abstract"].astype(str),
        min_df=3,
        max_df=0.8,
    )
    print(f"  行列サイズ: {tfidf_matrix.shape}")
    print(f"  語彙数: {len(feature_names)}")
    print(f"  語彙サンプル: {list(feature_names[:10])}")

    # ================================================================
    # 7. 統計分析
    # ================================================================
    section("7. 統計分析 (stats)")

    # HHI
    hhi_result = patiroha.calculate_hhi(app_counts.tolist())
    print(f"  HHI: {hhi_result.value:.4f} ({hhi_result.status})")

    # CAGR
    cagr_result = patiroha.calculate_cagr(df, year_col="year")
    if cagr_result.growth_rate is not None:
        print(f"  CAGR: {cagr_result.growth_rate:.1%} ({cagr_result.trend})")
    else:
        print(f"  CAGR: 計算不可")

    # 技術分野別HHI
    print("\n  技術分野推定 (IPC大分類別):")
    ipc_sections = [codes[0][:4] if codes else "不明" for codes in df["ipc_list"]]
    section_counts = pd.Series(ipc_sections).value_counts()
    for sec, count in section_counts.head(8).items():
        print(f"    {sec}: {count}件")

    # ================================================================
    # 8. 共起ネットワーク
    # ================================================================
    section("8. キーワード共起ネットワーク (network)")

    try:
        G = patiroha.build_cooccurrence_graph(
            df["keywords"].tolist(),
            top_n=30,
            threshold=0.03,
        )
        print(f"  ノード数: {len(G.nodes)}")
        print(f"  エッジ数: {len(G.edges)}")

        communities = patiroha.detect_communities(G)
        n_communities = len(set(communities.values()))
        print(f"  コミュニティ数: {n_communities}")

        hubs = patiroha.get_hub_keywords(G, top_n=10)
        print(f"  ハブキーワード:")
        for kw, score in hubs:
            print(f"    {kw}: {score:.3f}")

        # コミュニティ別キーワード
        print(f"\n  コミュニティ別キーワード:")
        comm_groups: dict[int, list[str]] = {}
        for node, comm_id in communities.items():
            comm_groups.setdefault(comm_id, []).append(node)
        for comm_id in sorted(comm_groups.keys()):
            members = comm_groups[comm_id][:5]
            print(f"    Community {comm_id}: {', '.join(members)}")

    except ImportError:
        print("  networkx未インストール — スキップ")

    # ================================================================
    # 9. クラスタ自動ラベリング
    # ================================================================
    section("9. クラスタ自動ラベリング (clustering.labeling)")

    import numpy as np

    # 疑似的なクラスタラベル (IPC大分類ベース)
    ipc_to_cluster = {}
    for i, sec in enumerate(section_counts.index[:7]):
        ipc_to_cluster[sec] = i

    df["pseudo_cluster"] = [
        ipc_to_cluster.get(codes[0][:4] if codes else "不明", -1)
        for codes in df["ipc_list"]
    ]

    labels = df["pseudo_cluster"].values.astype(np.intp)
    cluster_labels = patiroha.auto_label(
        df["abstract"].astype(str),
        labels,
        top_n=3,
    )
    print(f"  クラスタ数: {len([k for k in cluster_labels if k >= 0])}")
    for cid, label in sorted(cluster_labels.items()):
        count = (labels == cid).sum()
        print(f"    {label} ({count}件)")

    # ================================================================
    # 10. 出願人ポートフォリオ分析
    # ================================================================
    section("10. 出願人ポートフォリオ分析")

    top_applicants = app_counts.head(5).index.tolist()
    print(f"  上位5出願人の年別出願数:")
    print(f"  {'出願人':20s}", end="")
    years = sorted(df["year"].unique())
    for y in years:
        print(f" {int(y):>5d}", end="")
    print()

    for app_name in top_applicants:
        print(f"  {app_name:20s}", end="")
        for y in years:
            mask = df["applicant_list"].apply(lambda x: app_name in x) & (df["year"] == y)
            count = mask.sum()
            print(f" {count:>5d}", end="")
        print()

    # 出願人別HHI
    print(f"\n  出願人別 技術集中度 (上位5):")
    for app_name in top_applicants:
        mask = df["applicant_list"].apply(lambda x: app_name in x)
        app_df = df[mask]
        app_ipcs = [code for codes in app_df["ipc_list"] for code in codes]
        if app_ipcs:
            app_ipc_counts = pd.Series(app_ipcs).value_counts().tolist()
            result = patiroha.calculate_hhi(app_ipc_counts)
            print(f"    {app_name}: HHI={result.value:.3f} ({result.status})")

    # ================================================================
    # サマリ
    # ================================================================
    section("分析完了サマリ")
    print(f"  総特許数: {len(df)}")
    print(f"  出願年: {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"  ユニーク出願人数: {len(app_counts)}")
    print(f"  ユニークIPC数: {len(ipc_counts)}")
    print(f"  ユニークキーワード数: {len(kw_counts)}")
    print(f"  市場集中度(HHI): {hhi_result.value:.4f} ({hhi_result.status})")
    if cagr_result.growth_rate is not None:
        print(f"  成長率(CAGR): {cagr_result.growth_rate:.1%} ({cagr_result.trend})")


if __name__ == "__main__":
    main()
