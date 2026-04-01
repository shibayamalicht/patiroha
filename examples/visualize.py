"""
patiroha 可視化デモ — 共起ネットワーク・クラスタマップ・統計チャート

使い方:
    cd patiroha/
    source .venv/bin/activate
    python examples/visualize.py

出力: examples/output/ にHTMLファイルを生成しブラウザで表示
"""

from __future__ import annotations

import math
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import patiroha
from patiroha.metadata.ipc import IPC_SECTIONS, extract_ipc_parsed

DATA_PATH = Path(__file__).parent / "sample_patents_500.csv"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = [
    "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
    "#8ab17d", "#6c5ce7", "#fd79a8", "#00b894", "#0984e3",
    "#d63031", "#636e72", "#b2bec3", "#a29bfe", "#55efc4",
]


def load_and_preprocess() -> pd.DataFrame:
    """Load data and run all preprocessing."""
    print("データ読み込み・前処理...")
    df = patiroha.load_patent_data(DATA_PATH)

    df["parsed_date"] = patiroha.parse_date(df["date"])
    df["year"] = df["parsed_date"].dt.year
    df["ipc_list"] = df["ipc"].apply(lambda x: patiroha.extract_ipc(str(x)))
    df["ipc_parsed"] = df["ipc"].apply(lambda x: extract_ipc_parsed(str(x)))
    df["applicant_list"] = df["applicant"].apply(lambda x: patiroha.normalize_applicant(str(x)))
    df["keywords"] = df["abstract"].apply(lambda x: patiroha.extract_keywords(str(x)))

    # IPC階層
    df["ipc_section"] = df["ipc_parsed"].apply(
        lambda codes: codes[0].section.upper() if codes else "?"
    )
    df["ipc_class"] = df["ipc_parsed"].apply(
        lambda codes: codes[0].class_code.upper() if codes else "?"
    )
    df["ipc_subclass"] = df["ipc_parsed"].apply(
        lambda codes: codes[0].subclass.upper() if codes else "?"
    )
    df["ipc_section_name"] = df["ipc_section"].apply(
        lambda s: f"{s}: {IPC_SECTIONS.get(s.lower(), '不明')}"
    )

    return df


# ================================================================
# 1. 共起ネットワーク
# ================================================================
def build_network_chart(df: pd.DataFrame) -> go.Figure:
    """Build an interactive keyword co-occurrence network chart."""
    print("共起ネットワーク構築...")
    import networkx as nx

    G = patiroha.build_cooccurrence_graph(df["keywords"].tolist(), top_n=40, threshold=0.03)
    communities = patiroha.detect_communities(G)
    pos = nx.spring_layout(G, k=1.2, seed=42, iterations=80)

    max_w = max((d.get("weight", 0) for _, _, d in G.edges(data=True)), default=1)
    fig = go.Figure()

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = data.get("weight", 0)
        width = 0.5 + (w / max_w) * 4
        opacity = 0.2 + (w / max_w) * 0.6
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=width, color=f"rgba(150,150,150,{opacity})"),
            hoverinfo="text", text=f"{u} — {v}<br>Jaccard: {w:.3f}", showlegend=False,
        ))

    comm_ids = sorted(set(communities.values()))
    for comm_id in comm_ids:
        nodes = [n for n, c in communities.items() if c == comm_id]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text",
            marker=dict(
                size=[math.log(G.nodes[n].get("size", 1) + 1) * 12 for n in nodes],
                color=COLORS[comm_id % len(COLORS)],
                line=dict(width=1.5, color="white"),
            ),
            text=nodes, textposition="top center", textfont=dict(size=10),
            hovertext=[
                f"<b>{n}</b><br>出現: {G.nodes[n].get('size', 0)}回<br>"
                f"次数: {G.degree(n)}<br>Community: {comm_id}"
                for n in nodes
            ],
            hoverinfo="text", name=f"Community {comm_id}",
        ))

    fig.update_layout(
        title=dict(text="キーワード共起ネットワーク", font=dict(size=20)),
        showlegend=True, legend=dict(orientation="h", y=-0.05),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        width=1000, height=750, plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


# ================================================================
# 2. クラスタマップ (SBERT + HDBSCAN + t-SNE)
# ================================================================
def build_cluster_map(df: pd.DataFrame) -> go.Figure:
    """Build a 2D cluster scatter map using SBERT + UMAP + HDBSCAN."""
    import hdbscan
    import umap

    print("SBERT エンベディング生成中...")
    embedder = patiroha.SBERTEmbedder()
    vectors = embedder.encode(
        df, text_columns=["title", "abstract"],
        progress_callback=lambda p: print(f"  SBERT: {p:.0%}", end="\r"),
    )
    print()

    # UMAP dimensionality reduction
    print("UMAP 次元削減...")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric="cosine", random_state=42,
    )
    coords = reducer.fit_transform(vectors)
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    # HDBSCAN clustering on UMAP 2D
    print("HDBSCAN クラスタリング...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10, min_samples=5,
        metric="euclidean", cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords)
    df["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int(np.sum(labels == -1))
    print(f"  クラスタ数: {n_clusters}, ノイズ: {noise}件")

    # Auto-label
    cluster_labels = patiroha.auto_label(df["abstract"].astype(str), labels, top_n=3)
    df["cluster_label"] = [cluster_labels.get(int(l), "ノイズ") for l in labels]

    fig = go.Figure()
    unique_labels = sorted(df["cluster_label"].unique(), key=lambda x: ("ノイズ" in x, x))

    for i, label in enumerate(unique_labels):
        subset = df[df["cluster_label"] == label]
        is_noise = "ノイズ" in label
        short = label if len(label) < 45 else label[:42] + "..."

        fig.add_trace(go.Scatter(
            x=subset["umap_x"], y=subset["umap_y"], mode="markers",
            marker=dict(
                size=5 if is_noise else 8,
                color="#cccccc" if is_noise else COLORS[i % len(COLORS)],
                opacity=0.3 if is_noise else 0.8,
                line=dict(width=0.5, color="white"),
            ),
            name=short,
            hovertext=[
                f"<b>{row['title'][:60]}</b><br>"
                f"出願人: {', '.join(row['applicant_list'][:2])}<br>"
                f"IPC: {row['ipc']}<br>"
                f"年: {int(row['year'])}<br>"
                f"KW: {', '.join(row['keywords'][:5])}"
                for _, row in subset.iterrows()
            ],
            hoverinfo="text",
        ))

    fig.update_layout(
        title=dict(text="特許クラスタマップ (SBERT + UMAP + HDBSCAN)", font=dict(size=20)),
        xaxis=dict(title="UMAP 1", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="UMAP 2", showgrid=True, gridcolor="#eee"),
        width=1100, height=800, plot_bgcolor="white",
        legend=dict(font=dict(size=9), bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#ddd", borderwidth=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ================================================================
# 3. 出願トレンド (IPC階層対応)
# ================================================================
def build_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Build a stacked area chart of filing trends by IPC section."""
    print("出願トレンドチャート構築...")

    trend = df.groupby(["year", "ipc_section_name"]).size().unstack(fill_value=0)

    fig = go.Figure()
    for i, col in enumerate(trend.columns):
        fig.add_trace(go.Scatter(
            x=trend.index, y=trend[col], mode="lines+markers",
            name=col, stackgroup="one",
            line=dict(color=COLORS[i % len(COLORS)]),
        ))

    fig.update_layout(
        title=dict(text="IPCセクション別 出願トレンド", font=dict(size=20)),
        xaxis=dict(title="出願年", dtick=1),
        yaxis=dict(title="出願件数"),
        width=1000, height=500, plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
        margin=dict(l=40, r=20, t=60, b=80),
    )
    return fig


# ================================================================
# 4. IPC階層ツリーマップ
# ================================================================
def build_ipc_treemap(df: pd.DataFrame) -> go.Figure:
    """Build a treemap showing IPC hierarchy distribution."""
    print("IPC階層ツリーマップ構築...")

    rows = []
    for _, row in df.iterrows():
        for ipc in row["ipc_parsed"]:
            if ipc.section:
                sec_name = IPC_SECTIONS.get(ipc.section, "不明")
                rows.append({
                    "section": f"{ipc.section.upper()}: {sec_name}",
                    "subclass": ipc.subclass.upper() if ipc.subclass else "不明",
                    "full": ipc.raw.upper(),
                    "count": 1,
                })

    if not rows:
        return go.Figure()

    ipc_df = pd.DataFrame(rows)
    agg = ipc_df.groupby(["section", "subclass"]).size().reset_index(name="count")

    fig = px.treemap(
        agg,
        path=["section", "subclass"],
        values="count",
        color="count",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        title=dict(text="IPC階層分布 (セクション → サブクラス)", font=dict(size=20)),
        width=1000, height=600,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ================================================================
# 5. 出願人ポートフォリオ
# ================================================================
def build_portfolio_chart(df: pd.DataFrame) -> go.Figure:
    """Build a bubble chart of applicant portfolios with full diversity metrics."""
    print("出願人ポートフォリオチャート構築...")

    all_apps = [a for apps in df["applicant_list"] for a in apps]
    app_counts = pd.Series(all_apps).value_counts()
    top_apps = app_counts.head(12).index.tolist()

    rows = []
    for app_name in top_apps:
        mask = df["applicant_list"].apply(lambda x, a=app_name: a in x)
        app_df = df[mask]
        app_ipcs = [c for codes in app_df["ipc_list"] for c in codes]
        n_unique_ipc = len(set(app_ipcs))
        if app_ipcs:
            div = patiroha.calculate_diversity(pd.Series(app_ipcs).value_counts().tolist())
        else:
            div = patiroha.calculate_diversity([])
        rows.append({
            "出願人": app_name, "出願件数": len(app_df),
            "IPC多様性": n_unique_ipc,
            "HHI": div.hhi, "エントロピー": div.entropy, "ジニ係数": div.gini,
            "平均出願年": round(app_df["year"].mean(), 1),
        })

    port_df = pd.DataFrame(rows)
    fig = px.scatter(
        port_df, x="エントロピー", y="HHI",
        size="出願件数", color="出願人",
        hover_data=["出願件数", "IPC多様性", "ジニ係数", "平均出願年"],
        color_discrete_sequence=COLORS, size_max=40,
    )
    fig.update_layout(
        title=dict(text="出願人ポートフォリオ (エントロピー vs HHI)", font=dict(size=20)),
        xaxis=dict(title="技術エントロピー (高い = 多様)"),
        yaxis=dict(title="技術集中度 HHI (高い = 集中)"),
        width=1000, height=600, plot_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ================================================================
# 6. キーワード頻度チャート
# ================================================================
def build_keyword_chart(df: pd.DataFrame) -> go.Figure:
    """Build a horizontal bar chart of top keywords."""
    print("キーワード頻度チャート構築...")
    all_kw = [kw for kws in df["keywords"] for kw in kws]
    kw_counts = pd.Series(all_kw).value_counts().head(25)

    fig = go.Figure(go.Bar(
        x=kw_counts.values[::-1], y=kw_counts.index[::-1],
        orientation="h", marker=dict(color=COLORS[1]),
    ))
    fig.update_layout(
        title=dict(text="頻出キーワード Top 25", font=dict(size=20)),
        xaxis=dict(title="出現回数"),
        yaxis=dict(tickfont=dict(size=11)),
        width=800, height=650, plot_bgcolor="white",
        margin=dict(l=180, r=20, t=60, b=40),
    )
    return fig


# ================================================================
# ダッシュボード
# ================================================================
def build_dashboard(figs: dict[str, go.Figure], df: pd.DataFrame) -> str:
    """Build a single-page HTML dashboard with iframes."""
    print("ダッシュボードHTML生成...")

    all_apps = [a for apps in df["applicant_list"] for a in apps]
    app_counts = pd.Series(all_apps).value_counts()
    div = patiroha.calculate_diversity(app_counts.tolist())
    cagr = patiroha.calculate_cagr(df, year_col="year")
    n_clusters = df["cluster"].nunique() - (1 if -1 in df["cluster"].values else 0)

    cards = [
        ("#264653", "white", str(len(df)), "総特許数"),
        ("#2a9d8f", "white", str(len(app_counts)), "出願人数"),
        ("#e9c46a", "#333", f"{int(df['year'].min())}-{int(df['year'].max())}", "出願年"),
        ("#f4a261", "#333", f"{div.hhi:.3f}", f"HHI ({div.hhi_status})"),
        ("#e76f51", "white", f"{div.entropy:.2f}", "エントロピー"),
        ("#8ab17d", "white", f"{cagr.growth_rate:.1%}", f"CAGR ({cagr.trend})"),
        ("#6c5ce7", "white", str(n_clusters), "クラスタ数"),
    ]
    cards_html = "\n".join(
        f'<div style="background:{bg};color:{fg};padding:18px 28px;border-radius:8px;'
        f'min-width:130px;text-align:center;">'
        f'<div style="font-size:30px;font-weight:bold;">{val}</div>'
        f'<div style="font-size:12px;opacity:0.8;">{label}</div></div>'
        for bg, fg, val, label in cards
    )

    sections = [
        ("出願トレンド (IPCセクション別)", "trend.html", 550),
        ("IPC階層ツリーマップ", "ipc_treemap.html", 650),
        ("特許クラスタマップ (SBERT + HDBSCAN)", "cluster_map.html", 850),
        ("キーワード共起ネットワーク", "network.html", 800),
        ("出願人ポートフォリオ", "portfolio.html", 650),
        ("頻出キーワード", "keywords.html", 700),
    ]
    sections_html = "\n".join(
        f'<div class="chart-section"><h2>{title}</h2>'
        f'<iframe src="{src}" height="{h}"></iframe></div>'
        for title, src, h in sections
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>patiroha 特許情報分析ダッシュボード</title>
<style>
  body {{ font-family: "Helvetica Neue", Arial, "Hiragino Kaku Gothic ProN", sans-serif;
         background: #f5f5f5; margin: 0; padding: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #264653; margin-bottom: 5px; }}
  .subtitle {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
  .cards {{ display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }}
  .chart-section {{ background: white; border-radius: 10px; padding: 10px 20px;
                    margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .chart-section h2 {{ margin: 10px 0 5px 0; color: #264653; }}
  iframe {{ width: 100%; border: none; overflow: hidden; }}
</style>
</head>
<body>
<div class="container">
  <h1>patiroha 特許情報分析ダッシュボード</h1>
  <div class="subtitle">patiroha v{patiroha.__version__} — {len(df)}件の特許データを SBERT + HDBSCAN で分析</div>
  <div class="cards">{cards_html}</div>
  {sections_html}
  <div style="text-align:center;color:#999;font-size:12px;padding:20px;">
    Generated by patiroha v{patiroha.__version__}
  </div>
</div>
</body>
</html>"""


def main() -> None:
    print(f"patiroha v{patiroha.__version__} — 可視化デモ (SBERT + HDBSCAN)\n")

    df = load_and_preprocess()

    figs = {
        "network": build_network_chart(df),
        "cluster_map": build_cluster_map(df),
        "trend": build_trend_chart(df),
        "ipc_treemap": build_ipc_treemap(df),
        "portfolio": build_portfolio_chart(df),
        "keywords": build_keyword_chart(df),
    }

    for name, fig in figs.items():
        fig.write_html(OUTPUT_DIR / f"{name}.html")

    dashboard_html = build_dashboard(figs, df)
    dashboard_path = OUTPUT_DIR / "dashboard.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")

    print(f"\n出力ファイル:")
    for f in sorted(OUTPUT_DIR.glob("*.html")):
        print(f"  {f}")

    # Kill previous http server if running on port 8765
    import subprocess
    subprocess.run(["lsof", "-ti", ":8765"], capture_output=True)
    subprocess.run("lsof -ti :8765 | xargs kill -9 2>/dev/null", shell=True, capture_output=True)

    import http.server
    import os
    import threading

    os.chdir(str(OUTPUT_DIR))
    server = http.server.HTTPServer(("", 8765), http.server.SimpleHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print("\nHTTPサーバー起動: http://localhost:8765/dashboard.html")
    webbrowser.open("http://localhost:8765/dashboard.html")

    input("Enterキーで終了...")
    server.shutdown()


if __name__ == "__main__":
    main()
