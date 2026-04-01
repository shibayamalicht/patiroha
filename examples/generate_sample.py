"""Generate a 500-row synthetic patent dataset for testing patiroha."""

import csv
import random
from pathlib import Path

random.seed(42)

# Technology domains with associated terms
DOMAINS = {
    "CNF": {
        "titles": [
            "セルロースナノファイバー強化{matrix}",
            "CNF含有{matrix}の製造方法",
            "ナノセルロース複合{matrix}",
            "セルロースナノファイバー{product}",
            "CNF分散{matrix}の調製方法",
            "修飾セルロースナノファイバーを用いた{product}",
        ],
        "matrix": ["樹脂組成物", "ポリマー複合材料", "フィルム", "コーティング材", "接着剤組成物"],
        "product": ["透明フィルム", "バリアフィルム", "構造部材", "断熱材", "増粘剤"],
        "abstracts": [
            "セルロースナノファイバーを{matrix}中に均一分散させることにより、機械的強度と透明性を両立する複合材料を提供する。{detail}",
            "化学修飾したセルロースナノファイバーを含有する{matrix}に関する。ナノファイバーの表面処理により{matrix}との親和性を向上させ、{benefit}を実現する。",
            "水系分散したセルロースナノファイバーを用いた{product}の製造方法であって、乾燥工程における収縮を制御することにより均質な{product}を得る。",
        ],
        "detail": [
            "引張強度が従来比50%向上する", "熱膨張係数がガラスと同等まで低減される",
            "酸素透過度が10cc/m2/day以下となる", "ヘイズ値が1%以下の高透明性を示す",
        ],
        "benefit": ["高強度化", "軽量化", "高バリア性", "寸法安定性の向上"],
        "ipcs": ["C08L 1/02", "C08J 5/04", "B32B 23/00", "C08J 5/18", "D21H 11/18", "B29C 70/00"],
        "applicants": [
            "王子ホールディングス株式会社", "日本製紙株式会社", "大王製紙株式会社", "東レ株式会社",
            "大日本印刷株式会社", "凸版印刷株式会社", "旭化成株式会社", "花王株式会社",
        ],
    },
    "Battery": {
        "titles": [
            "リチウムイオン電池用{component}",
            "{component}を備えた二次電池",
            "全固体{type}電池",
            "{type}電池の製造方法",
            "電池用{component}の改良",
            "高エネルギー密度{type}電池",
        ],
        "component": ["正極材料", "負極材料", "電解質", "セパレータ", "集電体", "固体電解質"],
        "type": ["リチウムイオン", "リチウム硫黄", "ナトリウムイオン", "全固体"],
        "abstracts": [
            "リチウムイオン二次電池の{component}に関する。{detail}により、高容量かつ長サイクル寿命を実現する。",
            "{type}電池における{component}の新規構成に関する。界面抵抗を低減し、{benefit}を達成する。",
            "二次電池用{component}の製造方法であって、{process}により均質な{component}を得る方法を提供する。",
        ],
        "detail": [
            "ニッケルリッチ層状酸化物の表面コーティング", "シリコン系負極の膨張抑制構造",
            "硫化物系固体電解質の粒径制御", "ドライプロセスによる電極形成",
        ],
        "benefit": ["充放電効率99.5%以上", "エネルギー密度400Wh/kg超", "高速充電対応", "低温特性の改善"],
        "process": ["共沈法", "固相反応", "ゾルゲル法", "メカニカルミリング"],
        "ipcs": ["H01M 4/525", "H01M 10/052", "H01M 10/0562", "H01M 50/40", "C01G 53/00"],
        "applicants": [
            "パナソニック株式会社", "トヨタ自動車株式会社", "出光興産株式会社", "TDK株式会社",
            "村田製作所", "Samsung SDI", "LG Energy Solution", "CATL",
        ],
    },
    "Display": {
        "titles": [
            "有機EL{device}",
            "{device}の製造方法",
            "量子ドット{device}",
            "マイクロLED{device}",
            "フレキシブル{device}",
            "高解像度{device}の駆動回路",
        ],
        "device": ["表示装置", "発光素子", "ディスプレイ", "表示パネル", "発光デバイス"],
        "abstracts": [
            "有機エレクトロルミネッセンス{device}に関する。{detail}により、高輝度かつ長寿命の{device}を実現する。",
            "量子ドットを用いた{device}であって、{detail}により色再現性に優れた表示を可能にする。",
            "マイクロLEDを画素として用いた{device}の製造方法に関する。{process}により高精細な{device}を得る。",
        ],
        "detail": [
            "発光層の膜厚最適化", "電荷輸送層のバンド構造制御",
            "量子ドットのサイズ均一性向上", "インクジェット印刷による精密パターニング",
        ],
        "process": ["蒸着法", "塗布法", "転写法", "レーザーリフトオフ"],
        "ipcs": ["H10K 50/10", "H10K 71/00", "H01L 33/00", "G09G 3/32", "C09K 11/02"],
        "applicants": [
            "ソニー株式会社", "LG Display", "サムスン電子", "シャープ株式会社",
            "JDI", "BOE Technology", "京セラ株式会社", "住友化学株式会社",
        ],
    },
    "Autonomous": {
        "titles": [
            "自動運転{system}",
            "車両の{function}方法",
            "運転支援{system}",
            "{sensor}を用いた物体認識装置",
            "自動運転車両の{function}装置",
            "ロボットの{function}システム",
        ],
        "system": ["制御システム", "センシングシステム", "通信システム", "安全システム"],
        "function": ["経路計画", "障害物検出", "車線維持", "歩行者予測", "速度制御", "合流制御"],
        "sensor": ["LiDAR", "ミリ波レーダー", "ステレオカメラ", "超音波センサ"],
        "abstracts": [
            "自動運転レベル{level}に対応する{system}に関する。{sensor}データを{method}で処理し、リアルタイムの{function}を実現する。",
            "車両の{function}装置であって、{method}を用いて周囲環境を認識し、安全な走行を支援する。",
            "{sensor}とカメラの融合データに基づく{function}方法に関する。{detail}により認識精度を向上させる。",
        ],
        "level": ["3", "4", "5"],
        "method": ["ディープラーニング", "トランスフォーマーモデル", "強化学習", "カルマンフィルタ"],
        "detail": [
            "注意機構による特徴量統合", "点群データの意味分割",
            "時系列予測による行動推定", "マルチモーダル融合",
        ],
        "ipcs": ["G05D 1/02", "G06V 20/58", "G08G 1/16", "B60W 60/00", "G06N 3/08"],
        "applicants": [
            "トヨタ自動車株式会社", "日産自動車株式会社", "ホンダ技研工業株式会社", "Waymo",
            "デンソー株式会社", "日立Astemo", "Mobileye", "NVIDIA",
        ],
    },
    "Solar": {
        "titles": [
            "ペロブスカイト{cell}",
            "{cell}モジュール",
            "タンデム型{cell}",
            "{cell}の製造方法",
            "高効率{cell}",
            "フレキシブル{cell}",
        ],
        "cell": ["太陽電池", "光電変換素子", "太陽電池セル"],
        "abstracts": [
            "ペロブスカイト{cell}に関する。{detail}により光電変換効率を向上させ、低コストでの製造を可能にする。",
            "シリコン太陽電池とペロブスカイト太陽電池のタンデム構造に関する。中間層の最適化により{benefit}を達成する。",
            "{cell}の製造方法であって、{process}を用いて大面積化と高効率化の両立を実現する。",
        ],
        "detail": [
            "組成勾配構造の導入", "パッシベーション層の形成",
            "ペロブスカイト結晶の配向制御", "界面欠陥の低減",
        ],
        "benefit": ["変換効率30%超", "長期安定性の向上", "大面積モジュール化", "低照度での発電効率改善"],
        "process": ["スロットダイコーティング", "ブレードコーティング", "スプレー成膜", "蒸着法"],
        "ipcs": ["H01L 31/0725", "H10K 30/50", "H10K 30/30", "H01L 31/18"],
        "applicants": [
            "パナソニック株式会社", "東京大学", "京都大学", "理化学研究所",
            "積水化学工業株式会社", "カネカ", "Oxford PV", "First Solar",
        ],
    },
    "AI_Medical": {
        "titles": [
            "深層学習を用いた{target}検出方法",
            "医用画像の{task}システム",
            "AI支援{task}装置",
            "機械学習による{target}予測方法",
            "{modality}画像の自動{task}方法",
            "医療診断支援{system}",
        ],
        "target": ["肺結節", "腫瘍", "骨折", "網膜病変", "心電図異常"],
        "task": ["自動解析", "診断支援", "セグメンテーション", "分類", "予測"],
        "modality": ["CT", "MRI", "X線", "超音波", "内視鏡"],
        "system": ["システム", "装置", "プログラム"],
        "abstracts": [
            "医用画像の{task}に関する。畳み込みニューラルネットワークを用いて{modality}画像から{target}を高精度で検出する。{detail}",
            "深層学習モデルによる{target}の{task}方法であって、{detail}少量の教師データでも高い診断精度を達成する。",
            "{modality}画像における{target}の自動{task}システムに関する。{architecture}を採用し、リアルタイム処理を実現する。",
        ],
        "detail": [
            "転移学習により", "データ拡張と半教師あり学習を組み合わせることで",
            "マルチスケール特徴抽出により", "Attention機構の導入により",
        ],
        "architecture": ["U-Net", "ResNet", "Vision Transformer", "YOLO"],
        "ipcs": ["G06T 7/00", "A61B 6/03", "G16H 30/40", "G06N 3/08", "A61B 5/00"],
        "applicants": [
            "キヤノン株式会社", "富士フイルム株式会社", "オリンパス株式会社", "GEヘルスケア",
            "シーメンス", "東京大学", "国立がん研究センター", "NVIDIA",
        ],
    },
    "Recycling": {
        "titles": [
            "リサイクル可能な{material}",
            "{material}のケミカルリサイクル方法",
            "解重合可能な{material}の合成",
            "廃{material}の再資源化方法",
            "循環型{material}",
            "バイオマス由来{material}",
        ],
        "material": ["高分子材料", "プラスチック", "ポリエステル", "ポリアミド", "エポキシ樹脂"],
        "abstracts": [
            "解重合可能な{material}に関する。分子鎖中に{feature}を導入することで、温和な条件下でモノマーへの分解が可能な循環型{material}を実現する。",
            "廃{material}のケミカルリサイクル方法であって、{catalyst}を用いた{process}により高純度のモノマーを回収する。",
            "バイオマス原料から得られる{material}の合成方法に関する。{detail}により環境負荷の低い材料を提供する。",
        ],
        "feature": ["可逆結合部位", "刺激応答性結合", "動的共有結合", "エステル交換部位"],
        "catalyst": ["金属触媒", "酵素", "有機触媒", "イオン液体"],
        "process": ["加水分解", "メタノリシス", "グリコリシス", "酵素分解"],
        "detail": [
            "発酵プロセスの最適化", "バイオマスの前処理技術の改良",
            "触媒反応の選択性向上", "プロセスのエネルギー効率改善",
        ],
        "ipcs": ["C08J 11/10", "C08G 63/183", "C12P 7/62", "C08L 67/02", "B29B 17/00"],
        "applicants": [
            "三菱ケミカル株式会社", "東洋紡株式会社", "帝人株式会社", "BASF",
            "旭化成株式会社", "住友化学株式会社", "クラレ", "ダイセル",
        ],
    },
}

YEARS = list(range(2019, 2025))


def fill_template(template: str, domain: dict) -> str:
    """Fill a template string with random values from the domain dict."""
    result = template
    # Keep replacing until no more placeholders
    for _ in range(5):
        changed = False
        for key, values in domain.items():
            if isinstance(values, list) and isinstance(values[0], str):
                placeholder = "{" + key + "}"
                if placeholder in result:
                    result = result.replace(placeholder, random.choice(values), 1)
                    changed = True
        if not changed:
            break
    return result


def generate_row(domain_name: str, domain: dict) -> dict:
    title = fill_template(random.choice(domain["titles"]), domain)
    abstract = fill_template(random.choice(domain["abstracts"]), domain)
    applicants = random.sample(domain["applicants"], k=random.randint(1, 2))
    ipcs = random.sample(domain["ipcs"], k=random.randint(1, 3))
    year = random.choice(YEARS)
    month = random.randint(1, 12)
    day = random.randint(1, 28)

    return {
        "title": title,
        "abstract": abstract,
        "applicant": ";".join(applicants),
        "ipc": "; ".join(ipcs),
        "date": f"{year}-{month:02d}-{day:02d}",
    }


def main():
    output_path = Path(__file__).parent / "sample_patents_500.csv"
    rows = []

    # Distribute ~70 per domain (7 domains × ~71 = 497, add 3 extra)
    domain_names = list(DOMAINS.keys())
    for domain_name in domain_names:
        domain = DOMAINS[domain_name]
        for _ in range(71):
            rows.append(generate_row(domain_name, domain))

    # Add 3 more to reach 500
    for _ in range(3):
        dn = random.choice(domain_names)
        rows.append(generate_row(dn, DOMAINS[dn]))

    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "abstract", "applicant", "ipc", "date"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
