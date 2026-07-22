"""
4.6.4附圖_固定門檻誤判示意圖.py

對應論文 4.6.4「個人化校準機制之必要性」表24附加的視覺化版本：
以圖示說明「通用固定門檻」如何造成誤判（尤其是放鬆狀態即被誤判為「假達標」），
而「個人化門檻」因為隨每個人自己的基準值調整，不會出現同樣的系統性誤判。

資料來源與判定邏輯與 4.6.4_Facial_Yoga_個人化校準必要性_CV_WilsonCI.py 完全一致
（本檔直接重用其讀取／判定函式，不重新定義規則，避免兩份程式數字不一致）。

輸出：fig_table24_gap.png（放進論文/投影片 4.6.4 節或投影片46）
"""
import argparse
import importlib.util
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
SRC_MODULE_PATH = HERE / "4.6.4_Facial_Yoga_個人化校準必要性_CV_WilsonCI.py"

spec = importlib.util.spec_from_file_location("calib_mod", SRC_MODULE_PATH)
calib_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib_mod)

EX = calib_mod.EX
ZH = calib_mod.ZH

# --- 色票（依 dataviz skill 之 status palette / chart chrome） ---
C_TEXT_PRIMARY = "#0b0b0b"
C_TEXT_SECONDARY = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_BASELINE_AXIS = "#c3c2b7"
C_SURFACE = "#fcfcfb"

C_SERIES_BASELINE = "#2a78d6"   # 個人基準值（categorical slot 1: blue）
C_SERIES_PERSONAL_THR = "#1baf7a"  # 個人化門檻（categorical slot 2: aqua/green）
C_CRITICAL = "#d03b3b"          # 假達標（status: critical）
C_FIXED_LINE = "#52514e"        # 固定門檻線（secondary ink，非資料色，避免與序列色混淆）


def build_rows(ex, parts):
    thrs = [p["thr"][ex] for p in parts if ex in p["thr"]]
    Tg = st.median(thrs)
    rows = []
    for p in parts:
        if ex not in p["base"] or ex not in p["thr"]:
            continue
        b, t = p["base"][ex], p["thr"][ex]
        if ex == "Double_Chin":
            misjudged = b > Tg
        else:
            misjudged = b < Tg
        rows.append({"label": p["label"], "base": b, "thr": t, "misjudged": misjudged})
    rows.sort(key=lambda r: r["base"])
    for i, r in enumerate(rows, start=1):
        r["anon"] = f"P{i:02d}"
    return rows, Tg


def plot_panel(ax, ex, rows, Tg):
    n = len(rows)
    ys = list(range(n))

    # 固定門檻參考線（先畫，壓在最底層）
    ax.axvline(Tg, color=C_FIXED_LINE, linewidth=2, linestyle=(0, (5, 3)), zorder=1)

    # 每位受試者：基準值 -> 個人化門檻 的細連接線（顯示門檻隨基準值移動）
    for y, r in zip(ys, rows):
        ax.plot([r["base"], r["thr"]], [y, y], color=C_GRID, linewidth=2, zorder=2,
                solid_capstyle="round")

    # 個人基準值 dots（誤判者標 critical 紅色，其餘藍色）
    base_x = [r["base"] for r in rows]
    base_colors = [C_CRITICAL if r["misjudged"] else C_SERIES_BASELINE for r in rows]
    ax.scatter(base_x, ys, s=90, c=base_colors, zorder=4,
               edgecolors=C_SURFACE, linewidths=2)

    # 個人化門檻 dots
    thr_x = [r["thr"] for r in rows]
    ax.scatter(thr_x, ys, s=70, facecolors=C_SURFACE, edgecolors=C_SERIES_PERSONAL_THR,
               linewidths=2, zorder=5, marker="D")

    ax.set_yticks(ys)
    ax.set_yticklabels([r["anon"] for r in rows], fontsize=9, color=C_TEXT_SECONDARY)
    ax.set_ylim(-1, n)
    ax.invert_yaxis()

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(C_BASELINE_AXIS)
    ax.tick_params(axis="x", colors=C_TEXT_SECONDARY, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=C_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    n_mis = sum(r["misjudged"] for r in rows)
    ax.set_title(
        f"{ZH[ex]}　固定門檻下 {n_mis}/{n} 人放鬆即被誤判「假達標」",
        fontsize=12, color=C_TEXT_PRIMARY, pad=12, loc="left"
    )

    # 固定門檻線標籤
    ax.text(Tg, -0.6, "通用固定門檻", color=C_TEXT_SECONDARY, fontsize=8.5,
            ha="center", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / "原始資料"))
    ap.add_argument("--out", default=str(HERE / "fig_table24_gap.png"))
    opt = ap.parse_args()
    root = Path(opt.root)

    dirs = calib_mod.find_participants(root)
    parts = []
    for d in dirs:
        base, thr = calib_mod.load_calib(d)
        parts.append({"label": d.name, "base": base, "thr": thr})

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), facecolor=C_SURFACE)

    for ax, ex in zip(axes, EX):
        rows, Tg = build_rows(ex, parts)
        ax.set_facecolor(C_SURFACE)
        plot_panel(ax, ex, rows, Tg)
        ax.set_xlabel(
            "嘴角上提幅度基準值 C0（越小＝提得越高）" if ex == "Face_Lift"
            else "嘴唇開合幅度基準值 D0（越大＝張得越開）",
            fontsize=9.5, color=C_TEXT_SECONDARY
        )
        ax.margins(x=0.10)

    # --- 圖例（自建，因為用的是 dumbbell + 兩種 marker，不是單純色票） ---
    legend_handles = [
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=8,
                      markerfacecolor=C_SERIES_BASELINE, markeredgecolor=C_SURFACE,
                      markeredgewidth=1.5, label="個人放鬆基準值"),
        mlines.Line2D([], [], marker="o", linestyle="None", markersize=8,
                      markerfacecolor=C_CRITICAL, markeredgecolor=C_SURFACE,
                      markeredgewidth=1.5, label="個人放鬆基準值（固定門檻下被誤判為假達標）"),
        mlines.Line2D([], [], marker="D", linestyle="None", markersize=7,
                      markerfacecolor=C_SURFACE, markeredgecolor=C_SERIES_PERSONAL_THR,
                      markeredgewidth=1.8, label="個人化門檻（隨基準值調整）"),
        mlines.Line2D([], [], color=C_FIXED_LINE, linewidth=2, linestyle=(0, (5, 3)),
                      label="通用固定門檻（同一數值套用全體）"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
              fontsize=9.5, labelcolor=C_TEXT_SECONDARY, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "固定門檻 vs. 個人化門檻：個體差異下的誤判情形（N=15）",
        fontsize=14, color=C_TEXT_PRIMARY, y=1.02, x=0.02, ha="left", fontweight="bold"
    )

    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    fig.savefig(opt.out, dpi=200, facecolor=C_SURFACE, bbox_inches="tight")
    print(f"已輸出：{opt.out}")


if __name__ == "__main__":
    main()
