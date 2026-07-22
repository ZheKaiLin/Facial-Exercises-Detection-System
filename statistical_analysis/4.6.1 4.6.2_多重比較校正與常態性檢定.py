"""
4.6.1附_多重比較校正與常態性檢定.py

對應論文 4.6.1／4.6.2 節「多重比較校正」與「成對t檢定常態性/Cohen's dz」補充分析。
回應審查意見：
    「rmcorr p=.040效果極弱，且同時進行多種動作、多種相關與前後測檢定，
    未處理多重比較。成對t檢定亦未報告差值常態性及 Cohen's dz計算方式。」

本腳本直接匯入 4.6.1 與 4.6.2 既有腳本之資料載入函式，避免重複實作、
確保與正文表 23／表 24 使用完全相同之原始資料，計算：
    1) 4 項主要推論檢定（2 個 rmcorr + 2 個成對樣本 t 檢定）之 Holm-Bonferroni 校正後 p 值
    2) 成對樣本 t 檢定所用差異值之 Shapiro-Wilk 常態性檢定
    3) Cohen's dz 計算方式說明（dz = 平均差異 ÷ 差異值標準差，已用於 4.6.2 腳本）

執行：
    python "4.6.1附_多重比較校正與常態性檢定.py" --root "C:/.../原始資料"
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent


def _load_module(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_rmcorr_pvalues(root: Path):
    """重現 4.6.1 腳本之 rmcorr 計算，回傳 {exercise: (r, p)}"""
    mod = _load_module("4.6.1_Facial_Yoga_Score與動作幅度關係_Pearson_Spearman_RMCorr.py")
    parts = mod.find_participants(root)
    out = {}
    for ex in mod.EX:
        rows = []
        for d in parts:
            for score, amp in mod.load_pairs(d, ex):
                rows.append((d.name, score, amp))
        df = pd.DataFrame(rows, columns=["subject", "score", "amp"])
        rm = pg.rm_corr(data=df, x="score", y="amp", subject="subject")
        out[ex] = (float(rm["r"].iloc[0]), float(rm["pval"].iloc[0]))
    return out, mod.ZH


def compute_paired_t(root: Path):
    """重現 4.6.2 腳本之成對t檢定計算，回傳 {exercise: (t, p, diff_array)}"""
    mod = _load_module("4.6.2_Facial_Yoga_前後測改善與逐回合趨勢_成對t檢定.py")
    dirs = mod.find_participants(root)
    parts = [{"label": d.name, "amp": mod.load_amp(d)} for d in dirs]
    out = {}
    for ex in mod.EX:
        pre_m, post_m = [], []
        import statistics as st
        for p in parts:
            pre = p["amp"].get(("pre_test", ex), [])
            post = p["amp"].get(("post_test", ex), [])
            if pre and post:
                pre_m.append(st.mean(pre))
                post_m.append(st.mean(post))
        pre_a, post_a = np.array(pre_m), np.array(post_m)
        diff = post_a - pre_a
        t, p_val = stats.ttest_rel(post_a, pre_a)
        out[ex] = (float(t), float(p_val), diff)
    return out, mod.ZH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / "原始資料"),
                     help="所有受試者的父資料夾（預設：本腳本所在資料夾下的「原始資料」）")
    opt = ap.parse_args()
    root = Path(opt.root)
    if not root.exists():
        raise SystemExit(f"找不到資料夾：{root}")

    rmcorr_results, zh1 = compute_rmcorr_pvalues(root)
    ttest_results, zh2 = compute_paired_t(root)

    # ---------- 1) Holm-Bonferroni 多重比較校正（4 項主要推論檢定） ----------
    labels = []
    pvals = []
    for ex in rmcorr_results:
        labels.append(f"{zh1[ex]} rmcorr")
        pvals.append(rmcorr_results[ex][1])
    for ex in ttest_results:
        labels.append(f"{zh2[ex]} 成對t檢定")
        pvals.append(ttest_results[ex][1])

    reject, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")

    print("## 多重比較校正（Holm-Bonferroni，4項主要推論檢定家族）\n")
    print("| 檢定 | 原始 r/t | 原始 p 值 | Holm校正後 p 值 | 校正後是否顯著 |")
    print("|---|---|---|---|---|")
    idx = 0
    for ex in rmcorr_results:
        r, p = rmcorr_results[ex]
        print(f"| {zh1[ex]} rmcorr | r={r:.3f} | {p:.3f} | {p_holm[idx]:.3f} | "
              f"{'是' if reject[idx] else '否'} |")
        idx += 1
    for ex in ttest_results:
        t, p, _ = ttest_results[ex]
        print(f"| {zh2[ex]} 成對t檢定 | t={t:.3f} | {p:.3f} | {p_holm[idx]:.3f} | "
              f"{'是' if reject[idx] else '否'} |")
        idx += 1

    # ---------- 2) 成對t檢定差異值之 Shapiro-Wilk 常態性檢定 ----------
    print("\n## 成對t檢定差異值之常態性檢定（Shapiro-Wilk）\n")
    print("| 訓練項目 | n | W 統計量 | p 值 | 結論 |")
    print("|---|---|---|---|---|")
    for ex, (t, p, diff) in ttest_results.items():
        w, p_norm = stats.shapiro(diff)
        concl = "符合常態（不拒絕 H0）" if p_norm >= 0.05 else "偏離常態"
        print(f"| {zh2[ex]} | {len(diff)} | {w:.4f} | {p_norm:.4f} | {concl} |")

    # ---------- 3) Cohen's dz 計算方式 ----------
    print("\n## Cohen's dz 計算方式\n")
    print("dz = 平均差異（後測 − 前測） ÷ 差異值之標準差（ddof=1），")
    print("為相依樣本（成對）設計慣用之效果量公式，已用於 4.6.2 腳本之 Cohen's d 欄位。\n")
    for ex, (t, p, diff) in ttest_results.items():
        dz = float(np.mean(diff) / np.std(diff, ddof=1))
        print(f"  {zh2[ex]}：dz = {dz:.3f}")


if __name__ == "__main__":
    main()
