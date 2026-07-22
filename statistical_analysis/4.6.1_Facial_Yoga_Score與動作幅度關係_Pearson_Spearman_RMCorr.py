"""
4.6.2_Facial_Yoga_Score與動作幅度關係_Pearson_Spearman_RMCorr.py
對應論文 4.6.2「動作品質分數用於評分與動作幅度關係之分析」表21，三層分析：
  (1) Pooled（N=全部回合數，忽略受試者叢集結構，僅供探索性參考）
  (2) 受試者平均（每位受試者取其訓練回合之 Score / 動作幅度平均，N=受試者數）
  (3) rmcorr（重複量測相關，Bakdash & Marusich, 2017；採 pingouin.rm_corr 實作，
      以受試者為分群變數控制受試者間變異，檢驗受試者內之共同關聯趨勢）

資料夾結構與 Facial_Yoga_MakeTables.py 相同：
  父資料夾/
    受試者A/
      *_training_*_yoga.csv         （系統原始 Score，含 training session）
      backtest_amplitude/*_rounds.csv （先跑 Facial_Yoga_Backtest.py 產生，含 amp_mean_sustained）
    受試者B/ ...

執行：
  python Facial_Yoga_Table21_RMCorr.py --root "C:/.../總評估"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

EX = ["Face_Lift", "Double_Chin"]
ZH = {"Face_Lift": "臉部拉提訓練", "Double_Chin": "下顎線條塑型"}


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def _first(globiter):
    items = sorted(globiter)
    return items[0] if items else None


def has_required(d: Path):
    yoga = _first(d.glob("*_training_*_yoga.csv"))
    rounds = _first((d / "backtest_amplitude").glob("*_rounds.csv"))
    return yoga is not None and rounds is not None


def find_participants(root: Path):
    subs = [d for d in sorted(root.iterdir())
            if d.is_dir() and d.name != "backtest_amplitude"]
    parts = [d for d in subs if has_required(d)]
    if not parts and has_required(root):
        parts = [root]
    return parts


def load_pairs(d: Path, ex: str):
    """回傳該受試者、該訓練項目、training session 逐回合的 (round, score, amp_mean) 對齊資料。"""
    rounds_df = read_csv(_first((d / "backtest_amplitude").glob("*_rounds.csv")))
    yoga_df = read_csv(_first(d.glob("*_training_*_yoga.csv")))

    r = rounds_df[(rounds_df["session"] == "training") & (rounds_df["exercise"] == ex)]
    r = r.sort_values("round")
    y = yoga_df[(yoga_df["session"] == "training") & (yoga_df["exercise"] == ex)]
    y = y.sort_values("round")

    n = min(len(r), len(y))
    if n == 0:
        return []
    amp = r["amp_mean_sustained"].to_numpy(dtype=float)[:n]
    score = y["score"].to_numpy(dtype=float)[:n]
    mask = np.isfinite(amp) & np.isfinite(score)
    return list(zip(score[mask], amp[mask]))


def corr_ci(coef, n, kind="pearson"):
    if n is None or n < 5 or coef is None or not np.isfinite(coef) or abs(coef) >= 1:
        return float("nan"), float("nan")
    z = np.arctanh(coef)
    se = (1.03 if kind == "spearman" else 1.0) / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def sig(p):
    return "*" if p < .05 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent / "原始資料"),
                    help="所有受試者的父資料夾（預設：本腳本所在資料夾下的「原始資料」）")
    opt = ap.parse_args()
    root = Path(opt.root)
    if not root.exists():
        raise SystemExit(f"找不到資料夾：{root}")

    parts = find_participants(root)
    if not parts:
        raise SystemExit("找不到任何有效受試者資料夾")
    print(f"納入受試者 N = {len(parts)}：{', '.join(p.name for p in parts)}\n")

    print("## 表21、系統評分與動作幅度之關係分析（重新計算）\n")
    print("| 訓練項目 | 方法 | r / ρ | p 值 | 備註 |")
    print("|---|---|---|---|---|")

    for ex in EX:
        rows = []  # (subject_label, score, amp)
        for d in parts:
            for score, amp in load_pairs(d, ex):
                rows.append((d.name, score, amp))
        df = pd.DataFrame(rows, columns=["subject", "score", "amp"])
        n_pooled = len(df)

        # (1) Pooled
        r_p, p_p = stats.pearsonr(df["score"], df["amp"])
        rho_p, prho_p = stats.spearmanr(df["score"], df["amp"])

        # (2) 受試者平均
        subj_mean = df.groupby("subject")[["score", "amp"]].mean()
        n_subj = len(subj_mean)
        r_s, p_s = stats.pearsonr(subj_mean["score"], subj_mean["amp"])
        rho_s, prho_s = stats.spearmanr(subj_mean["score"], subj_mean["amp"])

        # (3) rmcorr（Bakdash & Marusich, 2017 / pingouin 實作）
        rm = pg.rm_corr(data=df, x="score", y="amp", subject="subject")
        r_rm = float(rm["r"].iloc[0])
        p_rm = float(rm["pval"].iloc[0])
        dof_rm = int(rm["dof"].iloc[0])
        ci_lo, ci_hi = corr_ci(r_rm, dof_rm + 2, "pearson")

        print(f"| {ZH[ex]} | Pooled (N={n_pooled}，探索性) | "
              f"Pearson r={r_p:.3f} / Spearman ρ={rho_p:.3f} | "
              f"{p_p:.3f}{sig(p_p)} / {prho_p:.3f}{sig(prho_p)} | 忽略重複量測結構 |")
        print(f"| | 受試者平均 (N={n_subj}) | "
              f"Pearson r={r_s:.3f} / Spearman ρ={rho_s:.3f} | "
              f"{p_s:.3f}{sig(p_s)} / {prho_s:.3f}{sig(prho_s)} | "
              f"{'顯著' if min(p_s,prho_s)<.05 else '不顯著'} |")
        print(f"| | **rmcorr（主要依據，dof={dof_rm}）** | **r={r_rm:.3f}** | **{p_rm:.3f}{sig(p_rm)}** | "
              f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]，{'顯著' if p_rm<.05 else '不顯著'} |")
    print()


if __name__ == "__main__":
    main()
