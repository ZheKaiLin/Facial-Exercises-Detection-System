"""
4.6.3_Facial_Yoga_前後測改善與逐回合趨勢_成對t檢定.py

對應論文 4.6.3「短期操作前後之動作幅度變化」：
  表22、前後測動作幅度改善摘要（成對樣本 t 檢定 + Cohen's d + 個體改善方向一致比例）
  表23、訓練節次內逐回合平均動作幅度（回合1/5/10 趨勢）

資料夾結構：
  父資料夾/
    受試者A/
      backtest_amplitude/*_rounds.csv （先跑 Facial_Yoga_Backtest.py 產生，含 amp_displacement）
    受試者B/ ...

執行：
  python "4.6.3_....py" --root "C:/.../總評估"
"""
import argparse
import statistics as st
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

EX = ["Face_Lift", "Double_Chin"]
ZH = {"Face_Lift": "臉部拉提訓練", "Double_Chin": "下顎線條塑型"}


def _first(globiter):
    items = sorted(globiter)
    return items[0] if items else None


def has_required(d: Path):
    return _first((d / "backtest_amplitude").glob("*_rounds.csv")) is not None


def find_participants(root: Path):
    subs = [d for d in sorted(root.iterdir())
            if d.is_dir() and d.name != "backtest_amplitude"]
    parts = [d for d in subs if has_required(d)]
    if not parts and has_required(root):
        parts = [root]
    return parts


def load_amp(d: Path):
    """回傳 dict：amp[(session,exercise)] = 依 round 排序之峰值幅度 list。"""
    rounds = pd.read_csv(_first((d / "backtest_amplitude").glob("*_rounds.csv")), encoding="utf-8-sig")
    amp = {}
    for _, r in rounds.iterrows():
        key = (r["session"], r["exercise"])
        amp.setdefault(key, []).append((int(r["round"]), float(r["amp_displacement"])))
    for k in amp:
        amp[k] = [v for _, v in sorted(amp[k])]
    return amp


def fmt(x, n=5):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent / "原始資料"),
                    help="所有受試者的父資料夾（預設：本腳本所在資料夾下的「原始資料」）")
    opt = ap.parse_args()
    root = Path(opt.root)
    if not root.exists():
        raise SystemExit(f"找不到資料夾：{root}")

    dirs = find_participants(root)
    if not dirs:
        raise SystemExit("找不到任何有效受試者資料夾（需含 backtest_amplitude/*_rounds.csv）")
    parts = [{"label": d.name, "amp": load_amp(d)} for d in dirs]
    N = len(parts)
    print(f"納入受試者 N = {N}：{', '.join(p['label'] for p in parts)}\n")

    # ============ 表22：前後測改善（跨受試者） ============
    print("## 表22、前後測動作幅度改善摘要\n")
    print("| 訓練項目 | 前測幅度 (SD) | 後測幅度 (SD) | ΔAmplitude | 改善率 | t 值 | p 值 | Cohen's d | 個體改善方向一致比例 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ex in EX:
        pre_m, post_m = [], []
        for p in parts:
            pre = p["amp"].get(("pre_test", ex), []); post = p["amp"].get(("post_test", ex), [])
            if pre and post:
                pre_m.append(st.mean(pre)); post_m.append(st.mean(post))
        if not pre_m:
            print(f"| {ZH[ex]} | — | — | — | — | — | — | — | — |"); continue
        pre_a, post_a = np.array(pre_m), np.array(post_m)
        diff = post_a - pre_a
        gpre, gpost = pre_a.mean(), post_a.mean()
        dmean = diff.mean()
        imp = dmean / gpre * 100 if gpre else float("nan")
        pre_sd = pre_a.std(ddof=1) if len(pre_a) > 1 else 0.0
        post_sd = post_a.std(ddof=1) if len(post_a) > 1 else 0.0
        n_up = int(np.sum(diff > 0))
        consist = f"{n_up}/{len(diff)} ({n_up/len(diff)*100:.1f}%)"
        if len(diff) >= 2 and diff.std(ddof=1) > 0:
            t, pp = stats.ttest_rel(post_a, pre_a)
            d = dmean / diff.std(ddof=1)
            tcell, pcell, dcell = f"{t:.3f}", f"{pp:.3f}", f"{d:.3f}"
        else:
            tcell = pcell = dcell = "(需 N≥2)"
        print(f"| {ZH[ex]} | {gpre:.5f} ({pre_sd:.5f}) | {gpost:.5f} ({post_sd:.5f}) | "
              f"{dmean:+.5f} | {imp:+.1f}% | {tcell} | {pcell} | {dcell} | {consist} |")
    print(f"\n（n(前後測皆有效)依動作而定；改善率 = 平均ΔAmplitude ÷ 前測平均 ×100%；"
          f"個體改善方向一致比例 = 該動作後測幅度大於前測幅度之受試者人數比例）\n")

    # ============ 表23：訓練節次內逐回合（跨受試者平均） ============
    print("## 表23、訓練節次內逐回合平均動作幅度\n")
    print("| 訓練項目 | 回合 1 | 回合 5 | 回合 10 | 回合間變化趨勢 |")
    print("|---|---|---|---|---|")
    for ex in EX:
        cols = {0: [], 4: [], 9: []}
        for p in parts:
            a = p["amp"].get(("training", ex), [])
            for idx in cols:
                if len(a) > idx:
                    cols[idx].append(a[idx])
        if not cols[0]:
            print(f"| {ZH[ex]} | — | — | — | — |"); continue
        r1, r5, r10 = (st.mean(cols[0]), st.mean(cols[4]) if cols[4] else float("nan"),
                       st.mean(cols[9]) if cols[9] else float("nan"))
        trend = "上升" if r10 > r1 * 1.05 else ("下降" if r10 < r1 * 0.95 else "大致持平")
        print(f"| {ZH[ex]} | {r1:.5f} | {fmt(r5)} | {fmt(r10)} | {trend} |")
    print()


if __name__ == "__main__":
    main()
