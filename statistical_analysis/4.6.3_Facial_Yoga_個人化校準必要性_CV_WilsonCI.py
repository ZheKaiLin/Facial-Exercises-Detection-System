"""
4.6.4_Facial_Yoga_個人化校準必要性_CV_WilsonCI.py

對應論文 4.6.4「個人化校準機制之必要性」：
  表24、個人化校準基準值之個體差異分析（平均數/標準差/全距/變異係數 CV）
  表24附加、固定門檻與個人化門檻之達標判定差異（觀察比例 + Wilson score 95% CI）

不對誤判比例進行對某個「可接受誤判率上限」之假設檢定——此類 H0 數值
（無論是外部借用之慣例值，或綁定系統特定參數）皆需額外假設支撐，且觀察
比例（53.3%、80.0%）與常見誤判率相距懸殊，顯著性幾乎必然成立，對解讀
助益有限。改以 Wilson score interval 呈現觀察比例之 95% CI，僅作描述性
推論（該比例之估計不確定範圍），不涉及是否顯著偏離任何門檻值。

資料夾結構：
  父資料夾/
    受試者A/
      *_training_*_calibration.csv （C0/D0 與門檻）
    受試者B/ ...

執行：
  python "4.6.4_....py" --root "C:/.../總評估"
"""
import argparse
import statistics as st
from pathlib import Path
import numpy as np
import pandas as pd

EX = ["Face_Lift", "Double_Chin"]
ZH = {"Face_Lift": "臉部拉提訓練", "Double_Chin": "下顎線條塑型"}
NAME = {"Face_Lift": "嘴角偏移基準值 C₀", "Double_Chin": "上下唇距離基準值 D₀"}


def _first(globiter):
    items = sorted(globiter)
    return items[0] if items else None


def has_required(d: Path):
    return _first(d.glob("*_training_*_calibration.csv")) is not None


def find_participants(root: Path):
    subs = [d for d in sorted(root.iterdir())
            if d.is_dir() and d.name != "backtest_amplitude"]
    parts = [d for d in subs if has_required(d)]
    if not parts and has_required(root):
        parts = [root]
    return parts


def load_calib(d: Path):
    calib = pd.read_csv(_first(d.glob("*_training_*_calibration.csv")), encoding="utf-8-sig")
    base = {r["exercise"]: float(r["baseline"]) for _, r in calib.iterrows()}
    thr = {r["exercise"]: float(r["threshold"]) for _, r in calib.iterrows()}
    return base, thr


def fmt(x, n=5):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def wilson_ci(x, n, z=1.96):
    """Wilson score interval，小樣本/極端比例下優於常態近似之 Wald CI。"""
    if n == 0:
        return float("nan"), float("nan")
    phat = x / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    adj = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - adj), min(1.0, center + adj)


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
        raise SystemExit("找不到任何有效受試者資料夾（需含 *_training_*_calibration.csv）")
    parts = []
    for d in dirs:
        base, thr = load_calib(d)
        parts.append({"label": d.name, "base": base, "thr": thr})
    N = len(parts)
    print(f"納入受試者 N = {N}：{', '.join(p['label'] for p in parts)}\n")

    # ============ 表24：校準基準（跨受試者）個體差異分析 ============
    print("## 表24、個人化校準基準值之個體差異分析\n")
    print("| 校準基準 | 平均數 | 標準差 | 全距 (最小 ～ 最大) | 變異係數 CV |")
    print("|---|---|---|---|---|")
    for ex in EX:
        vals = [p["base"][ex] for p in parts if ex in p["base"]]
        if not vals:
            print(f"| {NAME[ex]} | — | — | — | — |"); continue
        m = st.mean(vals); sd = st.stdev(vals) if len(vals) > 1 else 0.0
        cv = sd / abs(m) if m else float("nan")
        print(f"| {NAME[ex]} | {m:.6f} | {sd:.6f} | {min(vals):.6f} ～ {max(vals):.6f} | {fmt(cv,3)} |")
    print()

    # ============ 表24附加：固定門檻反事實 + 精確二項檢定 ============
    print("## 表24附加、固定門檻與個人化門檻之達標判定差異\n")
    print("（通用固定門檻 = 全體個人化門檻之中位數；判定僅用各人校準之基準值與門檻，量尺一致）\n")
    print("| 訓練項目 | 通用固定門檻 | 假達標人數 (比例) | 假未達標人數 (比例) | 個人化門檻誤判人數 |")
    print("|---|---|---|---|---|")
    ci_rows = []
    for ex in EX:
        thrs = [p["thr"][ex] for p in parts if ex in p["thr"]]
        bases = [(p["label"], p["base"][ex], p["thr"][ex]) for p in parts
                 if ex in p["base"] and ex in p["thr"]]
        if not thrs:
            print(f"| {ZH[ex]} | — | — | — | — |"); continue
        Tg = st.median(thrs)
        fp = ff = 0
        for _, b, t in bases:
            if ex == "Double_Chin":   # 達標：值 > 門檻（越大越好）
                if b > Tg: fp += 1        # 放鬆即超過 → 假達標
                if t < Tg: ff += 1        # 自身校準目標低於全域 → 難達 → 假未達標
            else:                      # Face_Lift 達標：值 < 門檻（越小越好）
                if b < Tg: fp += 1
                if t > Tg: ff += 1
        n = len(bases)
        print(f"| {ZH[ex]} | {Tg:.6f} | {fp} ({fp/n*100:.0f}%) | {ff} ({ff/n*100:.0f}%) | 0 |")
        ci_rows.append((f"{ZH[ex]} 假達標", fp, n))
        ci_rows.append((f"{ZH[ex]} 假未達標", ff, n))
    print("\n（個人化門檻依各人基準設定，放鬆狀態不會被系統性誤判，故個人化門檻誤判人數為 0。）\n")

    print("### 表24附加：固定門檻誤判比例之 95% 信賴區間 (Wilson score interval)\n")
    print("| 項目 | 誤判人數/N | 觀察比例 | 95% CI |")
    print("|---|---|---|---|")
    for name, x, n in ci_rows:
        lo, hi = wilson_ci(x, n)
        print(f"| {name} | {x}/{n} | {x/n:.1%} | [{lo:.1%}, {hi:.1%}] |")
    print("\n（採 Wilson score interval，較常態近似 Wald CI 適合小樣本與極端比例；"
          "本表僅呈現觀察比例之不確定範圍，不涉及對特定門檻值之顯著性檢定）")


if __name__ == "__main__":
    main()
