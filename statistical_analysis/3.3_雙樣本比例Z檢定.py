"""
Facial_Yoga_Table12_15附加_雙樣本比例Z檢定.py
補齊論文第3章「表13、表17」——YOLOv7-tiny / YOLOv9-tiny 於
「蔡旻均(2020)測試集 200張」vs「自行蒐集圖庫影像 100張」兩個分層測試子集，
對 Precision／Recall 進行雙樣本比例 z 檢定（常態近似法）。

【與舊版之差異，2026-07-05 修正】
1. Precision 分母改為 TP+FP（模型實際判定為正例之偵測框數），不再用表12/16
   之巨集平均 Precision 反推「out of 200/100」計數——因巨集平均是 5 個類別
   （各自分母不同）之平均比例，無法合法反推成單一分母之整數計數。
   本版直接從 YOLO test.py／val_dual.py 之原始預測框（labels/*.txt，含信心值）
   與 GT 標註逐框比對得出 TP/FP，取「5 類別平均 F1 最大」之信心門檻（對應
   test.py 內部 f1.mean(0).argmax() 選點邏輯），確保 Recall 精確重現表12/16
   之巨集平均數字，同時得到可validly pooled 之 Precision 分子/分母。
2. z 檢定與 95% CI 統一改採不合併（Wald）標準誤（兩組各自變異相加），
   不再對 z 檢定另外使用 pooled SE。原因：混用「p 值用 pooled SE、CI 用
   unpooled SE」會出現「p<.05 但 CI 卻涵蓋 0」之表面矛盾；改成兩者統一
   用同一（unpooled）SE 後，CI 是否涵蓋 0 與 p 值是否 <.05 保證方向一致，
   且不需要额外對「test-based CI」做特別說明。

執行：
  python 3.3_雙樣本比例Z檢定.py
  （預設讀取本腳本同資料夾下的 yolo原始資料\\，全部使用相對路徑，
   可直接連同整個「總評估」資料夾打包寄送，不依賴外部路徑：
     yolo原始資料\\
       gt_labels\\test_200\\*.txt、test_100\\*.txt   （GT 標註，v7/v9 共用）
       v7_test\\pooled_test200\\labels\\*.txt、pooled_test100\\labels\\*.txt
       v9_test\\pooled_test_200\\labels\\*.txt、pooled_test_100\\labels\\*.txt
   可用 --gt-root / --v7-root / --v9-root 覆寫）
"""
import argparse
from pathlib import Path
import numpy as np
from scipy import stats

IOU_THRES = 0.50
N_CLASSES = 5


def xywh_to_xyxy(x, y, w, h):
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def iou(b1, b2):
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = (ax2 - ax1) * (ay2 - ay1)
    a2 = (bx2 - bx1) * (by2 - by1)
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def load_dataset(gt_dir: Path, pred_dir: Path):
    """回傳 list of (gt_cls, gt_box, [(conf, cls, box), ...])，每張圖一筆。"""
    data = []
    for gt_file in sorted(gt_dir.glob("*.txt")):
        line = gt_file.read_text().strip().splitlines()[0]
        c, x, y, w, h = line.split()
        gt_cls, gt_box = int(c), xywh_to_xyxy(float(x), float(y), float(w), float(h))
        preds = []
        pf = pred_dir / gt_file.name
        if pf.exists():
            for pl in pf.read_text().strip().splitlines():
                if not pl.strip():
                    continue
                parts = pl.split()
                cls, x, y, w, h, conf = int(parts[0]), *map(float, parts[1:6])
                preds.append((conf, cls, xywh_to_xyxy(x, y, w, h)))
        preds.sort(key=lambda p: p[0], reverse=True)
        data.append((gt_cls, gt_box, preds))
    return data


def evaluate_at_threshold(data, conf_thres):
    tp_c = np.zeros(N_CLASSES, dtype=int)
    fp_c = np.zeros(N_CLASSES, dtype=int)
    n_gt_c = np.zeros(N_CLASSES, dtype=int)
    for gt_cls, gt_box, preds in data:
        n_gt_c[gt_cls] += 1
        matched = False
        for conf, cls, box in preds:
            if conf < conf_thres:
                continue
            if not matched and cls == gt_cls and iou(box, gt_box) >= IOU_THRES:
                tp_c[gt_cls] += 1
                matched = True
            else:
                fp_c[cls] += 1
    return tp_c, fp_c, n_gt_c - tp_c, n_gt_c


def best_threshold(data):
    """網格搜尋使 5 類別平均 F1 最大之信心門檻（對應 test.py 選點邏輯）。"""
    best_t, best_f1 = 0.0, -1.0
    for t in np.arange(0.00, 1.00, 0.01):
        tp_c, fp_c, fn_c, n_gt_c = evaluate_at_threshold(data, t)
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(tp_c + fp_c > 0, tp_c / (tp_c + fp_c), 0.0)
            rec = np.where(n_gt_c > 0, tp_c / n_gt_c, 0.0)
            f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        mean_f1 = f1.mean()
        if mean_f1 > best_f1:
            best_f1, best_t = mean_f1, t
    return best_t


def get_counts(gt_dir, pred_dir):
    data = load_dataset(gt_dir, pred_dir)
    t = best_threshold(data)
    tp_c, fp_c, fn_c, n_gt_c = evaluate_at_threshold(data, t)
    return int(tp_c.sum()), int(fp_c.sum()), int(n_gt_c.sum()), t


def two_prop_ztest(p1, n1, p2, n2):
    """回傳 (diff, ci_lo, ci_hi, z, p)。z 檢定與 CI 統一採不合併（Wald）標準誤。"""
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = diff / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_lo, ci_hi = diff - 1.96 * se, diff + 1.96 * se
    return diff, ci_lo, ci_hi, z, p_val


def sig(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", default=str(here / "yolo原始資料" / "gt_labels"))
    ap.add_argument("--v7-root", default=str(here / "yolo原始資料" / "v7_test"))
    ap.add_argument("--v9-root", default=str(here / "yolo原始資料" / "v9_test"))
    opt = ap.parse_args()
    gt_root, v7, v9 = Path(opt.gt_root), Path(opt.v7_root), Path(opt.v9_root)

    MODELS = {
        "YOLOv7-tiny（表13）": {
            200: (gt_root / "test_200", v7 / "pooled_test200" / "labels"),
            100: (gt_root / "test_100", v7 / "pooled_test100" / "labels"),
        },
        "YOLOv9-tiny（表17）": {
            200: (gt_root / "test_200", v9 / "pooled_test_200" / "labels"),
            100: (gt_root / "test_100", v9 / "pooled_test_100" / "labels"),
        },
    }

    for model, subsets in MODELS.items():
        counts = {}
        for n, (gt_dir, pred_dir) in subsets.items():
            tp, fp, n_images, t = get_counts(gt_dir, pred_dir)
            counts[n] = dict(tp=tp, fp=fp, n_images=n_images, t=t)
            print(f"{model} {n}張子集：門檻(mean-F1)={t:.2f}  影像數={n_images}  ΣTP={tp}  ΣFP={fp}")

        c200, c100 = counts[200], counts[100]
        print(f"\n## {model}、分層測試子集之比例檢定\n")
        print("| 指標 | 200 張子集 | 100 張子集 | 差異 [95% CI] | z 值 | p 值 |")
        print("|---|---|---|---|---|---|")

        # Recall：分母為影像數（= TP+FN，每張圖僅 1 個標註實例）
        p1, n1 = c200["tp"] / c200["n_images"], c200["n_images"]
        p2, n2 = c100["tp"] / c100["n_images"], c100["n_images"]
        diff, lo, hi, z, p = two_prop_ztest(p1, n1, p2, n2)
        pstr = f"{p:.4f}" if p < .01 else f"{p:.3f}"
        print(f"| Recall | {p1*100:.1f}% ({c200['tp']}/{n1}) | {p2*100:.1f}% ({c100['tp']}/{n2}) | "
              f"{diff*100:+.1f}% [{lo*100:+.1f}%, {hi*100:+.1f}%] | {z:.2f} | {pstr}{sig(p)} |")

        # Precision：分母為 TP+FP（模型實際判定為正例之偵測框數）
        n1p, n2p = c200["tp"] + c200["fp"], c100["tp"] + c100["fp"]
        p1p, p2p = c200["tp"] / n1p, c100["tp"] / n2p
        diff, lo, hi, z, p = two_prop_ztest(p1p, n1p, p2p, n2p)
        pstr = f"{p:.4f}" if p < .01 else f"{p:.3f}"
        print(f"| Precision | {p1p*100:.1f}% ({c200['tp']}/{n1p}) | {p2p*100:.1f}% ({c100['tp']}/{n2p}) | "
              f"{diff*100:+.1f}% [{lo*100:+.1f}%, {hi*100:+.1f}%] | {z:.2f} | {pstr}{sig(p)} |")
        print()

    print("（IoU>=0.50 逐框比對得出 TP/FP；信心門檻取 5 類別平均 F1 最大點，對應 test.py 選點邏輯；"
          "Recall 分母為影像數，Precision 分母為 TP+FP；z 檢定與 95% CI 統一採不合併（Wald）標準誤，"
          "確保 CI 是否涵蓋 0 與 p<.05 方向一致；雙尾檢定；* p<0.05，** p<0.01，*** p<0.001）")


if __name__ == "__main__":
    main()
