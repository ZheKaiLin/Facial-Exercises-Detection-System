"""
第三步：計算兩位標註者的一致性（Cohen's kappa + 框選 IoU）

比較對象：
    標註者 1（原始標籤）：v4_data_balanced/labels/test_100/*.txt
    標註者 2（盲標註）  ：annotation_reliability/labels_annotator2/*.txt
    對應關係            ：annotation_reliability/mapping_SECRET.csv（blind_name <-> original_name）

輸出：
    1) 主控台列印：表情分類一致率、Cohen's kappa、混淆矩陣、框選 IoU 統計
    2) agreement_detail.csv：逐張圖片的兩位標註結果與 IoU，供論文附錄或覆核
    3) agreement_summary.csv：彙總統計表，可直接整理進論文表格

使用方式：
    python analyze_agreement.py
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

CLASSES = ["angry", "happy", "neutral", "sad", "surprised"]

SCRIPT_DIR = Path(__file__).resolve().parent
MAPPING_CSV = SCRIPT_DIR / "mapping_SECRET.csv"
LABELS_A2_DIR = SCRIPT_DIR / "labels_annotator2"
LABELS_A1_DIR = SCRIPT_DIR.parent / "v4_data_balanced" / "labels" / "test_100"

IOU_AGREE_THRESHOLD = 0.5


def load_yolo_label(path):
    """回傳 (class_idx, (x1,y1,x2,y2) 正規化座標) 或 None"""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.split()
    c = int(parts[0])
    xc, yc, w, h = map(float, parts[1:5])
    x1, y1 = xc - w / 2, yc - h / 2
    x2, y2 = xc + w / 2, yc + h / 2
    return c, (x1, y1, x2, y2)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def main():
    if not MAPPING_CSV.exists():
        raise FileNotFoundError(f"找不到對照表，請先執行 prepare_blind_annotation.py：{MAPPING_CSV}")

    with open(MAPPING_CSV, "r", encoding="utf-8-sig") as f:
        mapping = list(csv.DictReader(f))

    rows = []
    missing_a2 = []
    for m in mapping:
        blind_name = m["blind_name"]
        original_name = m["original_name"]
        blind_stem = Path(blind_name).stem
        original_stem = Path(original_name).stem

        a1 = load_yolo_label(LABELS_A1_DIR / f"{original_stem}.txt")
        a2 = load_yolo_label(LABELS_A2_DIR / f"{blind_stem}.txt")

        if a1 is None:
            print(f"警告：找不到原始標籤 {original_stem}.txt，略過")
            continue
        if a2 is None:
            missing_a2.append(original_name)
            continue

        cls1, box1 = a1
        cls2, box2 = a2
        box_iou = iou(box1, box2)

        rows.append({
            "original_name": original_name,
            "blind_name": blind_name,
            "annotator1_class": CLASSES[cls1],
            "annotator2_class": CLASSES[cls2],
            "class_agree": cls1 == cls2,
            "iou": round(box_iou, 4),
            "box_agree_iou>=0.5": box_iou >= IOU_AGREE_THRESHOLD,
        })

    if missing_a2:
        print(f"\n注意：第二位標註者尚未完成 {len(missing_a2)} 張（總數 {len(mapping)}），"
              f"目前僅以已完成的 {len(rows)} 張計算統計量。")
        print("請先執行 annotate_gui.py 完成全部 100 張，再重跑本腳本以取得正式報告數據。\n")

    if not rows:
        print("目前沒有任何可比對的標註，無法計算統計量。")
        return

    y1 = [CLASSES.index(r["annotator1_class"]) for r in rows]
    y2 = [CLASSES.index(r["annotator2_class"]) for r in rows]

    kappa = cohen_kappa_score(y1, y2)
    agree_rate = float(np.mean([r["class_agree"] for r in rows]))
    cm = confusion_matrix(y1, y2, labels=list(range(len(CLASSES))))

    ious = [r["iou"] for r in rows]
    mean_iou = float(np.mean(ious))
    median_iou = float(np.median(ious))
    box_agree_rate = float(np.mean([r["box_agree_iou>=0.5"] for r in rows]))

    print("========== 表情分類一致性 ==========")
    print(f"樣本數：{len(rows)}")
    print(f"一致率（Percent Agreement）：{agree_rate * 100:.1f}%")
    print(f"Cohen's kappa：{kappa:.3f}")
    print("\n混淆矩陣（列=標註者1，欄=標註者2）：")
    header = "        " + "  ".join(f"{c:>10s}" for c in CLASSES)
    print(header)
    for i, c in enumerate(CLASSES):
        print(f"{c:>8s}" + "".join(f"{cm[i][j]:>12d}" for j in range(len(CLASSES))))

    print("\n========== 框選一致性（IoU）==========")
    print(f"平均 IoU：{mean_iou:.3f}")
    print(f"中位數 IoU：{median_iou:.3f}")
    print(f"IoU >= {IOU_AGREE_THRESHOLD} 之一致率：{box_agree_rate * 100:.1f}%")

    detail_csv = SCRIPT_DIR / "agreement_detail.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n逐張明細已存至：{detail_csv}")

    summary_csv = SCRIPT_DIR / "agreement_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["項目", "數值"])
        writer.writerow(["樣本數", len(rows)])
        writer.writerow(["表情分類一致率(%)", round(agree_rate * 100, 1)])
        writer.writerow(["Cohen's kappa", round(kappa, 3)])
        writer.writerow(["框選平均IoU", round(mean_iou, 3)])
        writer.writerow(["框選中位數IoU", round(median_iou, 3)])
        writer.writerow([f"框選一致率(IoU>={IOU_AGREE_THRESHOLD})(%)", round(box_agree_rate * 100, 1)])
    print(f"彙總表已存至：{summary_csv}")


if __name__ == "__main__":
    main()
