import os
import cv2
import shutil
import random
import numpy as np
from pathlib import Path

# =========================
# 你需要修改這裡
# =========================

SRC_ROOT = Path("v4_data")              # 原始資料集資料夾
DST_ROOT = Path("v4_data_balanced")          # 輸出的平衡資料集資料夾

CLASS_NAMES = ["angry", "happy", "neutral", "sad", "surprised"]

# 只對 train set 做少數類別擴增
# 這裡的 400 是 train set 中每一類希望接近的數量
# neutral 如果原本已經超過 400，就不會擴增
TARGET_PER_CLASS = {
    0: 400,   # angry
    1: 400,   # happy
    2: 400,   # neutral，不足才補，通常不會補
    3: 400,   # sad
    4: 400,   # surprised
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

random.seed(42)
np.random.seed(42)


def read_label(label_path):
    if not label_path.exists():
        return []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    return lines


def write_label(label_path, lines):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def get_first_class_id(label_lines):
    if len(label_lines) == 0:
        return None

    first = label_lines[0].split()
    if len(first) < 5:
        return None

    return int(float(first[0]))


def copy_dataset_structure():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"找不到來源資料集 -> {SRC_ROOT}（請確認資料夾位置，或修改本檔開頭的 SRC_ROOT）")

    if DST_ROOT.exists():
        print(f"偵測到 {DST_ROOT} 已存在，請先刪除或改名，避免混到舊資料。")
        raise SystemExit

    for split in ["train", "val", "test"]:
        src_img_dir = SRC_ROOT / "images" / split
        src_lbl_dir = SRC_ROOT / "labels" / split

        dst_img_dir = DST_ROOT / "images" / split
        dst_lbl_dir = DST_ROOT / "labels" / split

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if src_img_dir.exists():
            for img_path in src_img_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTS:
                    shutil.copy2(img_path, dst_img_dir / img_path.name)

        if src_lbl_dir.exists():
            for lbl_path in src_lbl_dir.iterdir():
                if lbl_path.suffix.lower() == ".txt":
                    shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

    print("原始 train / val / test 已複製完成。")


def collect_train_images_by_class():
    img_dir = DST_ROOT / "images" / "train"
    lbl_dir = DST_ROOT / "labels" / "train"

    class_to_items = {i: [] for i in range(len(CLASS_NAMES))}

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = lbl_dir / (img_path.stem + ".txt")
        label_lines = read_label(label_path)
        cls_id = get_first_class_id(label_lines)

        if cls_id is None:
            continue

        if cls_id in class_to_items:
            class_to_items[cls_id].append((img_path, label_path, label_lines))

    return class_to_items


def horizontal_flip(img, label_lines):
    flipped = cv2.flip(img, 1)

    new_lines = []
    for line in label_lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # YOLO 格式水平翻轉：x_center = 1 - x_center
        x = 1.0 - x

        new_line = f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        new_lines.append(new_line)

    return flipped, new_lines


def adjust_brightness_contrast(img, label_lines):
    alpha = random.uniform(0.75, 1.25)   # 對比
    beta = random.randint(-25, 25)       # 亮度
    aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return aug, label_lines


def add_noise(img, label_lines):
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    aug = img.astype(np.int16) + noise
    aug = np.clip(aug, 0, 255).astype(np.uint8)
    return aug, label_lines


def gaussian_blur(img, label_lines):
    aug = cv2.GaussianBlur(img, (3, 3), 0)
    return aug, label_lines


def gamma_correction(img, label_lines):
    gamma = random.uniform(0.8, 1.25)
    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    aug = cv2.LUT(img, table)
    return aug, label_lines


AUG_FUNCS = [
    ("flip", horizontal_flip),
    ("bright", adjust_brightness_contrast),
    ("noise", add_noise),
    ("blur", gaussian_blur),
    ("gamma", gamma_correction),
]


def augment_minority_classes():
    img_dir = DST_ROOT / "images" / "train"
    lbl_dir = DST_ROOT / "labels" / "train"

    class_to_items = collect_train_images_by_class()

    print("\n擴增前 train set 數量：")
    for cls_id, items in class_to_items.items():
        print(f"{cls_id} {CLASS_NAMES[cls_id]}: {len(items)}")

    for cls_id, target in TARGET_PER_CLASS.items():
        items = class_to_items[cls_id]
        current = len(items)

        if current >= target:
            print(f"\n{CLASS_NAMES[cls_id]} 已有 {current} 張，不需要擴增。")
            continue

        need = target - current
        print(f"\n開始擴增 {CLASS_NAMES[cls_id]}：目前 {current} 張，目標 {target} 張，需要新增 {need} 張。")

        for i in range(need):
            img_path, label_path, label_lines = random.choice(items)

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"無法讀取圖片：{img_path}")
                continue

            aug_name, aug_func = random.choice(AUG_FUNCS)
            aug_img, aug_label_lines = aug_func(img, label_lines)

            new_stem = f"{img_path.stem}_aug_{aug_name}_{i:04d}"
            new_img_path = img_dir / f"{new_stem}{img_path.suffix}"
            new_lbl_path = lbl_dir / f"{new_stem}.txt"

            cv2.imwrite(str(new_img_path), aug_img)
            write_label(new_lbl_path, aug_label_lines)

    class_to_items_after = collect_train_images_by_class()

    print("\n擴增後 train set 數量：")
    for cls_id, items in class_to_items_after.items():
        print(f"{cls_id} {CLASS_NAMES[cls_id]}: {len(items)}")


def main():
    print("來源資料集：", SRC_ROOT)
    print("輸出資料集：", DST_ROOT)

    copy_dataset_structure()
    augment_minority_classes()

    print("\n完成！請確認 balanced_data 是否已產生。")


if __name__ == "__main__":
    main()