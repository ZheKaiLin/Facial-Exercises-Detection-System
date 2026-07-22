"""
第一步：盲標註前處理

test_100 原始檔名（如 Freepik_sad1.jpg、Pixabay_angry1.jpg）本身就寫著表情類別，
若直接讓第二位標註者看到原檔名會洩漏答案，導致一致性統計失去意義。

本腳本將 test_100 的 100 張圖片複製一份、以隨機亂數重新命名為 img_001.jpg ~ img_100.jpg，
並以固定 random seed 打亂順序（同一顆 seed 之後可重現、但外部無法反推對應關係），
輸出：
    annotation_reliability/blind_images/           匿名圖片（給第二位標註者用）
    annotation_reliability/mapping_SECRET.csv       匿名檔名 <-> 原始檔名對照表（標註期間不可外流）

使用方式：
    python prepare_blind_annotation.py
"""
import csv
import random
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_IMAGES = SCRIPT_DIR.parent / "v4_data_balanced" / "images" / "test_100"
OUT_DIR = SCRIPT_DIR / "blind_images"
MAPPING_CSV = SCRIPT_DIR / "mapping_SECRET.csv"
SEED = 42


def main():
    if not SRC_IMAGES.exists():
        raise FileNotFoundError(f"找不到來源資料夾：{SRC_IMAGES}")

    image_paths = sorted(
        p for p in SRC_IMAGES.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"找到 {len(image_paths)} 張原始影像：{SRC_IMAGES}")

    rng = random.Random(SEED)
    shuffled = image_paths[:]
    rng.shuffle(shuffled)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, src_path in enumerate(shuffled, start=1):
        blind_name = f"img_{idx:03d}{src_path.suffix.lower()}"
        dst_path = OUT_DIR / blind_name
        shutil.copyfile(src_path, dst_path)
        rows.append({"blind_name": blind_name, "original_name": src_path.name})

    with open(MAPPING_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["blind_name", "original_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"已複製 {len(rows)} 張匿名圖片至：{OUT_DIR}")
    print(f"對照表（標註期間請勿提供給第二位標註者）已存至：{MAPPING_CSV}")
    print("\n下一步：請第二位標註者執行 annotate_gui.py 進行獨立標註。")


if __name__ == "__main__":
    main()
