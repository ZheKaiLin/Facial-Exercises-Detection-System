"""
獨立測試集（test_300）與訓練集（train, 2172 張）重複影像檢查。

對應論文 3.1.4 節「獨立測試資料建置」：驗證測試集完全未參與模型訓練，
以系統化雜湊比對取代純人工比對，避免資料夾層級的人工初篩錯誤（見 5.3 節說明）。

比對方法：
  1. MD5 精確比對：抓出位元組完全相同的檔案（例如同一張圖被誤存進兩個資料夾）。
  2. Perceptual hash（pHash，16x16 DCT，256-bit）近似比對：抓出裁切／縮放／
     重新編碼／壓縮後仍視覺相似的影像，MD5 無法偵測這類「內容相同但檔案不同」的重複。
     使用 256-bit 而非預設 64-bit，是為了在 2172 張訓練圖 x 300 張測試圖的規模下，
     取得更細緻的雜湊距離解析度，降低小差異被同一組粗粒度雜湊值掩蓋的機率。

輸出：
  - duplicate_report.csv：所有 pHash 距離 <= HAMMING_THRESHOLD 的配對（含 MD5 完全相同者）
  - 標準輸出摘要：MD5 完全重複數、pHash 疑似重複數、雜湊距離最接近的配對
"""

import csv
import hashlib
from pathlib import Path

import imagehash
from PIL import Image

BASE = Path(r"C:\Users\林哲楷\Desktop\Data processing\v4_data_final\images")
TEST_DIR = BASE / "test_300"
TRAIN_DIR = BASE / "train"

# pHash 雜湊大小：hash_size=16 -> 16x16=256-bit 雜湊值。
HASH_SIZE = 16
TOTAL_BITS = HASH_SIZE * HASH_SIZE

# 漢明距離門檻：256-bit 下取約 10% 位元差異（約 26 bits）視為疑似重複，
# 對應近乎相同或僅經裁切/縮放/重新編碼的影像；此為近似比對的業界慣例延伸。
HAMMING_THRESHOLD = round(TOTAL_BITS * 0.10)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUT_CSV = Path(__file__).parent / "duplicate_report.csv"


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_of(path: Path):
    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"), hash_size=HASH_SIZE)


def main():
    test_files = list_images(TEST_DIR)
    train_files = list_images(TRAIN_DIR)
    print(f"測試集（test_300）：{len(test_files)} 張")
    print(f"訓練集（train）：{len(train_files)} 張")

    print("計算訓練集 MD5 / pHash ...")
    train_md5 = {}
    train_phash = {}
    for p in train_files:
        train_md5[md5_of(p)] = p
        train_phash[p] = phash_of(p)

    print("計算測試集 MD5 / pHash 並逐一比對 ...")
    rows = []
    closest_pair = None  # (distance, test_path, train_path)

    for tp in test_files:
        t_md5 = md5_of(tp)
        t_hash = phash_of(tp)

        exact_match = train_md5.get(t_md5)

        best_train_path = None
        best_dist = None
        for train_path, train_hash in train_phash.items():
            dist = t_hash - train_hash  # 漢明距離（0~64）
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_train_path = train_path

        if closest_pair is None or best_dist < closest_pair[0]:
            closest_pair = (best_dist, tp, best_train_path)

        if exact_match is not None:
            rows.append({
                "test_image": tp.name,
                "train_image": exact_match.name,
                "match_type": "MD5_EXACT",
                "hamming_distance": 0,
            })
        elif best_dist is not None and best_dist <= HAMMING_THRESHOLD:
            rows.append({
                "test_image": tp.name,
                "train_image": best_train_path.name,
                "match_type": "PHASH_SIMILAR",
                "hamming_distance": best_dist,
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["test_image", "train_image", "match_type", "hamming_distance"])
        writer.writeheader()
        writer.writerows(rows)

    n_exact = sum(1 for r in rows if r["match_type"] == "MD5_EXACT")
    n_phash = sum(1 for r in rows if r["match_type"] == "PHASH_SIMILAR")

    print("\n===== 比對結果摘要 =====")
    print(f"MD5 完全相同（位元組級重複）：{n_exact} 組")
    print(f"pHash 疑似重複（漢明距離 <= {HAMMING_THRESHOLD}）：{n_phash} 組")
    if closest_pair:
        dist, tp, trp = closest_pair
        print(
            f"整體雜湊距離最接近的配對：test/{tp.name} vs train/{trp.name}，"
            f"漢明距離 = {dist} / {TOTAL_BITS} bits（差異約 {dist/TOTAL_BITS:.1%}）"
        )
    print(f"詳細報告已輸出：{OUT_CSV}")


if __name__ == "__main__":
    main()
