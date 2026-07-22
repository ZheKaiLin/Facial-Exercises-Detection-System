import os
import shutil
from sklearn.model_selection import train_test_split

# ================= 參數設定區 =================
# 根據你的資料夾架構，精準設定相對路徑
source_img_dir = os.path.join('images', 'trainval')
source_lbl_dir = os.path.join('labels', 'trainval')

# 為了安全起見，切好的資料會統一放在一個叫 'split_result' 的新資料夾中
output_base = './'
train_img_dir = os.path.join(output_base, 'images', 'train')
val_img_dir = os.path.join(output_base, 'images', 'val')
train_lbl_dir = os.path.join(output_base, 'labels', 'train')
val_lbl_dir = os.path.join(output_base, 'labels', 'val')

# 切割比例 (0.2 代表 80% 訓練集, 20% 驗證集)
val_ratio = 0.15
# ==============================================

# 🛑 執行前防呆檢查：確認來源資料夾真的存在
if not os.path.exists(source_img_dir):
    print(f"❌ 錯誤：找不到圖片資料夾 -> {source_img_dir}")
    exit()
if not os.path.exists(source_lbl_dir):
    print(f"❌ 錯誤：找不到標籤資料夾 -> {source_lbl_dir}")
    exit()

# 建立所有需要的輸出資料夾
for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# 1. 讀取所有檔案並萃取類別標籤
image_files = []
labels = []

print("正在掃描檔案...")
for img_name in os.listdir(source_img_dir):
    # 只抓取圖片檔
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
        
    base_name = os.path.splitext(img_name)[0]
    txt_name = base_name + '.txt'
    txt_path = os.path.join(source_lbl_dir, txt_name)
    
    # 確保圖片有對應的標籤檔
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            if lines:
                try:
                    # 讀取第一行的第一個數字作為該圖片的類別 (0, 1, 2, 3, 4)
                    class_id = int(lines[0].split()[0])
                    image_files.append(img_name)
                    labels.append(class_id)
                except ValueError:
                    pass

print(f"成功配對了 {len(image_files)} 筆有效的圖文資料。")

# 2. 進行分層抽樣切割
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    image_files, labels, 
    test_size=val_ratio, 
    random_state=42,     # 固定亂數種子，確保學術實驗的可重現性
    stratify=labels      # 核心：依照類別比例進行分層抽樣
)

print(f"切割完成！準備複製 -> 訓練集: {len(train_imgs)} 張, 驗證集: {len(val_imgs)} 張。")

# 3. 定義檔案複製函數
def copy_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for img_name in file_list:
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + '.txt'
        
        shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, img_name))
        shutil.copy(os.path.join(src_lbl_dir, txt_name), os.path.join(dst_lbl_dir, txt_name))

# 4. 執行複製
print("檔案複製中，請稍候...")
copy_files(train_imgs, source_img_dir, source_lbl_dir, train_img_dir, train_lbl_dir)
copy_files(val_imgs, source_img_dir, source_lbl_dir, val_img_dir, val_lbl_dir)

print(f"✅ 大功告成！請查看新資料夾: {output_base}")