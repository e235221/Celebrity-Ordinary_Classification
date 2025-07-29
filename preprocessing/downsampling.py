import os
import random
import shutil

# Path
SRC_DIR = '/home/student/e21/e215706/dm/info3dm_racial_classification/fairface_image/train'        # Input Image Directory 
DST_DIR = '/home/student/e21/e215706/dm/info3dm_racial_classification/fairface_image/train_down'        # Output Image Directory 
NUM_FILES = 32000       # 選ぶファイル数

# Directory生成
os.makedirs(DST_DIR, exist_ok=True)

# .jpg 抽出
jpg_files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith('.jpg')]

# Random選択
selected_files = random.sample(jpg_files, min(NUM_FILES, len(jpg_files)))

# Copy
for filename in selected_files:
    src_path = os.path.join(SRC_DIR, filename)
    dst_path = os.path.join(DST_DIR, filename)
    shutil.copy2(src_path, dst_path)
    print(f"Copied: {filename}")

print(f"\n総 {len(selected_files)}個の .jpgが {DST_DIR}にコピーされました.")

