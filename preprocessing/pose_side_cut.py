import pandas as pd
import os

# パス設定
image_dir = "/home/student/e21/e215706/dm/fix/info3dm_racial_classification/images_face_resized_300x300_fix_bbox" # 顔写真のパス
csv_path = "/home/student/e21/e215706/dm/fix/info3dm_racial_classification/test_imgs_pose.csv" # poseラベルありcsvのパス
output_csv_path = "/home/student/e21/e215706/dm/fix/info3dm_racial_classification/test_imgs_front.csv" # 出力csvパス

df = pd.read_csv(csv_path)
side_df = df[df['pose'] == 'side']

# 側面の写真を削除
for file_rel_path in side_df['file']:
    full_path = os.path.join(image_dir, file_rel_path)
    
    if os.path.exists(full_path):
        os.remove(full_path)
        print(f"Deleted: {full_path}")
    else:
        print(f"File not found (skipped): {full_path}")

# frontのcsvを保存
df_filtered = df[df['pose'] != 'side']
df_filtered.to_csv(output_csv_path, index=False)

print(f"saved: {output_csv_path}")

