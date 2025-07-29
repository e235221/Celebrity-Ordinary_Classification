import cv2
import numpy as np
import mediapipe as mp
import os
# === 設定 ===
input_folder = '/home/student/e21/e215706/dm/sorce/image/all_image/normal_train'
output_folder = '/home/student/e21/e215706/dm/sorce/white/FaceDetection/all_image/normal_train'  # 白背景の場合
# output_folder = 'test_background_black'  # 黒背景の場合
background_color = (255, 255, 255)  # 白 (255,255,255) か 黒 (0,0,0)

os.makedirs(output_folder, exist_ok=True)

mp_selfie_segmentation = mp.solutions.selfie_segmentation
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # RGBに変換
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(image_rgb)
            mask = results.segmentation_mask

            # マスクのしきい値を決めて人物領域を抽出
            condition = mask > 0.5
            bg_image = np.full(image.shape, background_color, dtype=np.uint8)
            output_image = np.where(condition[..., None], image, bg_image)

            # 保存
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, output_image)
            print(f"保存しました: {save_path}")


