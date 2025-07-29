import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# パス設定
csv_dir = "/home/student/e21/e215706/dm/sorce/image/csv/"
image_root = "/home/student/e21/e215706/dm/sorce/image/all_image"
output_root = "/home/student/e21/e215706/dm/sorce/remove_back/all_image"
csv_output_dir = "/home/student/e21/e215706/dm/sorce/remove_back/csv"
os.makedirs(csv_output_dir, exist_ok=True)

# CSV読み込み
train_csv_path = os.path.join(csv_dir, "all_train.csv")
test_csv_path = os.path.join(csv_dir, "all_test.csv")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# MediaPipe セグメンテーションの初期化
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# 画像処理と保存を行う関数
def process_and_save_images(df, name):
    updated_paths = []

    for rel_path in df['file']:
        input_path = os.path.join(image_root, rel_path)
        rel_png_path = rel_path.rsplit(".", 1)[0] + ".png"
        output_path = os.path.join(output_root, rel_png_path)

        # 出力フォルダが存在しない場合は作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 画像読み込み
        img = cv2.imread(input_path)
        if img is None:
            print(f"画像の読み込みに失敗しました: {input_path}")
            continue

        # RGBに変換してセグメンテーション実行
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = segmentor.process(img_rgb)

        if results.segmentation_mask is None:
            print(f"セグメンテーションに失敗しました: {input_path}")
            continue

        # マスク処理（True: 人物, False: 背景）
        mask = results.segmentation_mask > 0.5
        alpha = (mask * 255).astype(np.uint8)

        # 画像にアルファチャンネルを追加
        bg_removed = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bg_removed[:, :, 3] = alpha

        # PNG形式で保存（背景は透明）
        cv2.imwrite(output_path, bg_removed)
        updated_paths.append(rel_png_path)

        print(f"[{name}] 保存完了: {output_path}")

    # file列を.pngに更新し、CSVを保存
    df['file'] = updated_paths
    output_csv_path = os.path.join(csv_output_dir, f"{name}.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"[{name}] CSV保存完了: {output_csv_path}")

# 学習用とテスト用のCSVをそれぞれ処理
process_and_save_images(train_df, "all_train")
process_and_save_images(test_df, "all_test")

