import os
from PIL import Image, ImageOps

# 元画像ディレクトリ
SRC_DIR = '/home/student/e21/e215706/dm/fix/info3dm_racial_classification/train2'
# 反転画像保存先ディレクトリ
DST_DIR = '/home/student/e21/e215706/dm/fix/info3dm_racial_classification/train2'

# 対象拡張子
IMG_EXTS = ('.jpg', '.jpeg', '.png')


def augment_fliplr(src_dir, dst_dir):
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if not fname.lower().endswith(IMG_EXTS):
                continue
            if '_fliplr' in fname:
                continue  # 既に反転済み画像はスキップ

            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(root, src_dir)  # 相対パスで保存先を再現
            dst_subdir = os.path.join(dst_dir, rel_path)
            os.makedirs(dst_subdir, exist_ok=True)

            name, ext = os.path.splitext(fname)
            flip_name = f"{name}_fliplr{ext}"
            flip_path = os.path.join(dst_subdir, flip_name)

            if os.path.exists(flip_path):
                continue  # 既に存在する場合はスキップ

            try:
                with Image.open(src_path) as img:
                    flipped_img = ImageOps.mirror(img)
                    flipped_img.save(flip_path)
                    print(f"保存: {flip_path}")
            except Exception as e:
                print(f"エラー: {src_path} -> {e}")


def main():
    augment_fliplr(SRC_DIR, DST_DIR)


if __name__ == '__main__':
    main()
