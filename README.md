## 概要
「美男美女」または「一般人」に分類する機械学習モデル・スクリプトを提供する。[FairFace](https://github.com/joojs/fairface)を用いてデータセット構築・転移学習を行った。

本研究では，FairFace データセットを一般人データとして用い，独自に構築した美男美女データセットと併せ
て学習を行うことで，人物の外見に基づく分類を行う機械学習システムを構築した．さらに，Grad-CAM を
用いて各モデルの判断根拠を可視化し，信頼性の評価を試みた．転移学習を行ったFairFace モデルにおいて
は，顔領域を適切に着目しながら一般人・美男美女の分類精度100% を達成したが，新規学習したResNet18，
ResNet34，ResNet50，E\UTF{FB03}cientNet_b0 については，分類精度95% を超える高い分類精度を示したものの顔
ではなく画像の背景を見ていることが確認された．ここからデータセット構築時における内在的バイアスが含
まれていることがわかった．深層学習モデルが高精度を示す場合でも，その判断根拠が実験目的と合致してい
るとは限らないという，モデルの信頼性における重要な課題を浮き彫りにした．また，Custom Model では
Grad-Cam の結果より目元と口元を見て一般人と美男美女を分類していることから，これらが一般人と美男
美女の差異であると結論付けた．


レポート：[https://github.com/e235221/info3dm_FinalRep](https://github.com/e235221/info3dm_FinalRep)

---

## データセットについて

-   **有名人画像**：著名人の顔写真
-   **一般人画像**：一般人の顔写真（プライバシー配慮の上使用）

### 既存ラベル

既存のfairfaceラベル

-   **file**: `/home/student/e21/e215706/dm/sorce/image/all_image`以後のパス
-   **age**: 年齢
-   **gender**: 性別
-   **race**: 人種

詳細は[fairface](https://github.com/joojs/fairface)を参照

### 新規ラベル

今回の実験のために新しく追加したラベル

-   **looks**
    -   `0`: 一般人
    -   `1`: 美男美女(有名人)

---

## 使い方

### データセット構築

データセット構築に使用したコードは `preprocessing` ディレクトリに存在する

#### 美男美女

-   **画像収集**：`web_scraping.py` (web: Bing)
-   **初期csv作成**：`change_name_for_FairFaceCSV.py` (各画像のパスをcsvとして保存)
-   **リサイズ**：`resize.py` (300×300)
-   **側面画像除去**
    -   `pose.py` (`hopenet.py`, `pose_model/hopenet_alpha2.pkl`必要)：csvに正面と側面のラベルを追加
    -   `pose_side_cut.py`：csvを基に側面を画像を削除
-   **ラベリング**：csvに`file`、`age`、`gender`、`race` を追加(手作業)
-   **アップサンプリング**：`upsampling.py` (左右反転)

#### 一般人

-   **画像収集**：fairfaceから画像を用意
-   **ダウンサンプリング**：`downsampling.py` (ランダム選択)

#### 最後

1.  csvを結合
2.  csvに新規ラベル`looks`を追加
3.  `image/csv` にcsvを用意
4.  `image/all_image/good_train` に美男美女の学習データセットを用意
5.  `image/all_image/good_test` に美男美女の予測データセットを用意
6.  `image/all_image/normal_train` に一般人の学習データセットを用意
7.  `image/all_image/normal_test` に一般人の予測データセットを用意

### 実験

この段階では`image/all_image` ディレクトリにイメージ、`image/csv` にイメージに対するcsvを存在することを想定する（追加学習を行う場合`model` ディレクトリに学習済みモデルが存在することを想定）。
各コードのパス(`/home/student/e21/e215706/dm/sorce/`のところ)を正しく修正する。

#### 学習・予測

`code`ディレクトリにあるコードで学習及び予測を行う。

-   **追加学習コード**
    -   `custom_model.py`
-   **新規学習コード**
    -   `create_model_18.py` (model: resnet18)
    -   `create_model_34.py` (model: resnet34)
    -   `create_model_50.py` (model: resnet50)
    -   `create_model_efficientnet_b0.py` (model: efficientnet_b0)

実行結果は`log`ディレクトリに保存される

-   `custom_log.txt`: `custom_model.py`の結果
-   `resnet18_log.txt`: `create_model_18.py`の結果
-   `resnet34_log.txt`: `create_model_34.py`の結果
-   `resnet50_log.txt`: `create_model_50.py`の結果
-   `efficientnetb0_log.txt`: `create_model_efficientnet_b0.py`の結果

学習が終わったモデルは`model`ディレクトリに保存される

-   `custom/custom.pth`: `custom_model.py`で学習したモデル
-   `create/resnet18_looks_classifier.pth`: `create_model_18.py`で学習したモデル
-   `create/resnet34_looks_classifier.pth`: `create_model_34.py`で学習したモデル
-   `create/resnet50_looks_classifier.pth`: `create_model_50.py`で学習したモデル
-   `create/efficientnetb0_looks_classifier.pth`: `create_model_efficientnet_b0.py`で学習したモデル

#### gradcam検証

`analysis/gradcam_code` ディレクトリにあるコードでgradcamの検証を行う

-   **平均gradcam**
    -   `gradcam_custom_aver.py`: `custom.pth`の平均gradcam出力
    -   `gradcam_resnet18_aver.py`: `resnet18_looks_classifier.pth`の平均gradcam出力
    -   `gradcam_resnet34_aver.py`: `resnet34_looks_classifier.pth`の平均gradcam出力
    -   `gradcam_resnet50_aver.py`: `resnet50_looks_classifier.pth`の平均gradcam出力
    -   `gradcam_efficientnetb0_aver.py`: `efficientnetb0_looks_classifier.pth`の平均gradcam出力
-   **個別gradcam**
    -   `gradcam_custom.py`: `custom.pth`の個別gradcam出力
    -   `gradcam_resnet18.py`: `resnet18_looks_classifier.pth`の個別gradcam出力
    -   `gradcam_resnet34.py`: `resnet34_looks_classifier.pth`の個別gradcam出力
    -   `gradcam_resnet50.py`: `resnet50_looks_classifier.pth`の個別gradcam出力
    -   `gradcam_efficientnetb0_aver.py`: `efficientnetb0_looks_classifier.pth`の個別gradcam出力

`analysis/gradcam_results` ディレクトリにあるコードにgradcamの結果が保存される

-   `average`: 平均gradcamが保存されるディレクトリ
-   個別gradcamは各モデル名と同じディレクトリの保存

---

## ディレクトリ構成

-   `preprocessing`: 前処理コード
-   `image`: 画像とcsv
-   `code`: 学習・予測実行コード
-   `log`: 学習・予測結果
-   `model`: モデル
-   `anaanalysis`: gradcam検証
-   `white`: 白背景画像に対するディレクトリ
    -   `white/FaceDetection`: 白背景化コード及び白背景画像
    -   `white/code`: 学習・予測実行コード
    -   `white/log`: 学習・予測結果
    -   `white/model`: モデル
    -   `white/analysis`: gradcam検証
-   `remove_back`: 透明背景画像に対するディレクトリ
    -   `remove_back/all_imge`: 透明背景画像
    -   `remove_back/csv`: 画像のcsv
    -   `remove_back/code`: 学習・予測実行コード
    -   `remove_back/log`: 学習・予測結果
    -   `remove_back/model`: モデル
    -   `remove_back/analysis`: gradcam検証
-   `hyper`: ハイパーパラメータ調整後に対するディレクト
    -   `hyper/code`: 学習・予測実行コード
    -   `hyper/log`: 学習・予測結果
    -   `hyper/model`: モデル
    -   `hyper/analysis`: gradcam検証