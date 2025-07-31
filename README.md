## 概要
「有名人」または「一般人」に分類する機械学習モデル・スクリプトを提供する。本データセット構築・スクリプトにはFairFaceを用いた。


## データセットについて
- 有名人画像：著名人の顔写真：webスクレイピングで収集した。
- 一般人画像：一般人の顔写真：FairFaceデータセットの画像をラベル付し有名人画像とともに再学習。

## 使い方
- image/all_image ディレクトリにイメージ、image/csv にイメージに対するcsvを存在することを想定する（追加学習を行う場合model ディレクトリに学習済みモデルが存在することを想定）
- 各コードのパス(/home/student/e21/e215706/dm/sorce/のところ)を正しく修正する

### 学習
codeディレクトリにあるコードで学習及び予測を行う
  追加学習コード
  custom_model.py
  新規学習コード
  create_model_18.py (model:resnet18)
  create_model_34.py (model:resnet34)
  create_model_50.py (model:resnet50)
  create_model_efficientnet_b0.py (model:efficientnet_b0)
実行結果はlogディレクトリに保存される
  custom_log.txt: custom_model.pyの結果
  resnet18_log.txt: create_model_18.pyの結果
  resnet34_log.txt: create_model_34.pyの結果
  resnet50_log.txt: create_model_50.pyの結果
  efficientnetb0_log.txt: create_model_efficientnet_b0.pyの結果
学習が終わったモデルはmodelディレクトリに保存される
  custom/custom.pth: custom_model.pyで学習したモデル
  create/resnet18_looks_classifier.pth: create_model_18.pyで学習したモデル
  create/resnet34_looks_classifier.pth: create_model_34.pyで学習したモデル
  create/resnet50_looks_classifier.pth: create_model_50.pyで学習したモデル
  create/efficientnetb0_looks_classifier.pth: create_model_efficientnet_b0.pyで学習したモデル
### gradcam検証
analysis/gradcam_code ディレクトリにあるコードでgradcamの検証を行う
  平均gradcam
  gradcam_custom_aver.py: custom.pthの平均gradcam出力
  gradcam_resnet18_aver.py: resnet18_looks_classifier.pthの平均gradcam出力
  gradcam_resnet34_aver.py: resnet34_looks_classifier.pthの平均gradcam出力
  gradcam_resnet50_aver.py: resnet50_looks_classifier.pthの平均gradcam出力
  gradcam_efficientnetb0_aver.py: efficientnetb0_looks_classifier.pthの平均gradcam出力
  個別gradcam
  gradcam_custom.py: custom.pthの個別gradcam出力
  gradcam_resnet18.py: resnet18_looks_classifier.pthの個別gradcam出力
  gradcam_resnet34.py: resnet34_looks_classifier.pthの個別gradcam出力
  gradcam_resnet50.py: resnet50_looks_classifier.pthの個別gradcam出力
  gradcam_efficientnetb0_aver.py: efficientnetb0_looks_classifier.pthの個別gradcam出力
analysis/gradcam_results ディレクトリにあるコードにgradcamの結果が保存される
  average: 平均gradcamが保存されるディレクトリ
  個別gradcamは各モデル名と同じディレクトリの保存

## ディレクトリ構成

- preprocessing: 前処理コード
- image: 画像とcsv
- code: 学習・予測実行コード
- log: 学習・予測結果
- model: モデル
- anaanalysis: gradcam検証
- white: 白背景画像に対するディレクトリ
	- white/FaceDetection: 白背景化コード及び白背景画像
	- white/code: 学習・予測実行コード
	- white/log: 学習・予測結果
	- white/model: モデル
	- white/analysis: gradcam検証
- remove_back: 透明背景画像に対するディレクトリ
  - remove_back/all_imge: 透明背景画像
  - remove_back/csv: 画像のcsv
  - remove_back/code: 学習・予測実行コード
  - remove_back/log: 学習・予測結果
  - remove_back/model: モデル
  - remove_back/analysis: gradcam検証
- hyper: ハイパーパラメータ調整後に対するディレクトリ
  - hyper/code: 学習・予測実行コード
  - hyper/log: 学習・予測結果
  - hyper/model: モデル
  - hyper/analysis: gradcam検証
