# 顔画像の背景を白色にするプログラム

## 概要
有名人と一般人の顔画像はどのような特徴の差があるのかを学習するモデルを構築した。
機械学習を実行した後に，その機械学習のモデルがどこを見ているのかをヒートマップで表したものがある。そのヒートマップを見てると，モデルは顔ではなく顔以外の背景部分を見て学習をしていた。顔を見て，学習を行うようにモデルを改善したい。特定の区域，つまり顔だけを学習するにはどうすれば良いか？例えば顔のみを切り抜く，白背景にするなど，その対処を考えた。

長瀬が作成したのが，背景を白色にしてくれるプログラムだ。

### 実行方法
amane内でも同様に，`source FaceDetection/bin/activate`を実行する。
example:
```
e235221@amane:~/FairFace/FaceDetection% source FaceDetection/bin/activate
(FaceDetection) e235221@amane:~/FairFace/FaceDetection% pwd
/home/student/e23/e235221/FairFace/FaceDetection
(FaceDetection) e235221@amane:~/FairFace/FaceDetection% 
```


### 環境設定
```
pip install mediapipe opencv-python numpy

pip install opencv-python
```
