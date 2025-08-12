"""
models.py

このモジュールは、指定したアーキテクチャとクラス数に応じて PyTorch モデルを構築するための関数を提供します。
主な機能：
    - ResNet18 / ResNet34 / ResNet50 / EfficientNet-B0 のモデル構築
    - ImageNet 事前学習済み重みの使用有無を切り替え可能
    - 出力層を指定クラス数に合わせて再構成

依存ライブラリ：
    torchvision, torch.nn
"""

import torch.nn as nn
from torchvision import models

def build_model(arch: str, num_classes: int = 2, pretrained: bool = False):
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch in ("efficientnet_b0", "efficientnetb0"):
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m