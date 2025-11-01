# src/model.py
import torch
import torch.nn as nn
from typing import Dict, Any
from torchvision import models


class YourModel(nn.Module):
    """
    模型定义模板，以一个基于ResNet18的图像分类器为例。
    你可以直接使用它，或者将其替换为你自己的模型。
    """

    def __init__(self, model_params: Dict[str, Any], n_classes: int):
        super().__init__()

        pretrained = model_params.get('pretrained', True)
        dropout_rate = model_params.get('dropout_rate', 0.1)

        self.base_model = models.resnet18(pretrained=pretrained)
        in_features = self.base_model.fc.in_features

        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


def build_model(config: Dict[str, Any], n_classes: int) -> nn.Module:
    """工厂函数，用于根据配置构建并返回模型实例。"""
    model = YourModel(
        model_params=config.get('model_params', {}),
        n_classes=n_classes
    )
    return model