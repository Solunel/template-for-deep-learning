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

    def __init__(self, params: Dict[str, Any], n_classes: int):
        """
        Args:
            params (Dict[str, Any]): 从 config.yaml 中读取的参数字典。
            n_classes (int): 最终的分类类别数。
        """
        super().__init__()

        pretrained = params.get('pretrained', True)
        dropout_rate = params.get('dropout_rate', 0.1)

        # 加载预训练的ResNet18模型
        self.base_model = models.resnet18(pretrained=pretrained)

        # 获取原始全连接层的输入特征数
        in_features = self.base_model.fc.in_features

        # 替换原始的全连接层为一个新的、符合我们任务的分类头
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 预期形状为 (batch_size, 3, height, width)。
        Returns:
            torch.Tensor: Logits, 形状为 (batch_size, n_classes)。
        """
        return self.base_model(x)


def build_model(config: Dict[str, Any], n_classes: int) -> nn.Module:
    """工厂函数，用于根据配置构建并返回模型实例。"""
    model = YourModel(
        params=config.get('params', {}),
        n_classes=n_classes
    )
    return model