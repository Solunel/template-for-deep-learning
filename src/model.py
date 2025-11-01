# src/model.py

# --- 导入必要的库 ---
import torch  # PyTorch 深度学习框架
import torch.nn as nn  # PyTorch 神经网络模块，包含了所有网络层、激活函数等的定义
from typing import Dict, Any  # Python 类型提示
from torchvision import models  # PyTorch 计算机视觉库，包含了许多预训练好的模型


# --- 模型类的定义 ---
class YourModel(nn.Module):
    """
    模型定义模板，以一个基于ResNet18的图像分类器为例。
    这个类继承自 `nn.Module`，这是所有 PyTorch 模型的基类。
    你可以直接使用它，或者将其替换为你自己的模型。
    """

    # --- 初始化方法 ---
    def __init__(self, params, n_classes):
        """
        当创建这个类的实例时，这个方法会被调用，用于定义模型的结构。
        Args:
            params (Dict[str, Any]): 从 config.yaml 中读取的 'params' 部分的参数字典。
            n_classes (int): 最终的分类类别数。
        """
        # 调用父类 `nn.Module` 的初始化方法，这是必须的步骤
        super().__init__()

        # 从参数字典中获取 'pretrained' 的值，如果不存在则默认为 True
        # pretrained=True 表示加载在 ImageNet 数据集上预训练过的权重
        pretrained = params.get('pretrained', True)
        # 从参数字典中获取 'dropout_rate' 的值，如果不存在则默认为 0.1
        dropout_rate = params.get('dropout_rate', 0.1)

        # 1. 加载预训练的 ResNet18 模型
        #    `models.resnet18` 会返回一个 ResNet18 模型实例
        self.base_model = models.resnet18(pretrained=pretrained)

        # 2. 获取原始全连接层(fc)的输入特征数
        #    在 ResNet18 中，最后一个卷积层输出的特征图展平后，会送入一个全连接层
        #    `self.base_model.fc.in_features` 可以自动获取这个连接层的输入维度 (对于 ResNet18 是 512)
        in_features = self.base_model.fc.in_features

        # 3. 替换原始的全连接层为一个新的、符合我们任务的分类头
        #    原始的 ResNet18 是为 ImageNet (1000类) 设计的，我们需要把它换成我们自己的类别数 `n_classes`
        #    我们使用 `nn.Sequential` 来将多个层按顺序组合在一起
        self.base_model.fc = nn.Sequential(
            # a. 添加一个 Dropout 层，以一定的概率 `p` 随机将输入单元置为0，用于防止过拟合
            nn.Dropout(p=dropout_rate),
            # b. 添加一个新的线性(全连接)层，输入维度是 `in_features`，输出维度是我们的类别数 `n_classes`
            nn.Linear(in_features, n_classes)
        )

    # --- 前向传播方法 ---
    def forward(self, x):
        """
        这个方法定义了数据在网络中的流动方式（前向传播）。
        当调用 `model(input_data)` 时，这个方法会被自动执行。
        Args:
            x (torch.Tensor): 输入的图像数据，预期形状为 (batch_size, 3, height, width)。
        Returns:
            torch.Tensor: 模型的原始输出 (Logits)，形状为 (batch_size, n_classes)。
        """
        # 直接调用我们修改过的 `base_model` 来处理输入 `x` 并返回结果
        return self.base_model(x)


# --- 模型构建函数 (工厂模式) ---
def build_model(config, n_classes):
    """
    这是一个工厂函数，用于根据配置构建并返回模型实例。
    使用工厂函数的好处是，如果未来你想根据配置选择不同的模型 (比如 ResNet34, VGG)，
    你只需要在这个函数里添加 `if/else` 判断，而不需要修改 `train.py`。
    """
    # 创建 YourModel 类的一个实例
    model = YourModel(
        params=config.get('params', {}),  # 传入 'params' 配置
        n_classes=n_classes               # 传入类别数
    )
    # 返回创建好的模型实例
    return model