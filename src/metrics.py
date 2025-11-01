# src/metrics.py
import torch

def accuracy_fn(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算分类任务的准确率。

    Args:
        outputs (torch.Tensor): 模型的原始输出 (logits)，形状为 (N, C)。
        labels (torch.Tensor): 真实的标签，形状为 (N,).

    Returns:
        float: 该批次的平均准确率。
    """
    _, preds = torch.max(outputs, 1)
    # .eq() 比较预测和标签是否相等, .float() 转为浮点数, .mean() 计算平均值, .item() 获取Python标量
    return (preds == labels).float().mean().item()