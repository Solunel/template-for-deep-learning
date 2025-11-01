import torch

def accuracy_fn(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """计算分类任务的准确率。"""
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean().item()