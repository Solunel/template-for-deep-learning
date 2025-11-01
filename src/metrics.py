# src/metrics.py

# --- 导入必要的库 ---
import torch  # PyTorch 深度学习框架


# --- 准确率计算函数 ---
def accuracy_fn(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算分类任务的准确率。

    Args:
        outputs (torch.Tensor): 模型的原始输出 (logits)，形状为 (N, C)，
                                N 是批次大小，C 是类别数。
        labels (torch.Tensor): 真实的标签，形状为 (N,)。

    Returns:
        float: 该批次的平均准确率。
    """
    # 1. 使用 torch.max 在类别的维度(dim=1)上寻找最大值的索引。
    #    这个索引就代表了模型预测的类别。
    #    torch.max 会返回一个元组 (values, indices)，我们只需要索引，所以用 _ 忽略第一个返回值。
    _, preds = torch.max(outputs, 1)

    # 2. 比较预测的类别 `preds` 和真实的标签 `labels` 是否相等。
    #    `preds == labels` 会返回一个布尔类型的 Tensor，例如 [True, False, True, ...]。

    # 3. .float() 将布尔 Tensor 转换为浮点数 Tensor，即 [1.0, 0.0, 1.0, ...]。

    # 4. .mean() 计算这个 Tensor 的平均值，即 (正确的数量 / 总数量)，这就是准确率。

    # 5. .item() 从只有一个元素的 Tensor 中提取出 Python 的标量数值。
    return (preds == labels).float().mean().item()