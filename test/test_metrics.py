# tests/test_metrics.py

# --- 导入必要的库 ---
import torch  # PyTorch 核心库
import pytest  # 一个强大的 Python 测试框架
from src.metrics import accuracy_fn  # 从我们自己写的 metrics.py 中导入要测试的函数


# --- 测试用例 1: 完美得分 ---
def test_accuracy_perfect_score():
    """测试当所有预测都正确时，准确率是否为 1.0"""
    # 构造模型的输出 (logits)。每一行代表一个样本，每一列代表一个类别。
    # 第1行: [0.1, 0.9] -> 模型预测为类别 1
    # 第2行: [0.8, 0.2] -> 模型预测为类别 0
    # 第3行: [0.4, 0.6] -> 模型预测为类别 1
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

    # 构造真实的标签，与上面的模型预测完全一致
    labels = torch.tensor([1, 0, 1])

    # `assert` 语句用来断言一个条件为真。如果条件为假，测试将失败。
    # 我们断言 accuracy_fn 的计算结果应该等于 1.0
    assert accuracy_fn(outputs, labels) == 1.0


# --- 测试用例 2: 零分 ---
def test_accuracy_zero_score():
    """测试当所有预测都错误时，准确率是否为 0.0"""
    # 构造模型的输出 (同上)
    # 预测的类别分别为: 1, 0, 1
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

    # 构造真实的标签，与上面的模型预测完全相反
    labels = torch.tensor([0, 1, 0])

    # 断言准确率应该等于 0.0
    assert accuracy_fn(outputs, labels) == 0.0


# --- 测试用例 3: 部分得分 ---
def test_accuracy_partial_score():
    """测试当部分预测正确时，准确率是否计算正确"""
    # 构造模型的输出 (4个样本)
    # 预测的类别分别为: 1, 0, 1, 0
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])

    # 构造真实的标签
    # 真实标签为: 1, 1, 0, 0
    # 对比预测 [1, 0, 1, 0] 和真实 [1, 1, 0, 0]:
    # - 第1个样本: 预测 1, 真实 1 (正确)
    # - 第2个样本: 预测 0, 真实 1 (错误)
    # - 第3个样本: 预测 1, 真实 0 (错误)
    # - 第4个样本: 预测 0, 真实 0 (正确)
    labels = torch.tensor([1, 1, 0, 0])

    # 4个样本中有2个正确，准确率应为 0.5
    # `pytest.approx(0.5)` 用于比较浮点数，可以处理微小的精度误差
    assert accuracy_fn(outputs, labels) == pytest.approx(0.5)