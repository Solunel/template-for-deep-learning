# tests/test_metrics.py
import torch
import pytest
from src.metrics import accuracy_fn

def test_accuracy_perfect_score():
    """测试当所有预测都正确时，准确率是否为 1.0"""
    # 预测的类别分别为: 1, 0, 1
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    # 真实的标签
    labels = torch.tensor([1, 0, 1])
    assert accuracy_fn(outputs, labels) == 1.0

def test_accuracy_zero_score():
    """测试当所有预测都错误时，准确率是否为 0.0"""
    # 预测的类别分别为: 1, 0, 1
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    # 真实的标签与预测完全相反
    labels = torch.tensor([0, 1, 0])
    assert accuracy_fn(outputs, labels) == 0.0

def test_accuracy_partial_score():
    """测试当部分预测正确时，准确率是否计算正确"""
    # 预测的类别分别为: 1, 0, 1, 0
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
    # 真实的标签中，第一个和最后一个是正确的 (1, 0)
    labels = torch.tensor([1, 1, 0, 0])
    # 4个样本中有2个正确，准确率应为 0.5
    assert accuracy_fn(outputs, labels) == pytest.approx(0.5)