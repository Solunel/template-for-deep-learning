# tests/test_metrics.py
import torch
import pytest
from src.metrics import accuracy_fn

def test_accuracy_perfect_score():
    """测试当所有预测都正确时，准确率是否为 1.0"""
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    labels = torch.tensor()
    assert accuracy_fn(outputs, labels) == 1.0

def test_accuracy_zero_score():
    """测试当所有预测都错误时，准确率是否为 0.0"""
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    labels = torch.tensor()
    assert accuracy_fn(outputs, labels) == 0.0

def test_accuracy_partial_score():
    """测试当部分预测正确时，准确率是否计算正确"""
    outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
    labels = torch.tensor()
    assert accuracy_fn(outputs, labels) == pytest.approx(0.5)