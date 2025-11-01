# src/utils.py

import logging
import torch
import random
import numpy as np
import yaml
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math

logger = logging.getLogger(__name__)


def validate_config(config: Dict[str, Any]) -> None:
    """校验配置文件中必需的键是否存在。"""
    required_paths = ['data_root', 'checkpoint_dir', 'log_dir', 'output_path']
    for path in required_paths:
        if path not in config['paths']:
            raise KeyError(f"配置文件缺失必需的路径: paths.{path}")
    logger.info("配置文件校验通过。")


def init_env(config: Dict[str, Any]) -> None:
    """根据配置初始化程序运行环境（日志、目录、随机种子）。"""
    validate_config(config)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)

    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("环境初始化完成。")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载并解析 YAML 格式的配置文件。"""
    logger.info(f"正在从 {config_path} 加载配置...")
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_predictions(results: List[Tuple[str, str]], path: str) -> None:
    """将预测结果保存为 Kaggle 要求的 CSV 格式。"""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Category'])
        writer.writerows(results)
    logger.info(f"预测结果已保存至: {path}")


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    """创建 "预热 + 余弦衰减" 学习率调度器。"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)