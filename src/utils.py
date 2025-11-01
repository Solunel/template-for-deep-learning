# src/utils.py

# --- 导入必要的库 ---
import logging  # 日志记录库
import torch  # PyTorch 核心库
import random  # Python 内置的随机数库
import numpy as np  #
import yaml  # 用于读写 YAML 文件的库
import json  # 用于读写 JSON 文件的库
import csv  # 用于读写 CSV 文件的库
from pathlib import Path  # 面向对象的路径操作库
from typing import Dict, Any, Tuple, List  # Python 类型提示
from torch.optim import Optimizer  # 从 torch.optim 中导入优化器基类，用于类型提示
from torch.optim.lr_scheduler import LambdaLR  # 一种灵活的学习率调度器
import math  # Python 内置的数学库

# 获取一个名为 __name__ (即 'src.utils') 的 logger 实例
logger = logging.getLogger(__name__)


def validate_config(config):
    """校验配置文件中必需的键是否存在，实现“尽早失败”(fail-fast)策略。"""
    # 定义在 config['paths'] 中必须存在的键
    required_paths = ['data_root', 'checkpoint_dir', 'log_dir', 'output_path', 'class_map_file']
    # 遍历这个列表
    for path in required_paths:
        # 检查 'paths' 键是否存在，以及每个必需的路径键是否存在
        if 'paths' not in config or path not in config['paths']:
            # 如果任何一个不存在，就立即抛出 KeyError 异常，并给出清晰的错误信息
            raise KeyError(f"配置文件缺失必需的路径: paths.{path}")
    # 如果所有检查都通过，记录一条成功日志
    logger.info("配置文件校验通过。")


def init_env(config):
    """根据配置初始化程序运行环境（日志、目录、随机种子）。"""
    # 首先，校验配置文件是否合法
    validate_config(config)

    # --- 配置日志系统 ---
    logging.basicConfig(
        level=logging.INFO,  # 设置日志记录的最低级别为 INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 定义日志输出的格式
        datefmt='%Y-%m-%d %H:%M:%S',  # 定义日期的格式
    )

    # --- 创建必要的目录 ---
    # parents=True: 如果父目录不存在，也一并创建
    # exist_ok=True: 如果目录已经存在，不要抛出错误
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)

    # --- 设置随机种子以保证实验的可复现性 ---
    # 从配置中获取随机种子，如果没定义，则默认为 42
    seed = config.get('seed', 42)
    # 为 Python 内置的 random 模块设置种子
    random.seed(seed)
    # 为 NumPy 设置种子
    np.random.seed(seed)
    # 为 PyTorch 的 CPU 操作设置种子
    torch.manual_seed(seed)
    # 如果 CUDA (GPU) 可用，为所有 GPU 设置种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 记录日志，表示环境初始化完成
    logger.info("环境初始化完成。")


def load_config(config_path = "config.yaml"):
    """加载并解析 YAML 格式的配置文件。"""
    logger.info(f"正在从 {config_path} 加载配置...")
    # 使用 'with open' 语法打开文件，可以确保文件在使用后被正确关闭
    with open(config_path, "r", encoding='utf-8') as f:
        # 使用 yaml.safe_load() 来解析 YAML 文件内容，并返回一个 Python 字典
        return yaml.safe_load(f)


def save_predictions(results, path) -> None:
    """将预测结果保存为 Kaggle 要求的 CSV 格式。"""
    # 打开指定的路径用于写入 ('w')，newline='' 是为了防止写入空行
    with open(path, 'w', newline='') as f:
        # 创建一个 CSV 写入器对象
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['Id', 'Category'])
        # 写入所有结果行
        writer.writerows(results)
    # 记录日志，告知用户文件已保存
    logger.info(f"预测结果已保存至: {path}")


def resolve_class_map(config, train_dataset):
    """解决类别映射：存在则加载，不存在则从训练集创建并保存。"""
    # 获取检查点目录
    ckpt_dir = Path(config['paths']['checkpoint_dir'])
    # 构建类别映射文件的完整路径
    class_map_path = ckpt_dir / config['paths']['class_map_file']

    # 检查这个文件是否已经存在
    if class_map_path.exists():
        # 如果存在，就加载它
        logger.info(f"正在从 {class_map_path} 加载已有的类别映射...")
        with open(class_map_path, 'r') as f:
            # 使用 json.load() 从文件中读取并解析 JSON 数据
            class_to_idx = json.load(f)
    else:
        # 如果不存在，就从训练数据集中创建
        logger.info("未发现类别映射文件，将从训练集创建...")
        # `train_dataset` 是 Subset 对象, `dataset` 属性是原始的 YourDataset 对象, 再 `dataset` 是 ImageFolder
        class_to_idx = train_dataset.dataset.dataset.class_to_idx
        # 将创建的映射写入文件，以便下次使用和预测时加载
        with open(class_map_path, 'w') as f:
            # 使用 json.dump() 将 Python 字典写入文件
            # indent=4 让 JSON 文件格式化，更易读
            json.dump(class_to_idx, f, indent=4)
        logger.info(f"新的类别映射已创建并保存至: {class_map_path}")

    # 计算类别的总数
    n_classes = len(class_to_idx)
    # 返回 "类别名 -> 索引" 的映射字典和类别总数
    return class_to_idx, n_classes


def load_class_map(config):
    """在预测时加载类别映射。"""
    # 获取检查点目录和类别映射文件的路径
    ckpt_dir = Path(config['paths']['checkpoint_dir'])
    class_map_path = ckpt_dir / config['paths']['class_map_file']

    # 检查文件是否存在，如果不存在，这是一个严重错误，因为预测无法进行
    if not class_map_path.exists():
        # 抛出 FileNotFoundError 异常，并给出清晰的错误提示
        raise FileNotFoundError(f"错误: 类别映射文件 {class_map_path} 未找到！请先运行训练以生成该文件。")

    logger.info(f"正在从 {class_map_path} 加载类别映射...")
    # 打开并加载 "类别名 -> 索引" 的映射
    with open(class_map_path, 'r') as f:
        class_to_idx = json.load(f)

    # 预测时需要的是 "索引 -> 类别名" 的映射，所以我们反转字典的键和值
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # 计算类别总数
    n_classes = len(idx_to_class)
    # 返回反转后的映射字典和类别总数
    return idx_to_class, n_classes


def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps,
        num_cycles = 0.5, last_epoch = -1
) -> LambdaLR:
    """创建 "预热(warmup) + 余弦衰减(cosine decay)" 学习率调度器。"""

    # 定义一个函数，这个函数根据当前步数 `current_step` 计算学习率的缩放因子
    def lr_lambda(current_step):
        # --- 预热阶段 ---
        # 如果当前步数小于预热步数
        if current_step < num_warmup_steps:
            # 学习率从 0 线性增加到 1
            return float(current_step) / float(max(1, num_warmup_steps))

        # --- 余弦衰减阶段 ---
        # 计算在衰减阶段的进度 (从 0 到 1)
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # 使用余弦函数计算缩放因子。
        # 当 progress=0, cos(0)=1。当 progress=1, cos(pi)=-1。
        # 0.5 * (1 + cos(...)) 会使学习率从 1 平滑地下降到 0。
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    # 返回一个 LambdaLR 调度器实例，它会在每个 step 中调用我们定义的 lr_lambda 函数来更新学习率
    return LambdaLR(optimizer, lr_lambda, last_epoch)