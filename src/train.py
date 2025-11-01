# src/train.py (V2.4 - 持久化类别映射)
import logging
import torch
import torch.nn as nn
import json
from pathlib import Path
from dataset import prepare_dataloader
from engine import Engine
from model import build_model
from utils import init_env, load_config, get_cosine_schedule_with_warmup
import metrics as metrics_module

logger = logging.getLogger(__name__)


def train():
    config = load_config("config.yaml")
    init_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    train_loader, train_dataset = prepare_dataloader(config, 'train')
    dev_loader, _ = prepare_dataloader(config, 'dev')

    # 1. 模型 (从数据集中动态获取类别数)
    # 对于ImageFolder, 类别信息在 .dataset.dataset.classes (因为有random_split包装)
    n_classes = len(train_dataset.dataset.dataset.classes)
    class_to_idx = train_dataset.dataset.dataset.class_to_idx
    net = build_model(config, n_classes).to(device)

    # (新增) 持久化类别映射，确保训练和预测一致
    ckpt_dir = Path(config['paths']['checkpoint_dir'])
    class_map_path = ckpt_dir / config['paths']['class_map_file']
    with open(class_map_path, 'w') as f:
        json.dump(class_to_idx, f, indent=4)
    logger.info(f"类别映射已保存至: {class_map_path}")

    # 2. 损失函数 (源: torch.nn)
    criterion = nn.CrossEntropyLoss()

    # 3. 优化器 (源: torch.optim)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['hparams']['learning_rate'])

    # 4. 学习率调度器 (源: torch.optim.lr_scheduler 或自定义)
    scheduler = get_cosine_schedule_with_warmup(...)

    # 5. 评估指标
    metric_fns = {'accuracy': metrics_module.accuracy_fn}

    engine = Engine(...)
    engine.run_training()

    logger.info("训练流程完成!")
    logger.info(f"要查看训练过程，请在项目根目录运行: tensorboard --logdir {config['paths']['log_dir']}")


if __name__ == '__main__':
    train()