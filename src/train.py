# src/train.py
import logging
import torch
import torch.nn as nn
from dataset import prepare_dataloader
from engine import Engine
from model import build_model
from utils import init_env, load_config, get_cosine_schedule_with_warmup
import metrics as metrics_module

logger = logging.getLogger(__name__)


def train():
    """主训练函数"""
    config = load_config("config.yaml")
    init_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    train_loader, train_dataset = prepare_dataloader(config, 'train')
    dev_loader, _ = prepare_dataloader(config, 'dev')

    # --- 动态构建组件 ---
    # 1. 模型
    # 对于 ImageFolder, 类别数可以通过 `train_dataset.dataset.dataset.classes` 获取
    n_classes = len(train_dataset.dataset.dataset.classes)
    config['model_params']['n_classes'] = n_classes
    net = build_model(config, n_classes).to(device)

    # 2. 损失函数
    criterion = nn.CrossEntropyLoss()

    # 3. 优化器
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['hparams']['learning_rate'])

    # 4. 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['hparams']['warmup_steps'],
        num_training_steps=config['hparams']['total_steps']
    )

    # 5. 评估指标函数字典
    metric_fns = {'accuracy': metrics_module.accuracy_fn}

    # --- 实例化并启动引擎 ---
    engine = Engine(
        net=net, config=config, device=device, criterion=criterion,
        optimizer=optimizer, scheduler=scheduler, metric_fns=metric_fns,
        train_data=train_loader, dev_data=dev_loader
    )
    engine.run_training()

    logger.info("训练流程完成!")
    logger.info(f"要查看训练过程，请在项目根目录运行: tensorboard --logdir {config['paths']['log_dir']}")


if __name__ == '__main__':
    train()