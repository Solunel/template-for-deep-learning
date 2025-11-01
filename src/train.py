# src/train.py
import logging
import torch
import torch.nn as nn
from dataset import prepare_dataloader
from engine import Engine
from model import build_model
from utils import init_env, load_config, get_cosine_schedule_with_warmup, resolve_class_map
import metrics as metrics_module

logger = logging.getLogger(__name__)


def train():
    """主训练函数，负责组装所有组件并启动训练。"""
    # 1. 加载配置并初始化环境
    config = load_config("config.yaml")
    init_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 2. 准备数据加载器
    train_loader, train_dataset = prepare_dataloader(config, 'train')
    dev_loader, _ = prepare_dataloader(config, 'dev')

    # 3. 解析类别信息
    _, n_classes = resolve_class_map(config, train_dataset)

    # 4. 构建模型
    net = build_model(config, n_classes).to(device)

    # 5. 定义损失函数 (源: torch.nn)
    criterion = nn.CrossEntropyLoss()

    # 6. 定义优化器 (源: torch.optim)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['params']['learning_rate'])

    # 7. 定义学习率调度器 (源: torch.optim.lr_scheduler 或自定义)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['params']['warmup_steps'],
        num_training_steps=config['params']['total_steps']
    )

    # 8. 定义评估指标函数字典
    metric_fns = {'accuracy': metrics_module.accuracy_fn}

    # 9. 实例化引擎
    engine = Engine(
        net=net, config=config, device=device, criterion=criterion,
        optimizer=optimizer, scheduler=scheduler, metric_fns=metric_fns
    )

    # 10. 启动训练
    engine.run_training(train_loader, dev_loader)

    logger.info("训练流程完成!")
    logger.info(f"要查看训练过程，请在项目根目录运行: tensorboard --logdir {config['paths']['log_dir']}")


if __name__ == '__main__':
    train()