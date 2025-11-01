# src/predict.py
import logging
import torch
from dataset import prepare_dataloader
from engine import Engine
from model import build_model
from utils import init_env, load_config, save_predictions, load_class_map

logger = logging.getLogger(__name__)


def predict():
    """主预测函数，负责加载最佳模型并生成提交文件。"""
    # 1. 加载配置并初始化环境
    config = load_config("config.yaml")
    init_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 2. 准备测试数据加载器
    test_loader, _ = prepare_dataloader(config, 'test')

    try:
        # 3. 加载类别映射
        idx_to_class, n_classes = load_class_map(config)

        # 4. 构建模型
        net = build_model(config, n_classes).to(device)

        # 5. 实例化引擎 (预测时不需要损失、优化器等)
        engine = Engine(
            net=net, config=config, device=device,
            criterion=None, optimizer=None, scheduler=None, metric_fns=None
        )

        # 6. 执行预测
        ids, preds = engine.predict(test_loader)

        # 7. 格式化并保存结果
        pred_names = [idx_to_class[p] for p in preds]
        results = list(zip(ids, pred_names))
        save_predictions(results, config['paths']['output_path'])
        logger.info("预测流程完成!")

    except FileNotFoundError as e:
        logger.error(e)
        logger.error("无法执行预测。请确保已成功运行训练，并生成了所需的模型和类别映射文件。")


if __name__ == '__main__':
    predict()