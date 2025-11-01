# src/predict.py
import logging
import torch
from dataset import prepare_dataloader
from engine import Engine
from model import build_model
from utils import init_env, load_config, save_predictions

logger = logging.getLogger(__name__)

def predict():
    """主预测函数"""
    config = load_config("config.yaml")
    init_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    test_loader, test_dataset = prepare_dataloader(config, 'test')

    # 为了获取类别到名称的映射
    train_loader, train_dataset_for_map = prepare_dataloader(config, 'train')
    idx_to_class = {v: k for k, v in train_dataset_for_map.dataset.dataset.class_to_idx.items()}

    n_classes = len(idx_to_class)
    net = build_model(config, n_classes).to(device)

    engine = Engine(
        net=net, config=config, device=device, test_data=test_loader
    )

    try:
        ids, preds = engine.predict()
        pred_names = [idx_to_class[p] for p in preds]
        results = list(zip(ids, pred_names))
        save_predictions(results, config['paths']['output_path'])
        logger.info("预测流程完成!")
    except FileNotFoundError as e:
        logger.error(e)
        logger.error("无法执行预测。")

if __name__ == '__main__':
    predict()