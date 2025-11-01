import logging
import torch
import json
from pathlib import Path
from dataset import prepare_dataloader
from engine import Engine
from model import build_model
from utils import init_env, load_config, save_predictions

logger = logging.getLogger(__name__)


def predict():
    config = load_config("config.yaml")
    init_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    test_loader, _ = prepare_dataloader(config, 'test')

    # (修改) 从持久化的json文件加载类别映射，不再依赖训练集
    ckpt_dir = Path(config['paths']['checkpoint_dir'])
    class_map_path = ckpt_dir / config['paths']['class_map_file']
    if not class_map_path.exists():
        logger.error(f"错误: 类别映射文件 {class_map_path} 未找到！请先运行训练以生成该文件。")
        return

    with open(class_map_path, 'r') as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    n_classes = len(idx_to_class)

    net = build_model(config, n_classes).to(device)
    engine = Engine(net=net, config=config, device=device, test_data=test_loader)

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