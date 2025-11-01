# src/predict.py

# --- 导入必要的库 ---
import logging  # 日志记录库
import torch  # PyTorch 核心库
from dataset import get_test_loader  # 从 dataset.py 中导入测试数据加载器函数
from engine import Engine  # 从 engine.py 中导入核心引擎类
from model import build_model  # 从 model.py 中导入模型构建函数
from utils import init_env, load_config, save_predictions, load_class_map  # 从 utils.py 导入工具函数

# 获取一个名为 __name__ (即 'src.predict') 的 logger 实例
logger = logging.getLogger(__name__)


def predict():
    """主预测函数，负责加载最佳模型并生成提交文件。"""

    # --- 步骤 1: 加载配置并初始化环境 ---
    # 从 "config.yaml" 文件中加载所有配置项
    config = load_config("config.yaml")
    # 根据配置初始化环境 (主要是为了日志和目录)
    init_env(config)
    # 决定使用哪种设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 记录日志，告知用户当前使用的设备
    logger.info(f"使用设备: {device}")

    # --- 步骤 2: 准备测试数据加载器 ---
    # 调用 get_test_loader 函数来创建测试集的数据加载器
    test_loader = get_test_loader(config)

    # --- 使用 try...except 块来捕获可能发生的错误 ---
    # 比如类别映射文件不存在，这样程序可以优雅地退出并给出提示
    try:
        # --- 步骤 3: 加载类别映射 ---
        # 加载训练时保存的 "索引 -> 类别名" 的映射 和 总类别数
        # 这是必需的，因为模型预测输出的是类别索引 (如 0, 1, 2)，我们需要把它转换回原始的类别名
        idx_to_class, n_classes = load_class_map(config)

        # --- 步骤 4: 构建模型 ---
        # 再次构建一个和训练时结构完全相同的模型实例，并将其移动到指定设备
        net = build_model(config, n_classes).to(device)

        # --- 步骤 5: 实例化引擎 ---
        # 再次实例化引擎。注意，在预测时，我们不需要损失函数(criterion)、优化器(optimizer)等，
        # 所以这些参数都传入 None。
        engine = Engine(
            net=net, config=config, device=device,
            criterion=None, optimizer=None, scheduler=None, metric_fns=None
        )

        # --- 步骤 6: 执行预测 ---
        # 调用引擎的 predict 方法，它会加载最佳模型权重并对测试集进行预测
        # 返回两个列表：ids (文件名) 和 preds (预测的类别索引)
        ids, preds = engine.predict(test_loader)

        # --- 步骤 7: 格式化并保存结果 ---
        # 使用列表推导式，将预测的类别索引 `p` 通过 `idx_to_class` 映射转换回类别名
        pred_names = [idx_to_class[p] for p in preds]
        # 使用 zip 函数将文件名列表和预测的类别名列表打包成 (id, name) 对的列表
        results = list(zip(ids, pred_names))
        # 调用 save_predictions 函数将结果保存到配置文件指定的输出路径
        save_predictions(results, config['paths']['output_path'])
        # 记录日志，表示预测完成
        logger.info("预测流程完成!")

    except FileNotFoundError as e:
        # 如果在 try 块中发生了 FileNotFoundError (比如 class_map.json 找不到)
        # 记录这个错误
        logger.error(e)
        # 给出一个清晰的提示，告诉用户该如何解决这个问题
        logger.error("无法执行预测。请确保已成功运行训练，并生成了所需的模型和类别映射文件。")


# --- Python 的主程序入口 ---
if __name__ == '__main__':
    # 调用主预测函数
    predict()