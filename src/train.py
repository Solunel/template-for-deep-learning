# src/train.py

# --- 导入必要的库 ---
import logging  # 日志记录库
import torch  # PyTorch 核心库
import torch.nn as nn  # PyTorch 神经网络模块
from dataset import get_dataloaders  # 从我们自己写的 dataset.py 中导入函数
from engine import Engine  # 从我们自己写的 engine.py 中导入 Engine 类
from model import build_model  # 从我们自己写的 model.py 中导入函数
from utils import init_env, load_config, get_cosine_schedule_with_warmup, resolve_class_map  # 从 utils.py 导入工具函数
import metrics as metrics_module  # 将我们自己写的 metrics.py 导入，并起一个别名，避免命名冲突

# 获取一个名为 __name__ (即 'src.train') 的 logger 实例
logger = logging.getLogger(__name__)


def train():
    """主训练函数，负责组装所有组件并启动训练。"""

    # --- 步骤 1: 加载配置并初始化环境 ---
    # 从 "config.yaml" 文件中加载所有配置项
    config = load_config("config.yaml")
    # 根据配置初始化环境，包括设置随机种子、创建目录、配置日志等
    init_env(config)
    # 决定使用哪种设备：如果 CUDA (NVIDIA GPU) 可用，则使用 "cuda"，否则使用 "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 记录日志，告知用户当前使用的设备
    logger.info(f"使用设备: {device}")

    # --- 步骤 2: 准备数据加载器 ---
    logger.info("正在准备数据加载器...")
    # 调用 get_dataloaders 函数来创建训练和验证的数据加载器
    # train_dataset 对象后续会用来获取类别信息
    train_loader, dev_loader, train_dataset = get_dataloaders(config)
    logger.info("数据加载器准备完成。")

    # --- 步骤 3: 解析类别信息 ---
    # 从训练数据集中解析出 "类别名 -> 索引" 的映射 和 总类别数
    _, n_classes = resolve_class_map(config, train_dataset)

    # --- 步骤 4: 构建模型 ---
    # 使用 build_model 工厂函数创建模型实例，并将其移动到指定设备 (GPU/CPU)
    net = build_model(config, n_classes).to(device)

    # --- 步骤 5: 定义损失函数 ---
    # 对于多分类问题，交叉熵损失 (CrossEntropyLoss) 是最常用的损失函数
    # 它内部已经包含了 Softmax 操作，所以模型输出的原始 logits 可以直接传入
    criterion = nn.CrossEntropyLoss()

    # --- 步骤 6: 定义优化器 ---
    # AdamW 是 Adam 优化器的一个改进版本，通常在 Transformer 等模型中表现更好，也是一个稳健的选择
    # net.parameters() 会返回模型中所有需要训练的参数 (权重和偏置)
    # lr 是从配置文件中读取的学习率
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['params']['learning_rate'])

    # --- 步骤 7: 定义学习率调度器 ---
    # 使用我们自定义的 "预热 + 余弦衰减" 学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,  # 关联的优化器
        num_warmup_steps=config['params']['warmup_steps'],  # 预热阶段的步数
        num_training_steps=config['params']['total_steps']  # 总的训练步数
    )

    # --- 步骤 8: 定义评估指标函数字典 ---
    # 创建一个字典，键是指标的名称 (会显示在日志中)，值是对应的计算函数
    # 这样设计可以很方便地添加或删除评估指标
    metric_fns = {'accuracy': metrics_module.accuracy_fn}

    # --- 步骤 9: 实例化引擎 ---
    # 创建核心引擎 Engine 的一个实例，把所有之前创建的组件 (模型、配置、设备、损失函数等) 都传入
    engine = Engine(
        net=net, config=config, device=device, criterion=criterion,
        optimizer=optimizer, scheduler=scheduler, metric_fns=metric_fns
    )

    # --- 步骤 10: 启动训练 ---
    # 调用引擎的 run_training 方法，并传入训练和验证数据加载器，开始整个训练流程
    engine.run_training(train_loader, dev_loader)

    # --- 训练结束 ---
    logger.info("训练流程完成!")
    # 提示用户如何使用 TensorBoard 查看训练过程
    logger.info(f"要查看训练过程，请在项目根目录运行: tensorboard --logdir {config['paths']['log_dir']}")


# --- Python 的主程序入口 ---
# 当你直接运行 `python src/train.py` 时，`if __name__ == '__main__':` 下的代码块会被执行
if __name__ == '__main__':
    # 调用主训练函数
    train()