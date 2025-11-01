# 通用深度学习项目模板 (v2.3 - "传道之作")

这是一个高度模块化、配置驱动、经过专业审阅并对新手极其友好的PyTorch项目模板。它旨在成为任何深度学习任务的“黄金标准”起点，不仅提供强大的工具，更传授专业的工程思想。

## ✨ 核心特性

- **配置驱动**: 所有实验参数（路径、超参数、模型结构）均在 `config.yaml` 中统一管理。
- **模块化设计**: 职责分离，`模型`、`数据`、`训练引擎`、`评估指标` 各司其职，易于扩展。
- **专业训练流程**: 内置步数驱动训练、学习率调度（预热+余弦衰减）、早停法、以及检查点管理。
- **可复现性**: 通过全局随机种子和固定的数据集划分，确保实验结果的一致性。
- **质量保证**: 包含单元测试框架 (`pytest`)，确保核心逻辑的正确性。
- **深度实验追踪**: 集成 TensorBoard，自动记录损失、评估指标，并关联超参数，便于分析。
- **新手友好**: 提供详尽的引导式注释、安全的默认配置示例、以及本`README`文档，大幅降低上手门槛。

## 📂 项目结构

```
generic_dl_template/
├── README.md                 # <-- 你正在阅读的指南
├── config.yaml               # 实验的总控制中心 (你需要修改!)
├── requirements.txt          # 项目依赖清单
└── src/                      # 所有源代码
│   ├── train.py              # 训练入口脚本
│   ├── predict.py            # 预测入口脚本
│   ├── model.py              # 【你需要实现!】模型定义
│   ├── dataset.py            # 【你需要实现!】数据加载
│   ├── engine.py             # (通用) 核心训练引擎
│   ├── metrics.py            # 【你需要实现!】评估指标
│   └── utils.py              # (通用) 工具函数
└── tests/                    # 测试用例
    └── test_metrics.py       # (可选) 为你的指标编写测试
```

## 🚀 快速上手指南

**第1步: 环境设置**

```bash
# 克隆项目
git clone <your-repo-url>
cd generic_dl_template

# 创建并激活Python虚拟环境 (强烈推荐)
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 安装所有必需的库
pip install -r requirements.txt
```

**第2步: 准备你的数据**

将你的数据集（如图片、文本、特征文件等）放入根目录下的 `data/` 文件夹。模板中的 `src/dataset.py` 提供了一个使用`ImageFolder`的示例，假设数据结构如下：
```
data/
  train/
    class_a/
      1.jpg, 2.jpg, ...
    class_b/
      ...
  test/
    unknown/  (或任何子文件夹)
      1001.jpg, 1002.jpg, ...
```

**第3步: 配置你的实验 (`config.yaml`)**

打开 `config.yaml` 文件。这是你进行实验的**唯一**需要修改的文件。
- **首先**，参考文件顶部的注释，选择一套适合你硬件的预设配置（如 `lightweight_cpu_config`），将其参数复制到文件的主体部分。
- **然后**，根据你的任务修改 `TODO` 标记的参数（如路径、类别数等）。

**第4步: 实现你的核心逻辑 (三大“填空题”)**

你需要根据你的任务，完成以下三个文件中标记为 `TODO` 的部分：

1.  **`src/dataset.py`**: 实现 `YourDataset` 类，告诉程序如何加载你自己的数据。
2.  **`src/model.py`**: 实现 `YourModel` 类，定义你自己的神经网络结构。
3.  **`src/metrics.py`**: 实现你自己的评估函数，例如 `f1_score` 或 `MSE`。

我们在这些文件中提供了详尽的注释和针对常见任务（如图像分类）的示例代码来引导你。

**第5步: (可选但推荐) 编写并运行测试**

如果你在 `src/metrics.py` 中添加了新函数，建议先在 `tests/test_metrics.py` 中为它编写一个简单的测试用例，以确保其正确性。然后运行：
```bash
pytest
```
看到“passed”意味着你的核心工具是可靠的。

**第6步: 开始训练！**

```bash
python src/train.py
```
训练过程的日志会显示在终端，同时详细的图表会保存在 `logs/` 目录下（具体路径由`config.yaml`定义）。

**第7步: 分析你的实验**

```bash
# 启动 TensorBoard 服务
tensorboard --logdir logs
```
在浏览器中打开显示的链接 (通常是 `http://localhost:6006`)，进入 `HPARAMS` 标签页，你将看到超参数与性能指标的关联分析。

**第8步: 进行预测**

训练完成后，最佳模型会保存在 `checkpoints/` 目录下。运行预测脚本：
```bash
python src/predict.py
```
预测结果将保存在 `prediction.csv` (可在 `config.yaml` 中修改) 中。