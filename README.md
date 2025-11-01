# 通用深度学习项目模板

这是一个高度模块化、配置驱动、对新手极其友好的PyTorch项目模板。
## ✨ 核心特性

- **大道至简**: 采用扁平化的配置结构，所有参数在`config.yaml`中统一管理，清晰直观。
- **模块化设计**: 职责分离，`模型`、`数据`、`引擎`、`指标` 各司其职，易于扩展。
- **专业训练流程**: 内置步数驱动训练、学习率调度（预热+余弦衰减）、早停法、及检查点管理。
- **可复现性**: 通过全局随机种子和固定的数据集划分，确保实验结果的一致性。
- **质量保证**: 包含单元测试框架 (`pytest`)，确保核心逻辑的正确性。
- **深度实验追踪**: 集成 TensorBoard，自动记录并关联超参数与性能，便于分析。
- **新手友好**: 提供详尽的引导式注释、安全的默认配置、以及本`README`文档。
- **跨平台兼容**: 特别为 Windows 用户提供了清晰的指引和建议。

## 📂 项目结构

```
generic_dl_template/
├── README.md                 # <-- 你正在阅读的指南
├── config.yaml               # 实验的总控制中心 (你需要修改!)
├── requirements.txt          # 项目依赖清单
└── src/                      # 所有源代码
│   ├── __init__.py           # (空文件, 作用见注释)
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

# --- 根据你的操作系统选择以下命令之一 ---
# On Linux/macOS:
source venv/bin/activate
# On Windows (CMD):
venv\Scripts\activate

# 安装所有必需的库
pip install -r requirements.txt
```

**第2步: 准备你的数据 (重要!)**

将你的数据集放入根目录下的 `data/` 文件夹。模板中的 `src/dataset.py` 默认使用`ImageFolder`，它要求**严格的目录结构**：
```
data/
  ├── train/
  │   ├── class_a/
  │   │   ├── 1.jpg
  │   │   └── 2.jpg
  │   └── class_b/
  │       ├── 3.jpg
  │       └── 4.jpg
  └── test/
      └── unknown/  (测试集的子文件夹名可以是任意的)
          ├── 1001.jpg
          └── 1002.jpg
```
如果你使用其他数据格式（如CSV），请相应地修改 `src/dataset.py`。

**第3步: 配置你的实验 (`config.yaml`)**

打开 `config.yaml` 文件。这是你进行实验的**唯一**需要修改的文件。
- 根据你的任务修改 `paths` 和 `params` 中的参数。
- 特别注意 `num_workers`，Windows用户强烈建议保持为 `0`。

**第4步: 实现你的核心逻辑 (三大“填空题”)**

你需要根据你的任务，完成以下三个文件中标记为 `TODO` 的部分：
1.  **`src/dataset.py`**: 如果你的数据不是上述`ImageFolder`结构，请实现 `YourDataset` 类。
2.  **`src/model.py`**: 实现 `YourModel` 类，定义你自己的神经网络结构。
3.  **`src/metrics.py`**: 实现你自己的评估函数。

我们在这些文件中提供了详尽的注释和针对常见任务（如图像分类）的示例代码来引导你。

**第5步: (可选但推荐) 编写并运行测试**

```bash
pytest
```

**第6步: 开始训练！**

```bash
python src/train.py
```

**第7步: 分析你的实验**

```bash
tensorboard --logdir logs
```

**第8步: 进行预测**

```bash
python src/predict.py
```