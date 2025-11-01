# src/dataset.py (按照你的思路重构)

# --- 导入必要的库 ---
import torch  # PyTorch 深度学习框架
from torch.utils.data import Dataset, DataLoader, random_split  # PyTorch 数据处理工具
from typing import Dict, Any, Tuple  # Python 类型提示工具，增强代码可读性
from torchvision import datasets, transforms  # PyTorch 计算机视觉库，包含常见数据集和图像变换
from pathlib import Path  # 面向对象的路径操作库，比字符串拼接更安全、更跨平台


# --- 数据集类的定义 ---
# YourDataset 类保持不变，但我们现在可以把它看作是“训练/验证”数据集的基类
# TestDataset 也可以分开，但为了简化，我们仍然用 mode 来区分
class YourDataset(Dataset):
    """
    数据集类模板，使用 torchvision.datasets.ImageFolder。
    这个类是 PyTorch `Dataset` 类的子类，你需要实现 __init__, __len__, __getitem__ 三个核心方法。
    假设你的数据按以下结构存放: data/{train/test}/{class_name}/*.jpg
    """

    # --- 初始化方法 ---
    def __init__(self, config: Dict[str, Any], is_train: bool = True):
        """
        当创建这个类的实例时，这个方法会被调用。
        Args:
            config (Dict[str, Any]): 从 config.yaml 加载的配置字典。
            is_train (bool): 一个标志，用于区分是加载训练数据还是测试数据。
        """
        # 从配置中获取数据根目录，并用 Path() 转换为路径对象
        data_root = Path(config['paths']['data_root'])
        # 从配置中获取图像尺寸，如果没指定，则默认为 224
        image_size = config.get('params', {}).get('image_size', 224)
        # 将 is_train 标志保存为类的属性，方便其他方法使用
        self.is_train = is_train

        # --- 根据是训练模式还是测试模式，定义不同的数据增强/变换 ---
        if is_train:
            # 如果是训练模式...
            # 定义训练集数据所在的路径
            data_path = data_root / 'train'
            # 定义一系列针对训练集的数据变换操作
            self.transform = transforms.Compose([
                # 1. 调整所有图片的大小为 (image_size, image_size)
                transforms.Resize((image_size, image_size)),
                # 2. 以 50% 的概率对图片进行水平翻转，增加数据多样性，防止过拟合
                transforms.RandomHorizontalFlip(),
                # 3. 将 PIL Image 或 NumPy ndarray 转换为 PyTorch Tensor，并把像素值从 [0, 255] 缩放到 [0, 1]
                transforms.ToTensor(),
            ])
        else:  # test mode
            # 如果是测试/验证模式...
            # 定义测试集数据所在的路径
            data_path = data_root / 'test'
            # 定义一系列针对测试集的数据变换操作 (通常更简单，不做数据增强)
            self.transform = transforms.Compose([
                # 1. 调整所有图片的大小为 (image_size, image_size)，确保和训练时一致
                transforms.Resize((image_size, image_size)),
                # 2. 将图片转换为 PyTorch Tensor
                transforms.ToTensor(),
            ])

        # 使用 torchvision 自带的 ImageFolder 类来加载数据。
        # 它会自动从 `data_path` 的子文件夹名中推断出类别，并加载所有图片。
        # root: 数据路径
        # transform: 应用于每张图片的变换
        self.dataset = datasets.ImageFolder(root=data_path, transform=self.transform)

    # --- 获取数据集长度的方法 ---
    def __len__(self) -> int:
        """这个方法返回数据集中的样本总数。"""
        # 直接返回 ImageFolder 实例的长度
        return len(self.dataset)

    # --- 根据索引获取单个样本的方法 ---
    def __getitem__(self, index: int) -> Tuple:
        """
        这个方法定义了如何根据给定的索引 `index` 获取一个数据样本。
        DataLoader 会在后台调用这个方法来构建一个批次(batch)的数据。
        Args:
            index (int): 样本的索引。
        Returns:
            Tuple: 根据 `is_train` 标志返回不同的元组。
        """
        # 从 ImageFolder 数据集中获取指定索引的图像和标签
        image, label = self.dataset[index]

        # 根据模式返回不同的内容
        if self.is_train:
            # 如果是训练模式，返回图像和它的标签
            # 将标签转换为 long 类型的 Tensor，这是 PyTorch 损失函数通常要求的格式
            return image, torch.tensor(label).long()
        else:  # test mode
            # 如果是测试模式...
            # 通过 `self.dataset.imgs` 获取原始图片路径
            path, _ = self.dataset.imgs[index]
            # 我们需要返回图像和它的文件名(ID)，用于生成提交文件
            # Path(path).stem 可以获取不带后缀的文件名，例如 '.../1001.jpg' -> '1001'
            return image, Path(path).stem


# --- 创建数据加载器的函数 ---
def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dataset]:
    """
    为训练和验证创建并返回数据加载器。
    这个函数封装了数据集的加载、划分和DataLoader的创建，只为训练流程服务。
    """
    # 1. 创建完整的训练数据集实例 (这个实例包含了所有 'train' 文件夹下的数据)
    full_dataset = YourDataset(config, is_train=True)

    # 2. 划分训练集和验证集
    # 设置一个随机种子，确保每次划分的结果都一样，保证实验的可复现性
    generator = torch.Generator().manual_seed(config['seed'])
    # 计算训练集的长度，这里我们用 90% 的数据做训练
    train_len = int(0.9 * len(full_dataset))
    # 计算验证集的长度，即剩下的 10%
    lengths = [train_len, len(full_dataset) - train_len]
    # 使用 random_split 函数将完整数据集划分为两个子集
    train_set, dev_set = random_split(full_dataset, lengths, generator=generator)

    # 3. 分别为训练集和验证集创建 DataLoader
    # DataLoader 是一个迭代器，它能自动地将数据整理成批次(batch)、打乱数据、并使用多线程加载数据。
    train_loader = DataLoader(
        train_set,  # 要加载的数据集 (这里是划分后的训练子集)
        batch_size=config['params']['batch_size'],  # 每个批次的大小
        shuffle=True,  # 在每个 epoch 开始时打乱数据顺序，这对于训练很重要
        num_workers=config['params']['num_workers'],  # 使用多少个子进程来加载数据，加快速度
        pin_memory=True  # 如果为 True，数据加载器会将 Tensors 复制到 CUDA 固定内存中，这可以加速 GPU 的数据传输
    )
    dev_loader = DataLoader(
        dev_set,  # 要加载的数据集 (这里是划分后的验证子集)
        batch_size=config['params']['batch_size'],  # 批次大小
        shuffle=False,  # 验证时不需要打乱顺序
        num_workers=config['params']['num_workers'],  # 工作进程数
        pin_memory=True  # 锁定内存
    )

    # 返回所有需要的东西：训练加载器、验证加载器、以及训练子集对象(后续需要用它来获取类别映射)
    return train_loader, dev_loader, train_set


def get_test_loader(config: Dict[str, Any]) -> DataLoader:
    """为测试集创建并返回数据加载器。"""
    # 1. 创建测试数据集实例
    test_dataset = YourDataset(config, is_train=False)
    # 2. 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,  # 测试数据集
        batch_size=config['params']['batch_size'],  # 批次大小
        shuffle=False,  # 预测时绝不能打乱顺序，否则文件名和预测结果对不上
        num_workers=config['params']['num_workers'],  # 工作进程数
        pin_memory=True  # 锁定内存
    )
    # 预测时我们只需要加载器，不需要返回 dataset 对象
    return test_loader