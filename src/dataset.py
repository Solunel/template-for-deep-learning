# src/dataset.py (按照你的思路重构)

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Any, Tuple
from torchvision import datasets, transforms
from pathlib import Path


# YourDataset 类保持不变，但我们现在可以把它看作是“训练/验证”数据集的基类
# TestDataset 也可以分开，但为了简化，我们仍然用 mode 来区分
class YourDataset(Dataset):
    """
    数据集类模板，使用 torchvision.datasets.ImageFolder。
    假设你的数据按以下结构存放: data/{train/test}/{class_name}/*.jpg
    """

    def __init__(self, config: Dict[str, Any], is_train: bool = True):
        data_root = Path(config['paths']['data_root'])
        image_size = config.get('params', {}).get('image_size', 224)
        self.is_train = is_train

        if is_train:
            data_path = data_root / 'train'
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:  # test mode
            data_path = data_root / 'test'
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        self.dataset = datasets.ImageFolder(root=data_path, transform=self.transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple:
        image, label = self.dataset[index]
        if self.is_train:
            return image, torch.tensor(label).long()
        else:  # test mode
            path, _ = self.dataset.imgs[index]
            return image, Path(path).stem


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dataset]:
    """
    为训练和验证创建并返回数据加载器。
    这个函数封装了数据集的加载、划分和DataLoader的创建，只为训练流程服务。
    """
    # 1. 创建完整的训练数据集实例 (只加载一次！)
    full_dataset = YourDataset(config, is_train=True)

    # 2. 划分训练集和验证集
    generator = torch.Generator().manual_seed(config['seed'])
    train_len = int(0.9 * len(full_dataset))
    lengths = [train_len, len(full_dataset) - train_len]
    train_set, dev_set = random_split(full_dataset, lengths, generator=generator)

    # 3. 分别为训练集和验证集创建 DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=config['params']['batch_size'],
        shuffle=True,
        num_workers=config['params']['num_workers'],
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=config['params']['batch_size'],
        shuffle=False,
        num_workers=config['params']['num_workers'],
        pin_memory=True
    )

    # 返回所有需要的东西，注意这里的 train_set 是 Subset 对象
    return train_loader, dev_loader, train_set


def get_test_loader(config: Dict[str, Any]) -> DataLoader:
    """为测试集创建并返回数据加载器。"""
    test_dataset = YourDataset(config, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['params']['batch_size'],
        shuffle=False,
        num_workers=config['params']['num_workers'],
        pin_memory=True
    )
    # 预测时我们不需要返回 dataset 对象
    return test_loader