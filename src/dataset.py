# src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Any, Tuple
from torchvision import datasets, transforms
from pathlib import Path


class YourDataset(Dataset):
    """
    数据集类模板，使用 torchvision.datasets.ImageFolder。
    假设你的数据按以下结构存放: data/{train/test}/{class_name}/*.jpg
    """

    def __init__(self, config: Dict[str, Any], mode: str):
        """
        Args:
            config (Dict[str, Any]): 全局配置字典。
            mode (str): 'train', 'dev', 或 'test'。
        """
        self.mode = mode
        data_root = Path(config['paths']['data_root'])
        image_size = config.get('params', {}).get('image_size', 224)

        if mode in ['train', 'dev']:
            data_path = data_root / 'train'
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:  # test mode
            data_path = data_root / 'test'
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        # 使用 ImageFolder 自动从文件夹结构中加载数据和标签
        self.dataset = datasets.ImageFolder(root=data_path, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple:
        """获取单个数据点。"""
        # 从 ImageFolder 数据集中获取图像和标签
        image, label = self.dataset[index]

        if self.mode == 'test':
            # 对于测试集，我们返回图像和它的文件名（作为ID）
            path, _ = self.dataset.imgs[index]
            return image, Path(path).stem
        else:
            # 对于训练/验证集，确保标签是LongTensor
            return image, torch.tensor(label).long()


def prepare_dataloader(config: Dict[str, Any], mode: str) -> Tuple[DataLoader, Dataset]:
    """工厂函数，根据模式创建并返回数据加载器。"""
    if mode in ['train', 'dev']:
        full_dataset = YourDataset(config, mode='train')
        generator = torch.Generator().manual_seed(config['seed'])
        train_len = int(0.9 * len(full_dataset))
        lengths = [train_len, len(full_dataset) - train_len]
        train_set, dev_set = random_split(full_dataset, lengths, generator=generator)

        dataset = train_set if mode == 'train' else dev_set
        shuffle = (mode == 'train')
    elif mode == 'test':
        dataset = YourDataset(config, mode='test')
        shuffle = False
    else:
        raise ValueError(f"未知的模式: {mode}")

    dataloader = DataLoader(
        dataset, batch_size=config['params']['batch_size'],
        shuffle=shuffle, num_workers=config['params']['num_workers'],
        pin_memory=True
    )
    return dataloader, dataset