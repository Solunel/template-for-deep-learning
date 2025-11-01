# src/engine.py

# --- 导入必要的库 ---
import logging  # 用于记录程序运行信息的库
import os  # 用于与操作系统交互，比如检查文件是否存在
import torch  # PyTorch 深度学习框架
from torch.utils.tensorboard import SummaryWriter  # 用于将日志写入 TensorBoard 进行可视化
from tqdm import tqdm  # 一个快速、可扩展的 Python 进度条库
from typing import Dict, Any, Callable, Optional, Tuple, List  # Python 类型提示
from pathlib import Path  # 面向对象的路径操作库

# 获取一个名为 __name__ (即 'src.engine') 的 logger 实例
logger = logging.getLogger(__name__)


# --- 核心引擎类的定义 ---
class Engine:
    """通用核心引擎类 (V2.5)。这个类封装了训练、验证和预测的所有核心逻辑。"""

    # --- 初始化方法 ---
    def __init__(self,
                 net: torch.nn.Module,  # 神经网络模型
                 config: Dict[str, Any],  # 配置字典
                 device: torch.device,  # 运行设备 (CPU 或 GPU)
                 criterion: Optional[Callable] = None,  # 损失函数 (训练时需要)
                 optimizer: Optional[torch.optim.Optimizer] = None,  # 优化器 (训练时需要)
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,  # 学习率调度器 (训练时需要)
                 metric_fns: Optional[Dict[str, Callable]] = None):  # 评估指标函数字典 (训练时需要)

        # --- 将传入的参数保存为类的属性 ---
        self.net = net  # 模型
        self.config = config  # 配置
        self.device = device  # 设备
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化器
        self.scheduler = scheduler  # 学习率调度器
        self.metric_fns = metric_fns  # 评估指标

        # --- 初始化 TensorBoard 写入器 ---
        # SummaryWriter 会在指定的 log_dir 目录下创建日志文件
        self.writer = SummaryWriter(config['paths']['log_dir'])
        # 初始化当前训练步数为 0
        self.current_step = 0

        # --- 初始化早停和最佳模型保存相关的参数 ---
        # 从配置中获取参数
        params = config['params']
        # 获取用于判断模型好坏的主要指标名称，例如 'accuracy'
        self.primary_metric = params['primary_metric']
        # 获取该指标的模式，'max' 表示越大越好，'min' 表示越小越好
        self.primary_metric_mode = params['primary_metric_mode']
        # 初始化最佳指标值。如果是 'max' 模式，初始为负无穷；如果是 'min' 模式，初始为正无穷。
        self.best_metric = -float('inf') if self.primary_metric_mode == 'max' else float('inf')
        # 初始化一个计数器，记录验证集指标连续多少次没有提升
        self.epochs_no_improve = 0

    # --- 私有方法：保存模型检查点 ---
    def _save_checkpoint(self, is_best: bool = False) -> None:
        """保存模型检查点。检查点包含了在特定时间点恢复训练或评估所需的所有信息。"""
        # 创建一个字典，用于存放需要保存的状态
        checkpoint = {
            'step': self.current_step,  # 当前训练步数
            'model_state_dict': self.net.state_dict(),  # 模型的权重
            'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器的状态 (如动量等)
            'scheduler_state_dict': self.scheduler.state_dict(),  # 学习率调度器的状态
            'best_metric': self.best_metric  # 当前记录的最佳指标值
        }
        # 获取检查点要保存的目录
        ckpt_dir = Path(self.config['paths']['checkpoint_dir'])
        # 保存最新的模型检查点，用于意外中断后恢复训练
        torch.save(checkpoint, ckpt_dir / 'latest_model.pth')
        # 如果当前模型是最佳模型 (is_best=True)，则额外保存一份为 'best_model.pth'
        if is_best:
            torch.save(checkpoint, ckpt_dir / 'best_model.pth')

    # --- 私有方法：加载模型检查点 ---
    def _load_checkpoint(self, path: str, is_training: bool) -> None:
        """加载模型检查点。"""
        # 检查指定的路径是否存在文件
        if not os.path.exists(path):
            # 如果文件不存在，记录一条信息并直接返回，意味着将从头开始训练
            logger.info(f"检查点 {path} 未找到，将从零开始。")
            return

        # 如果文件存在，记录日志并开始加载
        logger.info(f"正在从 {path} 加载检查点...")
        # 加载检查点文件。map_location=self.device 确保模型被加载到正确的设备上
        checkpoint = torch.load(path, map_location=self.device)
        # 将加载的权重应用到当前模型中
        self.net.load_state_dict(checkpoint['model_state_dict'])

        # 如果是用于继续训练 (is_training=True)，我们还需要恢复优化器、调度器等的状态
        if is_training:
            # 恢复优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 恢复学习率调度器状态
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # 恢复训练步数
            self.current_step = checkpoint['step']
            # 恢复记录的最佳指标值
            self.best_metric = checkpoint.get('best_metric', self.best_metric)
            # 记录日志，告知用户训练已成功恢复
            logger.info(f"恢复训练完成。将从步数 {self.current_step} 开始。")

    # --- 私有方法：运行一次完整的验证 ---
    @torch.no_grad()  # 这是一个装饰器，告诉 PyTorch 在这个函数中不需要计算梯度，可以节省计算资源和内存
    def _run_validation(self, valid_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """在验证集上运行一次完整的评估。"""
        # 将模型切换到评估模式 (evaluation mode)。这会关闭 Dropout 和 BatchNorm 的训练行为。
        self.net.eval()
        # 初始化总损失为 0
        total_loss = 0
        # 初始化一个字典，用于累加每个评估指标的总值
        metric_totals = {name: 0.0 for name in self.metric_fns.keys()}

        # 使用 tqdm 创建一个进度条，遍历验证数据加载器中的所有批次
        for x, y in tqdm(valid_loader, desc="Validation", leave=False):
            # 将输入数据和标签移动到指定的设备 (GPU/CPU)
            x, y = x.to(self.device), y.to(self.device)
            # 模型进行前向传播，得到预测输出 (logits)
            output = self.net(x)
            # 使用损失函数计算当前批次的损失
            loss = self.criterion(output, y)
            # 累加损失。乘以 len(x) (即批次大小) 是为了后续能精确计算整个数据集的平均损失
            total_loss += loss.item() * len(x)

            # 遍历所有定义的评估指标函数
            for name, fn in self.metric_fns.items():
                # 计算当前批次的指标值，并乘以批次大小进行累加
                metric_totals[name] += fn(output, y) * len(x)

        # 计算整个验证集的平均损失
        avg_loss = total_loss / len(valid_loader.dataset)
        # 计算每个指标在整个验证集上的平均值
        avg_metrics = {name: total / len(valid_loader.dataset) for name, total in metric_totals.items()}
        # 将平均损失也加入到指标字典中
        avg_metrics['loss'] = avg_loss
        # 返回包含所有平均指标的字典
        return avg_metrics

    # --- 公开方法：启动完整的训练流程 ---
    def run_training(self, train_loader: torch.utils.data.DataLoader,
                     valid_loader: torch.utils.data.DataLoader) -> None:
        """启动完整的训练流程。"""
        # 记录日志，表示训练开始
        logger.info("开始训练流程...")
        # 定义最新模型检查点的路径
        latest_model_path = os.path.join(self.config['paths']['checkpoint_dir'], 'latest_model.pth')
        # 尝试从该路径加载检查点以恢复训练
        self._load_checkpoint(latest_model_path, is_training=True)

        # 从配置中获取训练参数
        params = self.config['params']
        # 创建一个训练数据加载器的迭代器。这样我们可以用 next() 来手动获取下一个批次
        train_iterator = iter(train_loader)
        # 创建一个总的训练进度条，初始值为当前步数，总步数为配置中定义的 total_steps
        pbar = tqdm(initial=self.current_step, total=params['total_steps'], desc="Training")

        # 主训练循环，当当前步数小于总步数时持续进行
        while self.current_step < params['total_steps']:
            # 将模型切换到训练模式 (training mode)，启用 Dropout 和 BatchNorm
            self.net.train()
            # 尝试从迭代器中获取下一个批次的数据
            try:
                x, y = next(train_iterator)
            except StopIteration:
                # 如果迭代器耗尽 (即遍历完了一轮数据)，则重新创建迭代器，开始新的 epoch
                train_iterator = iter(train_loader)
                x, y = next(train_iterator)

            # 将数据和标签移动到指定设备
            x, y = x.to(self.device), y.to(self.device)

            # 1. 梯度清零：清除上一轮计算的梯度，否则梯度会累加
            self.optimizer.zero_grad()
            # 2. 前向传播：模型根据输入 x 产生预测输出 output
            output = self.net(x)
            # 3. 计算损失：根据预测 output 和真实标签 y 计算损失
            loss = self.criterion(output, y)
            # 4. 反向传播：根据损失计算模型中所有可训练参数的梯度
            loss.backward()
            # 5. 更新权重：优化器根据计算出的梯度来更新模型的权重
            self.optimizer.step()
            # 6. 更新学习率：学习率调度器根据当前步数调整学习率
            self.scheduler.step()

            # 训练步数加一
            self.current_step += 1
            # 更新总进度条
            pbar.update(1)
            # 在进度条后面显示当前的损失值
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            # 将当前步的训练损失写入 TensorBoard，标签为 'Loss/train_step'
            self.writer.add_scalar('Loss/train_step', loss.item(), self.current_step)

            # --- 验证、保存和早停逻辑 ---
            # 判断是否达到了进行验证的步数
            if self.current_step % params['valid_steps'] == 0:
                # 运行一次完整的验证，并获取结果
                val_results = self._run_validation(valid_loader)
                # 构建一个日志字符串，用于打印验证结果
                log_str = f"Step {self.current_step}: " + ", ".join(
                    [f"Val {k}: {v:.4f}" for k, v in val_results.items()])
                # 记录验证结果日志
                logger.info(log_str)

                # 将所有验证指标写入 TensorBoard
                for name, value in val_results.items():
                    self.writer.add_scalar(f"Validation/{name}", value, self.current_step)

                # 保存一次最新的模型检查点
                self._save_checkpoint(is_best=False)

                # 获取当前验证集上的主要指标值
                current_metric = val_results.get(self.primary_metric, 0.0)
                # 判断当前指标是否比历史最佳指标更好
                is_better = (self.primary_metric_mode == 'max' and current_metric > self.best_metric) or \
                            (self.primary_metric_mode == 'min' and current_metric < self.best_metric)

                if is_better:
                    # 如果当前模型更好...
                    # 更新最佳指标值
                    self.best_metric = current_metric
                    # 重置“未提升”计数器
                    self.epochs_no_improve = 0
                    # 记录日志，发现新的最佳模型
                    logger.info(f"发现新最佳模型! {self.primary_metric}: {self.best_metric:.4f}")
                    # 保存这个最佳模型检查点
                    self._save_checkpoint(is_best=True)
                else:
                    # 如果当前模型没有更好...
                    # “未提升”计数器加一
                    self.epochs_no_improve += 1
                    # 记录日志，显示早停计数
                    logger.info(
                        f"验证集 {self.primary_metric} 未提升. 早停计数: {self.epochs_no_improve}/{params['patience']}")
                    # 检查是否满足早停条件
                    if params['enable_early_stopping'] and self.epochs_no_improve >= params['patience']:
                        logger.info(f"触发早停。")
                        # 跳出主训练循环
                        break

        # 训练结束后关闭进度条
        pbar.close()
        # 记录训练结束日志和最终的最佳指标
        logger.info(f"训练结束。最佳验证集 {self.primary_metric} 为: {self.best_metric:.4f}")

        # --- 记录超参数和最终结果到 TensorBoard ---
        # 准备一个只包含简单数据类型（如数字、字符串、布尔值）的超参数字典
        hparam_dict = {k: v for k, v in self.config['params'].items() if isinstance(v, (int, float, str, bool))}
        # 准备一个包含最终性能指标的字典
        metric_dict = {f'hparam/best_{self.primary_metric}': self.best_metric}
        # 将超参数和最终指标写入 TensorBoard，这样可以在 HPARAMS 标签页中进行比较
        self.writer.add_hparams(hparam_dict, metric_dict)
        # 关闭 TensorBoard 写入器
        self.writer.close()

    # --- 公开方法：使用最佳模型进行预测 ---
    @torch.no_grad()  # 预测时同样不需要计算梯度
    def predict(self, test_loader: torch.utils.data.DataLoader) -> Tuple[List, List]:
        """使用最佳模型进行预测。"""
        # 定义最佳模型检查点的路径
        best_model_path = os.path.join(self.config['paths']['checkpoint_dir'], 'best_model.pth')
        # 检查最佳模型文件是否存在
        if not os.path.exists(best_model_path):
            # 如果不存在，抛出一个文件未找到的异常
            raise FileNotFoundError(f"最佳模型文件未找到: {best_model_path}！请先完成训练。")

        # 加载最佳模型检查点
        checkpoint = torch.load(best_model_path, map_location=self.device)
        # 将加载的权重应用到当前模型
        self.net.load_state_dict(checkpoint['model_state_dict'])
        # 记录日志，告知用户模型加载成功
        logger.info(f"最佳模型加载成功 (来自 step {checkpoint['step']})。")

        # 将模型切换到评估模式
        self.net.eval()
        # 初始化两个列表，用于存放所有预测结果和对应的ID
        all_preds, all_ids = [], []
        # 遍历测试数据加载器
        for x, ids in tqdm(test_loader, desc="Predicting"):
            # 将输入数据移动到指定设备
            x = x.to(self.device)
            # 模型进行前向传播，得到预测输出 (logits)
            output = self.net(x)
            # 从 logits 中找出每个样本概率最高的类别的索引，这就是预测结果
            # torch.max(output, 1) 返回一个元组 (最大值, 最大值索引)，我们只需要索引
            _, preds = torch.max(output, 1)
            # 将当前批次的预测结果 (从 GPU 移到 CPU，再转为 numpy 数组) 添加到总列表中
            all_preds.extend(preds.cpu().numpy())
            # 将当前批次的 ID 添加到总列表中
            all_ids.extend(ids)
        # 返回包含所有 ID 和对应预测结果的两个列表
        return all_ids, all_preds