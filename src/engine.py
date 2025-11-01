# src/engine.py
import logging
import os
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Any, Callable, Optional, Tuple, List

logger = logging.getLogger(__name__)


class Engine:
    """通用核心引擎类 (V2.3)。"""

    def __init__(self,
                 net: torch.nn.Module,
                 config: Dict[str, Any],
                 device: torch.device,
                 criterion: Optional[Callable] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 metric_fns: Optional[Dict[str, Callable]] = None,
                 train_data: Optional[torch.utils.data.DataLoader] = None,
                 dev_data: Optional[torch.utils.data.DataLoader] = None,
                 test_data: Optional[torch.utils.data.DataLoader] = None):

        self.net = net
        self.config = config
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric_fns = metric_fns
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.writer = SummaryWriter(config['paths']['log_dir'])
        self.current_step = 0

        hparams = config['hparams']
        self.primary_metric = hparams['primary_metric']
        self.primary_metric_mode = hparams['primary_metric_mode']
        self.best_metric = -float('inf') if self.primary_metric_mode == 'max' else float('inf')
        self.epochs_no_improve = 0

    def _save_checkpoint(self, is_best: bool = False) -> None:
        checkpoint = {
            'step': self.current_step,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric
        }
        ckpt_dir = Path(self.config['paths']['checkpoint_dir'])
        torch.save(checkpoint, ckpt_dir / 'latest_model.pth')
        if is_best:
            torch.save(checkpoint, ckpt_dir / 'best_model.pth')

    def _load_checkpoint(self, path: str, is_training: bool) -> None:
        if not os.path.exists(path):
            logger.info(f"检查点 {path} 未找到。")
            return

        logger.info(f"正在从 {path} 加载检查点...")
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])

        if is_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_step = checkpoint['step']
            self.best_metric = checkpoint.get('best_metric', self.best_metric)
            logger.info(f"恢复训练完成。将从步数 {self.current_step} 开始。")

    @torch.no_grad()
    def _run_validation(self) -> Dict[str, float]:
        self.net.eval()
        total_loss = 0
        metric_totals = {name: 0.0 for name in self.metric_fns.keys()}

        for x, y in tqdm(self.dev_data, desc="Validation", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            output = self.net(x)
            loss = self.criterion(output, y)
            total_loss += loss.item() * len(x)

            for name, fn in self.metric_fns.items():
                metric_totals[name] += fn(output, y) * len(x)

        avg_loss = total_loss / len(self.dev_data.dataset)
        avg_metrics = {name: total / len(self.dev_data.dataset) for name, total in metric_totals.items()}
        avg_metrics['loss'] = avg_loss
        return avg_metrics

    def run_training(self) -> None:
        logger.info("开始训练流程...")
        latest_model_path = os.path.join(self.config['paths']['checkpoint_dir'], 'latest_model.pth')
        self._load_checkpoint(latest_model_path, is_training=True)

        train_iterator = iter(self.train_data)
        pbar = tqdm(initial=self.current_step, total=self.config['hparams']['total_steps'], desc="Training")

        while self.current_step < self.config['hparams']['total_steps']:
            self.net.train()
            try:
                x, y = next(train_iterator)
            except StopIteration:
                train_iterator = iter(self.train_data)
                x, y = next(train_iterator)

            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            output = self.net(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.current_step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            self.writer.add_scalar('Loss/train_step', loss.item(), self.current_step)

            if self.current_step % self.config['hparams']['valid_steps'] == 0:
                val_results = self._run_validation()
                log_str = f"Step {self.current_step}: " + ", ".join(
                    [f"Val {k}: {v:.4f}" for k, v in val_results.items()])
                logger.info(log_str)

                for name, value in val_results.items():
                    self.writer.add_scalar(f"Validation/{name}", value, self.current_step)

                self._save_checkpoint(is_best=False)

                current_metric = val_results.get(self.primary_metric, 0.0)
                is_better = (self.primary_metric_mode == 'max' and current_metric > self.best_metric) or \
                            (self.primary_metric_mode == 'min' and current_metric < self.best_metric)

                if is_better:
                    self.best_metric = current_metric
                    self.epochs_no_improve = 0
                    logger.info(f"发现新最佳模型! {self.primary_metric}: {self.best_metric:.4f}")
                    self._save_checkpoint(is_best=True)
                else:
                    self.epochs_no_improve += 1
                    patience = self.config['hparams']['patience']
                    logger.info(f"验证集 {self.primary_metric} 未提升. 早停计数: {self.epochs_no_improve}/{patience}")
                    if self.config['hparams']['enable_early_stopping'] and self.epochs_no_improve >= patience:
                        logger.info(f"触发早停。")
                        break

        pbar.close()
        logger.info(f"训练结束。最佳验证集 {self.primary_metric} 为: {self.best_metric:.4f}")

        hparam_dict = {k: v for k, v in self.config['hparams'].items() if isinstance(v, (int, float, str, bool))}
        metric_dict = {f'hparam/best_{self.primary_metric}': self.best_metric}
        self.writer.add_hparams(hparam_dict, metric_dict)
        self.writer.close()

    @torch.no_grad()
    def predict(self) -> Tuple[List, List]:
        best_model_path = os.path.join(self.config['paths']['checkpoint_dir'], 'best_model.pth')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"最佳模型文件未找到: {best_model_path}！请先完成训练。")

        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"最佳模型加载成功 (来自 step {checkpoint['step']})。")

        self.net.eval()
        all_preds, all_ids = [], []
        for x, ids in tqdm(self.test_data, desc="Predicting"):
            x = x.to(self.device)
            output = self.net(x)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(ids)
        return all_ids, all_preds