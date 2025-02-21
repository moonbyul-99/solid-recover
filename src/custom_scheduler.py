import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    带有线性预热和余弦退火学习率调度的类。

    参数:
        optimizer (Optimizer): 要调度的优化器。
        warmup_steps (int): 预热阶段的步数。
        anneal_steps (int): 余弦退火中止步数。
        min_lr (float): 最小学习率，默认为0.0。
        last_epoch (int): 最近一次训练的epoch数，默认为-1。
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, anneal_steps: int, min_lr: float = 0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.anneal_steps = anneal_steps
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._is_in_warmup:
            return [self.cosine_decay(base_lr).item() for base_lr in self.base_lrs]
        else:
            return [self.linear_warmup(base_lr).item() for base_lr in self.base_lrs]

    @property
    def _is_in_warmup(self):
        return self.last_epoch < self.warmup_steps

    def linear_warmup(self, base_lr):
        # 确保返回的是一个tensor
        return torch.tensor(base_lr * (self.last_epoch / self.warmup_steps)).to(torch.float32)

    def cosine_decay(self, base_lr):
        if self.last_epoch > self.anneal_steps:
            return torch.tensor(self.min_lr).to(torch.float32)
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.anneal_steps - self.warmup_steps)
            progress = torch.tensor(progress).to(torch.float32)
            return torch.tensor(self.min_lr + (base_lr - self.min_lr) * (0.5 * (1.0 + torch.cos(torch.pi * progress)))).to(torch.float32)