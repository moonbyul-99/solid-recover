import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import math

class sr_scheduler(_LRScheduler):
    '''
    Customized learning rate scheduler in solid recover model 

    Args:
        optimizer (torch.optim.Optimizer): the optimizer to be scheduled
        warmup_steps (int): number of warmup steps
        steady_1_steps (int): number of steps in the first steady phase
        cosine_anneal_steps (int): number of steps in the cosine annealing phase
        min_lr (float): minimum learning rate

        For all steps params, min value in 0

        The lr scheduler is first linear warmup, then change to constant lr, then cosine annealing, finally change to min lr constant
    '''
    def __init__(self, optimizer, warmup_steps, steady_1_steps, cosine_anneal_steps, min_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.steady_1_steps = steady_1_steps
        self.cosine_anneal_steps = cosine_anneal_steps
        self.min_lr = min_lr

        if isinstance(self.min_lr, (int, float)):
            self.min_lrs = [self.min_lr] * len(optimizer.param_groups)
        else:
            self.min_lrs = self.min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # get basic lr
        base_lrs = self.base_lrs
        current_step = self.last_epoch

        # linear Warmup
        if current_step < self.warmup_steps:
            return [base_lr * (current_step / self.warmup_steps) for base_lr in base_lrs]

        # constant phase 1
        elif current_step < self.warmup_steps + self.steady_1_steps:
            return base_lrs

        # consine annealing
        elif current_step < self.warmup_steps + self.steady_1_steps + self.cosine_anneal_steps:
            step_in_cosine = current_step - (self.warmup_steps + self.steady_1_steps)
            return [self.min_lrs[i] + 0.5 * (base_lrs[i] - self.min_lrs[i]) * (1 + math.cos(math.pi * step_in_cosine / self.cosine_anneal_steps))
                    for i, base_lr in enumerate(base_lrs)]
        
        # constant phase 2
        else:
            return self.min_lrs