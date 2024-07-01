from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up
            warmup_lr = [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
            return warmup_lr
        else:
            # Cosine annealing
            cos_lr = [self.min_lr + (base_lr - self.min_lr) * 
                      (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                      for base_lr in self.base_lrs]
            return cos_lr