import torch
from torch.optim import Optimizer
import math

class ARInverseAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, sharpness_threshold=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, sharpness_threshold=sharpness_threshold)
        super(ARInverseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        sharpness = 0
        total_params = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                sharpness += torch.sum(grad ** 2).item()
                total_params += grad.numel()

        sharpness /= total_params

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            sharpness_threshold = group['sharpness_threshold']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_inv'] = torch.zeros_like(p.data)
                    state['exp_avg_sq_inv'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg_inv, exp_avg_sq_inv = state['exp_avg_inv'], state['exp_avg_sq_inv']
                state['step'] += 1

                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                exp_avg_inv.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq_inv.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                denom_inv = (exp_avg_sq_inv.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size_inv = lr / bias_correction1

                if sharpness > sharpness_threshold:
                    ratio = min(1.0, max(0.0, 0.5 + 0.1 * (sharpness / sharpness_threshold - 1)))
                else:
                    ratio = max(0.0, min(1.0, 0.5 - 0.1 * (1 - sharpness / sharpness_threshold)))

                update = ratio * (exp_avg / denom) + (1 - ratio) * (exp_avg_inv * denom_inv)
                p.data.add_(-step_size, update)

        return loss