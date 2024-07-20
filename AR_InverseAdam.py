import torch
from torch.optim import Optimizer
import math

class ARInverseAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, delta=0.9, switch_rate=1e-2, lambda1=-1.0, lambda2=1.0, T=200*391):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, delta=delta, switch_rate=switch_rate, lambda1=lambda1, lambda2=lambda2, T=T)
        super(ARInverseAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            delta = group['delta']
            switch_rate = group['switch_rate']
            lambda1 = group['lambda1']
            lambda2 = group['lambda2']
            T = group['T']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['adam_ratio'] = 0.5  # Initial ratio for Adam and InverseAdam updates
                    state['mu_t'] = 0.0
                    state['sigma_t_square'] = 0.0

                mu_t, sigma_t_square = state['mu_t'], state['sigma_t_square']
                state['step'] += 1

                # Update the running average of the squared norm of the gradient
                grad_norm_square = torch.norm(grad).item() ** 2
                mu_t = delta * mu_t + (1 - delta) * grad_norm_square
                sigma_t_square = delta * sigma_t_square + (1 - delta) * (grad_norm_square - mu_t) ** 2
                state['mu_t'] = mu_t
                state['sigma_t_square'] = sigma_t_square

                # Update the sharpness threshold dynamically
                t = state['step']
                c_t = lambda1 *  t / T + lambda2 * (1 - t / T)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Update running averages of gradient and its square
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg_hat = exp_avg.div(bias_correction1)
                exp_avg_sq_hat = exp_avg_sq.div(bias_correction2)

                # Compute denominator
                denom = exp_avg_sq_hat.sqrt().add(eps)

                # Adjust the ratio based on the sharpness
                if grad_norm_square >= state['mu_t'] + c_t * math.sqrt(state['sigma_t_square']):
                    state['adam_ratio'] = min(1.0, state['adam_ratio'] + switch_rate)
                else:
                    state['adam_ratio'] = max(0.0, state['adam_ratio'] - switch_rate)

                adam_ratio = state['adam_ratio']
                inv_ratio = 1.0 - adam_ratio

                # Combine updates from Adam and InverseAdam
                update_adam = exp_avg_hat / denom
                update_inv = exp_avg_hat * denom
                update = adam_ratio * update_adam + inv_ratio * update_inv

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(-weight_decay * lr, p.data)

                p.data.add_(-lr, update)

        return loss
