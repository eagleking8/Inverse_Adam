import torch
from torch.optim.optimizer import Optimizer

class AdaSGDM(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.999, momentum=0.9, epsilon=1e-8, weight_decay=5e-4):
        defaults = dict(lr=lr, beta=beta, momentum=momentum, epsilon=epsilon, weight_decay=weight_decay)
        super(AdaSGDM, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta = group['beta']
                momentum = group['momentum']
                lr, epsilon, weight_decay = group['lr'], group['epsilon'], group['weight_decay']
                # # Apply L2 normalization
                # if weight_decay != 0:
                #     grad = grad.add(p.grad.data * weight_decay)

                state['step'] += 1

                # SGDM's moment estimates
                m.mul_(momentum).add_(grad)
                state['m'] = m
                v.mul_(beta).addcmul_(grad, grad, value=1 - beta)
                state['v'] = v

                # Bias-correction
                v_hat = v.div(1.0 - beta**state['step'])

                ada_norm = v_hat.sqrt().add(epsilon)

                update = m / ada_norm

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(-weight_decay * lr, p.data)

                # Combined update
                p.data.add_(-lr * update)

        return loss
