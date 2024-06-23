import torch
from torch.optim.optimizer import Optimizer

class InverseAdam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=0.01):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, alpha=alpha)
        super(InverseAdam, self).__init__(params, defaults)

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
                    state['inverse_adam_rate'] = 0.0

                m, v = state['m'], state['v']
                beta1, beta2 = group['beta1'], group['beta2']
                lr, epsilon, alpha = group['lr'], group['epsilon'], group['alpha']
                state['inverse_adam_rate'] = max(0.0, 1.0 - state['step'] * alpha)

                state['step'] += 1

                # Adam's moment estimates
                m.mul_(beta1).add_((1.0 - beta1) * grad)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected moment estimates
                m_hat = m.div(1.0 - beta1**state['step'])
                v_hat = v.div(1.0 - beta2**state['step'])

                # Inverse Adam part
                inverse_update = m_hat * (v_hat.sqrt().add(epsilon))

                # Adam part
                adam_update = m_hat / (v_hat.sqrt().add(epsilon))

                # Combined update
                p.data.add_(-lr * ((1.0 - state['inverse_adam_rate']) * adam_update + state['inverse_adam_rate'] * inverse_update))

        return loss
