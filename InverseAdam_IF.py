import torch
from torch.optim.optimizer import Optimizer

class InverseAdam_IF(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, switch_rate=0.01, weight_decay=0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, switch_rate=switch_rate,
                        weight_decay=weight_decay)
        super(InverseAdam_IF, self).__init__(params, defaults)

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
                beta1, beta2 = group['beta1'], group['beta2']
                lr, epsilon, switch_rate, weight_decay = group['lr'], group['epsilon'], group['switch_rate'], group['weight_decay']
                inverse_adam_rate = max(0.0, 1.0 - state['step'] * switch_rate)

                # # Apply L2 normalization
                # if weight_decay != 0:
                #     grad = grad.add(p.grad.data * weight_decay)

                state['step'] += 1

                # Adam's moment estimates
                m.mul_(beta1).add_((1.0 - beta1) * grad)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected moment estimates
                m_hat = m.div(1.0 - beta1**state['step'])
                v_hat = v.div(1.0 - beta2**state['step'])

                ada_norm = v_hat.sqrt().add(epsilon)

                # Inverse Adam part
                inverse_update = m_hat * ada_norm

                # Adam part
                adam_update = m_hat / ada_norm

                # Apply weight decay
                if weight_decay != 0:
                    p.data.add_(-weight_decay * lr, p.data)

                # Combined update
                p.data.add_(-lr * ((1.0 - inverse_adam_rate) * adam_update + inverse_adam_rate * inverse_update))

        return loss
