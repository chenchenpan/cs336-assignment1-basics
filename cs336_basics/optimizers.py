from __future__ import annotations

import math
import torch
from collections.abc import Callable, Iterable
from typing import Optional

class SGD(torch.optim.Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    We will implement a slight variation of SGD where the learning rate decays over training,
    starting with an initial learning rate `alpha` and taking successively smaller steps over time:
    alpha / sqrt(t+1).
    """
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p
                # print(state.keys())
                t = state.get("t", 0) # get iteration number from the state, or initial value
                grad = p.grad.data # get the gradient of loss with respect to p
                p.data -= lr / math.sqrt(t+1) * grad # update weight tensor in-place
                state["t"] = t + 1 # increment iteration number

        return loss

class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer with decoupled weight decay.
    state: exp_avg (m), exp_avg_sq (v), step (t).
    """
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameters: betas={betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lambd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad.data

                # Perform weight decay
                if lambd != 0:
                    p.data.mul_(1 - lr * lambd)
                
                # update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state["step"] = t + 1
                t += 1

                # bias correction
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t

                # according to Loshchilov and Hutter [2019].
                # lr_t = lr * sqrt(1-beta2**t) / (1-beta1**t)
                # θ <- θ - lr_t * m / (√v + ε)
                lr_t = lr * math.sqrt(bias2) / bias1
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add(eps), value=-lr_t)
        
        return loss
                

                
