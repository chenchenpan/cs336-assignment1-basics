from __future__ import annotations

import math
import torch
from torch import nn
from einops import einsum, rearrange

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    The function should take two parameters: 
    x: a tensor 
    dim: int Dimension i
    apply softmax to the i-th dimension of the input tensor. The output tensor should have the same shape as the input tensor,
    but its i-th dimension will have a normalized probability distribution. Use the trick of subtracting the maximum value in
    the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function takes in predicted logits (o) and target (t), and then computes the cross-entropy loss
    L = -log softmax(logits) [target] = negative log softmax at target index t
    --> L = - o_t +logsumexp(o) = - logits_target + logsumexp(logits)
    logits_target is just the logit value at the correct class index for each example. If logits has shape (..., V)
    and targets has shape (...,), then for each batch element: logits_target = logits[..., targets]. 
    So it's the model's unnormalized score for the true class, extracted from the last (vocab/class) dimension.
    logits: (..., V) # vocab_size
    targets: (...) 
    """
    # subtract the larget element to prevent overflow
    max_logits = logits.max(dim=-1, keepdim=True).values # (..., 1)
    shifted = logits - max_logits # (..., V)

    # logsumexp of shifted logits (still stable)
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1)) # (...,)

    # gather target logits
    targets = targets.long()
    logits_target = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) # (...,)

    # loss per example: -logits_target + max_logits + logsumexp(shifted)
    loss = - logits_target + max_logits.squeeze(-1) + logsumexp
    return loss.mean()