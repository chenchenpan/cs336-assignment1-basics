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
