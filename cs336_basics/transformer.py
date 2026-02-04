from __future__ import annotations

import math
import torch
from torch import nn
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features:int final dimension of the output
        device: torch.device | None=None Device to store the parameters on
        dtype:torch.dtype| None=None Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        # Store W (not W^T) for memory ordering reasons
        self.W = nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        # using torch.nn.init.trunc_normal_ to initialize weights
        std = math.sqrt(2.0/(in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module.This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        # embedding matrix of shape(vocab_size, d_model)
        self.embedding_matrix = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        # using torch.nn.init.trunc_normal_ to initialize the matrix
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: # inputs are token_ids
        # token_ids: (batch_size, sequence_length) long
        return self.embedding_matrix[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        """
        Construct the Root Mean Square Layer Normalization. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        result = x_norm * self.weight
        return result.to(in_dtype)

# position-wise feed-forward network: SwiGLU
# GLU defined as the element-wise product of a linear transformation passed through a sigmoid function and another linear transformation
# GLU(x) = (W1 @ x) * sigmoid(W2 @ x)
# SwiGLU(x) = (W1 @ x) * Swish(W2 @ x), where Swish(z) = z @ sigmoid(z)
# FFN(x) = W3 @ SwiGLU
# Note: @ means dot product; * means element-wise product

# The assignment's hint: You should set d_ff to approximately 8/3 * d_model in your implementation, while ensuring that
# the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware.
def _round_to_multiple(x, multiple):
    return int(math.ceil(x / multiple) * multiple)

class SwiGLU(nn.Module):
    # FFN(x) = SwiGLU(x,W1,W2,W3) = W2(SiLU(W1x)⊙ W3x)
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = _round_to_multiple((8 * d_model)/3, 64)
        self.d_model = d_model
        self.d_ff = d_ff
        factory_kwargs = {"device": device, "dtype": dtype}

        # W1 and W3 project up, W2 projects back down
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(x W1) = (x W1) ⊙  sigmoid(x W1)
        # SwiGLU: (x W1) ⊙ sigmoid(x W1) ⊙ (x W3), then project back
        x1 = self.W1(x)
        x2 = self.W3(x)
        silu = x1 * torch.sigmoid(x1)
        return self.W2(silu * x2)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # the angle Θ_(i,k) = i / Θ^[(2k−2)/d] for k ∈ {1, ..., d/2}
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = einsum(t, inv_freq, "i, j -> i j") # (max_seq_len, d_k/2)

        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        cos = self.cos[token_positions] # (..., seq_len, d_k/2)
        sin = self.sin[token_positions] # (..., seq_len, d_k/2)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
        return out.flatten(-2)


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

def scaled_dot_product_attention(Q, K, V, mask=None):
    """ 
    Q, K: (batch_size,  ..., seq_len, d_k)
    V: (batch_size, ..., seq_len, d_v)
    mask: (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k) # (batch_size, ..., seq_len, seq_len)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) # PyTorch broadcasts mask as (1, ..., seq_len, seq_len)

    attn = softmax(scores, dim=-1)
    return attn @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        q_proj_weight: torch.Tensor | None = None,
        k_proj_weight: torch.Tensor | None = None,
        v_proj_weight: torch.Tensor | None = None,
        o_proj_weight: torch.Tensor | None = None,
        use_rope: bool = False,
        rope_theta: float | None = None, 
        max_seq_len: int | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        """
        Implement causal multi-head self-attention as a torch.nn.Module.
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        Folllowing Vaswani et al. [2017], set d_k = d_v = d_model/h
        """
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if use_rope:
            self.rope = RoPE(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None
        
        def _init_proj(proj_weight: torch.Tensor | None) -> Linear:
            proj = Linear(d_model, d_model, device=device, dtype=dtype)
            if proj_weight is not None:
                proj.load_state_dict({"W": proj_weight.T})
            return proj

        # initialize Q, K, V, O projection weights
        self.q_proj = _init_proj(q_proj_weight)
        self.k_proj = _init_proj(k_proj_weight)
        self.v_proj = _init_proj(v_proj_weight)
        self.o_proj = _init_proj(o_proj_weight)
        

    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        b, s, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # change QKV from (batch_size, seq_len, d_model) to (b, h, s, d_k)
        Q = rearrange(Q, "b s (h d) -> b h s d", h=self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h=self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h=self.num_heads)

        # apply RoPE to Q and K
        if self.rope is not None and token_positions is not None:
            token_positions = token_positions.to(torch.long)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
        # causal_keep = ~mask
        causal_keep = torch.tril(torch.ones(s, s, device=x.device, dtype=torch.bool))

        out = scaled_dot_product_attention(Q, K, V, causal_keep) # (b, h, s, d)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        use_rope: bool = False,
        rope_theta: float | None = None, 
        max_seq_len: int | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        if token_positions is None:
            b, s, _ = x.shape
            token_positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """
    At minimum, this implementation should accept all the aforementioned construction parameters for the Transformerblock,
    as well as these additional parameters:
    vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
    context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
    num_layers: int The number of Transformer blocks to use.
    """
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        use_rope: bool = False,
        rope_theta: float | None = None, 
        max_seq_len: int | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.tok_emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.use_rope = use_rope
        if not use_rope:
            self.pos_emb = nn.Parameter(torch.zeros(context_len, d_model, device=device, dtype=dtype))
        else:
            self.pos_emb = None

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff = d_ff,
                eps=eps,
                use_rope=use_rope,
                rope_theta=rope_theta,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])

        self.ln_f = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor):
        x = self.tok_emb(token_ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb[:x.size(1)]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.out_proj(x)
            