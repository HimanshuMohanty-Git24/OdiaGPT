"""From-scratch causal multi-head self-attention."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .rope import RotaryEmbedding, apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.backend = config.attention_backend
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_base)
        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, time, width = x.shape
        if time > self.causal_mask.size(0):
            raise ValueError("Sequence exceeds configured context length")
        qkv = self.qkv(x).reshape(batch, time, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = self.rope(time, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.backend == "sdpa":
            result = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(
                ~self.causal_mask[:time, :time], torch.finfo(scores.dtype).min
            )
            weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
            weights = self.attn_dropout(weights)
            result = weights @ v
        result = result.transpose(1, 2).contiguous().reshape(batch, time, width)
        return self.resid_dropout(self.output(result))
