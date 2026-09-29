"""Rotary position embeddings (RoPE) for attention head tensors."""

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be positive and even")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inv_freq.to(device=device))
        angles = torch.repeat_interleave(frequencies, repeats=2, dim=-1)
        return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate pairs in [batch, heads, time, head_dim]."""
    if x.ndim != 4 or x.size(-1) % 2:
        raise ValueError("Expected [B, H, T, even head_dim]")
    if cos.shape != (x.size(-2), x.size(-1)) or sin.shape != cos.shape:
        raise ValueError("RoPE cosine/sine shapes do not match x")
    even = x[..., ::2]
    odd = x[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
    return x * cos[None, None, :, :] + rotated * sin[None, None, :, :]
