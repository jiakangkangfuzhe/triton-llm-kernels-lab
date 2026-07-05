from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x shaped [B, S, H, D]."""
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    c = cos[None, :, None, :].float()
    s = sin[None, :, None, :].float()
    y_even = x1 * c - x2 * s
    y_odd = x1 * s + x2 * c
    y = torch.empty_like(x)
    y[..., 0::2] = y_even.to(x.dtype)
    y[..., 1::2] = y_odd.to(x.dtype)
    return y


if triton is not None:
    @triton.jit
    def _rope_kernel(x_ptr, cos_ptr, sin_ptr, y_ptr, total_pairs: tl.constexpr, seq_len: tl.constexpr, n_heads: tl.constexpr, head_dim: tl.constexpr, half_dim: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_pairs

        d_pair = offs % half_dim
        tmp = offs // half_dim
        h = tmp % n_heads
        tmp = tmp // n_heads
        sidx = tmp % seq_len
        b = tmp // seq_len

        base = ((b * seq_len + sidx) * n_heads + h) * head_dim + d_pair * 2
        x0 = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + sidx * half_dim + d_pair, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + sidx * half_dim + d_pair, mask=mask, other=0.0).to(tl.float32)
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c
        tl.store(y_ptr + base, y0, mask=mask)
        tl.store(y_ptr + base + 1, y1, mask=mask)


def rope_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not installed")
    if x.dim() != 4:
        raise ValueError("x must have shape [B, S, H, D]")
    bsz, seq_len, n_heads, head_dim = x.shape
    half_dim = head_dim // 2
    if cos.shape != (seq_len, half_dim) or sin.shape != (seq_len, half_dim):
        raise ValueError("cos/sin shape mismatch")
    y = torch.empty_like(x)
    total_pairs = bsz * seq_len * n_heads * half_dim
    _rope_kernel[(triton.cdiv(total_pairs, block_size),)](
        x, cos, sin, y, total_pairs, seq_len, n_heads, head_dim, half_dim, BLOCK_SIZE=block_size
    )
    return y
