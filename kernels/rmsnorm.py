from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(var + eps)
    return (y * weight.float()).to(x.dtype)


if triton is not None:
    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows: tl.constexpr, hidden: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden
        x = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(x * x, axis=0) / hidden
        inv = tl.rsqrt(ss + eps)
        y = x * inv * w
        tl.store(y_ptr + row * hidden + offs, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_size: int | None = None) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not installed")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")
    hidden = x.shape[-1]
    if weight.numel() != hidden:
        raise ValueError("weight size must match hidden size")
    if block_size is None:
        block_size = triton.next_power_of_2(hidden)
    if block_size > 131072:
        raise ValueError("hidden dimension is too large for this simple kernel")
    y = torch.empty_like(x)
    n_rows = x.numel() // hidden
    _rmsnorm_kernel[(n_rows,)](x, weight, y, n_rows, hidden, eps, BLOCK_SIZE=block_size, num_warps=8)
    return y
