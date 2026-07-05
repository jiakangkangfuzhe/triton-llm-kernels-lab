from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def residual_add_torch(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return x + residual


if triton is not None:
    @triton.jit
    def _residual_add_kernel(x_ptr, r_ptr, y_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        r = tl.load(r_ptr + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, x + r, mask=mask)


def residual_add_triton(x: torch.Tensor, residual: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not installed")
    if not x.is_cuda or not residual.is_cuda:
        raise ValueError("x and residual must be CUDA tensors")
    if x.shape != residual.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {residual.shape}")
    y = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)
    _residual_add_kernel[grid](x, residual, y, n, BLOCK_SIZE=block_size)
    return y
