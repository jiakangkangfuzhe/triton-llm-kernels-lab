from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def fused_residual_rmsnorm_torch(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    added = x + residual
    var = added.float().pow(2).mean(dim=-1, keepdim=True)
    y = added.float() * torch.rsqrt(var + eps) * weight.float()
    return y.to(x.dtype), added


if triton is not None:
    @triton.jit
    def _fused_kernel(x_ptr, r_ptr, w_ptr, y_ptr, added_ptr, n_rows: tl.constexpr, hidden: tl.constexpr, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < hidden
        x = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        a = x + r
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        ss = tl.sum(a * a, axis=0) / hidden
        inv = tl.rsqrt(ss + eps)
        y = a * inv * w
        tl.store(y_ptr + row * hidden + offs, y, mask=mask)
        tl.store(added_ptr + row * hidden + offs, a, mask=mask)


def fused_residual_rmsnorm_triton(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_size: int | None = None):
    if triton is None:
        raise RuntimeError("triton is not installed")
    hidden = x.shape[-1]
    if block_size is None:
        block_size = triton.next_power_of_2(hidden)
    y = torch.empty_like(x)
    added = torch.empty_like(x)
    n_rows = x.numel() // hidden
    _fused_kernel[(n_rows,)](x, residual, weight, y, added, n_rows, hidden, eps, BLOCK_SIZE=block_size, num_warps=8)
    return y, added
