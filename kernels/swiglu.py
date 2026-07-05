from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def swiglu_torch(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


if triton is not None:
    @triton.jit
    def _swiglu_kernel(g_ptr, u_ptr, y_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (g / (1.0 + tl.exp(-g))) * u
        tl.store(y_ptr + offs, y, mask=mask)


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not installed")
    if gate.shape != up.shape:
        raise ValueError(f"shape mismatch: {gate.shape} vs {up.shape}")
    y = torch.empty_like(gate)
    n = gate.numel()
    _swiglu_kernel[(triton.cdiv(n, block_size),)](gate, up, y, n, BLOCK_SIZE=block_size)
    return y
