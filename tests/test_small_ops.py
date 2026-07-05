from __future__ import annotations

import pytest
import torch

from kernels.fused_residual_rmsnorm import fused_residual_rmsnorm_torch, fused_residual_rmsnorm_triton
from kernels.residual_add import residual_add_torch, residual_add_triton
from kernels.rmsnorm import rmsnorm_torch, rmsnorm_triton
from kernels.rope import build_rope_cache, rope_torch, rope_triton
from kernels.swiglu import swiglu_torch, swiglu_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _dtype():
    return torch.float16


def test_residual_add():
    x = torch.randn((2, 4, 128), device="cuda", dtype=_dtype())
    r = torch.randn_like(x)
    torch.testing.assert_close(residual_add_triton(x, r), residual_add_torch(x, r), rtol=1e-3, atol=1e-3)


def test_rmsnorm():
    x = torch.randn((2, 8, 256), device="cuda", dtype=_dtype())
    w = torch.randn(256, device="cuda", dtype=_dtype())
    torch.testing.assert_close(rmsnorm_triton(x, w), rmsnorm_torch(x, w), rtol=2e-3, atol=2e-3)


def test_fused_residual_rmsnorm():
    x = torch.randn((2, 8, 256), device="cuda", dtype=_dtype())
    r = torch.randn_like(x)
    w = torch.randn(256, device="cuda", dtype=_dtype())
    y_tri, a_tri = fused_residual_rmsnorm_triton(x, r, w)
    y_ref, a_ref = fused_residual_rmsnorm_torch(x, r, w)
    torch.testing.assert_close(a_tri, a_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(y_tri, y_ref, rtol=2e-3, atol=2e-3)


def test_swiglu():
    gate = torch.randn((2, 8, 512), device="cuda", dtype=_dtype())
    up = torch.randn_like(gate)
    torch.testing.assert_close(swiglu_triton(gate, up), swiglu_torch(gate, up), rtol=2e-3, atol=2e-3)


def test_rope():
    x = torch.randn((2, 16, 8, 64), device="cuda", dtype=_dtype())
    cos, sin = build_rope_cache(16, 64, x.device, _dtype())
    torch.testing.assert_close(rope_triton(x, cos, sin), rope_torch(x, cos, sin), rtol=2e-3, atol=2e-3)
