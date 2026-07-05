from __future__ import annotations

import pytest
import torch

from kernels.flash_attention_tiled import attention_torch_sdpa, attention_triton_tiled

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("seq_len", [16, 64])
@pytest.mark.parametrize("head_dim", [32, 64])
def test_flash_attention_style(seq_len: int, head_dim: int):
    q = torch.randn((1, 2, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ref = attention_torch_sdpa(q, k, v, causal=True)
    tri = attention_triton_tiled(q, k, v, causal=True, block_m=16, block_n=32)
    torch.testing.assert_close(tri, ref, rtol=5e-2, atol=5e-2)
