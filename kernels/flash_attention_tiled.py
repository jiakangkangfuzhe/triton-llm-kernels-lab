from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def attention_torch_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Reference attention for [B, H, S, D]. Materializes SxS scores."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        s = q.shape[-2]
        mask = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def attention_torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


if triton is not None:
    @triton.jit
    def _flash_attn_fwd_kernel_bh(
        q_ptr, k_ptr, v_ptr, o_ptr,
        seq_len: tl.constexpr, n_heads: tl.constexpr, head_dim: tl.constexpr,
        scale: tl.constexpr, causal: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // n_heads
        h = bh - b * n_heads

        offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        # Contiguous layout: [B, H, S, D]
        q_base = ((b * n_heads + h) * seq_len) * head_dim
        k_base = q_base
        v_base = q_base
        o_base = q_base

        q = tl.load(q_ptr + q_base + offs_m[:, None] * head_dim + offs_d[None, :],
                    mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
        q = q.to(tl.float32)

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            cols = start_n + offs_n
            k = tl.load(k_ptr + k_base + cols[:, None] * head_dim + offs_d[None, :],
                        mask=(cols[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0).to(tl.float32)
            qk = tl.dot(q, tl.trans(k)) * scale

            # Mask invalid keys and future positions for causal attention.
            qk = tl.where(cols[None, :] < seq_len, qk, -float("inf"))
            if causal:
                qk = tl.where(cols[None, :] <= offs_m[:, None], qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)

            v = tl.load(v_ptr + v_base + cols[:, None] * head_dim + offs_d[None, :],
                        mask=(cols[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0).to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(p, v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        out = acc / l_i[:, None]
        tl.store(o_ptr + o_base + offs_m[:, None] * head_dim + offs_d[None, :], out,
                 mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim))


def attention_triton_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    block_m: int = 16,
    block_n: int = 64,
    block_d: int | None = None,
) -> torch.Tensor:
    """Forward-only FlashAttention-style tiled attention.

    Args:
        q, k, v: contiguous tensors with shape [B, H, S, D].

    This is an educational kernel. It supports head_dim <= BLOCK_D and assumes
    contiguous [B, H, S, D] layout. It avoids materializing [S, S] attention.
    """
    if triton is None:
        raise RuntimeError("triton is not installed")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")
    if q.dim() != 4:
        raise ValueError("expected [B, H, S, D]")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    bsz, n_heads, seq_len, head_dim = q.shape
    block_d = block_d or triton.next_power_of_2(head_dim)
    if block_d < head_dim:
        raise ValueError("block_d must be >= head_dim")
    if block_d > 128:
        raise ValueError("this simple kernel is intended for head_dim <= 128")
    o = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, block_m), bsz * n_heads)
    _flash_attn_fwd_kernel_bh[grid](
        q, k, v, o,
        seq_len, n_heads, head_dim,
        1.0 / math.sqrt(head_dim), causal,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return o
