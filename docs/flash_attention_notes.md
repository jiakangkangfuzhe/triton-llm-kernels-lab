# FlashAttention-style tiled attention notes

## Problem with naive attention

For a sequence length `S`, naive attention materializes the full score matrix:

```text
scores = Q @ K^T       # [B, H, S, S]
probs  = softmax(scores)
out    = probs @ V
```

This is simple but expensive for long sequence lengths because the `S x S` matrix creates heavy HBM traffic and large temporary memory usage.

## Core idea

FlashAttention-style attention is IO-aware. Instead of materializing the full attention matrix, it:

1. Splits Q into row blocks.
2. Iterates over K/V column blocks.
3. Maintains online softmax statistics per query row.
4. Accumulates the final output block by block.
5. Avoids writing the full attention matrix to HBM.

## Online softmax update

For each query row, keep:

```text
m_i: running max logit
l_i: running normalization denominator
acc: running weighted value accumulator
```

When a new block of logits `qk` arrives:

```text
m_new = max(m_i, max(qk))
p     = exp(qk - m_new)
alpha = exp(m_i - m_new)
l_new = l_i * alpha + sum(p)
acc   = acc * alpha + p @ V_block
```

Finally:

```text
out = acc / l_i
```

## Causal mask

For decoder-only LLMs, token `i` cannot attend to future tokens `j > i`. In a tiled kernel, this mask is applied inside each `Q_block x K_block` tile:

```text
qk = where(k_pos <= q_pos, qk, -inf)
```

## What this repo implements

`kernels/flash_attention_tiled.py` implements a forward-only educational kernel for tensors shaped `[B, H, S, D]`. It is intended to make the core idea readable and benchmarkable.

Limitations:

- forward-only
- no dropout
- no backward kernel
- assumes contiguous layout
- intended for head_dim <= 128
- not a production replacement for FlashAttention-2/3

## Interview explanation

A safe explanation:

> I did not implement a full industrial FlashAttention-2/3 stack. I implemented a FlashAttention-style tiled forward prototype in Triton to understand block tiling, online softmax, causal masking, PV accumulation, and HBM/SRAM IO reduction. Then I compared it with naive attention and PyTorch SDPA using CUDA Event benchmarks.
