# Prefill / Decode kernel analysis

## Prefill

Prefill processes the full prompt. Query length is usually large, so attention and MLP matmuls dominate. Important factors:

- prompt length
- batch size
- head dimension
- attention implementation
- KV Cache write bandwidth
- tensor parallel communication, if any

For prefill, FlashAttention-style kernels matter because the attention matrix can be large.

## Decode

Decode generates one or a few new tokens per step. Query length is usually 1, but key/value length grows with context length. Bottlenecks shift:

- KV Cache read bandwidth
- small GEMM efficiency
- kernel launch overhead
- scheduler overhead under high concurrency
- batching strategy

For decode, isolated tiny kernels may not be fast because launch overhead dominates. Fusion and batching are often more important.

## Kernel-level implications

| Component | Prefill | Decode |
|---|---|---|
| RMSNorm | memory-bound | launch-overhead / memory-bound |
| RoPE | memory-bound | small-shape overhead |
| Attention | compute + IO heavy | KV Cache read heavy |
| SwiGLU | matmul dominates; elementwise gate memory-bound | small-shape overhead |
| Residual Add | memory-bound | launch-overhead dominated |

## Serving metrics connection

- High TTFT often points to prefill bottlenecks.
- High TPOT often points to decode bottlenecks.
- Long prompts stress prefill and KV Cache allocation.
- Long outputs stress decode scheduling and KV Cache reads.
- P95/P99 latency can be affected by batching, queueing, and memory fragmentation.
