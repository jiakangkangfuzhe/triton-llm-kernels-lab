# Attention benchmark template

Fill this file after running:

```bash
python benchmarks/bench_attention.py --dtype fp16 --device cuda --causal
```

Environment:

- GPU:
- CUDA:
- PyTorch:
- Triton:
- dtype:
- warmup:
- repeat:

| shape | mask | naive_ms | sdpa_ms | triton_ms | sdpa/triton | max_err |
|---|---|---:|---:|---:|---:|---:|
| B1H16S512D64 | causal | TBD | TBD | TBD | TBD | TBD |
| B1H16S1024D64 | causal | TBD | TBD | TBD | TBD | TBD |
| B1H16S2048D128 | causal | TBD | TBD | TBD | TBD | TBD |

Notes:

- Naive attention materializes the SxS matrix and becomes expensive for long sequences.
- PyTorch SDPA may already dispatch optimized kernels.
- This repo's Triton kernel is an educational FlashAttention-style prototype.
