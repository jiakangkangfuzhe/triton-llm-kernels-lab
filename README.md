# triton-llm-kernels-lab

A learning-oriented LLM inference kernel lab for **Triton**, **Transformer decoder blocks**, and **FlashAttention-style tiled attention**.

This repository is designed for interview-oriented LLM Infra practice. It contains forward-only Triton kernels, PyTorch baselines, CUDA Event micro-benchmarks, correctness tests, and notes for analyzing LLM inference bottlenecks.

> Scope: educational / experimental kernels, not production replacements for FlashAttention, xFormers, or vendor-tuned CUDA kernels.

## What is included

```text
kernels/
  residual_add.py              # residual add kernel + torch baseline
  rmsnorm.py                   # RMSNorm kernel + torch baseline
  fused_residual_rmsnorm.py    # fused residual add + RMSNorm
  swiglu.py                    # SwiGLU elementwise gate kernel
  rope.py                      # RoPE kernel for [B, S, H, D]
  flash_attention_tiled.py     # FlashAttention-style tiled attention prototype

benchmarks/
  bench_small_ops.py           # RMSNorm / RoPE / SwiGLU / residual benchmark
  bench_attention.py           # naive attention / PyTorch SDPA / Triton tiled attention
  bench_decode_path.py         # simplified decode-path microbenchmark

tests/
  test_small_ops.py
  test_flash_attention.py

docs/
  flash_attention_notes.md
  prefill_decode_kernel_analysis.md
  benchmark_methodology.md

results/
  small_ops_a100_fp16_template.md
  attention_a100_fp16_template.md
  kernel_summary_template.md
```

## Project focus

1. Implement LLM decoder-block operators in Triton.
2. Compare with PyTorch baselines for correctness and latency.
3. Use CUDA Event timing for stable GPU measurements.
4. Analyze memory-bound kernels such as RMSNorm, residual add, RoPE, and SwiGLU.
5. Implement a **FlashAttention-style tiled attention prototype** to understand tiling, online softmax, causal masking, and HBM/SRAM IO reduction.
6. Connect kernel-level observations to LLM inference stages: **prefill**, **decode**, **KV Cache**, and serving latency metrics.

## Installation

```bash
conda create -n triton-kernels python=3.10 -y
conda activate triton-kernels
pip install -r requirements.txt
```

Recommended environment:

- NVIDIA GPU with CUDA
- PyTorch with CUDA
- Triton >= 2.3
- A100 / H100 recommended for meaningful benchmark numbers

## Quick correctness tests

```bash
pytest -q tests/
```

If CUDA or Triton is unavailable, tests are skipped automatically.

## Run benchmarks

```bash
python benchmarks/bench_small_ops.py --dtype fp16 --device cuda
python benchmarks/bench_attention.py --dtype fp16 --device cuda --causal
python benchmarks/bench_decode_path.py --dtype fp16 --device cuda
```

Useful flags:

```bash
python benchmarks/bench_attention.py --seq-lens 512 1024 2048 --head-dims 64 128 --batch 1 --heads 16 --repeat 100
```

## Important notes

- `flash_attention_tiled.py` is an educational forward-only implementation.
- It implements the key idea of block-wise attention with online softmax and avoids materializing the full `S x S` attention matrix.
- It is not a full FlashAttention-2/3 reimplementation.
- Benchmark results depend on GPU, CUDA version, PyTorch version, Triton version, shape, dtype, warmup, and repeat count.
- Fill `results/*.md` with your own measured values before citing performance numbers.

## Resume mapping

This repo supports the resume project:

> 基于 Triton 的 LLM 推理核心算子与 FlashAttention-style Attention 优化实践

Suggested description:

- Implemented Triton kernels for LLM decoder-block operators including Residual Add, RMSNorm, Fused Residual RMSNorm, SwiGLU, RoPE, and a FlashAttention-style tiled attention prototype.
- Built CUDA Event benchmark scripts to compare latency, speedup, bandwidth, and max error against PyTorch baselines.
- Analyzed memory-bound operators, launch overhead, block size / num warps tuning, and prefill/decode bottlenecks in LLM inference.

## Repository status

This repository is intended to be continuously updated with:

- More benchmark shapes
- A100 / H100 result tables
- More optimized attention variants
- Decode attention with KV Cache layout experiments
- Tensor-core friendly matmul experiments
