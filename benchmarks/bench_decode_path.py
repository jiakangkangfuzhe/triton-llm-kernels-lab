#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.rmsnorm import rmsnorm_torch, rmsnorm_triton
from kernels.rope import build_rope_cache, rope_torch, rope_triton
from kernels.swiglu import swiglu_torch, swiglu_triton
from utils.timing import benchmark_ms, dtype_from_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--seq", type=int, default=1, help="decode step usually has query length = 1")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("CUDA GPU is required")
    dtype = dtype_from_name(args.dtype)
    x = torch.randn((args.batch, args.seq, args.hidden), device=args.device, dtype=dtype)
    w = torch.randn(args.hidden, device=args.device, dtype=dtype)
    gate = torch.randn_like(x)
    up = torch.randn_like(x)
    head_dim = args.hidden // args.heads
    x4 = torch.randn((args.batch, args.seq, args.heads, head_dim), device=args.device, dtype=dtype)
    cos, sin = build_rope_cache(args.seq, head_dim, x4.device, dtype)

    rows = []
    for name, torch_fn, triton_fn, args_tuple in [
        ("rmsnorm", rmsnorm_torch, rmsnorm_triton, (x, w)),
        ("rope", rope_torch, rope_triton, (x4, cos, sin)),
        ("swiglu", swiglu_torch, swiglu_triton, (gate, up)),
    ]:
        torch_ms = benchmark_ms(torch_fn, *args_tuple, warmup=args.warmup, repeat=args.repeat)
        triton_ms = benchmark_ms(triton_fn, *args_tuple, warmup=args.warmup, repeat=args.repeat)
        rows.append([name, f"{torch_ms:.4f}", f"{triton_ms:.4f}", f"{torch_ms/triton_ms:.2f}x"])

    print(tabulate(rows, headers=["decode_path_op", "torch_ms", "triton_ms", "speedup"], tablefmt="github"))
    print("\nNote: short decode shapes are often launch-overhead dominated; fusion is usually more important than isolated kernels.")


if __name__ == "__main__":
    main()
