#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.flash_attention_tiled import attention_torch_naive, attention_torch_sdpa, attention_triton_tiled
from utils.timing import benchmark_ms, dtype_from_name, max_abs_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--head-dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("CUDA GPU is required")
    dtype = dtype_from_name(args.dtype)
    rows = []

    for s in args.seq_lens:
        for d in args.head_dims:
            q = torch.randn((args.batch, args.heads, s, d), device=args.device, dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            ref = attention_torch_sdpa(q, k, v, causal=args.causal)
            tri = attention_triton_tiled(q, k, v, causal=args.causal, block_m=16, block_n=64)
            err = max_abs_err(ref, tri)

            # Naive attention is expensive for long S; still useful as a baseline at small S.
            naive_ms = None
            if s <= 1024:
                naive_ms = benchmark_ms(attention_torch_naive, q, k, v, args.causal, warmup=args.warmup, repeat=args.repeat)
            sdpa_ms = benchmark_ms(attention_torch_sdpa, q, k, v, args.causal, warmup=args.warmup, repeat=args.repeat)
            tri_ms = benchmark_ms(attention_triton_tiled, q, k, v, args.causal, warmup=args.warmup, repeat=args.repeat)
            rows.append([
                f"B{args.batch}H{args.heads}S{s}D{d}",
                "causal" if args.causal else "noncausal",
                "-" if naive_ms is None else f"{naive_ms:.4f}",
                f"{sdpa_ms:.4f}",
                f"{tri_ms:.4f}",
                f"{sdpa_ms/tri_ms:.2f}x",
                f"{err:.3e}",
            ])

    print(tabulate(rows, headers=["shape", "mask", "naive_ms", "sdpa_ms", "triton_ms", "sdpa/triton", "max_err"], tablefmt="github"))


if __name__ == "__main__":
    main()
