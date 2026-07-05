#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kernels.fused_residual_rmsnorm import fused_residual_rmsnorm_torch, fused_residual_rmsnorm_triton
from kernels.residual_add import residual_add_torch, residual_add_triton
from kernels.rmsnorm import rmsnorm_torch, rmsnorm_triton
from kernels.rope import build_rope_cache, rope_torch, rope_triton
from kernels.swiglu import swiglu_torch, swiglu_triton
from utils.timing import benchmark_ms, dtype_from_name, max_abs_err


def gbps(num_bytes: int, ms: float) -> float:
    return num_bytes / (ms / 1000.0) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1, 128, 1024])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("CUDA GPU is required for Triton benchmarks")

    dtype = dtype_from_name(args.dtype)
    rows = []

    for s in args.seq_lens:
        shape = (args.batch, s, args.hidden)
        x = torch.randn(shape, device=args.device, dtype=dtype)
        r = torch.randn_like(x)
        w = torch.randn(args.hidden, device=args.device, dtype=dtype)

        # Residual add.
        y_ref = residual_add_torch(x, r)
        y_tri = residual_add_triton(x, r)
        torch_ms = benchmark_ms(residual_add_torch, x, r, warmup=args.warmup, repeat=args.repeat)
        triton_ms = benchmark_ms(residual_add_triton, x, r, warmup=args.warmup, repeat=args.repeat)
        nbytes = x.numel() * x.element_size() * 3
        rows.append(["residual_add", str(list(shape)), f"{torch_ms:.4f}", f"{triton_ms:.4f}", f"{torch_ms/triton_ms:.2f}x", f"{gbps(nbytes, triton_ms):.2f}", f"{max_abs_err(y_ref, y_tri):.3e}"])

        # RMSNorm.
        y_ref = rmsnorm_torch(x, w)
        y_tri = rmsnorm_triton(x, w)
        torch_ms = benchmark_ms(rmsnorm_torch, x, w, warmup=args.warmup, repeat=args.repeat)
        triton_ms = benchmark_ms(rmsnorm_triton, x, w, warmup=args.warmup, repeat=args.repeat)
        nbytes = x.numel() * x.element_size() * 2 + w.numel() * w.element_size()
        rows.append(["rmsnorm", str(list(shape)), f"{torch_ms:.4f}", f"{triton_ms:.4f}", f"{torch_ms/triton_ms:.2f}x", f"{gbps(nbytes, triton_ms):.2f}", f"{max_abs_err(y_ref, y_tri):.3e}"])

        # Fused residual + RMSNorm.
        y_ref, a_ref = fused_residual_rmsnorm_torch(x, r, w)
        y_tri, a_tri = fused_residual_rmsnorm_triton(x, r, w)
        torch_ms = benchmark_ms(fused_residual_rmsnorm_torch, x, r, w, warmup=args.warmup, repeat=args.repeat)
        triton_ms = benchmark_ms(fused_residual_rmsnorm_triton, x, r, w, warmup=args.warmup, repeat=args.repeat)
        nbytes = x.numel() * x.element_size() * 4 + w.numel() * w.element_size()
        err = max(max_abs_err(y_ref, y_tri), max_abs_err(a_ref, a_tri))
        rows.append(["fused_res_rms", str(list(shape)), f"{torch_ms:.4f}", f"{triton_ms:.4f}", f"{torch_ms/triton_ms:.2f}x", f"{gbps(nbytes, triton_ms):.2f}", f"{err:.3e}"])

        # SwiGLU.
        gate = torch.randn(shape, device=args.device, dtype=dtype)
        up = torch.randn_like(gate)
        y_ref = swiglu_torch(gate, up)
        y_tri = swiglu_triton(gate, up)
        torch_ms = benchmark_ms(swiglu_torch, gate, up, warmup=args.warmup, repeat=args.repeat)
        triton_ms = benchmark_ms(swiglu_triton, gate, up, warmup=args.warmup, repeat=args.repeat)
        nbytes = gate.numel() * gate.element_size() * 3
        rows.append(["swiglu", str(list(shape)), f"{torch_ms:.4f}", f"{triton_ms:.4f}", f"{torch_ms/triton_ms:.2f}x", f"{gbps(nbytes, triton_ms):.2f}", f"{max_abs_err(y_ref, y_tri):.3e}"])

        # RoPE [B, S, H, D]
        head_dim = args.hidden // args.heads
        x4 = torch.randn((args.batch, s, args.heads, head_dim), device=args.device, dtype=dtype)
        cos, sin = build_rope_cache(s, head_dim, x4.device, dtype)
        y_ref = rope_torch(x4, cos, sin)
        y_tri = rope_triton(x4, cos, sin)
        torch_ms = benchmark_ms(rope_torch, x4, cos, sin, warmup=args.warmup, repeat=args.repeat)
        triton_ms = benchmark_ms(rope_triton, x4, cos, sin, warmup=args.warmup, repeat=args.repeat)
        nbytes = x4.numel() * x4.element_size() * 2 + cos.numel() * cos.element_size() * 2
        rows.append(["rope", str(list(x4.shape)), f"{torch_ms:.4f}", f"{triton_ms:.4f}", f"{torch_ms/triton_ms:.2f}x", f"{gbps(nbytes, triton_ms):.2f}", f"{max_abs_err(y_ref, y_tri):.3e}"])

    print(tabulate(rows, headers=["op", "shape", "torch_ms", "triton_ms", "speedup", "GB/s", "max_err"], tablefmt="github"))


if __name__ == "__main__":
    main()
