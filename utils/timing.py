from __future__ import annotations

from typing import Callable

import torch


def benchmark_ms(fn: Callable, *args, warmup: int = 30, repeat: int = 100) -> float:
    """Measure average GPU latency with CUDA Events.

    CUDA kernels are launched asynchronously, so wall-clock timing without
    synchronization is misleading. This helper records start/end CUDA events and
    returns elapsed time in milliseconds.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark_ms")

    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def dtype_from_name(name: str) -> torch.dtype:
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")
