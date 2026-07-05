# Kernel summary template

| Kernel | Implementation status | Correctness test | Benchmark script | Main bottleneck |
|---|---|---|---|---|
| Residual Add | Done | tests/test_small_ops.py | benchmarks/bench_small_ops.py | memory bandwidth / launch overhead |
| RMSNorm | Done | tests/test_small_ops.py | benchmarks/bench_small_ops.py | reduction + memory bandwidth |
| Fused Residual RMSNorm | Done | tests/test_small_ops.py | benchmarks/bench_small_ops.py | reduction + memory bandwidth |
| SwiGLU | Done | tests/test_small_ops.py | benchmarks/bench_small_ops.py | elementwise memory-bound |
| RoPE | Done | tests/test_small_ops.py | benchmarks/bench_small_ops.py | memory-bound |
| FlashAttention-style Attention | Prototype | tests/test_flash_attention.py | benchmarks/bench_attention.py | IO / tiling / occupancy |
