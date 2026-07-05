# Benchmark methodology

## Why CUDA Event

CUDA kernels are asynchronous. Timing with `time.time()` without synchronization measures CPU launch overhead more than GPU runtime. This repo uses CUDA Events:

1. warm up kernels
2. synchronize
3. record start event
4. run repeated calls
5. record end event
6. synchronize
7. divide elapsed time by repeat count

## Metrics

- `latency`: average kernel runtime in milliseconds
- `speedup`: baseline latency / Triton latency
- `GB/s`: approximate memory bandwidth for memory-bound kernels
- `max_err`: maximum absolute error compared with PyTorch baseline

## Caveats

- Small shapes can be launch-overhead dominated.
- Results vary across GPUs and software versions.
- PyTorch SDPA may dispatch optimized kernels internally.
- Triton kernels in this repo are educational, not fully tuned production kernels.

## Recommended result format

```text
GPU: NVIDIA A100 40GB
CUDA: xx.x
PyTorch: x.x.x
Triton: x.x.x
Dtype: fp16
Warmup: 30
Repeat: 100
```
