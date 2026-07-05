# Small ops benchmark template

Fill this file after running:

```bash
python benchmarks/bench_small_ops.py --dtype fp16 --device cuda
```

Environment:

- GPU:
- CUDA:
- PyTorch:
- Triton:
- dtype:
- warmup:
- repeat:

| op | shape | torch_ms | triton_ms | speedup | GB/s | max_err |
|---|---:|---:|---:|---:|---:|---:|
| residual_add | TBD | TBD | TBD | TBD | TBD | TBD |
| rmsnorm | TBD | TBD | TBD | TBD | TBD | TBD |
| fused_res_rms | TBD | TBD | TBD | TBD | TBD | TBD |
| swiglu | TBD | TBD | TBD | TBD | TBD | TBD |
| rope | TBD | TBD | TBD | TBD | TBD | TBD |

Notes:

- Short decode-like shapes may be launch-overhead dominated.
- RMSNorm and residual add are usually memory-bound.
