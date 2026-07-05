#!/usr/bin/env bash
set -euo pipefail
python benchmarks/bench_small_ops.py --dtype fp16 --device cuda --warmup 30 --repeat 100
python benchmarks/bench_attention.py --dtype fp16 --device cuda --causal --warmup 20 --repeat 50
python benchmarks/bench_decode_path.py --dtype fp16 --device cuda --warmup 30 --repeat 100
