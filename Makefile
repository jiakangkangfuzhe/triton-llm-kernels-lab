.PHONY: test bench-small bench-attn bench-decode

test:
	pytest -q tests/

bench-small:
	python benchmarks/bench_small_ops.py --dtype fp16 --device cuda

bench-attn:
	python benchmarks/bench_attention.py --dtype fp16 --device cuda --causal

bench-decode:
	python benchmarks/bench_decode_path.py --dtype fp16 --device cuda
