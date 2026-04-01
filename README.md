# parameter-golf-kernels

Standalone custom-kernel and optimizer-microbench experiments extracted from a parameter-golf training stack.

This repo intentionally excludes model checkpoints, datasets, and training artifacts. It only contains kernel code, orthogonalization backends, and the benchmark harnesses used to evaluate them.

## Included work

- Triton BF16 matmul for the exact projection and MLP GEMM shapes used in the training stack.
- Muon orthogonalization backends: Newton-Schulz 5-step, Polar Express 5-step, and 4-step ablations.
- Reproduction scripts for the benchmark numbers below.

## Approaches tried

1. Autotuned Triton matmul on exact training shapes.
   Result: slower than `torch.mm` backed by cuBLAS on every measured shape, so it was not wired into training.
2. Muon backend swap from Newton-Schulz 5-step to Polar Express 5-step.
   Result: Polar Express 5-step stayed roughly latency-neutral while improving approximation quality across every shard that was measured.
3. Four-step Muon ablations (`ns5_s4` and `polar_s4`).
   Result: these were faster, but the orthogonality and relative-error regressions were large enough that they were not the preferred drop-in choice.
4. Python-side cuBLASLt / `nvmath` probing.
   Result: attempted, but the target environment never produced a clean, benchmarkable import path, so there are no publishable timing numbers for that path in this repo.

## Repo layout

- `src/parameter_golf_kernels/triton_matmul.py`: Triton matmul kernel plus the exact GEMM shapes that were benchmarked.
- `src/parameter_golf_kernels/muon_ortho.py`: Newton-Schulz 5-step and Polar Express orthogonalization backends, including optional `torch.compile` wrapping.
- `benchmarks/bench_gemm.py`: cuBLAS vs Triton exact-shape GEMM benchmark.
- `benchmarks/bench_muon_ortho.py`: Muon orthogonalization benchmark on exact optimizer shard shapes.

## Environment notes

- Per the current PyTorch docs, stable PyTorch requires Python 3.10 or newer.
- Per the current Triton docs, binary wheels are available for CPython 3.10-3.14.
- This repo does not ship a lockfile because the correct PyTorch wheel index depends on the CUDA version installed on the target machine.

## Quick start

```bash
uv venv
source .venv/bin/activate

# Pick the wheel index that matches your machine from:
# https://pytorch.org/get-started/locally/
uv add --default-index https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# Triton docs show `pip install triton`; translated here to uv:
uv add triton

uv run benchmarks/bench_gemm.py
uv run benchmarks/bench_muon_ortho.py --compile
```

If your machine uses a different CUDA wheel, swap the `cu128` index for the current selector output from the PyTorch install page.

## Benchmark setup

- GPU: 1x `NVIDIA H100 80GB HBM3`
- GEMM benchmark: BF16 inputs, 5 warmup iterations, 50 measured iterations, one-time output validation against `torch.mm`
- Muon benchmark: compiled paths enabled, BF16 inputs, 5 warmup iterations, 50 measured iterations
- Muon shard-shape source: `d_model=512`, `n_layers=11`, `n_heads=8`, `n_kv_heads=4`, `mlp_mult=3.5`

## Exact GEMM shapes benchmarked

All GEMM rows use `(M, K, N)`:

| Name | Shape | cuBLAS ms | Triton ms | cuBLAS/Triton | cuBLAS TFLOP/s | Triton TFLOP/s | Valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q/Out proj | `(98304, 512, 512)` | 0.087 | 0.115 | 0.759x | 592.4 | 449.9 | true |
| K/V proj | `(98304, 512, 256)` | 0.056 | 0.070 | 0.803x | 457.7 | 367.8 | true |
| MLP up | `(98304, 512, 1536)` | 0.246 | 0.299 | 0.822x | 628.4 | 516.4 | true |
| MLP down | `(98304, 1536, 512)` | 0.214 | 0.258 | 0.830x | 722.1 | 599.6 | true |
| LM head | `(98304, 512, 1024)` | 0.162 | 0.208 | 0.781x | 634.7 | 495.9 | true |

Takeaway: the straightforward Triton kernel hit `0.759x-0.830x` of cuBLAS on the exact shapes we cared about, so it was kept as a benchmark artifact instead of a training replacement.

## Muon orthogonalization results

`relative_error_vs_exact_polar` uses exact SVD-based polar decomposition as the reference. Lower is better. `orthogonality_error` is also lower-is-better.

### 5-step candidates

| Shape | NS5 ms | Polar ms | Polar speedup vs NS5 | NS5 rel err | Polar rel err | NS5 ortho err | Polar ortho err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qo shard (3, 512, 512)` | 0.308 | 0.311 | 0.990x | 0.193 | 0.117 | 0.323 | 0.202 |
| `kv shard (3, 256, 512)` | 0.311 | 0.309 | 1.006x | 0.174 | 0.104 | 0.313 | 0.216 |
| `mlp_up shard (2, 1792, 512)` | 0.321 | 0.328 | 0.979x | 0.150 | 0.093 | 0.276 | 0.181 |
| `mlp_down shard (2, 512, 1792)` | 0.307 | 0.316 | 0.973x | 0.150 | 0.093 | 0.276 | 0.180 |

Takeaway: `polar_s5` was latency-neutral in practice (`0.973x-1.006x` vs `ns5_s5`) while consistently improving both relative error and orthogonality.

### 4-step ablations

| Shape | NS5 s4 ms | NS5 s4 speedup | NS5 s4 rel err | NS5 s4 ortho err | Polar s4 ms | Polar s4 speedup | Polar s4 rel err | Polar s4 ortho err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qo shard (3, 512, 512)` | 0.266 | 1.156x | 0.271 | 0.419 | 0.268 | 1.149x | 0.391 | 0.812 |
| `kv shard (3, 256, 512)` | 0.267 | 1.168x | 0.203 | 0.353 | 0.267 | 1.166x | 0.382 | 0.822 |
| `mlp_up shard (2, 1792, 512)` | 0.285 | 1.126x | 0.204 | 0.355 | 0.280 | 1.145x | 0.367 | 0.773 |
| `mlp_down shard (2, 512, 1792)` | 0.260 | 1.181x | 0.204 | 0.355 | 0.263 | 1.167x | 0.368 | 0.775 |

Takeaway: the 4-step variants bought `1.126x-1.181x` speedups, but the quality drop was large enough that they remained ablations rather than a default.

## Notes

- These are microbenchmarks, not end-to-end training throughput numbers.
- The code here is intentionally standalone and trimmed to the kernel experiments only.
- No model files are included.

## References

- GitHub CLI repo creation: <https://cli.github.com/manual/gh_repo_create>
- PyTorch install selector: <https://pytorch.org/get-started/locally/>
- Triton installation docs: <https://triton-lang.org/main/getting-started/installation.html>
