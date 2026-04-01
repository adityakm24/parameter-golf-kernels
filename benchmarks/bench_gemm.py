from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parameter_golf_kernels.triton_matmul import (  # noqa: E402
    PARAMETER_GOLF_GEMM_SHAPES,
    TRITON_AVAILABLE,
    TRITON_IMPORT_ERROR,
    triton_mm,
)


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _tflops(m: int, k: int, n: int, ms: float) -> float:
    flops = 2.0 * m * k * n
    return flops / (ms / 1000.0) / 1e12


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact Parameter Golf GEMM shapes.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per backend.")
    parser.add_argument("--iters", type=int, default=50, help="Measured iterations per backend.")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip one-time correctness check between cuBLAS and Triton outputs.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_gemm.py")

    device_name = torch.cuda.get_device_name(0)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"device={device_name}")
    print(f"triton_available={TRITON_AVAILABLE}")
    if not TRITON_AVAILABLE:
        print(f"triton_import_error={TRITON_IMPORT_ERROR}")
    print()
    print(
        f"{'Name':<15} {'cuBLAS ms':>10} {'Triton ms':>10} {'Speedup':>8} "
        f"{'cuBLAS TF':>10} {'Triton TF':>10} {'Valid':>8}"
    )
    print("-" * 82)

    results: list[dict[str, object]] = []

    for name, m, k, n in PARAMETER_GOLF_GEMM_SHAPES:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        cublas_ms = _time_ms(lambda: torch.mm(a, b), args.warmup, args.iters)
        cublas_tf = _tflops(m, k, n, cublas_ms)

        valid = None
        triton_ms = None
        triton_tf = None
        speedup = None

        if TRITON_AVAILABLE:
            if not args.skip_validation:
                ref = torch.mm(a, b)
                out = triton_mm(a, b)
                valid = bool(torch.allclose(ref, out, atol=1e-2, rtol=1e-2))
            triton_ms = _time_ms(lambda: triton_mm(a, b), args.warmup, args.iters)
            triton_tf = _tflops(m, k, n, triton_ms)
            speedup = cublas_ms / triton_ms

        print(
            f"{name:<15} {cublas_ms:>9.3f} "
            f"{(f'{triton_ms:>9.3f}' if triton_ms is not None else 'n/a'):>10} "
            f"{(f'{speedup:>7.2f}x' if speedup is not None else 'n/a'):>8} "
            f"{cublas_tf:>9.0f} "
            f"{(f'{triton_tf:>9.0f}' if triton_tf is not None else 'n/a'):>10} "
            f"{str(valid) if valid is not None else 'n/a':>8}"
        )

        results.append(
            {
                "name": name,
                "shape": [m, k, n],
                "cublas_ms": cublas_ms,
                "cublas_tflops": cublas_tf,
                "triton_ms": triton_ms,
                "triton_tflops": triton_tf,
                "speedup": speedup,
                "valid": valid,
            }
        )

    print()
    print("JSON_SUMMARY=" + json.dumps({"device": device_name, "results": results}))


if __name__ == "__main__":
    main()
