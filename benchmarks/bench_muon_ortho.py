from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parameter_golf_kernels.muon_ortho import (  # noqa: E402
    zeropower_via_newtonschulz5,
    zeropower_via_polarexpress5,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) / iters * 1000.0


def _polar_exact(g: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(g.float(), full_matrices=False)
    return u @ vh


def _orthogonality_error(x: torch.Tensor) -> float:
    xf = x.float()
    if xf.size(-2) >= xf.size(-1):
        gram = xf.mT @ xf
        eye = torch.eye(xf.size(-1), device=xf.device, dtype=xf.dtype)
    else:
        gram = xf @ xf.mT
        eye = torch.eye(xf.size(-2), device=xf.device, dtype=xf.dtype)
    err = (gram - eye).norm(dim=(-2, -1)) / math.sqrt(eye.size(0))
    return float(err.mean().item())


def _relative_error(x: torch.Tensor, ref: torch.Tensor) -> float:
    xf = x.float()
    err = (xf - ref).norm(dim=(-2, -1))
    denom = ref.norm(dim=(-2, -1)).clamp_min(1e-8)
    return float((err / denom).mean().item())


def _shard_batch(batch: int, world_size: int = 8) -> int:
    return ((batch + world_size - 1) // world_size) * world_size // world_size


def _shapes() -> list[tuple[str, tuple[int, int, int]]]:
    model_dim = 512
    num_layers = 11
    num_heads = 8
    num_kv_heads = 4
    mlp_mult = 3.5
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    mlp_dim = int(model_dim * mlp_mult)
    return [
        ("qo shard", (_shard_batch(2 * num_layers), model_dim, model_dim)),
        ("kv shard", (_shard_batch(2 * num_layers), kv_dim, model_dim)),
        ("mlp_up shard", (_shard_batch(num_layers), mlp_dim, model_dim)),
        ("mlp_down shard", (_shard_batch(num_layers), model_dim, mlp_dim)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Muon orthogonalization backends on exact shard shapes.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--compile", action="store_true", help="Enable compiled orthogonalization paths.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_muon_ortho.py")

    if args.compile:
        os.environ["MUON_COMPILE_NS5"] = "1"
        os.environ["MUON_COMPILE_POLAR"] = "1"
        os.environ.setdefault("COMPILE_ENABLED", "1")
    else:
        os.environ["MUON_COMPILE_NS5"] = "0"
        os.environ["MUON_COMPILE_POLAR"] = "0"

    torch.manual_seed(0)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"compile_enabled={args.compile}")
    print()

    configs = [
        ("ns5_s5", lambda x: zeropower_via_newtonschulz5(x, steps=5)),
        ("ns5_s4", lambda x: zeropower_via_newtonschulz5(x, steps=4)),
        ("polar_s5", lambda x: zeropower_via_polarexpress5(x, steps=5)),
        ("polar_s4", lambda x: zeropower_via_polarexpress5(x, steps=4)),
    ]

    results: list[dict[str, object]] = []
    print(
        f"{'Shape':<14} {'Backend':<10} {'ms':>8} {'speedup':>8} {'rel_err':>10} {'ortho_err':>10} {'finite':>7}"
    )
    print("-" * 78)

    for shape_name, shape in _shapes():
        g = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        ref = _polar_exact(g)
        baseline_ms = None
        for backend_name, fn in configs:
            y = fn(g)
            finite = bool(torch.isfinite(y).all().item())
            ms = _time_ms(lambda: fn(g), args.warmup, args.iters)
            rel_err = _relative_error(y, ref)
            ortho_err = _orthogonality_error(y)
            if backend_name == "ns5_s5":
                baseline_ms = ms
            speedup = (baseline_ms / ms) if baseline_ms is not None else 1.0
            print(
                f"{shape_name:<14} {backend_name:<10} {ms:>8.3f} {speedup:>7.2f}x {rel_err:>10.5f} {ortho_err:>10.5f} {str(finite):>7}"
            )
            results.append(
                {
                    "shape_name": shape_name,
                    "shape": list(shape),
                    "backend": backend_name,
                    "ms": ms,
                    "speedup_vs_ns5_s5": speedup,
                    "relative_error_vs_exact_polar": rel_err,
                    "orthogonality_error": ortho_err,
                    "finite": finite,
                }
            )

    print()
    print(
        "JSON_SUMMARY="
        + json.dumps(
            {
                "compile_enabled": args.compile,
                "results": results,
            }
        )
    )


if __name__ == "__main__":
    main()
