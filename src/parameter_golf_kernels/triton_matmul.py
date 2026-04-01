from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - environment dependent
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = repr(exc)
    TRITON_AVAILABLE = False
else:
    TRITON_IMPORT_ERROR = ""
    TRITON_AVAILABLE = True


PARAMETER_GOLF_GEMM_SHAPES: list[tuple[str, int, int, int]] = [
    ("Q/Out proj", 98_304, 512, 512),
    ("K/V proj", 98_304, 512, 256),
    ("MLP up", 98_304, 512, 1_536),
    ("MLP down", 98_304, 1_536, 512),
    ("LM head", 98_304, 512, 1_024),
]


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BM": 128, "BN": 256, "BK": 64, "G": 8}, num_stages=3, num_warps=8),
            triton.Config({"BM": 256, "BN": 128, "BK": 64, "G": 8}, num_stages=3, num_warps=8),
            triton.Config({"BM": 128, "BN": 128, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
            triton.Config({"BM": 128, "BN": 128, "BK": 32, "G": 8}, num_stages=4, num_warps=4),
            triton.Config({"BM": 64, "BN": 256, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
            triton.Config({"BM": 256, "BN": 64, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
            triton.Config({"BM": 64, "BN": 128, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
            triton.Config({"BM": 128, "BN": 64, "BK": 64, "G": 8}, num_stages=4, num_warps=4),
            triton.Config({"BM": 256, "BN": 256, "BK": 32, "G": 8}, num_stages=3, num_warps=8),
            triton.Config({"BM": 256, "BN": 256, "BK": 64, "G": 4}, num_stages=3, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _matmul_kernel(
        A,
        B,
        C,
        M,
        N,
        K,
        sAM,
        sAK,
        sBK,
        sBN,
        sCM,
        sCN,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        G: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_bm = tl.cdiv(M, BM)
        num_bn = tl.cdiv(N, BN)
        num_in_group = G * num_bn
        gid = pid // num_in_group
        first = gid * G
        gsz = min(num_bm - first, G)
        pid_m = first + ((pid % num_in_group) % gsz)
        pid_n = (pid % num_in_group) // gsz

        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)

        a_ptrs = A + (rm[:, None] * sAM + rk[None, :] * sAK)
        b_ptrs = B + (rk[:, None] * sBK + rn[None, :] * sBN)

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs, mask=rk[None, :] < K, other=0.0)
            b = tl.load(b_ptrs, mask=rk[:, None] < K, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK * sAK
            b_ptrs += BK * sBK
            rk += BK

        c = acc.to(tl.bfloat16)
        rm2 = pid_m * BM + tl.arange(0, BM)
        rn2 = pid_n * BN + tl.arange(0, BN)
        c_ptrs = C + (rm2[:, None] * sCM + rn2[None, :] * sCN)
        mask = (rm2[:, None] < M) & (rn2[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)


def triton_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("triton_mm expects rank-2 tensors")
    m, k = a.shape
    kb, n = b.shape
    if k != kb:
        raise ValueError(f"Incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")

    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


__all__ = [
    "PARAMETER_GOLF_GEMM_SHAPES",
    "TRITON_AVAILABLE",
    "TRITON_IMPORT_ERROR",
    "triton_mm",
]
